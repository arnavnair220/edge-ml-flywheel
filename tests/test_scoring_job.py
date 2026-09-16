"""Unit tests for the `CreateProcessingJob` request and the sets it is built over.

`test_training_job`'s argument for the other job a cycle runs: the request is a
dictionary nothing type-checks at the boundary, so the properties worth stating
are the ones a job would otherwise discover by running and producing the wrong
thing.

`TestTheLabelWall` is the one that matters most. This job's whole claim is that
it reads images and no boxes, and IAM is only half of that -- a policy says what
a role may reach, and these say the request never asks. An absence is exactly the
kind of property that survives being written down and then quietly stops holding.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from edge_ml_flywheel.conventions import (
    SCORED_COHORTS,
    Buckets,
    Cohort,
    Cycle,
    ImageId,
    ModelArtifact,
    PartitionVersion,
    RunId,
    Seed,
    detections_prefix,
    model_artifact_key,
    new_model_version,
    scoring_manifest_key,
    training_code_key,
)
from edge_ml_flywheel.evaluation.match import MAX_DETS
from edge_ml_flywheel.oracle.cohorts import Cohorts
from edge_ml_flywheel.scoring import cohorts as sets
from edge_ml_flywheel.scoring import job
from edge_ml_flywheel.training import job as training

BUCKETS = Buckets.for_account("123456789012")
REGION = "us-east-1"
ROLE = "arn:aws:iam::123456789012:role/edge-ml-flywheel-scoring"
RUN = RunId("20260812t143355z-v1-uncertainty")
ATTEMPT = datetime(2026, 9, 9, 14, 33, 55, tzinfo=UTC)

V0 = PartitionVersion(0)


def an_image_id(index: int) -> ImageId:
    """A BDD100K-shaped ID: two 8-character hex groups."""
    return ImageId(f"{index:08x}-{index ^ 0x5F5E0FF:08x}")


def a_target(cycle: int = 1, seed: int = 1) -> job.Target:
    return job.Target(
        buckets=BUCKETS,
        region=REGION,
        role_arn=ROLE,
        version=new_model_version(RUN, Cycle(cycle)),
        seed=Seed(seed),
    )


def a_request(cycle: int = 1, seed: int = 1, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "target": a_target(cycle, seed),
        "scoring": job.Scoring(),
        "compute": job.Compute(),
        "attempt": ATTEMPT,
    }
    return job.processing_job(**{**arguments, **overrides})


def channel(request: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next(
        (entry for entry in request["ProcessingInputs"] if entry["InputName"] == name), None
    )


def output(request: dict[str, Any], name: str) -> dict[str, Any] | None:
    outputs = request["ProcessingOutputConfig"]["Outputs"]
    return next((entry for entry in outputs if entry["OutputName"] == name), None)


def flag(request: dict[str, Any], name: str) -> str:
    arguments = request["AppSpecification"]["ContainerArguments"]
    return arguments[arguments.index(name) + 1]


class TestTheLabelWall:
    def test_no_channel_reaches_a_label_prefix(self) -> None:
        """Not `raw/labels/`, not a cohort's boxes, not a purchase. The role's
        policy denies all three; this says the job never asks for them."""
        forbidden = ("raw/labels", "labels/cohort=", "derived/purchases")
        for entry in a_request()["ProcessingInputs"]:
            uri = entry["S3Input"]["S3Uri"]
            assert not any(prefix in uri for prefix in forbidden), uri

    def test_the_only_data_bucket_reads_are_images(self) -> None:
        """A manifest names keys under the image prefix and nothing else, so the
        one bucket holding ground truth is reached for pixels alone."""
        for entry in a_request()["ProcessingInputs"]:
            uri = entry["S3Input"]["S3Uri"]
            if BUCKETS.data in uri:
                assert "raw/images/" in uri or "/scoring/" in uri, uri

    def test_it_writes_only_detections(self) -> None:
        """Narrower than the cycle prefix: a scoring job has no reason to be able
        to overwrite a gate report or a model filed under the same cycle."""
        for entry in a_request()["ProcessingOutputConfig"]["Outputs"]:
            assert "/detections/" in entry["S3Output"]["S3Uri"]


class TestChannels:
    def test_each_cohort_arrives_as_its_own_manifest(self) -> None:
        """Two documents rather than one, because a manifest names keys below one
        prefix and `eval` draws from `val` while `pool` draws from `train`."""
        for cohort in SCORED_COHORTS:
            source = channel(a_request(cycle=2), cohort.value)
            assert source is not None, cohort
            s3 = source["S3Input"]

            assert s3["S3DataType"] == "ManifestFile"
            assert s3["S3Uri"].endswith(scoring_manifest_key(RUN, Cycle(2), cohort))

    def test_the_model_channel_names_the_checkpoint_and_not_the_seed_prefix(self) -> None:
        """The seed prefix also holds `_sagemaker/`, whose tarball is a second
        copy of the same weights. A prefix channel pointed one level up would
        download both and leave the container choosing between them."""
        source = channel(a_request(cycle=3, seed=2), job.MODEL_CHANNEL)
        assert source is not None

        version = new_model_version(RUN, Cycle(3))
        expected = model_artifact_key(version, Seed(2), ModelArtifact.TORCH)
        assert source["S3Input"]["S3Uri"].endswith(expected)
        assert source["S3Input"]["S3Uri"].endswith(ModelArtifact.TORCH.value)

    def test_the_code_is_the_archive_the_cycle_trained_from(self) -> None:
        """One archive per cycle, unpacked by both jobs, so the code that scored
        a model is the tree that trained it."""
        source = channel(a_request(cycle=4), job.CODE_CHANNEL)
        assert source is not None
        assert source["S3Input"]["S3Uri"].endswith(training_code_key(RUN, Cycle(4)))

    def test_every_channel_is_file_mode(self) -> None:
        for entry in a_request()["ProcessingInputs"]:
            assert entry["S3Input"]["S3InputMode"] == "File"

    def test_each_cohort_writes_to_its_own_prefix(self) -> None:
        version = new_model_version(RUN, Cycle(1))
        for cohort in SCORED_COHORTS:
            entry = output(a_request(cycle=1, seed=1), cohort.value)
            assert entry is not None, cohort
            assert entry["S3Output"]["S3Uri"].endswith(detections_prefix(version, Seed(1), cohort))

    def test_the_upload_waits_for_the_job_to_finish(self) -> None:
        """A partially uploaded detections file is one a later step reads as a
        complete answer, and there is nothing to watch in progress."""
        for entry in a_request()["ProcessingOutputConfig"]["Outputs"]:
            assert entry["S3Output"]["S3UploadMode"] == "EndOfJob"


class TestTheContainerCommand:
    def test_it_unpacks_installs_and_runs_in_that_order(self) -> None:
        """A Processing job has no framework toolkit, so the three steps script
        mode performs invisibly for the training job are written out here."""
        script = job.container_entrypoint()[2]
        assert script.index("tar xzf") < script.index("pip install") < script.index("exec python")

    def test_it_fails_on_the_first_failed_step(self) -> None:
        """Without `-e`, a failed unpack is a job that goes on to report an
        import error several minutes after the real fault."""
        assert job.container_entrypoint()[2].startswith("set -euo pipefail")

    def test_the_trailing_word_is_what_makes_the_arguments_arrive(self) -> None:
        """`bash -c <script> <name>` sets `$0`, which is what leaves `"$@"` to
        expand to the arguments SageMaker appends rather than eating the first."""
        command = job.container_entrypoint()
        assert command[:2] == ["bash", "-c"]
        assert command[3] == job.ENTRY_POINT
        assert '"$@"' in command[2]


class TestArguments:
    def test_every_value_is_a_string(self) -> None:
        arguments = a_request()["AppSpecification"]["ContainerArguments"]
        assert all(isinstance(value, str) for value in arguments)

    def test_the_run_and_cycle_are_not_passed_beside_the_version(self) -> None:
        """They are inside it, for `training.job`'s reason."""
        arguments = a_request(cycle=3)["AppSpecification"]["ContainerArguments"]
        assert flag(a_request(cycle=3), "--version") == new_model_version(RUN, Cycle(3))
        assert "--run_id" not in arguments
        assert "--cycle" not in arguments

    def test_the_resolution_is_the_one_the_model_was_trained_at(self) -> None:
        """A detector run at a resolution it was not trained at is a worse
        detector for a reason that appears nowhere in the report."""
        assert int(flag(a_request(), "--image_size")) == training.Recipe(epochs=1).image_size


class TestScoring:
    def test_the_confidence_floor_is_low_enough_to_sweep_the_curve(self) -> None:
        """AP integrates the low-confidence tail, so a floor at something that
        looks like a detection would truncate it and lower every number the
        project reports."""
        assert job.Scoring().confidence_floor < 0.05

    def test_a_floor_of_zero_is_refused(self) -> None:
        with pytest.raises(ValueError, match="0 is not a floor"):
            job.Scoring(confidence_floor=0.0)

    def test_the_detection_cap_is_the_one_the_match_cache_is_built_at(self) -> None:
        """A job emitting more rows would write boxes no metric can read back."""
        assert job.Scoring().max_detections == MAX_DETS

    def test_a_cap_above_the_cache_is_refused(self) -> None:
        with pytest.raises(ValueError, match="match cache is built"):
            job.Scoring(max_detections=MAX_DETS + 1)


class TestTheImage:
    def test_it_is_the_pinned_training_container(self) -> None:
        """Processing overrides the entry point, so the toolkit it ships is
        unused -- and reusing the tag means one pinned container in the project
        rather than a second that can drift a torch version from the weights."""
        image = a_request()["AppSpecification"]["ImageUri"]
        assert image.endswith(f":{training.DLC_TAG}")
        assert REGION in image


class TestTheImagesToScore:
    """`scoring.cohorts`: what each manifest names, before anything writes one."""

    def _cohorts(self, pool: int = 5, evaluation: int = 3) -> Cohorts:
        of_image = {an_image_id(index): Cohort.POOL for index in range(pool)}
        of_image |= {an_image_id(100 + index): Cohort.EVAL for index in range(evaluation)}
        of_image[an_image_id(200)] = Cohort.BOOTSTRAP
        return Cohorts(partition_version=V0, of_image=of_image)

    def test_the_pool_is_what_is_left_after_the_purchases(self) -> None:
        bought = [an_image_id(0), an_image_id(1)]
        remaining = sets.to_score(self._cohorts(), Cohort.POOL, bought)

        assert remaining == (an_image_id(2), an_image_id(3), an_image_id(4))

    def test_the_eval_cohort_is_the_same_list_every_cycle(self) -> None:
        """Purchases subtract from the pool and cannot touch this, because the
        oracle refuses to sell an eval image and the two sets are disjoint."""
        index = self._cohorts()
        first = sets.to_score(index, Cohort.EVAL, [])
        later = sets.to_score(index, Cohort.EVAL, [an_image_id(0), an_image_id(1)])

        assert first == later == (an_image_id(100), an_image_id(101), an_image_id(102))

    def test_a_cohort_nothing_scores_has_no_image_list(self) -> None:
        """`bootstrap` is the training set, so a model's confidence over it
        measures nothing anyone acts on."""
        with pytest.raises(ValueError, match="not a scored cohort"):
            sets.to_score(self._cohorts(), Cohort.BOOTSTRAP, [])

    def test_a_pool_bought_to_exhaustion_is_refused_rather_than_empty(self) -> None:
        """An empty manifest is a job that fails on a channel error; this says
        the run has spent its pool, which is the thing that happened."""
        index = self._cohorts(pool=2)
        with pytest.raises(ValueError, match="nothing left to score"):
            sets.to_score(index, Cohort.POOL, [an_image_id(0), an_image_id(1)])

    def test_a_purchase_outside_the_pool_is_refused(self) -> None:
        """It subtracts from nothing, so the pool comes out the right size and
        the cycle proceeds -- while the training set holds a frame the partition
        calls `eval`."""
        with pytest.raises(ValueError, match="not in the pool"):
            sets.check_purchases(self._cohorts(), [an_image_id(0), an_image_id(100)])

    def test_the_refusal_names_the_cohort_the_image_is_actually_in(self) -> None:
        """A bare "not in the pool" sends someone looking for a typo; "in eval"
        says the oracle sold a frame it should never have been asked for."""
        with pytest.raises(ValueError, match="eval"):
            sets.check_purchases(self._cohorts(), [an_image_id(100)])

    def test_an_ordinary_ledger_passes(self) -> None:
        sets.check_purchases(self._cohorts(), [an_image_id(0), an_image_id(1)])
