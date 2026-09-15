"""Unit tests for the `CreateTrainingJob` request.

The request is a dictionary that nothing type-checks at the boundary -- SageMaker
either accepts it or returns a validation error minutes later -- so the
properties worth stating are the ones that would otherwise be discovered by a
job that ran and produced the wrong thing.

`TestNoResume` is the one that matters most. The absence of `CheckpointConfig`
is the interrupt policy (design section 3), and an absence is exactly the kind of
property that survives being written down and then quietly stops holding.
"""

from datetime import UTC, datetime
from typing import Any

import pytest

from edge_ml_flywheel.conventions import (
    Buckets,
    Cycle,
    ModelVersion,
    PartitionVersion,
    RunId,
    Seed,
    new_model_version,
    training_manifest_key,
)
from edge_ml_flywheel.training import job

BUCKETS = Buckets.for_account("123456789012")
REGION = "us-east-1"
ROLE = "arn:aws:iam::123456789012:role/edge-ml-flywheel-training"
RUN = RunId("20260812t143355z-v1-uncertainty")
ATTEMPT = datetime(2026, 9, 9, 14, 33, 55, tzinfo=UTC)

V0 = PartitionVersion(0)


def a_target(cycle: int = 1, seed: int = 1) -> job.Target:
    return job.Target(
        buckets=BUCKETS,
        region=REGION,
        role_arn=ROLE,
        version=new_model_version(RUN, Cycle(cycle)),
        seed=Seed(seed),
        partition_version=V0,
    )


def a_request(cycle: int = 1, seed: int = 1, **overrides: Any) -> dict[str, Any]:
    arguments: dict[str, Any] = {
        "target": a_target(cycle, seed),
        "recipe": job.Recipe(epochs=1),
        "compute": job.Compute(),
        "attempt": ATTEMPT,
    }
    return job.training_job(**{**arguments, **overrides})


def channel(request: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next(
        (entry for entry in request["InputDataConfig"] if entry["ChannelName"] == name), None
    )


class TestNoResume:
    def test_the_request_carries_no_checkpoint_config(self) -> None:
        """A spot job with nowhere to checkpoint restarts rather than resumes.

        The rule is a property of the job definition, so this is where it is
        enforced. Adding `CheckpointConfig` to make an interrupt cheaper would
        make seed k stop fixing the run, which is the thing the matched-seed
        comparison shares between champion and challenger.
        """
        assert "CheckpointConfig" not in a_request()

    def test_spot_is_on_and_the_wait_covers_the_run(self) -> None:
        request = a_request()
        assert request["EnableManagedSpotTraining"] is True
        assert (
            request["StoppingCondition"]["MaxWaitTimeInSeconds"]
            >= request["StoppingCondition"]["MaxRuntimeInSeconds"]
        )

    def test_on_demand_carries_no_wait_time(self) -> None:
        """`MaxWaitTimeInSeconds` is only meaningful for a spot job, and
        SageMaker refuses it on one that is not."""
        request = a_request(compute=job.Compute(use_spot=False))
        assert request["EnableManagedSpotTraining"] is False
        assert "MaxWaitTimeInSeconds" not in request["StoppingCondition"]

    def test_a_spot_wait_shorter_than_the_run_is_refused(self) -> None:
        with pytest.raises(ValueError, match="must cover its max runtime"):
            job.Compute(max_runtime_seconds=7200, max_wait_seconds=3600)


class TestChannels:
    def test_the_images_arrive_as_a_manifest(self) -> None:
        """Named object by object, because the labeled set is a scattered subset
        of one flat prefix holding all 70,000 train images."""
        source = channel(a_request(cycle=1), job.IMAGES_CHANNEL)
        assert source is not None
        s3 = source["DataSource"]["S3DataSource"]

        assert s3["S3DataType"] == "ManifestFile"
        assert s3["S3Uri"].endswith(training_manifest_key(RUN, Cycle(1)))
        assert BUCKETS.artifacts in s3["S3Uri"]

    def test_cycle_zero_has_no_purchases_channel(self) -> None:
        """Nothing has been bought yet, and SageMaker fails a job whose channel
        prefix matches no object."""
        assert channel(a_request(cycle=0), job.PURCHASES_CHANNEL) is None

    def test_a_later_cycle_takes_the_whole_run_of_purchases(self) -> None:
        """The cumulative labeled set as one prefix: every cycle before this one,
        because a cycle buys after it trains."""
        source = channel(a_request(cycle=4), job.PURCHASES_CHANNEL)
        assert source is not None

        s3_uri = source["DataSource"]["S3DataSource"]["S3Uri"]
        assert s3_uri.endswith(f"derived/purchases/run_id={RUN}/")
        assert "cycle=" not in s3_uri

    def test_the_bootstrap_channel_names_one_cohort(self) -> None:
        """`cohort=bootstrap/` and never the `labels/` subtree above it, which
        also holds the eval boxes every cycle is scored against."""
        source = channel(a_request(), job.BOOTSTRAP_CHANNEL)
        assert source is not None

        s3_uri = source["DataSource"]["S3DataSource"]["S3Uri"]
        assert s3_uri.endswith("labels/cohort=bootstrap/")
        assert "cohort=eval" not in s3_uri

    def test_every_channel_is_file_mode(self) -> None:
        """`File` copies the channel to disk once and every epoch reads disk."""
        request = a_request()
        assert request["AlgorithmSpecification"]["TrainingInputMode"] == "File"
        assert all(entry["InputMode"] == "File" for entry in request["InputDataConfig"])

    def test_nothing_reaches_for_a_withheld_label(self) -> None:
        """No channel addresses `raw/labels/`. The bucket policy and the role
        both deny it; this says the job never asks."""
        for entry in a_request()["InputDataConfig"]:
            assert "raw/labels" not in entry["DataSource"]["S3DataSource"]["S3Uri"]


class TestHyperParameters:
    def test_every_value_is_a_string(self) -> None:
        """SageMaker rejects a hyperparameter that is not, and an int here is the
        easiest way to write a request that only fails on the call."""
        assert all(isinstance(value, str) for value in a_request()["HyperParameters"].values())

    def test_the_run_and_cycle_are_not_passed_beside_the_version(self) -> None:
        """They are inside it. Three spellings of two facts is how a job comes to
        write its artifacts under a cycle it did not train."""
        parameters = a_request(cycle=3)["HyperParameters"]
        assert parameters["version"] == new_model_version(RUN, Cycle(3))
        assert "run_id" not in parameters
        assert "cycle" not in parameters

    def test_script_mode_is_pointed_at_the_cycle_it_belongs_to(self) -> None:
        parameters = a_request(cycle=2)["HyperParameters"]
        assert parameters["sagemaker_program"] == job.ENTRY_POINT
        assert parameters["sagemaker_submit_directory"].endswith(
            "cycle=002/training/sourcedir.tar.gz"
        )


class TestJobName:
    def test_it_reads_as_the_model_and_the_seed(self) -> None:
        version = new_model_version(RUN, Cycle(7))
        assert job.job_name(version, Seed(3), ATTEMPT) == f"{version}-s3-143355"

    def test_a_thirty_one_character_slug_fits_exactly(self) -> None:
        """Everything but the slug is 33 characters of the 63 SageMaker allows,
        at the widest cycle and seed a key can hold. This is the boundary, and
        the alternative to testing it is finding it on the cycle that cannot
        train."""
        longest = RunId(f"20260812t143355z-{'a' * 31}")
        name = job.job_name(new_model_version(longest, Cycle(999)), Seed(9), ATTEMPT)
        assert len(name) == job.MAX_JOB_NAME

    def test_a_slug_one_character_longer_is_refused_with_the_fix(self) -> None:
        """`RUN_SLUG_MAX_LEN` allows 32, which is one past what a job name holds.
        The refusal names the slug rather than the length, because shortening the
        slug is the only thing the operator can act on."""
        overlong = RunId(f"20260812t143355z-{'a' * 32}")
        with pytest.raises(ValueError, match="run slug is what to shorten"):
            job.job_name(new_model_version(overlong, Cycle(999)), Seed(9), ATTEMPT)

    def test_an_overlong_name_is_refused_rather_than_truncated(self) -> None:
        with pytest.raises(ValueError, match="over SageMaker's"):
            job.job_name(ModelVersion(f"{'a' * 70}-c001"), Seed(1), ATTEMPT)


class TestOutput:
    def test_sagemakers_tarball_lands_beside_the_seed_and_not_in_it(self) -> None:
        """SageMaker writes `<job name>/output/model.tar.gz` under whatever path
        it is given, so it is given an underscore-prefixed one: the artifacts
        `conventions` addresses stay the only things at the seed prefix, and Glue
        walks past the rest."""
        path = a_request(cycle=1, seed=2)["OutputDataConfig"]["S3OutputPath"]
        assert path.endswith(
            "models/version=20260812t143355z-v1-uncertainty-c001/seed=2/_sagemaker/"
        )

    def test_the_image_is_pinned(self) -> None:
        """A tag and not `latest`: the container is half of what a recipe
        version means."""
        image = a_request()["AlgorithmSpecification"]["TrainingImage"]
        assert image.endswith(f":{job.DLC_TAG}")
        assert REGION in image


class TestRecipe:
    def test_the_default_resolution_is_the_handicap_the_design_chose(self) -> None:
        assert job.Recipe(epochs=1).image_size == 416

    def test_a_resolution_the_model_cannot_take_is_refused(self) -> None:
        with pytest.raises(ValueError, match="multiple of 32"):
            job.Recipe(epochs=1, image_size=500)

    def test_zero_epochs_is_refused(self) -> None:
        with pytest.raises(ValueError, match="epochs must be positive"):
            job.Recipe(epochs=0)
