"""The evaluation request, and one run of the container that consumes it.

Two kinds of evidence, for two kinds of mistake.

`TestTheEvalWall` and the channel tests are `test_scoring_job`'s argument
inverted. That job's claim is that it reads no box; this one's is that it reads
exactly one prefix of boxes and no other, which is a harder thing to keep true --
a role with one label grant is the role somebody widens. An absence is exactly
the kind of property that survives being written down and then quietly stops
holding.

`TestTheJobEndToEnd` runs `entrypoint.main` against two buckets in moto, with its
inputs at the keys a cycle writes them to. Every piece it calls is already tested
on its own, so what this adds is the wiring the unit tests cannot reach: that the
job finds its inputs in the store, that the outputs land at the keys their
readers use, that a cycle with no champion reaches a verdict, and that a cycle
with one produces a delta with the sign the fixture was built to have. Those are
the failures that otherwise appear as a green job and an object nothing looks
for.
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
from moto import mock_aws

from edge_ml_flywheel import storage
from edge_ml_flywheel.conventions import (
    CLASS_SET,
    COHORT_SPLIT,
    Buckets,
    Cohort,
    Cycle,
    DetectionRow,
    ImageId,
    ModelVersion,
    PartitionVersion,
    Precision,
    RunId,
    Seed,
    cohort_labels_prefix,
    detections_key,
    detections_prefix,
    eval_matches_key,
    eval_metrics_key,
    eval_prefix,
    gate_report_key,
    gate_report_prefix,
    model_version_cycle,
    new_model_version,
    scoring_manifest_key,
    training_code_key,
)
from edge_ml_flywheel.evaluation import entrypoint, job
from edge_ml_flywheel.gates import Gate
from edge_ml_flywheel.ingest.labels import Box
from edge_ml_flywheel.partition import cohort_labels
from edge_ml_flywheel.scoring import detections as detection_rows
from edge_ml_flywheel.storage import StorageConfig
from edge_ml_flywheel.storage.s3 import S3Store, client
from edge_ml_flywheel.training import images
from edge_ml_flywheel.training import job as training

BUCKETS = Buckets.for_account("123456789012")
REGION = "us-east-1"
ROLE = "arn:aws:iam::123456789012:role/edge-ml-flywheel-evaluation"
RUN = RunId("20260812t143355z-v1-uncertainty")
ATTEMPT = datetime(2026, 9, 15, 14, 33, 55, tzinfo=UTC)

V0 = PartitionVersion(0)

CHALLENGER = new_model_version(RUN, Cycle(2))
CHAMPION = new_model_version(RUN, Cycle(1))


def a_target(**overrides: Any) -> job.Target:
    arguments: dict[str, Any] = {
        "buckets": BUCKETS,
        "region": REGION,
        "role_arn": ROLE,
        "version": CHALLENGER,
        "seeds": (Seed(1),),
        "partition_version": V0,
    }
    return job.Target(**{**arguments, **overrides})


def a_request(**overrides: Any) -> dict[str, Any]:
    return job.processing_job(target=a_target(**overrides), compute=job.Compute(), attempt=ATTEMPT)


def channel(request: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next(
        (entry for entry in request["ProcessingInputs"] if entry["InputName"] == name), None
    )


def output(request: dict[str, Any], name: str) -> dict[str, Any] | None:
    outputs = request["ProcessingOutputConfig"]["Outputs"]
    return next((entry for entry in outputs if entry["OutputName"] == name), None)


class TestTheEvalWall:
    """The one grant this role holds, and the three it must not."""

    def test_the_only_label_channel_is_the_eval_cohort(self) -> None:
        """`raw/labels/`, `cohort=bootstrap/` and `derived/purchases/` are denied
        in the role's own policy; this says the job never asks for them.

        Bootstrap and purchases matter as much as the withheld pool here. They
        are the model's training set, and a job that could read them could report
        a number measured over what the model learned from."""
        forbidden = ("raw/labels", "labels/cohort=bootstrap", "derived/purchases")
        for entry in a_request(champion=CHAMPION)["ProcessingInputs"]:
            uri = entry["S3Input"]["S3Uri"]
            assert not any(prefix in uri for prefix in forbidden), uri

    def test_the_eval_boxes_arrive_on_their_own_channel(self) -> None:
        source = channel(a_request(), job.LABELS_CHANNEL)
        assert source is not None
        assert source["S3Input"]["S3Uri"].endswith(cohort_labels_prefix(V0, Cohort.EVAL))

    def test_it_is_handed_no_checkpoint(self) -> None:
        """The job that holds ground truth never sees a model. Scoring already
        ran it, and the detections are what arrive here -- so the identity that
        could contaminate the eval and the identity that can read the answer key
        are not the same one."""
        for entry in a_request(champion=CHAMPION)["ProcessingInputs"]:
            assert "/models/" not in entry["S3Input"]["S3Uri"]

    def test_it_reads_no_image(self) -> None:
        """Pixels are scoring's business. A channel of 5,000 JPEGs here would be
        gigabytes copied for a job that never decodes one."""
        for entry in a_request()["ProcessingInputs"]:
            assert "raw/images/" not in entry["S3Input"]["S3Uri"]

    def test_it_writes_only_its_caches_and_its_verdict(self) -> None:
        """Narrower than the cycle prefix: an evaluation job produces a report,
        so it has no reason to overwrite a model or a selection record filed
        under the same cycle."""
        for entry in a_request()["ProcessingOutputConfig"]["Outputs"]:
            uri = entry["S3Output"]["S3Uri"]
            assert "/eval/" in uri or "/gates/" in uri, uri

    def test_the_pool_detections_are_not_read(self) -> None:
        """62,000 images of boxes that decide nothing here. The uncertainty
        ranking is selection's input, read by the step that spends the budget."""
        for entry in a_request()["ProcessingInputs"]:
            assert f"cohort={Cohort.POOL.value}" not in entry["S3Input"]["S3Uri"]


class TestChannels:
    def test_the_manifest_says_which_images_were_scored(self) -> None:
        """A prefix channel over one document, not the `ManifestFile` scoring was
        handed: what is wanted is the list, not the images it names."""
        source = channel(a_request(), job.MANIFEST_CHANNEL)
        assert source is not None
        assert source["S3Input"]["S3DataType"] == "S3Prefix"
        assert source["S3Input"]["S3Uri"].endswith(scoring_manifest_key(RUN, Cycle(2), Cohort.EVAL))

    def test_each_seed_brings_its_own_detections(self) -> None:
        request = a_request(seeds=(Seed(1), Seed(2)))
        for seed in (Seed(1), Seed(2)):
            source = channel(request, job.detections_channel(seed))
            assert source is not None, seed
            assert source["S3Input"]["S3Uri"].endswith(
                detections_prefix(CHALLENGER, seed, Cohort.EVAL)
            )

    def test_the_champion_arrives_as_one_prefix_of_cached_arrays(self) -> None:
        """Every seed's cache in one channel, under the cycle that produced the
        champion rather than this one -- a champion is re-compared without being
        re-scored."""
        source = channel(a_request(champion=CHAMPION), job.CHAMPION_CHANNEL)
        assert source is not None
        assert source["S3Input"]["S3Uri"].endswith(eval_prefix(CHAMPION))

    def test_a_first_cycle_has_no_champion_channel(self) -> None:
        """Absence rather than an empty prefix: SageMaker fails a job whose
        channel matches no object."""
        assert channel(a_request(), job.CHAMPION_CHANNEL) is None

    def test_the_code_is_the_archive_the_cycle_trained_from(self) -> None:
        source = channel(a_request(), job.CODE_CHANNEL)
        assert source is not None
        assert source["S3Input"]["S3Uri"].endswith(training_code_key(RUN, Cycle(2)))

    def test_every_channel_is_file_mode(self) -> None:
        for entry in a_request(champion=CHAMPION)["ProcessingInputs"]:
            assert entry["S3Input"]["S3InputMode"] == "File"

    def test_the_caches_and_the_report_land_where_their_keys_say(self) -> None:
        caches = output(a_request(), job.EVAL_OUTPUT)
        report = output(a_request(), job.GATES_OUTPUT)
        assert caches is not None and report is not None

        assert caches["S3Output"]["S3Uri"].endswith(eval_prefix(CHALLENGER))
        assert report["S3Output"]["S3Uri"].endswith(gate_report_prefix(RUN, Cycle(2)))

    def test_the_upload_waits_for_the_job_to_finish(self) -> None:
        """A partially uploaded report is one a later step reads as a complete
        verdict."""
        for entry in a_request()["ProcessingOutputConfig"]["Outputs"]:
            assert entry["S3Output"]["S3UploadMode"] == "EndOfJob"


class TestTheTarget:
    def test_a_job_over_no_seed_is_refused(self) -> None:
        with pytest.raises(ValueError, match="no seed"):
            a_target(seeds=())

    def test_a_repeated_seed_is_refused(self) -> None:
        """It would name one detections channel twice and weight that seed twice
        in the mean."""
        with pytest.raises(ValueError, match="appears twice"):
            a_target(seeds=(Seed(1), Seed(1)))

    def test_a_model_cannot_be_compared_against_itself(self) -> None:
        """A paired delta between one model and itself is zero by construction,
        which the gate would read as a challenger that failed to improve."""
        with pytest.raises(ValueError, match="cannot be compared against itself"):
            a_target(champion=CHALLENGER)

    def test_more_seeds_than_channels_is_refused_here(self) -> None:
        """SageMaker caps a Processing job at ten inputs, and an API error naming
        a limit does not name the seed list that hit it."""
        with pytest.raises(ValueError, match="input channels"):
            a_target(seeds=tuple(Seed(seed) for seed in range(1, 9)), champion=CHAMPION)


class TestTheContainerCommand:
    def test_it_runs_the_evaluation_entry_point(self) -> None:
        command = a_request()["AppSpecification"]["ContainerEntrypoint"]
        assert command[3] == job.ENTRY_POINT
        assert job.ENTRY_POINT in command[2]

    def test_it_is_the_cpu_build_of_the_cycles_container(self) -> None:
        """Nothing here loads a model, so the GPU image would be gigabytes of
        CUDA pulled for drivers nothing opens -- at the same framework version,
        so the pyarrow that reads a detections file is the one that wrote it."""
        image = a_request()["AppSpecification"]["ImageUri"]
        assert image.endswith(f":{training.DLC_CPU_TAG}")
        assert "-cpu-" in image
        assert REGION in image

    def test_it_runs_on_a_cpu_instance(self) -> None:
        assert not job.Compute().instance_type.startswith("ml.g")


class TestArguments:
    def test_every_value_is_a_string(self) -> None:
        for value in a_request(champion=CHAMPION)["AppSpecification"]["ContainerArguments"]:
            assert isinstance(value, str)

    def test_the_seeds_are_passed_as_a_list(self) -> None:
        arguments = a_request(seeds=(Seed(2), Seed(1)))["AppSpecification"]["ContainerArguments"]
        at = arguments.index("--seeds")
        assert arguments[at + 1 : at + 3] == ["1", "2"]

    def test_a_first_cycle_passes_no_champion(self) -> None:
        assert "--champion" not in a_request()["AppSpecification"]["ContainerArguments"]

    def test_no_threshold_is_an_argument(self) -> None:
        """They are the pre-declared promotion rule. An argument for one would be
        a way to gate two cycles of a run differently, and the report's record of
        the numbers it applied would stop being evidence."""
        arguments = a_request(champion=CHAMPION)["AppSpecification"]["ContainerArguments"]
        for flag in ("--min_mean_delta", "--resamples", "--confidence"):
            assert flag not in arguments


class TestTheEdgeMeasurement:
    """What the job is given so it can judge the artifact that ships.

    The size arrives as an argument rather than being read off the object,
    because this role holds no model grant at all -- `infra/evaluation.tf` keeps
    `models/*` off every statement, which is what lets the account answer "what
    could have contaminated the eval" with two identities and one answer each.
    """

    SIZE = 4 * 1024 * 1024

    def test_the_int8_boxes_arrive_on_their_own_channel(self) -> None:
        source = channel(a_request(artifact_bytes=self.SIZE), job.INT8_CHANNEL)
        assert source is not None
        assert source["S3Input"]["S3Uri"].endswith(
            detections_prefix(CHALLENGER, Seed(1), Cohort.EVAL, Precision.INT8)
        )

    def test_it_is_the_deployed_seed_that_is_measured(self) -> None:
        """Exactly one artifact ships, so exactly one is quantized. The lowest
        seed, matching `entrypoint.deployed_seed`."""
        source = channel(
            a_request(seeds=(Seed(3), Seed(1)), artifact_bytes=self.SIZE), job.INT8_CHANNEL
        )
        assert source is not None
        assert "seed=1/" in source["S3Input"]["S3Uri"]

    def test_the_size_is_passed_to_the_container(self) -> None:
        arguments = a_request(artifact_bytes=self.SIZE)["AppSpecification"]["ContainerArguments"]
        assert arguments[arguments.index("--artifact_bytes") + 1] == str(self.SIZE)

    def test_a_cycle_with_no_export_takes_neither_the_channel_nor_the_flag(self) -> None:
        """Both halves are absent together: no export means no int8 detections
        either, so the cycle reports no edge verdict rather than half of one."""
        request = a_request()

        assert channel(request, job.INT8_CHANNEL) is None
        assert "--artifact_bytes" not in request["AppSpecification"]["ContainerArguments"]

    def test_an_empty_artifact_is_refused_rather_than_measured(self) -> None:
        """Zero bytes is a failed upload, and gating on it would report a size
        pass for a file that is not a model."""
        with pytest.raises(ValueError, match="is not an artifact"):
            a_target(artifact_bytes=0)

    def test_the_channel_counts_against_the_input_cap(self) -> None:
        """SageMaker caps a Processing job at ten inputs, and the int8 channel is
        one of them -- so the seed ceiling drops by one when an edge verdict is
        being produced, and that is refused here rather than by an API error
        naming a limit without naming the seed list."""
        seeds = tuple(Seed(seed) for seed in range(1, 8))

        with pytest.raises(ValueError, match="input channels"):
            a_target(seeds=seeds, champion=CHAMPION, artifact_bytes=self.SIZE)


class TestOutputNames:
    """What the container calls each file, relative to its output channel.

    Derived from the key builders rather than spelled, so a job that succeeds
    cannot leave an object nothing looks for.
    """

    def test_the_metrics_name_completes_its_key(self) -> None:
        names = job.output_names(CHALLENGER, [Seed(1)])
        assert eval_prefix(CHALLENGER) + names["metrics"] == eval_metrics_key(CHALLENGER)

    def test_each_seeds_cache_completes_its_key(self) -> None:
        names = job.output_names(CHALLENGER, [Seed(1), Seed(2)])
        for seed in (Seed(1), Seed(2)):
            relative = names[f"matches-{seed}"]
            assert eval_prefix(CHALLENGER) + relative == eval_matches_key(CHALLENGER, seed)

    def test_the_report_name_completes_its_key(self) -> None:
        names = job.output_names(CHALLENGER, [Seed(1)])
        assert gate_report_prefix(RUN, Cycle(2)) + names["report"] == gate_report_key(RUN, Cycle(2))


# --- The container, against a channel tree on disk ----------------------------

IMAGES = 14

# Every class in every image, which is what keeps the collapse check out of the
# way: a class with no ground truth in eval is a hard failure by design, so a
# fixture missing one would fail every test here for the wrong reason.
CATEGORIES = CLASS_SET.names

# Boxes per class per image, varying by image between one and this. Varying is
# load-bearing for the bootstrap: with every image contributing identically,
# every resample returns the same AP and the band collapses onto the point
# estimate -- so the tests below would pass without the resampling having done
# anything.
MAX_SLOTS = 3


def an_image(position: int) -> ImageId:
    return ImageId(f"{position:08x}-{position:08x}")


def slots_in(position: int) -> int:
    return 1 + position % MAX_SLOTS


def boxes_for(position: int) -> list[Box]:
    """One to three boxes of every class, on a grid inside the native frame.

    Laid out so no two boxes overlap, which makes the greedy assignment
    unambiguous: a detection placed on a box matches that box and nothing else,
    so what a test changes is how many boxes were found rather than how well.
    """
    return [
        Box(
            category=category,
            x1=20.0 + column * 135 + position,
            y1=20.0 + slot * 220,
            x2=20.0 + column * 135 + position + 100,
            y2=20.0 + slot * 220 + 180,
        )
        for column, category in enumerate(CATEGORIES)
        for slot in range(slots_in(position))
    ]


LABELS = {an_image(position): boxes_for(position) for position in range(IMAGES)}


def found_boxes(image_id: ImageId, found: int) -> list[Box]:
    """The first `found` boxes *of each class*, which is what a model finding
    more of them looks like.

    Per class rather than over the flat list, because the flat list is ordered by
    class: taking a prefix of it would be a model that detects the first few
    classes and none of the rest, which the collapse check fails outright -- a
    different verdict from the one about how good a model is.
    """
    kept: list[Box] = []
    for category in CATEGORIES:
        of_class = [box for box in LABELS[image_id] if box.category == category]
        kept.extend(of_class[:found])
    return kept


@dataclass(frozen=True, slots=True)
class Stores:
    """The two buckets the job reads and writes, and scratch for fixtures."""

    data: S3Store
    artifacts: S3Store
    scratch: Path

    def put(self, store: S3Store, local: Path, key: str) -> None:
        store.upload(local, key)

    def json(self, key: str) -> dict[str, Any]:
        return dict(json.loads(self.artifacts.get(key)))


@pytest.fixture
def stores(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> Iterator[Stores]:
    """Both buckets in moto, configured the way a cluster configures the job.

    The credentials are fake for `test_base_weights`' reason: a test whose mock
    failed to engage would otherwise reach a real account.
    """
    for name, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_DEFAULT_REGION", REGION),
        (storage.DATA_BUCKET, BUCKETS.data),
        (storage.ARTIFACTS_BUCKET, BUCKETS.artifacts),
    ):
        monkeypatch.setenv(name, value)
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    monkeypatch.delenv(storage.ENDPOINT_URL, raising=False)
    monkeypatch.setattr(entrypoint, "WORK", tmp_path / "work")

    with mock_aws():
        config = StorageConfig(region=REGION)
        for bucket in (BUCKETS.data, BUCKETS.artifacts):
            client(config).create_bucket(Bucket=bucket)
        yield Stores(
            data=S3Store.connect(BUCKETS.data, config),
            artifacts=S3Store.connect(BUCKETS.artifacts, config),
            scratch=tmp_path / "fixtures",
        )


def write_channels(
    stores: Stores,
    *,
    found: int,
    seeds: tuple[Seed, ...] = (Seed(1),),
    named: Sequence[ImageId] | None = None,
    version: ModelVersion = CHALLENGER,
) -> None:
    """Everything one evaluation reads, in the buckets at the keys it reads them from.

    `found` is how many boxes of each class the model detected, which is what
    makes the sign of a delta known by construction: a challenger that finds more
    of them than the champion is better, and any band disagreeing with that is a
    bug rather than a close call.

    `named` overrides which images were scored, and the detections follow it --
    they are the output of a pass over exactly the manifest's images, so a
    fixture where the two disagree would be testing a scoring job that ignored
    the list it was handed. Shrinking it is the one way two cycles come to score
    different eval cohorts: a changed `max_images`.

    The labels always cover every image, because the cohort file is the whole
    frozen 5,000 whatever a cycle scored.
    """
    scored = sorted(LABELS) if named is None else sorted(named)
    local = stores.scratch / f"c{model_version_cycle(version)}"

    images.write(local / "manifest.json", BUCKETS.data, scored, COHORT_SPLIT[Cohort.EVAL])
    stores.put(
        stores.artifacts,
        local / "manifest.json",
        scoring_manifest_key(RUN, model_version_cycle(version), Cohort.EVAL),
    )

    cohort_labels.write_parquet(
        [
            cohort_labels.CohortLabel(image_id=image_id, boxes=tuple(boxes))
            for image_id, boxes in sorted(LABELS.items())
        ],
        local / "labels.parquet",
    )
    stores.put(
        stores.data,
        local / "labels.parquet",
        f"{cohort_labels_prefix(V0, Cohort.EVAL)}part-00000.parquet",
    )

    for seed in seeds:
        path = local / f"detections-{seed}.parquet"
        _write_detections(path, scored, found)
        stores.put(stores.artifacts, path, detections_key(version, seed, Cohort.EVAL))


def _write_detections(path: Path, scored: Sequence[ImageId], found: int) -> None:
    """One detections parquet, placed exactly on the boxes the model found.

    Exactly on the box, so it matches at every IoU threshold and what moves
    between two fixtures is how many boxes were found rather than how well.
    """
    detection_rows.write(
        [
            DetectionRow(
                image_id=image_id,
                category=box.category,
                x1=box.x1,
                y1=box.y1,
                x2=box.x2,
                y2=box.y2,
                score=0.9,
            )
            for image_id in scored
            for box in found_boxes(image_id, found)
        ],
        path,
    )


def run_container(
    version: ModelVersion,
    seeds: tuple[Seed, ...],
    champion: ModelVersion | None = None,
) -> None:
    """`entrypoint.main`, reading from and writing to the `stores` buckets."""
    argv = [
        "--version",
        str(version),
        "--partition_version",
        str(V0),
        "--seeds",
        *(str(seed) for seed in seeds),
    ]
    if champion is not None:
        argv.extend(("--champion", str(champion)))
    entrypoint.main(argv)


def stage_champion(stores: Stores, found: int, seeds: tuple[Seed, ...]) -> None:
    """Run the job once for the champion, which leaves its caches in the bucket.

    Built by the entry point rather than by hand, because that is what a real
    champion's cache is: the previous cycle's output of this same code. A fixture
    assembled another way would be testing the comparison against something no
    cycle ever writes.
    """
    write_channels(stores, found=found, seeds=seeds, version=CHAMPION)
    run_container(CHAMPION, seeds)


class TestTheJobEndToEnd:
    def _produced(self, stores: Stores) -> tuple[dict[str, Any], dict[str, Any]]:
        metrics = stores.json(eval_metrics_key(CHALLENGER))
        report = stores.json(gate_report_key(RUN, model_version_cycle(CHALLENGER)))
        return metrics, report

    def test_a_first_cycle_caches_its_arrays_and_reaches_a_verdict(self, stores: Stores) -> None:
        """No champion, so the verdict is the collapse check and the cache is the
        thing the next cycle will compare against."""
        write_channels(stores, found=2)
        run_container(CHALLENGER, (Seed(1),))

        assert stores.artifacts.exists(eval_matches_key(CHALLENGER, Seed(1)))

        metrics, report = self._produced(stores)
        assert metrics["images"] == IMAGES
        assert set(metrics["seeds"]) == {"1"}
        assert report["champion"] is None
        assert report["delta"] is None
        assert report["passed"]

    def test_the_files_land_at_the_keys_their_readers_use(self, stores: Stores) -> None:
        """The failure this catches is a job that succeeds and an object nothing
        looks for, which no unit test of either half can see."""
        write_channels(stores, found=2)
        run_container(CHALLENGER, (Seed(1),))

        assert set(stores.artifacts.keys(eval_prefix(CHALLENGER))) == {
            eval_metrics_key(CHALLENGER),
            eval_matches_key(CHALLENGER, Seed(1)),
        }
        assert set(stores.artifacts.keys(gate_report_prefix(RUN, Cycle(2)))) == {
            gate_report_key(RUN, Cycle(2))
        }

    def test_the_verdict_records_the_thresholds_it_applied(self, stores: Stores) -> None:
        """A verdict without the numbers behind it is not checkable six weeks
        later without a git checkout."""
        write_channels(stores, found=2)
        run_container(CHALLENGER, (Seed(1),))

        _, report = self._produced(stores)
        assert report["thresholds"]["min_mean_delta"] == 0.005
        assert [gate["gate"] for gate in report["gates"]] == [Gate.QUALITY.value]

    def test_the_two_unimplemented_gates_are_absent_rather_than_green(self, stores: Stores) -> None:
        """`edge` and `canary` read measurements nothing produces yet. A pass
        recorded for a check that never ran is the one entry in this document
        that would be a lie."""
        write_channels(stores, found=2)
        run_container(CHALLENGER, (Seed(1),))

        _, report = self._produced(stores)
        reported = {gate["gate"] for gate in report["gates"]}
        assert Gate.EDGE.value not in reported
        assert Gate.CANARY.value not in reported

    def test_a_better_challenger_clears_the_band(self, stores: Stores) -> None:
        """The champion finds one box per image and the challenger finds three,
        so the sign is known by construction."""
        write_channels(stores, found=3)
        stage_champion(stores, found=1, seeds=(Seed(1),))
        run_container(CHALLENGER, (Seed(1),), CHAMPION)

        _, report = self._produced(stores)
        assert report["delta"]["observed"] > 0
        assert report["delta"]["lower"] > 0
        assert report["passed"]

    def test_an_identical_challenger_does_not_promote(self, stores: Stores) -> None:
        """The A/A test's expected verdict, at the smallest scale that produces
        it: the same detections on both sides cancel exactly on every resample,
        so the delta is zero and a gate that promoted it would have a false
        positive."""
        write_channels(stores, found=2)
        stage_champion(stores, found=2, seeds=(Seed(1),))
        run_container(CHALLENGER, (Seed(1),), CHAMPION)

        _, report = self._produced(stores)
        assert report["delta"]["observed"] == pytest.approx(0.0)
        assert not report["passed"]
        assert "under the promotion threshold" in report["gates"][0]["reason"]

    def test_a_model_that_detects_nothing_fails_rather_than_raising(self, stores: Stores) -> None:
        """Design section 4.2 hard fails a challenger whose classes collapse, and
        the gate can only report it if the job does not die first."""
        write_channels(stores, found=0)
        run_container(CHALLENGER, (Seed(1),))

        _, report = self._produced(stores)
        assert not report["passed"]
        assert "scored zero AP" in report["gates"][0]["reason"]

    def test_a_champion_scored_over_another_cohort_is_refused(self, stores: Stores) -> None:
        """Two caches over different eval sets is the one mismatch that produces
        arithmetic instead of an error at every step but the last, so it is named
        as the input problem it is rather than left to the bootstrap."""
        stage_champion(stores, found=1, seeds=(Seed(1),))
        write_channels(stores, found=3, named=sorted(LABELS)[:-1])

        with pytest.raises(SystemExit, match="eval images"):
            run_container(CHALLENGER, (Seed(1),), CHAMPION)

    def test_a_seed_the_champion_never_ran_is_refused(self, stores: Stores) -> None:
        """A paired delta compares matching seeds. Dropping the unmatched one
        would compare a two-seed mean against a one-seed one."""
        write_channels(stores, found=3, seeds=(Seed(1), Seed(2)))
        stage_champion(stores, found=1, seeds=(Seed(1),))

        with pytest.raises(SystemExit, match="no cached matches for seed 2"):
            run_container(CHALLENGER, (Seed(1), Seed(2)), CHAMPION)

    def test_no_label_outside_the_eval_cohort_is_read(self, stores: Stores) -> None:
        """The eval wall from the job's side. The bootstrap labels sit beside the
        eval labels in the bucket, so a job that reached for them would find them
        -- and nothing it fetched holds their bytes."""
        withheld = b"bootstrap labels, which this job must never fetch"
        stores.data.put(f"{cohort_labels_prefix(V0, Cohort.BOOTSTRAP)}part-00000.parquet", withheld)
        write_channels(stores, found=2)

        run_container(CHALLENGER, (Seed(1),))

        fetched = [path for path in entrypoint.WORK.rglob("*") if path.is_file()]
        assert fetched
        assert all(path.read_bytes() != withheld for path in fetched)
