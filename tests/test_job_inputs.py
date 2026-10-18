"""Tests for what each job fetches from the store before it runs.

The jobs read every input from the object store, and which keys they read is
the whole of each job's reach into the data. So the properties worth pinning
are about keys: each job fetches exactly its inputs, lands them where its
entrypoint looks, and never fetches a label it is not allowed to hold.

**Every bucket here also holds what the job must not read.** A test that passed
against a bucket containing only the right objects would say nothing about
whether the job stays away from the wrong ones; these fixtures put the
withheld labels beside the permitted ones, so reaching for them would succeed.
"""

import json
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from moto import mock_aws

from edge_ml_flywheel.conventions import (
    Buckets,
    Cohort,
    Cycle,
    ImageId,
    ModelArtifact,
    PartitionVersion,
    Precision,
    RunId,
    Seed,
    Split,
    base_weights_key,
    cohort_labels_prefix,
    detections_key,
    eval_matches_key,
    model_artifact_key,
    new_model_version,
    purchases_run_prefix,
    raw_image_key,
    raw_label_key,
    scoring_manifest_key,
    training_manifest_key,
)
from edge_ml_flywheel.evaluation import inputs as evaluation
from edge_ml_flywheel.scoring import inputs as scoring
from edge_ml_flywheel.storage import StorageConfig, tree
from edge_ml_flywheel.storage.s3 import S3Store, client
from edge_ml_flywheel.training import images
from edge_ml_flywheel.training import inputs as training

REGION = "us-east-1"
BUCKETS = Buckets(data="inputs-data", artifacts="inputs-artifacts", telemetry="inputs-telemetry")
RUN = RunId("20260812t143355z-v1-uncertainty")
V0 = PartitionVersion(0)
CHALLENGER = new_model_version(RUN, Cycle(2))
CHAMPION = new_model_version(RUN, Cycle(1))

# Bytes no job may ever fetch. Each fixture files them under a label prefix the
# job under test is denied, and each test checks no fetched file holds them.
WITHHELD = b"a label this job must never read"


def an_image_id(index: int) -> ImageId:
    """A BDD100K-shaped ID: two 8-character hex groups."""
    return ImageId(f"{index:08x}-{index ^ 0x5F5E0FF:08x}")


@dataclass(frozen=True, slots=True)
class Stores:
    data: S3Store
    artifacts: S3Store


@pytest.fixture
def stores(monkeypatch: pytest.MonkeyPatch) -> Iterator[Stores]:
    """Both buckets in moto, with fake credentials for `test_base_weights`' reason."""
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.delenv("AWS_SESSION_TOKEN", raising=False)
    with mock_aws():
        config = StorageConfig(region=REGION)
        for bucket in (BUCKETS.data, BUCKETS.artifacts):
            client(config).create_bucket(Bucket=bucket)
        yield Stores(
            data=S3Store.connect(BUCKETS.data, config),
            artifacts=S3Store.connect(BUCKETS.artifacts, config),
        )


def manifest(image_ids: Sequence[ImageId], split: Split) -> bytes:
    return json.dumps(images.document(BUCKETS.data, image_ids, split)).encode()


def fetched(root: Path) -> dict[str, bytes]:
    """Every file under `root`, by its path relative to it."""
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in root.rglob("*")
        if path.is_file()
    }


def withhold(stores: Stores, *prefixes: str) -> None:
    for prefix in prefixes:
        stores.data.put(f"{prefix}part-00000.parquet", WITHHELD)


class TestFetch:
    def test_objects_land_relative_to_their_prefix(self, stores: Stores, tmp_path: Path) -> None:
        stores.data.put("a/b/c.json", b"c")
        stores.data.put("a/b/d/e.json", b"e")

        landed = tree.fetch_prefix(stores.data, "a/b/", tmp_path)

        assert landed == 2
        assert fetched(tmp_path) == {"c.json": b"c", "d/e.json": b"e"}

    def test_no_prefix_is_the_tree_as_the_bucket(self, stores: Stores, tmp_path: Path) -> None:
        stores.data.put("a/b/c.json", b"c")

        tree.fetch(stores.data, ["a/b/c.json"], tmp_path)

        assert fetched(tmp_path) == {"a/b/c.json": b"c"}

    def test_a_key_outside_the_prefix_is_refused_before_any_read(
        self, stores: Stores, tmp_path: Path
    ) -> None:
        with pytest.raises(ValueError, match="not under"):
            tree.fetch(stores.data, ["elsewhere/x"], tmp_path, prefix="a/")

        assert fetched(tmp_path) == {}

    def test_an_empty_prefix_is_a_count_of_zero(self, stores: Stores, tmp_path: Path) -> None:
        assert tree.fetch_prefix(stores.data, "nothing/", tmp_path) == 0


class TestManifestKeys:
    def test_entries_become_full_keys(self) -> None:
        ids = [an_image_id(1), an_image_id(2)]

        keys = images.manifest_keys(manifest(ids, Split.TRAIN), Split.TRAIN)

        assert keys == tuple(raw_image_key(image_id, Split.TRAIN) for image_id in sorted(ids))

    def test_another_splits_manifest_is_refused(self) -> None:
        """A training job handed a `val` manifest would be reading the split
        `eval` is drawn from."""
        with pytest.raises(ValueError, match="reads only"):
            images.manifest_keys(manifest([an_image_id(1)], Split.VAL), Split.TRAIN)

    def test_an_entry_that_walks_out_of_the_prefix_is_refused(self) -> None:
        document = json.loads(manifest([an_image_id(1)], Split.TRAIN))
        document.append("../../labels/x.jpg")

        with pytest.raises(ValueError, match="file name"):
            images.manifest_keys(json.dumps(document).encode(), Split.TRAIN)

    def test_a_header_with_no_image_is_refused(self) -> None:
        document = json.loads(manifest([an_image_id(1)], Split.TRAIN))[:1]

        with pytest.raises(ValueError, match="naming any image"):
            images.manifest_keys(json.dumps(document).encode(), Split.TRAIN)


class TestTrainingInputs:
    LABELED = (an_image_id(1), an_image_id(2))

    def _cycle(self, stores: Stores, purchased: bool = True) -> None:
        """A cycle's inputs, plus a `train` image the manifest leaves out and the
        two label prefixes training is denied: the pool's and `eval`'s."""
        stores.artifacts.put(
            training_manifest_key(RUN, Cycle(2)), manifest(self.LABELED, Split.TRAIN)
        )
        for image_id in [*self.LABELED, an_image_id(3)]:
            stores.data.put(raw_image_key(image_id, Split.TRAIN), image_id.encode())
        stores.data.put(f"{cohort_labels_prefix(V0, Cohort.BOOTSTRAP)}part-00000.parquet", b"b")
        if purchased:
            stores.data.put(f"{purchases_run_prefix(RUN)}cycle=001/part-00000.parquet", b"p")
        stores.artifacts.put(base_weights_key(), b"weights")

        withhold(stores, cohort_labels_prefix(V0, Cohort.EVAL))
        stores.data.put(raw_label_key(an_image_id(3), Split.TRAIN), WITHHELD)

    def test_every_input_lands_where_the_entrypoint_reads_it(
        self, stores: Stores, tmp_path: Path
    ) -> None:
        self._cycle(stores)

        staged = training.stage(stores.data, stores.artifacts, CHALLENGER, V0, tmp_path)

        assert sorted(fetched(staged.images)) == sorted(
            f"{image_id}.jpg" for image_id in self.LABELED
        )
        assert [sorted(fetched(root).values()) for root in staged.labels] == [[b"b"], [b"p"]]
        assert staged.base.read_bytes() == b"weights"

    def test_no_withheld_label_is_fetched(self, stores: Stores, tmp_path: Path) -> None:
        self._cycle(stores)

        training.stage(stores.data, stores.artifacts, CHALLENGER, V0, tmp_path)

        assert WITHHELD not in fetched(tmp_path).values()

    def test_a_cycle_with_no_purchases_yet_trains_on_the_bootstrap(
        self, stores: Stores, tmp_path: Path
    ) -> None:
        self._cycle(stores, purchased=False)

        staged = training.stage(stores.data, stores.artifacts, CHALLENGER, V0, tmp_path)

        assert fetched(staged.labels[1]) == {}

    def test_a_partition_with_no_bootstrap_labels_is_refused(
        self, stores: Stores, tmp_path: Path
    ) -> None:
        self._cycle(stores)

        with pytest.raises(SystemExit, match="holds no labels"):
            training.stage(stores.data, stores.artifacts, CHALLENGER, PartitionVersion(1), tmp_path)

    def test_a_manifest_naming_val_is_refused_before_anything_is_fetched(
        self, stores: Stores, tmp_path: Path
    ) -> None:
        self._cycle(stores)
        stores.artifacts.put(
            training_manifest_key(RUN, Cycle(2)), manifest(self.LABELED, Split.VAL)
        )

        with pytest.raises(ValueError, match="reads only"):
            training.stage(stores.data, stores.artifacts, CHALLENGER, V0, tmp_path)

        assert fetched(tmp_path) == {}


class TestScoringInputs:
    SCORED = (an_image_id(4), an_image_id(5))

    def _cycle(self, stores: Stores) -> None:
        stores.artifacts.put(
            scoring_manifest_key(RUN, Cycle(2), Cohort.EVAL), manifest(self.SCORED, Split.VAL)
        )
        for image_id in self.SCORED:
            stores.data.put(raw_image_key(image_id, Split.VAL), image_id.encode())
        for artifact in (ModelArtifact.TORCH, ModelArtifact.ONNX):
            stores.artifacts.put(
                model_artifact_key(CHALLENGER, Seed(1), artifact), artifact.value.encode()
            )
        withhold(
            stores,
            cohort_labels_prefix(V0, Cohort.EVAL),
            cohort_labels_prefix(V0, Cohort.BOOTSTRAP),
        )

    @pytest.mark.parametrize(
        ("precision", "artifact"),
        [(Precision.FP32, ModelArtifact.TORCH), (Precision.INT8, ModelArtifact.ONNX)],
    )
    def test_the_precision_picks_the_model_file(
        self,
        stores: Stores,
        tmp_path: Path,
        precision: Precision,
        artifact: ModelArtifact,
    ) -> None:
        self._cycle(stores)

        staged = scoring.stage(
            stores.data, stores.artifacts, scoring.Pass(CHALLENGER, Seed(1), precision), tmp_path
        )

        assert staged.model.read_bytes() == artifact.value.encode()

    def test_each_cohort_is_exactly_its_manifest_in_order(
        self, stores: Stores, tmp_path: Path
    ) -> None:
        self._cycle(stores)

        staged = scoring.stage(
            stores.data, stores.artifacts, scoring.Pass(CHALLENGER, Seed(1)), tmp_path
        )

        assert [path.stem for path in staged.images[Cohort.EVAL]] == sorted(self.SCORED)

    def test_no_label_of_any_cohort_is_fetched(self, stores: Stores, tmp_path: Path) -> None:
        self._cycle(stores)

        scoring.stage(stores.data, stores.artifacts, scoring.Pass(CHALLENGER, Seed(1)), tmp_path)

        assert WITHHELD not in fetched(tmp_path).values()


class TestEvaluationInputs:
    def _cycle(self, stores: Stores) -> None:
        stores.artifacts.put(
            scoring_manifest_key(RUN, Cycle(2), Cohort.EVAL),
            manifest([an_image_id(4)], Split.VAL),
        )
        stores.data.put(f"{cohort_labels_prefix(V0, Cohort.EVAL)}part-00000.parquet", b"eval")
        for seed in (Seed(1), Seed(2)):
            stores.artifacts.put(detections_key(CHALLENGER, seed, Cohort.EVAL), f"s{seed}".encode())
        stores.artifacts.put(
            detections_key(CHALLENGER, Seed(1), Cohort.EVAL, Precision.INT8), b"int8"
        )
        stores.artifacts.put(eval_matches_key(CHAMPION, Seed(1)), b"champion")
        withhold(stores, cohort_labels_prefix(V0, Cohort.BOOTSTRAP), purchases_run_prefix(RUN))

    def test_every_input_lands_where_the_entrypoint_reads_it(
        self, stores: Stores, tmp_path: Path
    ) -> None:
        self._cycle(stores)
        comparison = evaluation.Comparison(
            version=CHALLENGER,
            seeds=(Seed(1), Seed(2)),
            partition_version=V0,
            champion=CHAMPION,
            gates_edge=True,
        )

        staged = evaluation.stage(stores.data, stores.artifacts, comparison, tmp_path)

        assert staged.manifest.is_file()
        assert list(fetched(staged.labels).values()) == [b"eval"]
        assert {seed: list(fetched(path).values()) for seed, path in staged.detections.items()} == {
            Seed(1): [b"s1"],
            Seed(2): [b"s2"],
        }
        assert staged.int8 is not None
        assert list(fetched(staged.int8).values()) == [b"int8"]
        assert staged.champion is not None
        assert list(fetched(staged.champion).values()) == [b"champion"]

    def test_a_first_cycle_has_no_champion_and_no_int8_pass(
        self, stores: Stores, tmp_path: Path
    ) -> None:
        self._cycle(stores)
        comparison = evaluation.Comparison(
            version=CHALLENGER, seeds=(Seed(1),), partition_version=V0
        )

        staged = evaluation.stage(stores.data, stores.artifacts, comparison, tmp_path)

        assert staged.champion is None
        assert staged.int8 is None

    def test_only_the_eval_cohorts_labels_are_fetched(self, stores: Stores, tmp_path: Path) -> None:
        """Bootstrap and the purchases are the model's training set, and sit in
        the same bucket as the eval labels."""
        self._cycle(stores)
        comparison = evaluation.Comparison(
            version=CHALLENGER, seeds=(Seed(1),), partition_version=V0
        )

        evaluation.stage(stores.data, stores.artifacts, comparison, tmp_path)

        assert WITHHELD not in fetched(tmp_path).values()
