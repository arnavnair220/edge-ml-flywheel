"""Selection against an account: rank the pool the scoring job wrote.

The pure layer has its own tests, and they cover the arithmetic. What is this
module's own is the wiring -- that the images ranked are the ones the manifest
named rather than the ones the detections happened to mention, that the tail the
scoring job emits is dropped before it reaches a score, and that the file written
is the one the oracle later charges against.

Run against `moto` rather than a fake client, for `test_purchase`'s reason: what
is under test is a function that reads four objects out of two buckets and a
registration out of DynamoDB, and a fake would be a fake written to hand back
whatever the assertion wanted.
"""

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import boto3
import pyarrow.parquet as pq
import pytest
from moto import mock_aws

from edge_ml_flywheel.conventions import (
    Buckets,
    Cohort,
    Cycle,
    DetectionRow,
    ImageId,
    PartitionVersion,
    RecipeVersion,
    RunId,
    RunRegistration,
    Seed,
    Split,
    Table,
    detections_key,
    new_model_version,
    scoring_manifest_key,
    selection_ranking_key,
    table_name,
)
from edge_ml_flywheel.run import registration as reg
from edge_ml_flywheel.scoring import detections
from edge_ml_flywheel.selection import launch as selecting
from edge_ml_flywheel.selection import ranking
from edge_ml_flywheel.selection.score import BLIND_SPOT, DECISIVE
from edge_ml_flywheel.training import images

RUN = RunId("20260812t143355z-v1-uncertainty")
CYCLE = Cycle(1)
VERSION = new_model_version(RUN, CYCLE)
SEED = Seed(1)
REGION = "us-east-1"
ACCOUNT = "123456789012"
BUCKETS = Buckets.for_account(ACCOUNT)

# A budget small enough that the batch is a prefix of a readable pool, and large
# enough that the ranking has to choose. The arithmetic is the same at 1,000.
BUDGET = 3

POOL = tuple(ImageId(f"0000000{n}-0000000{n}") for n in range(1, 8))


def a_detection(image_id: ImageId, score: float, category: str = "car") -> DetectionRow:
    return DetectionRow(
        image_id=image_id, category=category, x1=1.0, y1=2.0, x2=30.0, y2=40.0, score=score
    )


@pytest.fixture
def aws(monkeypatch: pytest.MonkeyPatch) -> Iterator[boto3.Session]:
    """An account holding the two buckets and the runs table.

    Fake credentials for `test_run`'s reason: a mock that failed to engage would
    otherwise reach a real account and write into a write-once cycle prefix.
    """
    for name, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_SESSION_TOKEN", "testing"),
        ("AWS_DEFAULT_REGION", REGION),
    ):
        monkeypatch.setenv(name, value)

    with mock_aws():
        session = boto3.Session(region_name=REGION)
        for bucket in (BUCKETS.data, BUCKETS.artifacts):
            session.client("s3").create_bucket(Bucket=bucket)

        session.resource("dynamodb").create_table(
            TableName=table_name(Table.RUNS),
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        _register(session)
        yield session


def _register(aws: boto3.Session, **overrides: Any) -> None:
    values: dict[str, Any] = {
        "run_id": RUN,
        "created_at": datetime(2026, 8, 12, 14, 33, 55, tzinfo=UTC),
        "git_commit": "d" * 40,
        "partition_version": PartitionVersion(0),
        "recipe_version": RecipeVersion(1),
        "label_budget_per_cycle": BUDGET,
        "note": "selection tests",
    }
    table = reg.runs_table(aws.resource("dynamodb"))
    table.put_item(Item=reg.to_item(RunRegistration(**(values | overrides))))


def _manifest(aws: boto3.Session, image_ids: tuple[ImageId, ...], tmp_path: Path) -> None:
    """The record of what this cycle scored, which is what selection ranks."""
    local = tmp_path / "pool.manifest"
    images.write(local, BUCKETS.data, image_ids, Split.TRAIN)
    aws.client("s3").upload_file(
        str(local), BUCKETS.artifacts, scoring_manifest_key(RUN, CYCLE, Cohort.POOL)
    )


def _detections(aws: boto3.Session, rows: list[DetectionRow], tmp_path: Path) -> None:
    local = tmp_path / "detections.parquet"
    detections.write(rows, local)
    aws.client("s3").upload_file(
        str(local), BUCKETS.artifacts, detections_key(VERSION, SEED, Cohort.POOL)
    )


def _ranking(aws: boto3.Session, tmp_path: Path) -> tuple[Any, ...]:
    local = tmp_path / "read-back.parquet"
    aws.client("s3").download_file(BUCKETS.artifacts, selection_ranking_key(RUN, CYCLE), str(local))
    return ranking.read(local)


class TestRankingThePool:
    def test_the_batch_is_the_top_of_the_ranking(self, aws: boto3.Session, tmp_path: Path) -> None:
        """Descending uncertainty, so the images nearest 0.5 are bought."""
        _manifest(aws, POOL, tmp_path)
        _detections(
            aws,
            [a_detection(image_id, 0.5 + index * 0.05) for index, image_id in enumerate(POOL)],
            tmp_path,
        )

        ranked = selecting.rank(aws, VERSION, SEED)
        rows = _ranking(aws, tmp_path)

        assert ranked.pool == len(POOL)
        assert ranked.batch == BUDGET
        assert [row.image_id for row in rows][:BUDGET] == list(POOL[:BUDGET])
        assert [row.selected for row in rows] == [True] * BUDGET + [False] * (len(POOL) - BUDGET)

    def test_every_ranked_image_is_in_the_file_not_only_the_batch(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The ledger records what was bought; only this says what it was bought
        over, which is the comparison the ranking rests on."""
        _manifest(aws, POOL, tmp_path)
        _detections(aws, [a_detection(image_id, 0.5) for image_id in POOL], tmp_path)

        selecting.rank(aws, VERSION, SEED)

        assert len(_ranking(aws, tmp_path)) == len(POOL)

    def test_the_rank_column_is_the_order_the_file_is_written_in(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        _manifest(aws, POOL, tmp_path)
        _detections(
            aws,
            [a_detection(image_id, 0.5 + index * 0.05) for index, image_id in enumerate(POOL)],
            tmp_path,
        )

        selecting.rank(aws, VERSION, SEED)
        rows = _ranking(aws, tmp_path)

        assert [row.rank for row in rows] == list(range(len(POOL)))


class TestWhatTheModelNeverSaw:
    def test_an_image_with_no_detections_at_all_ranks_top(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The blind spot, and the reason the pool is read off the manifest: the
        detections name the images with boxes, and this one has none."""
        _manifest(aws, POOL, tmp_path)
        _detections(aws, [a_detection(image_id, 0.9) for image_id in POOL[1:]], tmp_path)

        selecting.rank(aws, VERSION, SEED)
        rows = _ranking(aws, tmp_path)

        assert rows[0].image_id == POOL[0]
        assert rows[0].score == BLIND_SPOT

    def test_an_image_carrying_only_the_low_confidence_tail_also_ranks_top(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The case the scoring job's 0.001 floor creates, and the one this
        wiring exists to get right. Without the floor these rows would read as
        detections the model was decisive about, and the frame it actually saw
        nothing in would rank last instead of first.
        """
        _manifest(aws, POOL, tmp_path)
        _detections(
            aws,
            [a_detection(POOL[0], 0.002), a_detection(POOL[0], 0.03)]
            + [a_detection(image_id, 0.9) for image_id in POOL[1:]],
            tmp_path,
        )

        selecting.rank(aws, VERSION, SEED)
        rows = _ranking(aws, tmp_path)

        assert rows[0].image_id == POOL[0]
        assert rows[0].score == BLIND_SPOT

    def test_blind_spots_in_the_batch_are_reported(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The number that says whether the ranking is working or buying a
        thousand frames the detector simply failed on."""
        _manifest(aws, POOL, tmp_path)
        _detections(aws, [a_detection(image_id, 0.99) for image_id in POOL[2:]], tmp_path)

        assert selecting.rank(aws, VERSION, SEED).blind_spots == 2

    def test_a_confident_frame_ranks_last(self, aws: boto3.Session, tmp_path: Path) -> None:
        """Decisive is the opposite end from a blind spot, and the two must not
        collapse together."""
        _manifest(aws, POOL, tmp_path)
        _detections(
            aws,
            [a_detection(POOL[0], 0.99)] + [a_detection(image_id, 0.5) for image_id in POOL[1:]],
            tmp_path,
        )

        selecting.rank(aws, VERSION, SEED)
        rows = _ranking(aws, tmp_path)

        assert rows[-1].image_id == POOL[0]
        assert rows[-1].score == DECISIVE


class TestRefusals:
    def test_a_cycle_that_never_scored_is_refused(self, aws: boto3.Session, tmp_path: Path) -> None:
        with pytest.raises(SystemExit, match="which images this cycle ranked"):
            selecting.rank(aws, VERSION, SEED)

    def test_a_seed_with_no_detections_is_refused(self, aws: boto3.Session, tmp_path: Path) -> None:
        """Naming the seed, because "the scoring job did not run" and "selection
        is broken" are different problems with the same symptom."""
        _manifest(aws, POOL, tmp_path)

        with pytest.raises(SystemExit, match="never scored over the pool"):
            selecting.rank(aws, VERSION, SEED)

    def test_ranking_twice_is_refused(self, aws: boto3.Session, tmp_path: Path) -> None:
        """The file is what the oracle charges against, so rewriting it after a
        purchase leaves the ledger pointing at a batch the record no longer
        names."""
        _manifest(aws, POOL, tmp_path)
        _detections(aws, [a_detection(image_id, 0.5) for image_id in POOL], tmp_path)
        selecting.rank(aws, VERSION, SEED)

        with pytest.raises(SystemExit, match="already exists"):
            selecting.rank(aws, VERSION, SEED)

    def test_replace_permits_a_rerun(self, aws: boto3.Session, tmp_path: Path) -> None:
        _manifest(aws, POOL, tmp_path)
        _detections(aws, [a_detection(image_id, 0.5) for image_id in POOL], tmp_path)
        selecting.rank(aws, VERSION, SEED)

        assert selecting.rank(aws, VERSION, SEED, replace=True).batch == BUDGET

    def test_a_budget_larger_than_the_pool_is_refused(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The end of a run, and a skeleton cycle capped below its budget. Loud
        rather than a batch quietly smaller than the run registered, because the
        budget is the denominator of the headline number.
        """
        _manifest(aws, POOL[:2], tmp_path)
        _detections(aws, [a_detection(image_id, 0.5) for image_id in POOL[:2]], tmp_path)

        with pytest.raises(ValueError, match="end of the run"):
            selecting.rank(aws, VERSION, SEED)

    def test_a_detection_for_an_image_outside_the_manifest_is_refused(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """Inference over the wrong image set -- the eval cohort, most
        consequentially. A ranking drawn from a set it was not given is a bug
        worth stopping for."""
        _manifest(aws, POOL[:4], tmp_path)
        _detections(aws, [a_detection(image_id, 0.5) for image_id in POOL], tmp_path)

        with pytest.raises(ValueError, match="not in the pool"):
            selecting.rank(aws, VERSION, SEED)

    def test_a_category_outside_the_class_set_is_refused(
        self, aws: boto3.Session, tmp_path: Path
    ) -> None:
        """The archive spells three categories differently from `det_20`, and the
        mismatch is otherwise silent."""
        _manifest(aws, POOL, tmp_path)
        _detections(aws, [a_detection(image_id, 0.5, "pedestrian") for image_id in POOL], tmp_path)

        with pytest.raises(ValueError, match="outside the class set"):
            selecting.rank(aws, VERSION, SEED)


class TestTheRankingFile:
    def test_it_is_written_where_the_oracle_looks(self) -> None:
        key = selection_ranking_key(RUN, CYCLE)
        assert key.startswith(f"run_id={RUN}/cycle=001/selection/")
        assert key.endswith(".parquet")

    def test_a_ranking_with_nothing_selected_has_no_batch(self) -> None:
        """The oracle refuses a purchase of no images anyway; this refuses it
        with the file named rather than the batch."""
        rows = ranking.rows(POOL, dict.fromkeys(POOL, 0.5), [])
        with pytest.raises(ValueError, match="no image as selected"):
            ranking.selected(rows)

    def test_a_batch_outside_the_ranking_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not in the ranking"):
            ranking.rows(POOL[:2], dict.fromkeys(POOL, 0.5), [POOL[5]])

    def test_it_round_trips(self, tmp_path: Path) -> None:
        rows = ranking.rows(POOL, dict.fromkeys(POOL, 0.5), POOL[:2])
        path = tmp_path / "ranking.parquet"

        ranking.write(rows, path)
        assert ranking.read(path) == rows

    def test_it_is_written_in_a_codec_a_lambda_can_open(self, tmp_path: Path) -> None:
        """Both readers of this file are Lambdas, and the managed pyarrow layer is
        built without zstd -- which is what stopped the first real execution at
        `Prepare`. Read off the file rather than asserted on the constant,
        because what a reader has to cope with is the file.
        """
        path = tmp_path / "ranking.parquet"
        ranking.write(ranking.rows(POOL, dict.fromkeys(POOL, 0.5), POOL[:1]), path)

        codecs = {
            pq.ParquetFile(path).metadata.row_group(0).column(index).compression
            for index in range(pq.ParquetFile(path).metadata.num_columns)
        }
        assert codecs == {"SNAPPY"}
