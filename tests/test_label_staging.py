"""Tests for copying the label documents a partition names onto local disk.

Two properties, and neither is about speed -- the reason this copy moved into the
package is measured in a build log, not assertable here.

**The tree is the bucket.** `cohort_labels.read` opens `stage_dir / <key>` with
no path convention of its own, so a stager that lands the bytes anywhere else
produces a build whose labels step fails with a missing file after the download
has already been paid for.

**Only the labeled cohorts are fetched.** The partition role can read all 80,000
label documents, so the guarantee is not that a `pool` key is refused when asked
for -- it is that `stage-labels` never asks. The CLI test is where that is
pinned, and it is why the command derives its own key list rather than reading
the one the buildspec prints.
"""

import json
from collections.abc import Iterator
from pathlib import Path

import boto3
import pytest
from moto import mock_aws

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    LABELED_COHORTS,
    AssignmentRow,
    Cohort,
    ImageId,
    PartitionVersion,
    assignments_key,
    raw_label_key,
)
from edge_ml_flywheel.partition import assign, cohort_labels, fetch
from edge_ml_flywheel.partition.__main__ import main

REGION = "us-east-1"

# What moto's STS hands back, which is what `Buckets.for_account` names the
# bucket from.
ACCOUNT = "123456789012"
BUCKET = f"edge-ml-flywheel-data-{ACCOUNT}"

V0 = PartitionVersion(0)

# Two of each, for `test_cohort_labels`' reason: the cohorts are distinguished by
# name rather than by size, and a set comparison over eight images fails the same
# way it would over 13,000.
COHORTS = (
    Cohort.BOOTSTRAP,
    Cohort.BOOTSTRAP,
    Cohort.POOL,
    Cohort.POOL,
    Cohort.EVAL,
    Cohort.EVAL,
    Cohort.RESERVE,
    Cohort.RESERVE,
)


def an_image_id(index: int) -> ImageId:
    """A BDD100K-shaped ID: two 8-character hex groups."""
    return ImageId(f"{index:08x}-{index ^ 0x5F5E0FF:08x}")


def assignments() -> tuple[AssignmentRow, ...]:
    return tuple(
        AssignmentRow(image_id=an_image_id(index), cohort=cohort)
        for index, cohort in enumerate(COHORTS)
    )


def a_document(image_id: ImageId) -> bytes:
    """Distinct per image, so a document landing under the wrong key is visible.

    Shapeless on purpose: this module moves bytes and never parses them, and a
    test that fed it archive-shaped JSON would imply otherwise.
    """
    return json.dumps({"name": f"{image_id}.jpg"}).encode("utf-8")


@pytest.fixture
def data(monkeypatch: pytest.MonkeyPatch) -> Iterator[boto3.Session]:
    """A data bucket holding a label document for **every** cohort.

    The unlabeled ones are present here and absent from `test_cohort_labels`'
    staged tree, and the difference is the point: there a `pool` read fails as a
    missing file, and here it would succeed. What the CLI test asserts is that
    nothing reached for one anyway.

    The credentials are fake for `test_base_weights`' reason: a test whose mock
    failed to engage would otherwise reach a real account.
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
        s3 = session.client("s3")
        s3.create_bucket(Bucket=BUCKET)
        for row in assignments():
            s3.put_object(
                Bucket=BUCKET,
                Key=raw_label_key(row.image_id, COHORT_SPLIT[row.cohort]),
                Body=a_document(row.image_id),
            )
        yield session


def staged(root: Path) -> set[str]:
    """Every key that landed, spelled as a key rather than as a path."""
    return {path.relative_to(root).as_posix() for path in root.rglob("*.json") if path.is_file()}


class TestStage:
    def test_every_key_lands_at_its_own_path(self, data: boto3.Session, tmp_path: Path) -> None:
        keys = [raw_label_key(an_image_id(0), COHORT_SPLIT[Cohort.BOOTSTRAP])]

        landed = fetch.stage(fetch.client(data), BUCKET, keys, tmp_path)

        assert landed == 1
        assert (tmp_path / keys[0]).read_bytes() == a_document(an_image_id(0))

    def test_the_whole_list_is_attempted_before_a_failure_is_raised(
        self, data: boto3.Session, tmp_path: Path
    ) -> None:
        """A partial stage fails the build, and says how partial.

        One missing document and a role that cannot read the prefix at all are
        the two things that go wrong here, they want different fixes, and the
        count is what tells them apart -- which a stager that stopped at the
        first failure could not report.
        """
        present = raw_label_key(an_image_id(0), COHORT_SPLIT[Cohort.BOOTSTRAP])
        absent = raw_label_key(an_image_id(404), COHORT_SPLIT[Cohort.BOOTSTRAP])

        with pytest.raises(RuntimeError, match=r"1 of 2 label documents"):
            fetch.stage(fetch.client(data), BUCKET, [present, absent], tmp_path)

        assert (tmp_path / present).is_file()

    def test_the_bucket_comes_from_the_account(self, data: boto3.Session) -> None:
        assert fetch.data_bucket(data) == BUCKET


class TestStageLabelsCommand:
    def test_it_stages_the_labeled_cohorts_and_nothing_else(
        self, data: boto3.Session, tmp_path: Path
    ) -> None:
        rows = assignments()
        assign.write_parquet(rows, tmp_path / assignments_key(V0))

        main(["stage-labels", "--stage-dir", str(tmp_path), "--partition-version", "0"])

        assert staged(tmp_path) == {
            raw_label_key(row.image_id, COHORT_SPLIT[row.cohort])
            for row in rows
            if row.cohort in LABELED_COHORTS
        }

    def test_what_it_stages_is_what_the_labels_step_reads(
        self, data: boto3.Session, tmp_path: Path
    ) -> None:
        """The staged tree and `cohort_labels.read` agree on where a document is.

        Asserted through `label_key` rather than through `read`, which parses:
        this module's contract is the path, and the archive shape is
        `test_cohort_labels`' subject.
        """
        rows = assignments()
        assign.write_parquet(rows, tmp_path / assignments_key(V0))

        main(["stage-labels", "--stage-dir", str(tmp_path), "--partition-version", "0"])

        for cohort in sorted(LABELED_COHORTS):
            for image_id in cohort_labels.image_ids(rows, cohort):
                assert (tmp_path / cohort_labels.label_key(cohort, image_id)).is_file()
