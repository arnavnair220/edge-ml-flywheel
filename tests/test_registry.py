"""Tests for the registration step: the manifest, the request, and the refusals.

Three kinds of evidence, split the way the package is.

The manifest document is pure, so it round-trips through a dict. That is the
whole of its risk: it is written once under a key no second write corrects, and a
field that does not survive the trip is a fact lost permanently.

The request builder is pure too, and what is tested is the part a reviewer would
otherwise have to take on trust -- that the approval status is the gates' verdict
and cannot be anything else, and that a rejected model is registered rather than
dropped.

`register` itself runs against `moto`, with a bucket holding exactly what a real
cycle would have left behind: a gate report, a digest per seed, and SageMaker's
own tarball under the job-named directory it invents. The refusals are the point
of those tests -- a missing verdict and a missing artifact are the two ways a
cycle arrives here with nothing to register, and both should stop before a
manifest is written.
"""

import json
from collections.abc import Iterator
from datetime import UTC, datetime
from typing import Any

import boto3
import pytest
from moto import mock_aws

from edge_ml_flywheel.conventions import (
    Buckets,
    Cohort,
    GateResult,
    ModelArtifact,
    ModelManifest,
    PartitionVersion,
    RecipeVersion,
    RunId,
    RunRegistration,
    Seed,
    Table,
    columns,
    gate_report_key,
    model_artifact_key,
    model_manifest_key,
    model_package_group,
    model_seed_prefix,
    new_model_version,
    sha256sums_document,
    table_name,
)
from edge_ml_flywheel.registry import launch, package
from edge_ml_flywheel.registry import manifest as document
from edge_ml_flywheel.run import registration as reg

REGION = "us-east-1"
ACCOUNT = "123456789012"

RUN = RunId("20260812t143355z-v0-skeleton")
OTHER = RunId("20260812t143355z-v0-control")
VERSION = new_model_version(RUN, 3)

COMMIT = "0" * 40
DIGEST = "a" * 64
OTHER_DIGEST = "b" * 64

# The checkpoint's digest, deliberately not `DIGEST`. Both artifacts are listed
# in one `model.sha256`, and a fixture where they matched would pass whichever
# line the registration happened to read -- which is the thing being checked.
TORCH_DIGEST = "c" * 64

PASSED = GateResult(gate="quality", passed=True, reason="mean paired delta +0.0120")
FAILED = GateResult(gate="quality", passed=False, reason="the band does not clear zero")


def manifest(**overrides: Any) -> ModelManifest:
    """A manifest of the shape a single-seed cycle produces."""
    fields: dict[str, Any] = {
        "version": VERSION,
        "created_at": datetime(2026, 8, 12, 15, 0, tzinfo=UTC),
        "git_commit": COMMIT,
        "partition_version": PartitionVersion(0),
        "recipe_version": RecipeVersion(1),
        "cohorts_trained_on": frozenset({Cohort.BOOTSTRAP}),
        "labels_spent": 0,
        "deployed_seed": Seed(1),
        "artifact_sha256": {Seed(1): DIGEST},
        "gates": (PASSED,),
    }
    return ModelManifest(**{**fields, **overrides})


class TestTheDocument:
    def test_round_trips(self) -> None:
        built = manifest()
        assert document.from_document(document.to_document(built)) == built

    def test_survives_json(self) -> None:
        """The document is written as JSON, which loses the type of every mapping
        key: a seed read back as `"1"` is not the seed the digest was filed
        under, and `deployed_seed in artifact_sha256` is then false for a manifest
        that is perfectly well formed."""
        built = manifest(artifact_sha256={Seed(1): DIGEST, Seed(2): OTHER_DIGEST})
        landed = document.from_document(json.loads(json.dumps(document.to_document(built))))

        assert landed == built
        assert landed.artifact_sha256[Seed(1)] == DIGEST

    def test_encodes_every_field_of_the_schema(self) -> None:
        """The drift check inside `to_document` as an assertion rather than as a
        raise nobody sees: a field added to `ModelManifest` and forgotten here is
        a document missing it permanently."""
        encoded = document.to_document(manifest())
        assert set(columns(ModelManifest)) <= set(encoded)

    def test_does_not_restate_the_run_and_cycle(self) -> None:
        """Both are already inside `version`, and a document that repeated them
        would be one two readers could disagree about."""
        encoded = document.to_document(manifest())
        assert "run_id" not in encoded
        assert "cycle" not in encoded

    def test_is_stable_across_two_writes_of_the_same_facts(self) -> None:
        """Sorted where order is not meaningful, so a diff of two cycles is about
        what changed rather than about set iteration order."""
        cohorts = {Cohort.POOL, Cohort.BOOTSTRAP}
        first = document.to_document(manifest(cohorts_trained_on=frozenset(cohorts)))
        second = document.to_document(
            manifest(cohorts_trained_on=frozenset(reversed(list(cohorts))))
        )
        assert first == second


class TestReadingTheGateReport:
    def report(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "version": VERSION,
            "passed": True,
            "gates": [{"gate": "quality", "passed": True, "reason": "clears the floor"}],
        }
        return {**base, **overrides}

    def test_reads_the_verdicts(self) -> None:
        gates = document.gates(self.report(), VERSION)
        assert len(gates) == 1
        assert gates[0].passed
        assert gates[0].reason == "clears the floor"

    def test_refuses_a_report_about_another_model(self) -> None:
        """The report is addressed by run and cycle and the model by its version.
        A mismatch means the manifest would record another model's verdict under
        a name that looks right."""
        with pytest.raises(ValueError, match="judges"):
            document.gates(self.report(version=new_model_version(RUN, 4)), VERSION)

    def test_refuses_a_report_with_no_gate_in_it(self) -> None:
        """ "No check ran" and "the model was rejected" are different facts about
        different bugs, and `gates_passed` would fold them together."""
        with pytest.raises(ValueError, match="no gate"):
            document.gates(self.report(gates=[]), VERSION)


class TestTheRequest:
    def request(self, **overrides: Any) -> dict[str, Any]:
        return package.create_model_package(
            manifest=manifest(**overrides),
            buckets=Buckets.for_account(ACCOUNT),
            image="an.ecr.uri/pytorch-training:tag",
            model_data_url="s3://artifacts/model.tar.gz",
        )

    def test_the_group_is_the_run(self) -> None:
        """One ladder per run, because a partition or recipe change forces a new
        run and a re-baselined champion (design section 5)."""
        assert self.request()["ModelPackageGroupName"] == RUN

    def test_it_carries_no_tags(self) -> None:
        """SageMaker refuses tags on a package version and says to put them on
        the group, which is the one place a per-version fact cannot go -- the
        group is the run, opened once by whichever cycle reaches it first.

        Nothing was lost by dropping them. They were `project`, `run_id`,
        `cycle` and `recipe_version`, and the metadata below already carries
        every per-model one, so the request held two spellings of the same facts
        and the API rejected the redundant one -- after a cycle had trained,
        scored twice and evaluated.
        """
        request = self.request()
        assert "Tags" not in request

        metadata = request["CustomerMetadataProperties"]
        for field in ("run_id", "cycle", "recipe_version"):
            assert field in metadata

    def test_approval_is_the_gates_verdict(self) -> None:
        assert self.request()["ModelApprovalStatus"] == package.APPROVED
        assert self.request(gates=(FAILED,))["ModelApprovalStatus"] == package.REJECTED

    def test_a_model_whose_gates_never_ran_is_not_approved(self) -> None:
        """`gates_passed` is false for an empty tuple rather than vacuously true,
        and this is the request where that matters."""
        assert package.approval_status(manifest(gates=())) == package.REJECTED

    def test_the_rejection_carries_its_reason(self) -> None:
        """A verdict is never recorded without one (design section 5), and this
        is the copy someone reads without opening an S3 object."""
        assert "does not clear zero" in self.request(gates=(FAILED,))["ModelPackageDescription"]

    def test_a_long_verdict_is_truncated_rather_than_refused(self) -> None:
        """A gate that failed several checks at once is the one worth not losing
        the whole request over, so the description says it was cut."""
        wordy = GateResult(gate="quality", passed=False, reason="x" * 2000)
        described = self.request(gates=(wordy,))["ModelPackageDescription"]

        assert len(described) <= package.MAX_DESCRIPTION
        assert described.endswith(package.DESCRIPTION_ELLIPSIS)

    def test_the_manifest_is_attached_by_reference(self) -> None:
        """A metadata value caps at 256 characters and the manifest has a gate
        reason in it, so the map holds a pointer and the fields anyone filters
        on."""
        metadata = self.request()["CustomerMetadataProperties"]

        assert metadata["manifest"].endswith(model_manifest_key(VERSION))
        assert metadata["artifact_sha256"] == DIGEST
        assert metadata["deployed_seed"] == "1"
        assert all(0 < len(value) <= package.MAX_METADATA_VALUE for value in metadata.values())

    def test_the_manifest_is_also_a_model_card(self) -> None:
        """The SageMaker-native home for what a manifest says. The file stays --
        a device cannot call an API to read a card -- but a reviewer who knows
        SageMaker looks for one."""
        card = json.loads(self.request()["ModelCard"]["ModelCardContent"])

        assert card["model_overview"]["model_name"] == VERSION
        assert card["additional_information"]["custom_details"]["artifact_sha256"] == DIGEST

    def test_the_card_restates_the_manifest_rather_than_adding_to_it(self) -> None:
        """One spelling of what the model is. The custom details are `metadata`
        verbatim, so the card cannot drift from the document it describes."""
        built = manifest()
        buckets = Buckets.for_account(ACCOUNT)
        card = json.loads(package.card_content(built, buckets))

        assert card["additional_information"]["custom_details"] == package.metadata(built, buckets)

    def test_the_card_status_describes_the_document_not_the_model(self) -> None:
        """A rejected model still gets a finished card saying why. The verdict is
        `ModelApprovalStatus`, and the two are deliberately not wired together."""
        rejected = self.request(gates=(FAILED,))

        assert rejected["ModelApprovalStatus"] == package.REJECTED
        assert rejected["ModelCard"]["ModelCardStatus"] == package.CARD_STATUS

    def test_the_card_fits_the_size_the_api_allows(self) -> None:
        wordy = GateResult(gate="quality", passed=False, reason="x" * 5000)
        assert (
            len(self.request(gates=(wordy,))["ModelCard"]["ModelCardContent"]) <= package.MAX_CARD
        )

    def test_the_metrics_are_pointed_at_rather_than_copied(self) -> None:
        """`metrics.json` is per seed and this request is per model, so a figure
        copied here would be one of several with nothing to say which."""
        statistics = self.request()["ModelMetrics"]["ModelQuality"]["Statistics"]
        assert statistics["S3Uri"].endswith("metrics.json")


class TestTheGroupName:
    def test_is_the_run(self) -> None:
        assert model_package_group(RUN) == RUN

    def test_the_longest_run_a_slug_permits_still_fits(self) -> None:
        """SageMaker caps an entity name at 63 characters. A prefix here would
        make a legal run unregisterable at its first cycle, which is the worst
        place to discover a name is too long."""
        longest = RunId(f"20260812t143355z-{'a' * 32}")
        assert len(model_package_group(longest)) <= 63


@pytest.fixture
def account(monkeypatch: pytest.MonkeyPatch) -> Iterator[boto3.Session]:
    """A bucket and a runs table holding what a cycle would have left behind.

    Fake credentials for `test_run`'s reason: without them a test whose mock
    failed to engage reaches a real account.
    """
    for name, value in (
        ("AWS_ACCESS_KEY_ID", "testing"),
        ("AWS_SECRET_ACCESS_KEY", "testing"),
        ("AWS_SESSION_TOKEN", "testing"),
        ("AWS_DEFAULT_REGION", REGION),
    ):
        monkeypatch.setenv(name, value)

    with mock_aws():
        aws = boto3.Session(region_name=REGION)
        buckets = Buckets.for_account(ACCOUNT)
        for bucket in (buckets.artifacts, buckets.data):
            aws.client("s3").create_bucket(Bucket=bucket)

        resource = aws.resource("dynamodb")
        resource.create_table(
            TableName=table_name(Table.RUNS),
            KeySchema=[{"AttributeName": "run_id", "KeyType": "HASH"}],
            AttributeDefinitions=[{"AttributeName": "run_id", "AttributeType": "S"}],
            BillingMode="PAY_PER_REQUEST",
        )
        reg.register(
            reg.runs_table(resource),
            RunRegistration(
                run_id=RUN,
                created_at=datetime(2026, 8, 12, 14, 33, 55, tzinfo=UTC),
                git_commit=COMMIT,
                partition_version=PartitionVersion(0),
                recipe_version=RecipeVersion(1),
                label_budget_per_cycle=1000,
                note="a cycle to register",
            ),
        )
        yield aws


def cycle_output(
    aws: boto3.Session,
    seeds: tuple[Seed, ...] = (Seed(1),),
    gates: tuple[GateResult, ...] = (PASSED,),
    tarball: bool = True,
) -> None:
    """Everything the cycle's jobs wrote, as they wrote it.

    The tarball goes under a job-named directory rather than at a predictable
    key, because that is what SageMaker does to an `OutputDataConfig` prefix and
    it is the reason `model_data_url` is a listing.
    """
    client = aws.client("s3")
    artifacts = Buckets.for_account(ACCOUNT).artifacts

    client.put_object(
        Bucket=artifacts,
        Key=gate_report_key(RUN, 3),
        Body=json.dumps(
            {
                "version": VERSION,
                "passed": all(gate.passed for gate in gates),
                "gates": [
                    {"gate": gate.gate, "passed": gate.passed, "reason": gate.reason}
                    for gate in gates
                ],
            }
        ).encode(),
    )

    for seed in seeds:
        client.put_object(
            Bucket=artifacts,
            Key=model_artifact_key(VERSION, seed, ModelArtifact.SHA256),
            Body=sha256sums_document(
                {ModelArtifact.ONNX: DIGEST, ModelArtifact.TORCH: TORCH_DIGEST}
            ).encode(),
        )
        if tarball:
            client.put_object(
                Bucket=artifacts,
                Key=(
                    f"{model_seed_prefix(VERSION, seed)}{launch.SAGEMAKER_PREFIX}"
                    f"{VERSION}-s{seed}-150000/{launch.SAGEMAKER_MODEL}"
                ),
                Body=b"a tarball",
            )


class TestRegistering:
    def test_writes_a_manifest_and_approves_a_passing_model(self, account: boto3.Session) -> None:
        cycle_output(account)
        registered = launch.register(account, RUN, VERSION, (Seed(1),))

        assert registered.passed
        assert registered.group == RUN
        assert registered.request["ModelApprovalStatus"] == package.APPROVED

        landed = json.loads(
            account.client("s3")
            .get_object(
                Bucket=Buckets.for_account(ACCOUNT).artifacts, Key=model_manifest_key(VERSION)
            )["Body"]
            .read()
        )
        assert document.from_document(landed).artifact_sha256[Seed(1)] == DIGEST

    def test_a_failed_gate_is_registered_rather_than_dropped(self, account: boto3.Session) -> None:
        """The rejection log is a feature (design section 5). A rejected cycle
        still writes its manifest and still records a version -- what it does not
        do is promote."""
        cycle_output(account, gates=(FAILED,))
        registered = launch.register(account, RUN, VERSION, (Seed(1),))

        assert not registered.passed
        assert registered.request["ModelApprovalStatus"] == package.REJECTED

    def test_the_manifest_records_the_verdict_it_was_given(self, account: boto3.Session) -> None:
        cycle_output(account, gates=(FAILED,))
        launch.register(account, RUN, VERSION, (Seed(1),))

        landed = json.loads(
            account.client("s3")
            .get_object(
                Bucket=Buckets.for_account(ACCOUNT).artifacts, Key=model_manifest_key(VERSION)
            )["Body"]
            .read()
        )
        assert document.from_document(landed).gates == (FAILED,)

    def test_a_cycle_that_bought_nothing_trained_on_the_bootstrap_alone(
        self, account: boto3.Session
    ) -> None:
        """The honest answer for every cycle until the purchase step lands, and
        the one `ModelManifest` refuses to see `eval` in."""
        cycle_output(account)
        launch.register(account, RUN, VERSION, (Seed(1),))

        landed = json.loads(
            account.client("s3")
            .get_object(
                Bucket=Buckets.for_account(ACCOUNT).artifacts, Key=model_manifest_key(VERSION)
            )["Body"]
            .read()
        )
        built = document.from_document(landed)
        assert built.cohorts_trained_on == frozenset({Cohort.BOOTSTRAP})
        assert built.labels_spent == 0

    def test_every_seed_is_recorded_and_the_lowest_one_ships(self, account: boto3.Session) -> None:
        """Seed 1 by convention (design section 4.2), never the best-scoring one.
        The rest are kept because the matched-seed saving depends on them."""
        cycle_output(account, seeds=(Seed(1), Seed(2)))
        launch.register(account, RUN, VERSION, (Seed(1), Seed(2)))

        landed = json.loads(
            account.client("s3")
            .get_object(
                Bucket=Buckets.for_account(ACCOUNT).artifacts, Key=model_manifest_key(VERSION)
            )["Body"]
            .read()
        )
        built = document.from_document(landed)
        assert built.deployed_seed == 1
        assert set(built.artifact_sha256) == {1, 2}

    def test_refuses_a_cycle_with_no_verdict(self, account: boto3.Session) -> None:
        """A model is not registered without one, and the absence means the
        evaluation job never finished."""
        cycle_output(account)
        account.client("s3").delete_object(
            Bucket=Buckets.for_account(ACCOUNT).artifacts, Key=gate_report_key(RUN, 3)
        )

        with pytest.raises(SystemExit, match="does not exist"):
            launch.register(account, RUN, VERSION, (Seed(1),))

    def test_refuses_a_seed_that_published_no_digest(self, account: boto3.Session) -> None:
        """A model whose artifact cannot be identified is one a device cannot
        verify before loading (design section 6)."""
        cycle_output(account)

        with pytest.raises(SystemExit, match="published no digest"):
            launch.register(account, RUN, VERSION, (Seed(1), Seed(2)))

    def test_refuses_a_model_with_no_tarball_to_point_at(self, account: boto3.Session) -> None:
        cycle_output(account, tarball=False)

        with pytest.raises(SystemExit, match="never wrote one"):
            launch.register(account, RUN, VERSION, (Seed(1),))

    def test_refuses_a_version_belonging_to_another_run(self, account: boto3.Session) -> None:
        """The execution claimed its cycle under one run and the version was
        built several states later, so these are two values by the time they
        meet."""
        cycle_output(account)

        with pytest.raises(SystemExit, match="belongs to run"):
            launch.register(account, OTHER, VERSION, (Seed(1),))

    def test_refuses_a_cycle_that_trained_no_seed(self, account: boto3.Session) -> None:
        with pytest.raises(SystemExit, match="no seed"):
            launch.register(account, RUN, VERSION, ())

    def test_nothing_is_written_when_the_verdict_is_missing(self, account: boto3.Session) -> None:
        """The order the failures are worth having in: everything the manifest is
        made of is read before any of it is written."""
        cycle_output(account)
        artifacts = Buckets.for_account(ACCOUNT).artifacts
        account.client("s3").delete_object(Bucket=artifacts, Key=gate_report_key(RUN, 3))

        with pytest.raises(SystemExit):
            launch.register(account, RUN, VERSION, (Seed(1),))

        found = account.client("s3").list_objects_v2(
            Bucket=artifacts, Prefix=model_manifest_key(VERSION)
        )
        assert not found.get("Contents")
