"""Tests for the run summary: the rows, the carry, and the stored document.

All pure. The impure half of `reporting` is S3 reads of documents other tests
already cover -- the manifest in `test_registry`, the gate report in
`test_evaluation_job`, the telemetry reduction in `test_fleet_telemetry` -- so
what is left to establish here is the part that is this module's own: the shape
that goes in the bucket, and the one field not read straight off an artifact.

The carry gets the most attention because it is the only inference in the
deliverable. `deployed` is what a reader uses to tell a promotion from a
rejection from a rollback, and getting it wrong would not raise anywhere: the
document would simply describe a run in which the wrong model was on the device,
and nothing downstream would contradict it.
"""

import json
from datetime import UTC, datetime
from typing import Any

import pytest

from edge_ml_flywheel.conventions import (
    Cohort,
    CycleSummary,
    GateResult,
    ModelManifest,
    PartitionVersion,
    RecipeVersion,
    RunId,
    Seed,
    new_model_version,
    run_prefix,
    run_summary_key,
)
from edge_ml_flywheel.reporting import summary

RUN = RunId("20260812t143355z-v0-skeleton")
COMMIT = "0" * 40
DIGEST = "a" * 64

DATA = GateResult(gate="data", passed=True, reason="1000 new images")
QUALITY = GateResult(gate="quality", passed=True, reason="mean paired delta +0.0120")
EDGE = GateResult(gate="edge", passed=True, reason="int8 within 1.1% of fp32")
REFUSED = GateResult(gate="quality", passed=False, reason="the band does not clear zero")
CANARY_KEPT = GateResult(gate="canary", passed=True, reason="p95 43 ms, digest verified")
CANARY_LOST = GateResult(gate="canary", passed=False, reason="the digest is not the manifest's")


def manifest(cycle: int, **overrides: Any) -> ModelManifest:
    """A manifest of the shape a single-seed cycle produces."""
    fields: dict[str, Any] = {
        "version": new_model_version(RUN, cycle),
        "created_at": datetime(2026, 8, 12, 15, 0, tzinfo=UTC),
        "git_commit": COMMIT,
        "partition_version": PartitionVersion(0),
        "recipe_version": RecipeVersion(1),
        "cohorts_trained_on": frozenset({Cohort.BOOTSTRAP}),
        "labels_spent": 8000 + 1000 * cycle,
        "deployed_seed": Seed(1),
        "artifact_sha256": {Seed(1): DIGEST},
        "gates": (DATA, QUALITY, EDGE),
    }
    return ModelManifest(**{**fields, **overrides})


def facts(cycle: int, **overrides: Any) -> summary.CycleFacts:
    fields: dict[str, Any] = {
        "manifest": manifest(cycle),
        "delta": (0.012, 0.005, 0.019),
        "canary": CANARY_KEPT,
        "p95_ms": 43.0,
    }
    return summary.CycleFacts(**{**fields, **overrides})


class TestTheKey:
    def test_sits_above_every_cycle(self) -> None:
        """A statement about the run, so it is not under one of its turns. A
        reader fetching a run's record should not have to know which cycle to
        look in, and a cycle prefix is write-once evidence about one turn."""
        key = run_summary_key(RUN)
        assert key == f"{run_prefix(RUN)}summary.json"
        assert "cycle=" not in key


class TestTheCarry:
    def test_a_promotion_deploys_its_own_model(self) -> None:
        rows = summary.rows([facts(0), facts(1)])
        assert [row.deployed for row in rows] == [
            new_model_version(RUN, 0),
            new_model_version(RUN, 1),
        ]

    def test_a_rejection_leaves_the_champion_in_place(self) -> None:
        """The cycle keeps its labels and its version, and the device keeps the
        model it had. A row whose `deployed` moved to a rejected challenger would
        describe a rollout that never happened."""
        rejected = facts(
            1, manifest=manifest(1, gates=(DATA, REFUSED, EDGE)), canary=None, p95_ms=None
        )
        rows = summary.rows([facts(0), rejected])

        assert rows[1].version == new_model_version(RUN, 1)
        assert rows[1].deployed == new_model_version(RUN, 0)
        assert rows[1].failed == ("quality",)

    def test_a_rollback_leaves_the_champion_in_place(self) -> None:
        """The other way a cycle can fail, and the reason one rule covers both:
        the challenger cleared the cloud gates and shipped, the device refused
        it, and the version the device ends on is the previous one either way."""
        lost = facts(1, canary=CANARY_LOST)
        rows = summary.rows([facts(0), lost])

        assert rows[1].deployed == new_model_version(RUN, 0)
        assert rows[1].failed == ("canary",)

    def test_a_rejection_does_not_strand_later_promotions(self) -> None:
        """A failed cycle is not the end of a run. The next challenger trains on
        the larger set and can promote, which is the whole reason a rejection
        keeps its labels."""
        rejected = facts(
            1, manifest=manifest(1, gates=(DATA, REFUSED, EDGE)), canary=None, p95_ms=None
        )
        rows = summary.rows([facts(0), rejected, facts(2)])

        assert [row.deployed for row in rows] == [
            new_model_version(RUN, 0),
            new_model_version(RUN, 0),
            new_model_version(RUN, 2),
        ]

    def test_nothing_is_deployed_before_the_first_promotion(self) -> None:
        """A run whose first cycle failed has no champion at all, which is not
        the same as one whose champion is cycle 0."""
        refused = facts(
            0, manifest=manifest(0, gates=(DATA, REFUSED, EDGE)), canary=None, p95_ms=None
        )
        assert summary.rows([refused])[0].deployed is None

    def test_cycles_are_ordered_before_the_fold(self) -> None:
        """The carry is order-dependent, so a caller listing cycles out of order
        would otherwise produce a document naming the wrong champion -- and
        nothing downstream would contradict it."""
        rejected = facts(
            1, manifest=manifest(1, gates=(DATA, REFUSED, EDGE)), canary=None, p95_ms=None
        )
        rows = summary.rows([facts(2), rejected, facts(0)])

        assert [row.cycle for row in rows] == [0, 1, 2]
        assert rows[1].deployed == new_model_version(RUN, 0)


class TestTheVerdicts:
    def test_the_canary_is_appended_last(self) -> None:
        """The order the gates were reached in, which is the one thing a flat
        list of verdicts can say about when each was asked."""
        gates = summary.rows([facts(0)])[0].gates
        assert [gate.gate for gate in gates] == ["data", "quality", "edge", "canary"]

    def test_a_model_that_never_shipped_carries_three(self) -> None:
        """Absent rather than recorded as passing. A canary that never ran is not
        a canary that was satisfied, and the project refuses that substitution
        everywhere else too."""
        rejected = facts(
            0, manifest=manifest(0, gates=(DATA, REFUSED, EDGE)), canary=None, p95_ms=None
        )
        assert [gate.gate for gate in summary.rows([rejected])[0].gates] == [
            "data",
            "quality",
            "edge",
        ]

    def test_no_verdict_is_not_a_rejection(self) -> None:
        """An empty gate list means no check ran. `failed` is empty for it, so a
        reader has to look at `gates` to tell it from a promotion -- which is
        why the outcome table in the doc asks about the empty case."""
        nothing = facts(0, manifest=manifest(0, gates=()), canary=None, p95_ms=None)
        row = summary.rows([nothing])[0]

        assert row.gates == ()
        assert row.failed == ()
        assert row.deployed is None


class TestTheDocument:
    def test_round_trips(self) -> None:
        rows = summary.rows([facts(0), facts(1)])
        assert summary.from_document(summary.to_document(RUN, rows)) == rows

    def test_survives_json(self) -> None:
        """It is stored as JSON and meant to be downloaded, so the trip through
        text is the one it actually makes."""
        rows = summary.rows([facts(0), facts(1)])
        landed = summary.from_document(json.loads(json.dumps(summary.to_document(RUN, rows))))
        assert landed == rows

    def test_carries_the_run_it_describes(self) -> None:
        """Unlike the manifest, which omits what its version already names. This
        document is the one meant to be read away from its prefix, and a file on
        a desk has to say which run it is about."""
        assert summary.to_document(RUN, summary.rows([facts(0)]))["run_id"] == RUN

    def test_nulls_survive_as_nulls(self) -> None:
        """The three absences the schema distinguishes from zero. A `p95_ms` of
        0.0 would read as a model that cost nothing per frame, and a `delta` of
        zero as a comparison that came out even rather than one never made."""
        baseline = facts(0, delta=None, canary=None, p95_ms=None)
        stored = summary.to_document(RUN, summary.rows([baseline]))
        row = stored["cycles"][0]

        assert row["delta"] is None
        assert row["p95_ms"] is None
        assert row["deployed"] == new_model_version(RUN, 0)

        assert summary.from_document(stored)[0].delta is None

    def test_an_unknown_schema_is_refused(self) -> None:
        """This document outlives the code that wrote it by design, so a shape
        from another version can actually arrive. Refused rather than parsed on
        the assumption the fields still mean what they did."""
        stored = summary.to_document(RUN, summary.rows([facts(0)]))
        stored["schema"] = summary.SCHEMA + 1

        with pytest.raises(ValueError, match="schema"):
            summary.from_document(stored)


class TestTheRowSchema:
    def test_a_delta_outside_its_band_is_refused(self) -> None:
        """Three floats in one tuple, so a caller can transpose them. The band
        brackets the observation by construction, and a row where it does not is
        one read out of the wrong fields."""
        with pytest.raises(ValueError, match="outside its own band"):
            CycleSummary(
                cycle=1,
                version=new_model_version(RUN, 1),
                labels_spent=9000,
                gates=(QUALITY,),
                delta=(0.019, 0.005, 0.012),
                deployed=None,
                p95_ms=None,
            )

    def test_a_p95_of_zero_is_refused(self) -> None:
        """A latency nothing measured. `None` is how the schema says that."""
        with pytest.raises(ValueError, match="never measured"):
            CycleSummary(
                cycle=1,
                version=new_model_version(RUN, 1),
                labels_spent=9000,
                gates=(QUALITY,),
                delta=None,
                deployed=None,
                p95_ms=0.0,
            )


class TestTheDelta:
    def test_read_from_a_gate_report(self) -> None:
        report = {
            "delta": {
                "observed": 0.012,
                "lower": 0.005,
                "upper": 0.019,
                "resamples": 2000,
                "confidence": 0.95,
            }
        }
        assert summary.delta_of(report) == (0.012, 0.005, 0.019)

    def test_a_baseline_report_has_none(self) -> None:
        """The first cycle has no champion, and its report records that as a null
        rather than as a delta of zero."""
        assert summary.delta_of({"delta": None}) is None
