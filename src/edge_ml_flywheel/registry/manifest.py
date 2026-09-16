"""The manifest document, and the gate report it takes its verdict from.

Both halves are pure. `to_document` and `from_document` hold the entire storage
schema of a manifest; `gates` holds the reading of a report written by a job in
another container. So the encoding -- the part written once, under a key no
second write corrects -- round-trips in a test with no credentials and no model.

**The manifest is a witness, not a copy.** `ModelManifest.disagreements` is only
able to catch a model that ran a configuration its run never declared because the
manifest states its own view of the versions. So nothing here fills a field in
from the run registration to make it agree; the caller supplies what the cycle
actually did, and the disagreement is checked afterwards.

**The gate report is parsed rather than trusted.** It arrives as JSON written by
another process, so every field it names is checked on the way in and a verdict
with no gates behind it is refused here rather than becoming a manifest whose
`gates_passed` is quietly false for a reason nobody can see.
"""

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Any

from edge_ml_flywheel.conventions import (
    Cohort,
    GateResult,
    ModelManifest,
    ModelVersion,
    PartitionVersion,
    RecipeVersion,
    Seed,
    columns,
    parse_model_version,
)


def to_document(manifest: ModelManifest) -> dict[str, Any]:
    """A manifest as the JSON object at `conventions.model_manifest_key`.

    Every field encoded explicitly rather than by walking the dataclass, for
    `run.registration.to_item`'s reason: four of them need a decision -- a
    `datetime` has no JSON type, a `frozenset` has no stable order, a mapping
    keyed by `Seed` becomes string keys, and a tuple of dataclasses has to be
    flattened. What a generic walk would have bought is drift protection, so that
    is bought directly: the field set is checked against the schema of record on
    the way out, and a field added to `ModelManifest` and forgotten here fails at
    the write rather than being missing from a document permanently.

    `run_id` and `cycle` are deliberately not written. They are properties rather
    than fields, already inside `version`, and a document that restated them
    would be a document two readers could disagree about.

    Sorted where order is not meaningful -- the cohorts, the seeds -- so two
    manifests of the same facts are the same bytes, and a diff of two cycles is
    about what changed.
    """
    document: dict[str, Any] = {
        "version": str(manifest.version),
        "created_at": manifest.created_at.isoformat(),
        "git_commit": manifest.git_commit,
        "partition_version": int(manifest.partition_version),
        "recipe_version": int(manifest.recipe_version),
        "cohorts_trained_on": sorted(cohort.value for cohort in manifest.cohorts_trained_on),
        "labels_spent": manifest.labels_spent,
        "deployed_seed": int(manifest.deployed_seed),
        "artifact_sha256": {
            str(seed): digest for seed, digest in sorted(manifest.artifact_sha256.items())
        },
        "gates": [
            {"gate": str(gate.gate), "passed": gate.passed, "reason": gate.reason}
            for gate in manifest.gates
        ],
    }

    expected = set(columns(ModelManifest))
    missing = sorted(expected - set(document))
    if missing:
        raise ValueError(f"manifest fields with no encoding here: {missing}")

    return document


def from_document(document: Mapping[str, Any]) -> ModelManifest:
    """The inverse, with `ModelManifest.__post_init__` as the validator.

    Nothing here re-checks a value the dataclass already refuses. What it does
    have to get right is the types JSON cannot carry: a seed key read back as
    `"1"` is not the `Seed` the digest was filed under, and a mapping keyed by
    strings would make `deployed_seed in artifact_sha256` false for a manifest
    that is perfectly well formed.

    Used on the read-back after the write, which is where it earns its place: a
    document that parses is a document the promotion step can act on, and one
    that does not is a failure at the cycle that produced it rather than at the
    cycle that tried to promote it.
    """
    return ModelManifest(
        version=parse_model_version(str(document["version"])),
        created_at=datetime.fromisoformat(str(document["created_at"])),
        git_commit=str(document["git_commit"]),
        partition_version=PartitionVersion(int(document["partition_version"])),
        recipe_version=RecipeVersion(int(document["recipe_version"])),
        cohorts_trained_on=frozenset(Cohort(name) for name in document["cohorts_trained_on"]),
        labels_spent=int(document["labels_spent"]),
        deployed_seed=Seed(int(document["deployed_seed"])),
        artifact_sha256={
            Seed(int(seed)): str(digest)
            for seed, digest in dict(document["artifact_sha256"]).items()
        },
        gates=tuple(_gate(entry) for entry in document["gates"]),
    )


def _gate(entry: Mapping[str, Any]) -> GateResult:
    """One verdict, however it was serialized.

    Shared by the manifest's own round trip and by the gate report, because the
    two carry the same three fields in the same spelling -- `report_document` and
    `to_document` write one shape, so reading it twice would be one shape read
    two ways, and the copy that drifts is the one nothing round-trips.
    """
    return GateResult(
        gate=str(entry["gate"]),
        passed=bool(entry["passed"]),
        reason=str(entry["reason"]),
    )


def gates(report: Mapping[str, Any], version: ModelVersion) -> tuple[GateResult, ...]:
    """The verdicts out of a gate report, checked against the model they judge.

    The version check is the one that matters. A report is addressed by run and
    cycle and a model by its version, and the two are built from the same run and
    cycle -- so a mismatch here means the report at this cycle's key was written
    about another model, which is a manifest that would record the wrong verdict
    under a name that looks right.

    An empty gate list is refused rather than passed through. `gates_passed` is
    already false for it, so the manifest would be honest; but it would record
    "this model failed" where the truth is "no gate ran", and those are different
    facts about different bugs.
    """
    judged = str(report.get("version", ""))
    if judged != version:
        raise ValueError(
            f"the gate report at this cycle's key judges {judged!r}, not {version}. A manifest "
            f"built from it would record another model's verdict."
        )

    reported: Sequence[Any] = report.get("gates", ())
    if not reported:
        raise ValueError(
            f"the gate report for {version} lists no gate, so no check ran. A model with no "
            f"verdict is not a model that was rejected."
        )

    return tuple(_gate(entry) for entry in reported)
