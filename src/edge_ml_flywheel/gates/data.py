"""The data gate: is this cycle's batch usable at all?

Four checks over what was bought, before anything is asked about the model. All
four are pure functions of the purchase, the partition and the run's budget --
nothing here opens a file or a socket, so a gate verdict is reproducible from a
saved purchase.

**Leakage is checked again, downstream, on purpose.** `oracle.cohorts` already
refuses a batch reaching into `eval`, and this re-runs the identical judgement on
what the ledger says was actually bought. That is not redundancy: the oracle's
check runs inside the oracle, on the list it was handed, and this one runs in the
cycle, on the labels that came back. A bug that lets the two disagree is exactly
the bug worth catching, and the design calls this the single most valuable check
in the system (section 4.1). It reuses `refusals` rather than restating it, so
there is still one implementation of what `eval` means.

**Integrity checks are deliberately absent.** Design section 4.1 also lists
corrupt files, wrong resolution and blank frames. Those are not predicates over a
purchase -- they decode a thousand JPEGs -- and ingest already validated
resolution against `NATIVE_IMAGE_SIZE` and recorded a sha256 per image, so the
failure they look for has been ruled out upstream. Verifying the bytes still
match their recorded digest is a job for whatever stages them, not for a gate.

**Distribution shift is not checked, and never will be here.** A concentrated
snow purchase is the selector working. The batch's composition is charted against
the pool's base rates and gated on by nothing (design section 4.1).
"""

from collections import Counter
from collections.abc import Sequence
from typing import Final

from edge_ml_flywheel.conventions import ClassSet, GateResult
from edge_ml_flywheel.gates.thresholds import DEFAULT, Gate, Thresholds
from edge_ml_flywheel.oracle.cohorts import PURCHASABLE, Cohorts, refusals
from edge_ml_flywheel.oracle.labels import SoldLabel

# How many offending items a reason lists before summarizing. `cohorts` uses the
# same figure for the same reason: enough to see a pattern without putting a
# whole batch into a string that lands in a manifest.
_REPORTED: Final = 5


def _leakage(cohorts: Cohorts, labels: Sequence[SoldLabel]) -> str | None:
    """Every purchased image is in the pool, or the cycle stops.

    Hard, with no override and no threshold to soften it. An image from `eval`
    in the training set invalidates every number the run has produced, and it
    does so silently -- the metric goes up, which is what a leak looks like.

    Covers `reserve` and an unassigned ID in the same pass, and covers the
    design's cheap second form -- that every purchase came from `train` -- by
    construction: `pool` is the only cohort drawn from `train` that is sellable,
    so anything from `val` is refused by being outside it.
    """
    wrong = refusals(cohorts, [label.image_id for label in labels])
    if not wrong:
        return None
    listed = ", ".join(f"{image} ({reason})" for image, reason in sorted(wrong.items())[:_REPORTED])
    more = f", and {len(wrong) - _REPORTED} more" if len(wrong) > _REPORTED else ""
    return (
        f"{len(wrong)} of {len(labels)} purchased images are not in {PURCHASABLE.value}: "
        f"{listed}{more}"
    )


def _volume(labels: Sequence[SoldLabel], thresholds: Thresholds) -> str | None:
    """The batch is big enough to have taught the challenger anything."""
    if len(labels) >= thresholds.min_new_images:
        return None
    return (
        f"{len(labels)} newly labeled images is under the floor of {thresholds.min_new_images}, "
        f"so a delta measured against it would not be comparable to any other cycle's"
    )


def _budget(labels: Sequence[SoldLabel], budget_per_cycle: int) -> str | None:
    """The batch is inside the cap the run registered.

    The oracle's ledger refuses an overspend at purchase time, so this failing
    means the two disagree -- labels exist that the budget never saw. Checked
    here because the cost-per-label figure is the project's headline number and a
    denominator nobody verified is not a measurement.
    """
    if len(labels) <= budget_per_cycle:
        return None
    return (
        f"{len(labels)} labels is over the registered budget of {budget_per_cycle} per cycle, so "
        f"the ledger and the purchase disagree"
    )


def _coverage(labels: Sequence[SoldLabel], classes: ClassSet, thresholds: Thresholds) -> str | None:
    """Every class in the class set got enough new instances to learn from.

    Counted over boxes rather than images: an image is the unit of purchase, but
    a class is taught by its boxes, and one image can carry twenty cars and no
    bus. Categories outside the class set are ignored -- the archive carries ten
    and a class set names four or nine (design section 3), so boxes the model
    does not predict are not evidence about it either way.
    """
    counts = Counter(box.category for label in labels for box in label.boxes)
    short = {
        name: counts[name]
        for name in classes.names
        if counts[name] < thresholds.min_instances_per_class
    }
    if not short:
        return None
    listed = ", ".join(f"{name} ({count})" for name, count in sorted(short.items())[:_REPORTED])
    return (
        f"{len(short)} class(es) got under {thresholds.min_instances_per_class} new instances "
        f"this cycle: {listed}"
    )


def data_gate(
    cohorts: Cohorts,
    labels: Sequence[SoldLabel],
    classes: ClassSet,
    budget_per_cycle: int,
    thresholds: Thresholds = DEFAULT,
) -> GateResult:
    """One verdict over the whole batch, naming every check that failed.

    Every check runs even after one has failed, and the reason lists all of them.
    A gate that short-circuits reports the first problem and hides the rest, so a
    cycle that leaked *and* underbought takes two runs to diagnose -- and each run
    is a training job.

    Leakage is reported first when several fail, because it is the one that says
    the run's earlier numbers are also suspect rather than that this cycle is.
    """
    failures = [
        failure
        for failure in (
            _leakage(cohorts, labels),
            _volume(labels, thresholds),
            _budget(labels, budget_per_cycle),
            _coverage(labels, classes, thresholds),
        )
        if failure is not None
    ]

    if failures:
        return GateResult(gate=Gate.DATA, passed=False, reason="; ".join(failures))

    boxes = sum(len(label.boxes) for label in labels)
    return GateResult(
        gate=Gate.DATA,
        passed=True,
        reason=(
            f"{len(labels)} images and {boxes} boxes, all in {PURCHASABLE.value}, within a budget "
            f"of {budget_per_cycle}, every one of {len(classes.names)} classes at "
            f"{thresholds.min_instances_per_class}+ new instances"
        ),
    )
