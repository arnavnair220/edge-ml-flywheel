"""Pass/fail checks with pre-declared thresholds, and the reason behind each.

A gate is a pure function returning a `GateResult`. Nothing here trains, scores,
reads S3 or decodes an image: the inputs are a purchase, a partition and numbers
`evaluation` already computed, so a promotion decision is reproducible from a
saved cycle and testable without a model.

Two properties hold across all of them.

**A verdict is never recorded without its reason.** `GateResult` carries both,
and a rejection with its reason is the artifact the project is built to produce
(design section 5). A bare `False` six weeks later is a fact nobody can act on.

**A gate reports every condition that failed, not the first.** The next attempt
costs a training run, so short-circuiting turns one diagnosis into two cycles.

`Gate` names four checks and this package implements all four, but not at the
same moment in a cycle. `DATA`, `QUALITY` and `EDGE` are computed from a cycle's
own artifacts and decide whether the challenger is registered and promoted.
`CANARY` is computed from what a device reported after the promoted artifact
reached it, so it is asked afterwards and what it decides is whether the rollout
continues or rolls back. Its input is a `ReplayReport`, which is a reduction of
telemetry rather than an AWS call, so it stays a pure function like the rest.
"""

from edge_ml_flywheel.gates.canary import canary_gate
from edge_ml_flywheel.gates.data import data_gate
from edge_ml_flywheel.gates.edge import edge_gate
from edge_ml_flywheel.gates.quality import quality_gate
from edge_ml_flywheel.gates.thresholds import DEFAULT, Gate, Thresholds

__all__ = [
    "DEFAULT",
    "Gate",
    "Thresholds",
    "canary_gate",
    "data_gate",
    "edge_gate",
    "quality_gate",
]
