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

`Gate` names four checks and this package implements two. `EDGE` and `CANARY`
read p95 latency on an ARM64 task and two replay hours of device telemetry, and
nothing produces either yet -- writing them now would mean inventing their input
shape and testing against the invention. They land with their producers, in
phases 6 and 7; `thresholds.Gate` reserves the names in the meantime so the
report format and the state machine can already be written against them.
"""

from edge_ml_flywheel.gates.data import data_gate
from edge_ml_flywheel.gates.quality import quality_gate
from edge_ml_flywheel.gates.thresholds import DEFAULT, Gate, Thresholds

__all__ = ["DEFAULT", "Gate", "Thresholds", "data_gate", "quality_gate"]
