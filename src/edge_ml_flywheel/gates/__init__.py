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

`Gate` names four checks and this package implements three. `EDGE` is complete:
both its thresholds -- artifact size and quantized accuracy against fp32 -- are
properties of a file and a number, and speed is reported off the fleet rather
than gated (design section 4.3), so nothing about it waits on silicon. `CANARY`
reads two replay hours of device telemetry and has no predicate at all, because
nothing produces that input yet and one written now would be tested against an
invented shape. `thresholds.Gate` reserves its name in the meantime so the report
format and the state machine can already be written against it.
"""

from edge_ml_flywheel.gates.data import data_gate
from edge_ml_flywheel.gates.edge import edge_gate
from edge_ml_flywheel.gates.quality import quality_gate
from edge_ml_flywheel.gates.thresholds import DEFAULT, Gate, Thresholds

__all__ = ["DEFAULT", "Gate", "Thresholds", "data_gate", "edge_gate", "quality_gate"]
