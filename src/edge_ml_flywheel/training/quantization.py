"""Which part of the exported graph keeps its precision, and why.

int8 quantization rounds every weight and activation onto 256 steps. Applied
flat across the whole graph it costs a small detector more relative mAP than the
edge gate allows (design section 4.3), and the two settings that recover most of
it are settings rather than discoveries -- so they are the starting
configuration here, not the remedy after a cycle spent failing.

**Per-channel weight scales**, which `export` passes and this module does not
decide. A convolution's output channels differ in range by orders of magnitude,
and one scale shared across them quantizes the narrow channels to a constant:
the filter stops contributing at all.

**The detection head left in float**, which is what this module names. Its
convolutions emit the box coordinates and the class logits directly, so a
rounded value there displaces a box rather than blurring a feature, and a box
displaced past the IoU threshold is scored as a miss instead of as a slightly
worse detection. The head is a small share of a model whose parameters are
overwhelmingly in the backbone, so excluding it costs the artifact-size
threshold almost nothing.

Naming the head is the only part that needs to read the graph. Ultralytics
exports each node under its module path -- `/model.<n>/...` -- numbered in
definition order, and `Detect` is the last module defined. So the highest `<n>`
present is the head, which is a property of how the model is written rather than
of any one release's layer count.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

# The op every quantizable weight in this graph belongs to. Named because the
# guards below are about convolutions specifically: a head with none is not the
# head, and a graph whose convolutions are all in the head has nothing to
# quantize.
CONV: Final = "Conv"

_MODULE_PATH: Final = re.compile(r"^/model\.(\d+)/")


@dataclass(frozen=True, slots=True)
class Node:
    """One graph node, reduced to what the exclusion rule reads.

    A shape of this project's own rather than `onnx.NodeProto`, for the reason
    the gates are pure functions over match arrays: `onnx` is a container
    package, absent from the lockfile the way `ultralytics` and `torch` are, and
    a rule expressible only with it installed is a rule the test suite never
    runs. `export` adapts the real graph onto this in one comprehension.
    """

    name: str
    op_type: str


def head_nodes(nodes: Sequence[Node]) -> list[str]:
    """The Detect head's node names, which quantization leaves at full precision.

    Every node under the highest `/model.<n>/` path, not only its convolutions:
    the statement is that the head is not quantized, and it stays true if the set
    of quantized op types ever widens beyond `Conv`.

    Both failures below raise rather than degrade. A graph this rule cannot read
    is a graph whose head would be quantized silently, and that is precisely the
    accuracy loss the rule exists to prevent -- so an export that cannot name its
    head does not produce an artifact at all.
    """
    by_module: dict[int, list[Node]] = {}
    for node in nodes:
        found = _MODULE_PATH.match(node.name)
        if found is not None:
            by_module.setdefault(int(found.group(1)), []).append(node)

    if not by_module:
        raise ValueError(
            "no node in the exported graph carries a /model.<n>/ path, so the detection head "
            "cannot be named and would be quantized with everything else"
        )

    last = max(by_module)
    head = by_module[last]
    if not any(node.op_type == CONV for node in head):
        raise ValueError(
            f"/model.{last}/ holds no {CONV}, so it is not the detection head this rule assumes "
            f"the last module to be"
        )

    names = {node.name for node in head}
    if not any(node.op_type == CONV and node.name not in names for node in nodes):
        raise ValueError(
            f"every {CONV} in the graph is under /model.{last}/, so excluding the head would "
            f"leave nothing to quantize"
        )

    return sorted(names)
