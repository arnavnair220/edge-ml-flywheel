"""Tests for the one part of the export that is a rule rather than a library call.

`training.quantization` names the nodes int8 quantization must leave alone. It
reads a shape of this project's own rather than `onnx.NodeProto` precisely so
these run with no container packages installed, which is the arrangement the
gates and the bootstrap already use.

The graphs below are Ultralytics' node naming in miniature: every node under
`/model.<n>/`, numbered in definition order, with `Detect` last.
"""

import pytest

from edge_ml_flywheel.training.quantization import CONV, Node, head_nodes


def conv(name: str) -> Node:
    return Node(name=name, op_type=CONV)


def graph(head: int = 23, depth: int = 3) -> list[Node]:
    """A backbone of convolutions and a head holding a convolution and the box
    decoding arithmetic that follows it."""
    body = [conv(f"/model.{index}/conv/Conv") for index in range(depth)]
    return [
        *body,
        conv(f"/model.{head}/cv2.0/conv/Conv"),
        conv(f"/model.{head}/cv3.0/conv/Conv"),
        Node(name=f"/model.{head}/Sigmoid", op_type="Sigmoid"),
        Node(name=f"/model.{head}/Concat", op_type="Concat"),
    ]


class TestNamingTheHead:
    def test_excludes_every_node_of_the_last_module(self) -> None:
        """Every node and not only its convolutions: the claim is that the head
        is not quantized, and it survives the set of quantized op types widening
        past `Conv`."""
        assert head_nodes(graph()) == [
            "/model.23/Concat",
            "/model.23/Sigmoid",
            "/model.23/cv2.0/conv/Conv",
            "/model.23/cv3.0/conv/Conv",
        ]

    def test_leaves_the_backbone_quantizable(self) -> None:
        excluded = set(head_nodes(graph()))

        assert not any(node.name in excluded for node in graph()[:3])

    def test_finds_the_head_by_the_highest_index_not_by_a_literal(self) -> None:
        """The head's index is a property of the model definition, so a release
        that adds a layer moves it. Numbering, not counting."""
        assert head_nodes(graph(head=31)) == sorted(node.name for node in graph(head=31)[3:])

    def test_orders_the_indices_numerically(self) -> None:
        """`/model.9/` sorts above `/model.10/` as a string, and picking the head
        that way would quantize it and leave a backbone layer in float."""
        nodes = [conv("/model.9/conv/Conv"), conv("/model.10/cv2.0/conv/Conv")]

        assert head_nodes(nodes) == ["/model.10/cv2.0/conv/Conv"]


class TestRefusals:
    """All three raise rather than fall back. A graph this rule cannot read is a
    graph whose head would be quantized silently, which is the accuracy loss the
    rule exists to prevent -- so the export produces no artifact at all."""

    def test_refuses_a_graph_with_no_module_paths(self) -> None:
        nodes = [conv("Conv_0"), conv("Conv_1")]

        with pytest.raises(ValueError, match="cannot be named"):
            head_nodes(nodes)

    def test_refuses_a_last_module_holding_no_convolution(self) -> None:
        """Then it is not the detection head this rule assumes the last module to
        be, and excluding it would leave the real head quantized."""
        nodes = [conv("/model.0/conv/Conv"), Node(name="/model.1/Concat", op_type="Concat")]

        with pytest.raises(ValueError, match="not the detection head"):
            head_nodes(nodes)

    def test_refuses_a_graph_that_is_all_head(self) -> None:
        """Excluding it would leave nothing to quantize, so the artifact would be
        fp32 under an int8 filename."""
        nodes = [conv("/model.23/cv2.0/conv/Conv"), conv("/model.23/cv3.0/conv/Conv")]

        with pytest.raises(ValueError, match="nothing to quantize"):
            head_nodes(nodes)
