"""ONNX export and int8 quantization, as the training job runs them.

The deployed artifact is not the checkpoint. The fleet runs ONNX Runtime on an
ARM CPU (design section 6), so what ships is an int8 ONNX file and what the
manifest's digest names is that file. Both steps run here, in the job that
produced the weights, so the artifact and its digest are taken in one place --
the same rule that puts `model.pt`'s digest inside the container rather than
beside the object in the bucket.

Ultralytics' own `int8=True` targets TensorRT and OpenVINO rather than ONNX
Runtime, so the export is fp32 and the quantization is a second pass over it
through `onnxruntime.quantization`. `quantization` holds the part of that pass
worth testing; everything here talks to a library.

Calibration frames come from the training images already on the job's local
disk. `eval` frames would fold the cohort the model is scored on into how the
model is built, and this job could not read them in any case -- the label wall
denies it the split `eval` is drawn from.
"""

import logging
import random
import shutil
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import onnx
from numpy.typing import NDArray
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process
from ultralytics import YOLO

from edge_ml_flywheel.training import quantization

log = logging.getLogger("edge_ml_flywheel.training")

# Pinned rather than left to Ultralytics' default, for the reason the container
# tag is pinned: the opset is half of what a runtime has to implement, so a
# floating one is a deployed artifact that changes shape without anyone changing
# it. 17 is old enough that every ONNX Runtime the fleet could run supports it.
OPSET = 17

# Enough frames for the activation ranges to settle and few enough that
# calibration is a minute rather than an hour. The usual range is 100-500.
CALIBRATION_FRAMES = 256

# Ultralytics letterboxes to this grey, and calibration sees what inference will.
_PAD = 114

_MEGABYTE = 1024 * 1024


def to_onnx(checkpoint: Path, image_size: int, destination: Path) -> Path:
    """The fp32 graph, with NMS deliberately left outside it.

    Two reasons pointing the same way. The quantizer has no int8 form for the
    NMS ops and would leave a float island mid-graph; and the runtime on the
    device applies NMS itself, over raw outputs, exactly as the scoring job's
    Ultralytics backend does. Keeping it out is what makes the artifact the same
    shape in both places.

    `dynamic=False` because the device runs one frame at a time at one
    resolution, and a static shape is what lets the quantizer fold shapes into
    constants. `simplify=False` because the simplifier is a second package to
    pin and `quant_pre_process` performs the same folding on the way in.

    Precision is left at the default rather than asked for. Ultralytics' `half`
    is deprecated in favour of a `quantize` argument, and neither is the lever
    here: the export is fp32 by construction and `quantize` below is the pass
    that makes it int8, through ONNX Runtime rather than through a format
    Ultralytics targets at TensorRT.
    """
    exported = YOLO(str(checkpoint)).export(
        format="onnx",
        imgsz=image_size,
        opset=OPSET,
        dynamic=False,
        simplify=False,
        nms=False,
    )
    produced = Path(str(exported))
    if not produced.is_file():
        raise SystemExit(f"the ONNX export reported {produced}, which is not a file")

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(produced), destination)
    log.info(
        "exported fp32 ONNX at %d px, opset %d, %.1f MB",
        image_size,
        OPSET,
        destination.stat().st_size / _MEGABYTE,
    )
    return destination


def calibration_frames(images_dir: Path, seed: int, count: int = CALIBRATION_FRAMES) -> list[Path]:
    """A fixed sample of this cycle's training images.

    Drawn from its own `Random` rather than the seeded global one: this runs
    after training, so it cannot disturb the augmentation order pairing depends
    on, and seeding it from the job's seed still makes the draw a property of
    the run rather than of the order the filesystem listed.
    """
    every = sorted(path for path in images_dir.iterdir() if path.is_file())
    if not every:
        raise SystemExit(f"{images_dir} holds no image to calibrate against")
    if len(every) <= count:
        return every
    return sorted(random.Random(seed).sample(every, count))


def _tensor(path: Path, size: int) -> NDArray[np.float32]:
    """One frame, preprocessed the way inference will preprocess it.

    Letterboxed rather than stretched: calibration measures the range of values
    each layer sees, and a distribution collected from differently shaped inputs
    would set the scales for images the model never receives.
    """
    image = cv2.imread(str(path))
    if image is None:
        raise SystemExit(f"{path} could not be decoded, so calibration cannot read it")

    height, width = image.shape[:2]
    scale = min(size / height, size / width)
    resized = cv2.resize(
        image, (round(width * scale), round(height * scale)), interpolation=cv2.INTER_LINEAR
    )

    canvas = np.full((size, size, 3), _PAD, dtype=np.uint8)
    top = (size - resized.shape[0]) // 2
    left = (size - resized.shape[1]) // 2
    canvas[top : top + resized.shape[0], left : left + resized.shape[1]] = resized

    chw = canvas[..., ::-1].transpose(2, 0, 1)
    return np.ascontiguousarray(chw, dtype=np.float32)[None] / 255.0


# `CalibrationDataReader` is `Any` to mypy -- `onnxruntime` is a container
# package with no stub, the same standing `ultralytics` and `torch` have -- and
# strict mode refuses to subclass `Any`. The alternative is a local protocol
# duplicating the one method, which would type-check against itself rather than
# against the library.
class _Frames(CalibrationDataReader):  # type: ignore[misc]
    """Feeds the calibrator one preprocessed frame at a time.

    A reader rather than an array because the frames are decoded lazily: 256
    images at 416 px is small, but the interface is a stream and holding the
    whole batch resident buys nothing.
    """

    def __init__(self, frames: Sequence[Path], input_name: str, image_size: int) -> None:
        self._input = input_name
        self._size = image_size
        self._frames = iter(frames)

    def get_next(self) -> dict[str, Any] | None:
        path = next(self._frames, None)
        if path is None:
            return None
        return {self._input: _tensor(path, self._size)}


def quantize(
    fp32: Path, destination: Path, frames: Sequence[Path], image_size: int, work: Path
) -> Path:
    """int8 over the exported graph, at the settings `quantization` argues for.

    QDQ format, because it is what ONNX Runtime's CPU provider optimizes for and
    what leaves the excluded nodes genuinely in float rather than wrapped in a
    cast. Convolutions only, so the box decoding arithmetic after the head keeps
    its precision without being named. Per-channel weights and the head excluded
    for the reasons `quantization` gives.
    """
    prepared = work / "prepared.onnx"
    quant_pre_process(fp32, prepared)

    graph = onnx.load(str(prepared))
    nodes = [quantization.Node(name=node.name, op_type=node.op_type) for node in graph.graph.node]
    excluded = quantization.head_nodes(nodes)
    input_name = str(graph.graph.input[0].name)

    quantize_static(
        prepared,
        destination,
        _Frames(frames, input_name, image_size),
        quant_format=QuantFormat.QDQ,
        op_types_to_quantize=[quantization.CONV],
        per_channel=True,
        activation_type=QuantType.QInt8,
        weight_type=QuantType.QInt8,
        nodes_to_exclude=excluded,
        calibrate_method=CalibrationMethod.MinMax,
    )
    prepared.unlink()

    log.info(
        "quantized to int8 over %d calibration frames, %d head node(s) left in float, %.1f MB",
        len(frames),
        len(excluded),
        destination.stat().st_size / _MEGABYTE,
    )
    return destination


def deployable(
    checkpoint: Path, frames: Sequence[Path], destination: Path, image_size: int, work: Path
) -> Path:
    """The artifact the fleet runs: export, then quantize over the export.

    The calibration frames are chosen by the caller rather than here, because
    choosing them is a question about the cycle's dataset and the two steps below
    are questions about a graph.

    Intermediates live under `work` rather than beside the destination, because
    the destination is the directory SageMaker collects into `model.tar.gz` and
    a leftover fp32 graph there would ship as part of the model.
    """
    fp32 = work / "model.fp32.onnx"
    to_onnx(checkpoint, image_size, fp32)
    quantize(fp32, destination, frames, image_size, work)
    fp32.unlink()
    return destination
