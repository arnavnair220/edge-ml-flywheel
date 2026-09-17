"""The replay component itself, as it runs on the device.

Started by the nucleus with the flags `component.run_script` writes. Everything
it needs arrives as an artifact Greengrass has already verified or as a
configuration value the deploying side resolved, so the component makes no
discovery call and has no opinion about where it is.

**It exits when the replay is done, and that is the design.** A component whose
run script returns zero is `FINISHED` to Greengrass and is not restarted; one
that returns non-zero is restarted and then marked broken. So the exit code is
the health signal, `starts` is what the restart looks like from the telemetry
side, and neither needs a daemon that stays up doing nothing between cycles on a
device sized for one process.

**Nothing on the device writes a file to S3** (design section 7). It publishes to
IoT Core and a rule lands the object, which is what keeps the device's write
grant to a topic rather than to a bucket.

**The frames are the cycle's own sample**, shipped as an artifact rather than
chosen here. A device that picked its own frames would be a device whose
telemetry could not be joined to a scoring decision, which is the property design
section 7.2 asks for by name.
"""

import argparse
import hashlib
import io
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Final

import boto3
import numpy as np
import onnxruntime
from numpy.typing import NDArray
from PIL import Image

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    TELEMETRY_BATCH,
    Cohort,
    FrameRow,
    ImageId,
    ModelVersion,
    RunId,
    parse_image_id,
    parse_model_version,
    parse_run_id,
    raw_image_key,
)
from edge_ml_flywheel.fleet import detect, telemetry

log = logging.getLogger("edge_ml_flywheel.fleet.replay")

# Ultralytics' letterbox fill. Matched rather than chosen, for `detect`'s reason:
# the device pass and the cloud pass should differ by the silicon, not by the
# grey they pad with.
_PAD: Final = (114, 114, 114)

# How much of the model file is hashed at a time. The artifact is a few megabytes
# and the device has half a gigabyte, so this is about not reading the file twice
# rather than about not fitting it.
_DIGEST_CHUNK: Final = 1024 * 1024

# The split the pool is drawn from, resolved through `COHORT_SPLIT` rather than
# spelled `train`. The device's whole reach into the data bucket is built from
# this, so taking it from the mapping that defines the partition is what keeps a
# device unable to address a `val` image -- which is where `eval` lives.
_SPLIT: Final = COHORT_SPLIT[Cohort.POOL]


def _parser() -> argparse.ArgumentParser:
    """The flags `component.run_script` writes, underscored to match the jobs."""
    parser = argparse.ArgumentParser(prog="replay")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--frames", required=True, type=Path)
    parser.add_argument("--work", required=True, type=Path)
    parser.add_argument("--thing", required=True)
    parser.add_argument("--run_id", required=True)
    parser.add_argument("--version", required=True)
    parser.add_argument("--topic", required=True)
    parser.add_argument("--endpoint", required=True, help="The account's ATS data endpoint.")
    parser.add_argument("--bucket", required=True, help="The data bucket the frames are read from.")
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--confidence_floor", type=float, required=True)
    return parser


def digest(model: Path) -> str:
    """The sha256 of the file this component is about to load.

    Computed over the bytes on disk rather than taken from the recipe, which is
    the whole point of reporting it: Greengrass checking its own download against
    its own recipe is a closed loop, and this is the half that can disagree with
    `ModelManifest.artifact_sha256` when a recipe was built over the wrong
    object.
    """
    hasher = hashlib.sha256()
    with model.open("rb") as handle:
        while chunk := handle.read(_DIGEST_CHUNK):
            hasher.update(chunk)
    return hasher.hexdigest()


def starts(work: Path, version: ModelVersion) -> int:
    """How many times this component has started for this version, counting now.

    Kept in the work directory, which Greengrass preserves across a restart of
    the same component, and keyed by version because it is preserved across a
    *new* one as well -- an unkeyed counter would report the previous cycle's
    restarts against this cycle's model.

    Read-then-write rather than a conditional anything, and that is safe here for
    a reason it is not in `run.control`: the nucleus runs one instance of a
    component at a time, so there is no second writer to race with.
    """
    counter = work / f"starts-{parse_model_version(version)}.txt"
    previous = int(counter.read_text().strip()) if counter.is_file() else 0
    now = previous + 1
    counter.write_text(f"{now}\n")
    return now


def frame_ids(frames: Path) -> tuple[ImageId, ...]:
    """The sampled image IDs, validated before a single image is fetched.

    Every ID at once rather than as they are used, because the failure worth
    catching is a manifest built wrong, and finding it on frame 400 means having
    spent the replay to discover it.
    """
    document = json.loads(frames.read_text())
    if not isinstance(document, list) or not document:
        raise SystemExit(f"{frames} is not a non-empty list of image IDs")
    return tuple(parse_image_id(str(value)) for value in document)


def letterbox(image: Image.Image, size: int) -> NDArray[np.float32]:
    """One image as the network's input tensor: padded square, RGB, CHW, 0 to 1.

    Aspect ratio preserved and the remainder padded, rather than a plain resize.
    A 1280x720 frame squashed to a square moves every box's shape, and the model
    was trained and exported against the padded form -- so a plain resize would
    measure the same graph on an input distribution it has never seen and report
    the difference as the device's.
    """
    scale = min(size / image.width, size / image.height)
    width, height = max(1, round(image.width * scale)), max(1, round(image.height * scale))

    canvas = Image.new("RGB", (size, size), _PAD)
    canvas.paste(
        image.resize((width, height), Image.Resampling.BILINEAR),
        ((size - width) // 2, (size - height) // 2),
    )

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def _fetch(s3: Any, bucket: str, image_id: ImageId, size: int) -> NDArray[np.float32]:
    """One frame, fetched and preprocessed.

    Deliberately outside the timed section in `main`. What design section 4.3
    wants per version is what the model costs on this silicon, and a download
    over the public internet is a measurement of the network on the afternoon it
    ran.
    """
    body = s3.get_object(Bucket=bucket, Key=raw_image_key(image_id, _SPLIT))["Body"].read()
    with Image.open(io.BytesIO(body)) as image:
        return letterbox(image.convert("RGB"), size)


def _publish(client: Any, topic: str, document: dict[str, Any]) -> None:
    client.publish(topic=topic, qos=1, payload=json.dumps(document).encode())


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _parser().parse_args(argv)

    run_id: RunId = parse_run_id(args.run_id)
    version: ModelVersion = parse_model_version(args.version)
    started = starts(args.work, version)
    artifact = digest(args.model)
    ids = frame_ids(args.frames)
    log.info("%s replaying %d frames of %s, start %d", args.thing, len(ids), version, started)

    s3 = boto3.client("s3")
    iot = boto3.client("iot-data", endpoint_url=f"https://{args.endpoint}")

    # Cold start is the session plus the first inference, and deliberately not
    # the process. Interpreter startup and the onnxruntime import cost the same
    # for every version, so including them would put a constant in front of the
    # number design section 4.3 asks to be charted *per version* -- which is the
    # part that depends on the artifact.
    #
    # The two halves are timed separately because the first frame's download sits
    # between them, and a cold start measured across that would be a measurement
    # of the network on the afternoon it ran.
    loading = time.perf_counter()
    session = onnxruntime.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    feed = session.get_inputs()[0].name
    session_ms = (time.perf_counter() - loading) * 1000.0

    source = telemetry.Source(run_id=run_id, version=version, thing=args.thing)
    batch: list[FrameRow] = []
    seq = 0
    cold_start_ms = 0.0

    for index, image_id in enumerate(ids):
        tensor = _fetch(s3, args.bucket, image_id, args.image_size)

        began = time.perf_counter()
        output = session.run(None, {feed: tensor})[0]
        elapsed_ms = (time.perf_counter() - began) * 1000.0

        if index == 0:
            cold_start_ms = session_ms + elapsed_ms
            log.info(
                "cold start %.0f ms, of which %.0f ms was the session", cold_start_ms, session_ms
            )

        # Warmup frames are inferred and not recorded. The first passes through a
        # fresh session pay for lazy allocation and a cold instruction cache, and
        # a p95 that included them would describe the start of the run rather
        # than the run.
        if index < args.warmup:
            continue

        batch.append(
            FrameRow(
                image_id=image_id,
                inference_ms=elapsed_ms,
                scores=detect.scores_of(output.astype(np.float32), args.confidence_floor),
            )
        )
        if len(batch) >= TELEMETRY_BATCH:
            _publish(iot, args.topic, telemetry.frames_document(source, seq, batch))
            seq += 1
            batch = []

    if batch:
        _publish(iot, args.topic, telemetry.frames_document(source, seq, batch))

    # Last, so that its presence means the run finished rather than that it
    # began -- `telemetry.replay_document` states the same rule from the reader's
    # side.
    _publish(
        iot,
        args.topic,
        telemetry.replay_document(
            source=source,
            artifact_sha256=artifact,
            cold_start_ms=cold_start_ms,
            starts=started,
            replayed=len(ids) - args.warmup,
        ),
    )
    log.info("replayed %d frames after %d warmup", len(ids) - args.warmup, args.warmup)


if __name__ == "__main__":
    main()
