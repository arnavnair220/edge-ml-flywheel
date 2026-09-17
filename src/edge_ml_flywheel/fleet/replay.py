"""The scoring component itself, as it runs on the device.

Started by the nucleus with the flags `component.run_script` writes. Everything
it needs arrives as an artifact Greengrass has already verified or as a
configuration value the deploying side resolved, so the component makes no
discovery call and has no opinion about where it is.

**This is the cycle's pass over the unlabeled pool.** The predictions it produces
are what selection ranks and what the label budget is spent against, so the
component is not an observer of the loop but a step in it -- the execution that
deployed it is blocked on the task token it was handed, and resuming that
execution is this component's last act.

**It exits when the pass is done, and that is the design.** A component whose run
script returns zero is `FINISHED` to Greengrass and is not restarted; one that
returns non-zero is restarted and then marked broken. So the exit code is the
health signal, `starts` is what the restart looks like from the telemetry side,
and neither needs a daemon that stays up doing nothing between cycles on a device
sized for one process.

**Two outputs, by two routes, because they fail differently.** The detections go
to S3 as one parquet, since a ranking computed from whichever MQTT messages
survived is not a ranking. The latencies and the summary go to IoT Core, where a
lost message costs a percentile rather than a purchase. Design section 7 said
nothing on the device writes a file; that held while the device only reported on
itself, and the write grant it buys is one prefix in the artifacts bucket.

**The frames are the cycle's own sample**, shipped as an artifact rather than
chosen here. A device that picked its own frames would be a device whose
predictions could not be joined to a scoring decision, which is the property
design section 7.2 asks for by name.
"""

import argparse
import hashlib
import io
import json
import logging
import sys
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import boto3
import numpy as np
import onnxruntime
from numpy.typing import NDArray
from PIL import Image

from edge_ml_flywheel.conventions import (
    COHORT_SPLIT,
    CYCLE_DIGITS,
    NATIVE_IMAGE_SIZE,
    TELEMETRY_BATCH,
    Cohort,
    DetectionRow,
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
from edge_ml_flywheel.scoring import detections

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
    parser.add_argument(
        "--detections_bucket", required=True, help="The artifacts bucket the parquet is written to."
    )
    parser.add_argument(
        "--detections_key", required=True, help="Where this cycle's pool detections land."
    )
    # Empty rather than absent when nobody is waiting. A pass run by hand after a
    # device failure produces the same file and the same telemetry; what it must
    # not do is resume an execution that has already timed out and moved on.
    parser.add_argument(
        "--task_token", default="", help="The cycle's task token, if one is waiting."
    )
    parser.add_argument(
        "--cycle", type=int, required=True, help="The cycle this deployment belongs to."
    )
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--image_size", type=int, required=True)
    parser.add_argument("--confidence_floor", type=float, required=True)
    return parser


@dataclass(slots=True)
class Pass:
    """One device pass over one cycle's sample: what it runs with, and what it saw.

    An object rather than nine arguments to a generator, and the two measured
    fields are the reason it has to be one. `rows` yields detections, because
    that is what the parquet writer consumes, so the cold start and the frame
    count the summary is built from have nowhere to be returned. Holding them
    here rather than in a module-level variable means a second pass in one
    process reports its own numbers and not the first one's.
    """

    session: Any
    feed: str
    s3: Any
    iot: Any
    args: argparse.Namespace
    source: telemetry.Source
    session_ms: float
    cold_start_ms: float = 0.0
    measured: int = 0

    def rows(self, ids: tuple[ImageId, ...]) -> Iterator[DetectionRow]:
        """Every frame's detections, publishing the latencies on the way past.

        A generator because the parquet writer takes one, and the whole point of
        it taking one is that a pass over `POOL_SAMPLE` frames never exists in
        memory as a list. At up to `MAX_DETECTIONS` boxes a frame that list would
        be millions of dataclasses on a device with two gigabytes.

        The telemetry publish sits inside the loop for the same reason: batches
        go out as they fill, so a long pass reports progress and a device that
        dies halfway has still told the truth about the half it did.

        **Warmup frames are scored but not timed.** Their detections belong in
        the file -- a frame excluded from the ranking because it happened to be
        drawn first would be a hole in the sample nothing recorded -- while their
        latencies describe a cold session rather than the device, which is what
        design section 4.3 excludes them from.
        """
        args = self.args
        fit = detect.Letterbox(
            source_width=NATIVE_IMAGE_SIZE[0],
            source_height=NATIVE_IMAGE_SIZE[1],
            size=args.image_size,
        )
        batch: list[FrameRow] = []
        seq = 0

        for index, image_id in enumerate(ids):
            tensor = _fetch(self.s3, args.bucket, image_id, args.image_size)

            began = time.perf_counter()
            output = self.session.run(None, {self.feed: tensor})[0]
            elapsed_ms = (time.perf_counter() - began) * 1000.0

            if index == 0:
                self.cold_start_ms = self.session_ms + elapsed_ms
                log.info(
                    "cold start %.0f ms, of which %.0f ms was the session",
                    self.cold_start_ms,
                    self.session_ms,
                )

            yield from detect.detections_of(
                output.astype(np.float32), image_id, fit, args.confidence_floor
            )

            if index < args.warmup:
                continue

            self.measured += 1
            batch.append(FrameRow(image_id=image_id, inference_ms=elapsed_ms))
            if len(batch) >= TELEMETRY_BATCH:
                self._publish_frames(seq, batch)
                seq += 1
                batch = []

        if batch:
            self._publish_frames(seq, batch)

    def _publish_frames(self, seq: int, batch: list[FrameRow]) -> None:
        _publish(self.iot, self.args.topic, telemetry.frames_document(self.source, seq, batch))


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


def starts(work: Path, cycle: int) -> int:
    """How many times this component has started for this cycle, counting now.

    Kept in the work directory, which Greengrass preserves across a restart of
    the same component and across a new component version as well -- so an
    unkeyed counter would report every earlier cycle's starts against this one.

    **Keyed by cycle and not by model version**, which is the difference between
    a counter and a wrong answer. A cycle that rejects its challenger redeploys
    the standing champion, so one model version can be deployed three cycles
    running: keyed by it, the second cycle's clean install would report two
    starts and fail the canary's completion check for something that never
    happened. A cycle deploys exactly once, which is what makes it the key.

    Read-then-write rather than a conditional anything, and that is safe here for
    a reason it is not in `run.control`: the nucleus runs one instance of a
    component at a time, so there is no second writer to race with.
    """
    if cycle < 0:
        raise ValueError(f"cycle {cycle} is not a cycle this counter can be keyed by")
    counter = work / f"starts-c{cycle:0{CYCLE_DIGITS}d}.txt"
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

    The geometry comes from `detect.Letterbox` rather than being computed here,
    because `detect.rescale` has to undo exactly this paste. Two copies of the
    arithmetic would be a box offset by a few pixels in a file nothing checks.
    """
    fit = detect.Letterbox(source_width=image.width, source_height=image.height, size=size)

    canvas = Image.new("RGB", (size, size), _PAD)
    canvas.paste(
        image.resize((fit.width, fit.height), Image.Resampling.BILINEAR),
        (fit.pad_x, fit.pad_y),
    )

    array = np.asarray(canvas, dtype=np.float32) / 255.0
    return np.ascontiguousarray(array.transpose(2, 0, 1)[None])


def _fetch(s3: Any, bucket: str, image_id: ImageId, size: int) -> NDArray[np.float32]:
    """One frame, fetched and preprocessed.

    Deliberately outside the timed section in `scored`. What design section 4.3
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
    """Score the sample, or tell the cycle why it could not.

    The failure path is the reason this wraps `_run` rather than being it. An
    execution blocked on a task token has no other way to learn that the device
    gave up: without this it waits out the state's two hours to report a timeout,
    when the device knew the cause in the first minute. The exception is
    re-raised after, because the exit code is what Greengrass reads.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    args = _parser().parse_args(argv)
    try:
        _run(args)
    except Exception as error:
        _abandon(args.task_token, error)
        raise


def _abandon(task_token: str, error: Exception) -> None:
    """Fail the waiting task, and never at the expense of the real error.

    A token that has already timed out is refused by the service, and a raise
    here would replace the cause of the failure with the failure to report it.
    """
    if not task_token:
        return
    try:
        boto3.client("stepfunctions").send_task_failure(
            taskToken=task_token,
            error=type(error).__name__,
            cause=str(error)[:32768],
        )
        log.error("failed the cycle waiting on this pass: %s", error)
    except Exception:
        # Reporting a failure must never replace it. Whatever went wrong on the
        # device is the interesting error; that the token was also stale is a
        # line in the log.
        log.exception("could not report the failure to the waiting cycle")


def _run(args: argparse.Namespace) -> None:
    run_id: RunId = parse_run_id(args.run_id)
    version: ModelVersion = parse_model_version(args.version)
    started = starts(args.work, int(args.cycle))
    artifact = digest(args.model)
    ids = frame_ids(args.frames)
    log.info("%s scoring %d frames with %s, start %d", args.thing, len(ids), version, started)

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

    over = Pass(
        session=session,
        feed=feed,
        s3=s3,
        iot=iot,
        args=args,
        source=telemetry.Source(run_id=run_id, version=version, thing=args.thing),
        session_ms=session_ms,
    )

    parquet = Path(args.work) / Path(args.detections_key).name
    rows = detections.write(over.rows(ids), parquet)

    # The file before the summary, and the summary before the token. Each step
    # is the evidence for the one after it: a reader that sees the summary can
    # open the detections, and an execution that is resumed can rank them. The
    # reverse order would resume a cycle onto a key that is still uploading.
    s3.upload_file(str(parquet), args.detections_bucket, args.detections_key)
    log.info("wrote %d detections to s3://%s/%s", rows, args.detections_bucket, args.detections_key)
    parquet.unlink(missing_ok=True)

    _publish(
        iot,
        args.topic,
        telemetry.replay_document(
            source=over.source,
            artifact_sha256=artifact,
            cold_start_ms=over.cold_start_ms,
            starts=started,
            replayed=over.measured,
        ),
    )
    log.info("scored %d frames, %d of them measured", len(ids), over.measured)

    resume(args.task_token, args.detections_key, rows)


def resume(task_token: str, detections_key: str, rows: int) -> None:
    """Hand the cycle back, which is the last thing this component does.

    **The token is the completion signal, and nothing else is.** A pass that dies
    partway writes no parquet, publishes no summary and calls nothing here, so
    the execution waiting on it times out rather than ranking a fraction of a
    sample. That is why `select` does not have to count the device's rows against
    the draw: an incomplete pass never reaches a state where anything reads it.

    An empty token is a pass nobody is waiting on -- re-run by hand after a
    device failure, or driven from the CLI. It produces the same file and the
    same telemetry and resumes nothing, because the execution that issued the
    original token has long since timed out and a stale token is refused anyway.
    """
    if not task_token:
        log.info("no task token, so nothing is waiting on this pass")
        return

    boto3.client("stepfunctions").send_task_success(
        taskToken=task_token,
        output=json.dumps({"detections_key": detections_key, "detections": rows}),
    )
    log.info("resumed the cycle waiting on this pass")


if __name__ == "__main__":
    main()
