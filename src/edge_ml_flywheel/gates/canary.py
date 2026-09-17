"""The canary gate: does the promoted artifact behave on a device?

Three checks over one device's replay, and every one of them is operational
rather than statistical. A detector is deterministic -- the same image gives the
same answer every time -- so there is no run-to-run spread on a device and any
"within N standard deviations" test over one is vacuous (design section 4.5). It
either always passes or divides by zero. What can genuinely fail is the
plumbing, so that is what this asks about.

**The bytes that ran are the bytes the gates judged.** The digest the device
computed over the file it loaded, against the one the manifest recorded where the
file was produced. Greengrass verifies its own copy against its own recipe on
download, which is a closed loop that cannot catch a recipe built over the wrong
object; this is the independent half, and it is the check the other two are
worthless without.

**The component ran once and finished.** Greengrass restarts a component that
exits, so a second start is the agent catching something that crashed -- the
failure design section 4.5 calls hot-swapping breaking the agent. A replay that
stopped short is the same finding arriving by a different route, and both are
read off the counts rather than off an error nobody was watching for.

**It is no slower than the model it replaces.** The only comparison here, and the
only check needing a champion. What it catches is a graph that reached the device
intact and runs at a fraction of the speed, which every cloud pass would miss:
the int8 pass scores on a cloud CPU, and what an ARM device makes of the same
operators is not a question that hardware answers.

**Three conditions, and design section 4.5 lists five.** Memory flatness over two
replay hours and the distance between two confidence distributions are both
measurements a single device on a single pass cannot make -- one needs the hours,
the other needs a champion replaying the same frames at the same time, which is
shadow mode and a second component. They are absent rather than approximated,
because a leak test over four minutes and a distribution distance over one sample
are checks that pass by construction and report themselves as evidence.
"""

from edge_ml_flywheel.conventions import GateResult, ReplayReport
from edge_ml_flywheel.gates.thresholds import DEFAULT, Gate, Thresholds


def _digest(report: ReplayReport, expected: str) -> str | None:
    """The device loaded the file the manifest names.

    Compared against `ModelManifest.artifact_sha256` for the deployed seed, which
    is taken where the bytes were produced. A mismatch is not a corrupt download
    -- Greengrass would have refused that -- it is a component pointed at a
    different object, which is a recipe built over the wrong version and a
    rollout of a model no gate ever saw.
    """
    if report.artifact_sha256 == expected:
        return None
    return (
        f"{report.thing} loaded an artifact with digest {report.artifact_sha256[:12]}, and the "
        f"manifest for {report.version} records {expected[:12]}"
    )


def _completed(report: ReplayReport) -> str | None:
    """The component started once and replayed every frame it was given.

    Both halves in one check because they are one question -- did this run
    finish cleanly -- and because a restart is usually *why* a replay came up
    short. Reporting them separately would put one cause under two headings.
    """
    failures = []
    if report.starts != 1:
        failures.append(
            f"the component started {report.starts} times, so Greengrass restarted something "
            f"that exited"
        )
    if not report.complete:
        failures.append(
            f"{report.reported} of {report.replayed} replayed frames arrived, so the run did not "
            f"finish or its telemetry did not"
        )
    return "; ".join(failures) if failures else None


def _throughput(
    report: ReplayReport, champion: ReplayReport | None, thresholds: Thresholds
) -> str | None:
    """The challenger is within the allowance of the champion's frame rate.

    A missing champion is not a failure and not a skip. A run's first promotion
    has nothing to be slower than, which is the same situation the quality gate
    meets at cycle 0 and answers the same way: the model is its own baseline, and
    the number is recorded so the next cycle has one.

    Only a drop is checked. A challenger that runs faster than the champion has
    not failed a canary -- it has quantized to a cheaper graph, which is a result
    rather than a regression.
    """
    if champion is None:
        return None

    floor = champion.throughput_fps * (1.0 - thresholds.max_throughput_drop)
    if report.throughput_fps >= floor:
        return None

    drop = (champion.throughput_fps - report.throughput_fps) / champion.throughput_fps
    return (
        f"throughput is {report.throughput_fps:.1f} frames/s against the champion's "
        f"{champion.throughput_fps:.1f}, a {drop:.1%} drop over the "
        f"{thresholds.max_throughput_drop:.0%} allowance"
    )


def canary_gate(
    report: ReplayReport,
    expected_sha256: str,
    champion: ReplayReport | None = None,
    thresholds: Thresholds = DEFAULT,
) -> GateResult:
    """One verdict over one device's replay.

    `champion` is that model's own replay, read from the telemetry of the cycle
    it was deployed in rather than measured again. A replay is a function of a
    model and a device, and re-running the champion's would be re-deriving an
    answer already in the bucket -- the same argument that keeps the champion's
    eval scores cached (design section 7).

    Every check runs even after one has failed, for `edge_gate`'s reason. The
    next attempt here costs a rollback and a redeploy rather than a training run,
    but the verdict is also the record of what the rollout found, and one that
    stopped at the first failure would describe less than what happened.

    A pass reports the two numbers the writeup wants beside the verdict (design
    section 4.3): they are measured here and gated nowhere, so this reason is the
    only place a cycle records what the model did on real ARM silicon.
    """
    failures = [
        failure
        for failure in (
            _digest(report, expected_sha256),
            _completed(report),
            _throughput(report, champion, thresholds),
        )
        if failure is not None
    ]

    if failures:
        return GateResult(gate=Gate.CANARY, passed=False, reason="; ".join(failures))

    against = (
        f" against the champion's {champion.throughput_fps:.1f}"
        if champion is not None
        else ", the run's first deployment and its own baseline"
    )
    return GateResult(
        gate=Gate.CANARY,
        passed=True,
        reason=(
            f"{report.thing} replayed {report.reported} frames on one start, digest verified, "
            f"p95 {report.p95_ms:.0f} ms, cold start {report.cold_start_ms:.0f} ms, "
            f"{report.throughput_fps:.1f} frames/s{against}"
        ),
    )


def detections_stand(report: ReplayReport, expected_sha256: str) -> bool:
    """Whether what the device wrote may be ranked, and a batch bought from it.

    **A different question from whether the rollout stands**, over two of the
    same three checks. `canary_gate` decides what the fleet keeps running;
    this decides what the cycle is allowed to spend its budget on, and the two
    are not the same judgement over the same evidence.

    The digest and the completion check are statements about the detections. The
    wrong bytes produced them, or the pass did not finish -- either way the file
    is not a ranking of this cycle's sample by the model the gates were reported
    over, and buying a thousand labels from it spends real budget on a fiction.

    Throughput is not. A model that ran correctly and slowly wrote exactly the
    detections a fast one would have; the rollout is rolled back for being slow
    and the ranking is untouched by it. That asymmetry is the whole reason this
    is a second function rather than a field on the verdict.

    The thresholds are not an argument, because neither check has one. Both are
    equalities -- the digest matches or it does not, the frames arrived or they
    did not -- which is what makes them safe to read this way.
    """
    return _digest(report, expected_sha256) is None and _completed(report) is None
