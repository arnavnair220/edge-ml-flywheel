"""What `fleet.deploy` asks Greengrass for.

`test_fleet_component` covers the request builders, which take the deploying
cycle and have always honoured it. This covers the layer above: whether the
callers actually pass one. That is the seam the builders cannot defend, because
a caller that omits an optional argument gets a documented fallback rather than
an error.
"""

from typing import Any

from edge_ml_flywheel.conventions import Cycle, RunId, new_model_version
from edge_ml_flywheel.fleet import deploy

RUN = RunId("20260812t143355z-v0-skeleton")
TARGET = "arn:aws:iot:us-east-1:123456789012:thinggroup/edge-ml-flywheel-devices"
COMPONENT = "edge-ml-flywheel.20260812t143355z-v0-skeleton"


class _Greengrass:
    """Captures the one request `redeploy` makes."""

    def __init__(self) -> None:
        self.request: dict[str, Any] = {}

    def create_deployment(self, **request: Any) -> dict[str, str]:
        self.request = request
        return {"deploymentId": "d-1"}


class _Session:
    def __init__(self) -> None:
        self.greengrass = _Greengrass()

    def client(self, name: str) -> Any:
        assert name == "greengrassv2"
        return self.greengrass


class TestTheCycleARolloutDeploysUnder:
    """A rollout is numbered by the cycle making it, never by the model's own.

    The two agree on every promoting cycle, which is why this went unseen until
    a cycle rejected its challenger: the champion was then redeployed under the
    component version of the cycle that trained it, whose recipe carries that
    cycle's number. The device keys its start counter by the cycle it is told,
    so the pass arrived looking like the earlier cycle starting a second time
    and the canary refused a run that had in fact started once and finished.
    """

    def test_a_rollout_keeping_the_champion_is_numbered_by_the_deploying_cycle(self) -> None:
        aws = _Session()
        champion = new_model_version(RUN, Cycle(0))

        deploy.redeploy(aws, champion, TARGET, task_token="opaque", cycle=Cycle(1))

        entry = aws.greengrass.request["components"][COMPONENT]
        assert entry["componentVersion"] == "0.1.0"

    def test_a_rollout_promoting_its_challenger_is_numbered_the_same_either_way(self) -> None:
        """The case that always worked, asserted so that a fix here cannot quietly
        renumber the cycles that were never broken."""
        aws = _Session()
        promoted = new_model_version(RUN, Cycle(3))

        deploy.redeploy(aws, promoted, TARGET, task_token="opaque", cycle=Cycle(3))

        entry = aws.greengrass.request["components"][COMPONENT]
        assert entry["componentVersion"] == "0.3.0"

    def test_a_rollback_stays_on_the_champions_own_component(self) -> None:
        """A rollback passes no cycle, and that is deliberate: it puts back the
        component the champion was published as, rather than minting one for a
        cycle that is already over. Nothing waits on a rollback, so no counter
        is read off it."""
        aws = _Session()
        champion = new_model_version(RUN, Cycle(2))

        deploy.redeploy(aws, champion, TARGET)

        entry = aws.greengrass.request["components"][COMPONENT]
        assert entry["componentVersion"] == "0.2.0"
