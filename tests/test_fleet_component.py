"""Tests for the Greengrass documents a deployment is made of.

`registry.package`'s suite for the step after promotion: the recipe and the
deployment are pure functions, so what is asserted is the document -- the
artifacts it names, the platform it is scoped to, the paths it hands the
component.

**Recipe variables are asserted as literal braces.** `{artifacts:path}` and
`{iot:thingName}` are strings the nucleus substitutes, not Python format fields,
and the failure when one is accidentally interpolated is a component pointed at a
path that does not exist. So the tests read them the way Greengrass does.
"""

import json
from typing import Any

import pytest

from edge_ml_flywheel.conventions import (
    POOL_SAMPLE,
    Buckets,
    Cycle,
    ModelVersion,
    RunId,
    Seed,
    new_model_version,
    replay_code_key,
    replay_manifest_key,
)
from edge_ml_flywheel.fleet import component
from edge_ml_flywheel.selection.score import BAND_LOW

RUN = RunId("20260812t143355z-v0-skeleton")
VERSION = ModelVersion("20260812t143355z-v0-skeleton-c003")
COMMIT = "b" * 40
BUCKETS = Buckets.for_account("123456789012")
ENDPOINT = "a1b2c3d4e5f6g7-ats.iot.us-east-1.amazonaws.com"

RELEASE = component.Release(version=VERSION, seed=Seed(1), git_commit=COMMIT)


def a_recipe(**overrides: Any) -> dict[str, Any]:
    fields: dict[str, Any] = {
        "release": RELEASE,
        "buckets": BUCKETS,
        "iot_endpoint": ENDPOINT,
        "replay": component.STANDARD,
    }
    return component.recipe(**(fields | overrides))


def artifact_uris(recipe: dict[str, Any]) -> list[str]:
    return [artifact["URI"] for artifact in recipe["Manifests"][0]["Artifacts"]]


class TestTheRelease:
    def test_a_short_commit_is_refused_before_anything_is_uploaded(self) -> None:
        """`replay_code_key` refuses it too, but only once a component version
        has already been named. Refusing here means a release that cannot be
        staged fails before the first put."""
        with pytest.raises(ValueError, match="40-character git commit"):
            component.Release(version=VERSION, seed=Seed(1), git_commit="abc123")

    def test_a_version_that_is_not_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="not a model version"):
            component.Release(version=ModelVersion("nope"), seed=Seed(1), git_commit=COMMIT)


class TestTheReplaySettings:
    def test_the_sample_is_the_cycle_s_whole_ranking_universe(self) -> None:
        """`POOL_SAMPLE` frames after 50 warmup, at the size the model was
        exported at. Not design section 4.3's 500: this pass is what selection
        ranks, so the p95 falls out of a set far larger than it was specified
        over rather than the set being sized for the p95."""
        assert component.STANDARD.frames == POOL_SAMPLE
        assert component.STANDARD.warmup == 50
        assert component.STANDARD.image_size == 416

    def test_the_floor_is_the_band_the_selector_reads(self) -> None:
        """Not the cloud pass's 0.001. Nothing integrates the pool, and
        `image_score` discards everything below the band before it scores
        anything, so the tail is a tenfold larger file and an identical
        ranking."""
        assert component.STANDARD.confidence_floor == BAND_LOW

    def test_measured_is_what_the_summary_will_claim(self) -> None:
        assert component.Replay(frames=500, warmup=50).measured == 450

    def test_a_warmup_that_eats_the_whole_replay_is_refused(self) -> None:
        """A replay recording no frame reports as one that lost its telemetry,
        which sends the next attempt looking at the rule rather than the flags."""
        with pytest.raises(ValueError, match="leaves nothing measured"):
            component.Replay(frames=50, warmup=50)

    def test_a_confidence_floor_outside_zero_to_one_is_refused(self) -> None:
        with pytest.raises(ValueError, match="admits nothing or everything"):
            component.Replay(confidence_floor=1.5)


class TestTheRecipe:
    def test_it_is_addressed_by_the_run_and_the_cycle(self) -> None:
        recipe = a_recipe()

        assert recipe["ComponentName"] == "edge-ml-flywheel.20260812t143355z-v0-skeleton"
        assert recipe["ComponentVersion"] == "0.3.0"

    def test_it_names_the_model_the_frames_and_the_code(self) -> None:
        """Three artifacts and no more. Greengrass hashes each when the version
        is created and verifies it on download, so everything the device must not
        be wrong about is in this list."""
        uris = artifact_uris(a_recipe())

        assert len(uris) == 3
        assert uris[0].endswith(
            "models/version=20260812t143355z-v0-skeleton-c003/seed=1/model.onnx"
        )
        assert uris[1].endswith(replay_manifest_key(RUN, Cycle(3)))
        assert uris[2].endswith(replay_code_key(COMMIT))

    def test_only_the_code_is_unarchived(self) -> None:
        """The model and the frame list are read where they land. Unarchiving
        either would be asking Greengrass to unpack a file that is not an
        archive."""
        artifacts = a_recipe()["Manifests"][0]["Artifacts"]
        unarchived = [artifact for artifact in artifacts if "Unarchive" in artifact]

        assert len(unarchived) == 1
        assert unarchived[0]["Unarchive"] == "ZIP"

    def test_the_seed_that_ships_decides_which_artifact_is_named(self) -> None:
        """The manifest records the deployed seed, and a recipe naming a
        different one produces a digest the canary rejects -- which is why the
        seed is a field rather than a constant."""
        other = component.Release(version=VERSION, seed=Seed(4), git_commit=COMMIT)

        assert "seed=4/model.onnx" in artifact_uris(a_recipe(release=other))[0]

    def test_it_is_scoped_to_arm(self) -> None:
        """A manifest with no platform matches every device, and an x86 one would
        install a graph exported for ARM and fail at its first inference rather
        than at the deployment."""
        assert a_recipe()["Manifests"][0]["Platform"]["architecture"] == "aarch64"

    def test_the_endpoint_is_configuration_rather_than_a_device_lookup(self) -> None:
        """It is account-specific, so a device that discovered it would need
        `iot:DescribeEndpoint` and a call before its first publish."""
        assert a_recipe()["ComponentConfiguration"]["DefaultConfiguration"]["iotEndpoint"] == (
            ENDPOINT
        )

    def test_the_replay_settings_reach_the_device(self) -> None:
        replay = component.Replay(frames=80, warmup=8, image_size=320, confidence_floor=0.05)
        configuration = a_recipe(replay=replay)["ComponentConfiguration"]["DefaultConfiguration"]

        assert configuration["warmup"] == 8
        assert configuration["imageSize"] == 320
        assert configuration["confidenceFloor"] == 0.05

    def test_the_frame_count_is_not_sent_to_the_device(self) -> None:
        """The device replays the list it was given. A count beside that list
        would be a second statement of its length, and the two could disagree."""
        configuration = a_recipe()["ComponentConfiguration"]["DefaultConfiguration"]

        assert "frames" not in configuration


class TestWhatTheDeviceIsTold:
    def test_the_device_names_itself_in_the_topic(self) -> None:
        """So a second device needs no edit here and cannot be configured into
        publishing under the first one's name."""
        topic = a_recipe()["ComponentConfiguration"]["DefaultConfiguration"]["topic"]

        assert topic.endswith("/{iot:thingName}")
        assert str(RUN) in topic

    def test_the_run_script_uses_recipe_variables_for_every_path(self) -> None:
        """Greengrass decides where an artifact lands and where a component may
        write, so a path guessed here is one that works until the nucleus changes
        its layout."""
        script = component.run_script()

        assert "{artifacts:path}/model.onnx" in script
        assert "{artifacts:path}/replay.json" in script
        assert "{work:path}" in script
        assert "{iot:thingName}" in script

    def test_the_code_path_and_the_archive_name_agree(self) -> None:
        """A disagreement between the two is a component that starts and cannot
        import itself, which is why one is derived from the other."""
        script = a_recipe()["Manifests"][0]["Lifecycle"]["Run"]["Script"]

        assert script.startswith(
            f"PYTHONPATH={{artifacts:decompressedPath}}/{component.CODE_DIRECTORY} "
        )
        assert replay_code_key(COMMIT).endswith(f"/{component.CODE_DIRECTORY}.zip")

    def test_the_path_is_in_the_command_rather_than_the_lifecycle(self) -> None:
        """A component that set `PYTHONPATH` through the lifecycle's `Setenv`
        started with it empty: three failures in 250 ms on `No module named
        edge_ml_flywheel`, and broken before the nucleus had logged that config
        node arriving. Recipe variables were never the problem -- the same script
        printed a fully resolved `{artifacts:decompressedPath}`.

        An assignment prefixed to the command is one the shell makes before the
        process exists, so there is nothing left to arrive late.
        """
        run = a_recipe()["Manifests"][0]["Lifecycle"]["Run"]

        assert "Setenv" not in run
        assert "PYTHONPATH=" in run["Script"]

    def test_it_runs_the_package_as_a_module(self) -> None:
        """Which is why the archive carries no entry point beside the package."""
        assert "-m edge_ml_flywheel.fleet.replay" in component.run_script()


class TestTheDeployment:
    def test_it_names_one_component_version_on_one_target(self) -> None:
        target = component.thing_group_arn("us-east-1", "123456789012", "edge-ml-flywheel-devices")
        request = component.deployment(VERSION, target)

        assert request["targetArn"] == target
        assert request["components"] == {
            "edge-ml-flywheel.20260812t143355z-v0-skeleton": {
                "componentVersion": "0.3.0",
                "configurationUpdate": {"merge": json.dumps({"version": VERSION})},
            }
        }

    def test_a_failed_install_rolls_the_device_back_by_itself(self) -> None:
        """The health check design section 6 describes. The canary gate is the
        other half: a deployment that installed perfectly well and is slower."""
        request = component.deployment(VERSION, "arn:aws:iot:us-east-1:123456789012:thinggroup/x")

        assert request["deploymentPolicies"]["failureHandlingPolicy"] == "ROLLBACK"

    def test_a_rollback_is_the_same_request_naming_an_earlier_version(self) -> None:
        """There is no rollback builder, so the path a rollback takes is the path
        every cycle has already exercised."""
        target = "arn:aws:iot:us-east-1:123456789012:thinggroup/x"
        name = "edge-ml-flywheel.20260812t143355z-v0-skeleton"

        forward = component.deployment(VERSION, target)
        back = component.deployment(new_model_version(RUN, Cycle(2)), target)

        assert set(forward) == set(back)
        assert forward["components"][name]["componentVersion"] == "0.3.0"
        assert back["components"][name]["componentVersion"] == "0.2.0"

    def test_a_cycle_that_rejected_deploys_the_champion_under_its_own_number(self) -> None:
        """A component version names the cycle that deployed it, not the cycle
        that trained the model: the recipe carries this cycle's sample, and a
        component version is immutable once published. The deployment says which
        model it is, because the number no longer can."""
        target = "arn:aws:iot:us-east-1:123456789012:thinggroup/x"
        name = "edge-ml-flywheel.20260812t143355z-v0-skeleton"

        request = component.deployment(VERSION, target, cycle=Cycle(5))
        entry = request["components"][name]

        assert entry["componentVersion"] == "0.5.0"
        assert json.loads(entry["configurationUpdate"]["merge"])["version"] == VERSION

    def test_the_task_token_rides_the_deployment(self) -> None:
        """A token does not exist when a component version is published and a
        component version is immutable once it does, so the deployment is the
        only half of the pair that can carry one."""
        target = "arn:aws:iot:us-east-1:123456789012:thinggroup/x"
        name = "edge-ml-flywheel.20260812t143355z-v0-skeleton"

        request = component.deployment(VERSION, target, task_token="opaque")
        merged = json.loads(request["components"][name]["configurationUpdate"]["merge"])

        assert merged["taskToken"] == "opaque"

    def test_a_rollback_hands_nobody_a_token(self) -> None:
        """Nothing is waiting on a rollback. The cycle that was waiting is the
        one that failed."""
        target = "arn:aws:iot:us-east-1:123456789012:thinggroup/x"
        name = "edge-ml-flywheel.20260812t143355z-v0-skeleton"

        request = component.deployment(VERSION, target)
        merged = json.loads(request["components"][name]["configurationUpdate"]["merge"])

        assert "taskToken" not in merged

    def test_the_deployment_name_carries_the_version_it_puts_out(self) -> None:
        request = component.deployment(VERSION, "arn:aws:iot:us-east-1:123456789012:thinggroup/x")

        assert request["deploymentName"].endswith("-0.3.0")
