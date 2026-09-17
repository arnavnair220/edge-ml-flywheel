"""What one model version is, as the `CreateModelPackage` request that records it.

`training.job`'s shape for the step after the cycle's last job: a pure function
returning the argument dictionary, with no client and no call, so the state
machine and a test read one definition rather than two.

**The approval status is the gate's verdict and nothing else.** It is computed
from the manifest's own gates -- `ModelManifest.gates_passed`, which is false for
a model whose gates never ran -- rather than taken as an argument, so there is no
signature here through which a caller could register an approval the gates did
not produce.

**A rejected model is registered too.** The rejection log is a feature (design
section 5): a cycle that refused to promote is the evidence the project is built
to produce, and a registry holding only the approvals is a registry that cannot
show it. `Rejected` is what the status field is for.

**The container spec is a formality and is kept to one.** SageMaker wants an
image and a tarball before it will hold a version, so the request names the
training image and the `model.tar.gz` that job produced. Nothing deploys this
package: the artifact reaches a device as a Greengrass component (design section
6), and the digest it verifies is the manifest's. So there are no supported
instance types here -- listing them would describe an endpoint nobody creates.
"""

import json
from typing import Any, Final

from edge_ml_flywheel.conventions import (
    Buckets,
    ModelArtifact,
    ModelManifest,
    eval_metrics_key,
    model_artifact_key,
    model_manifest_key,
    model_package_group,
    uri,
)

# What the model takes and what it returns, as the registry spells them. Honest
# rather than nominal: the scoring job hands the model a JPEG and writes down
# boxes, which is the same pair of types whatever runs it.
CONTENT_TYPE: Final = "image/jpeg"
RESPONSE_TYPE: Final = "application/json"

# SageMaker's ceiling on a package description. The gate reasons are joined into
# it and they are written to be read, so the cap is reached by a verdict that
# failed several checks at once -- which is exactly the one worth not truncating
# silently. `DESCRIPTION_ELLIPSIS` is what says it was.
MAX_DESCRIPTION: Final = 1024
DESCRIPTION_ELLIPSIS: Final = " [truncated; see the manifest]"

# A metadata value is capped at 256 characters and may not be empty, so what goes
# in the map below is identifiers and numbers. The reasons are in the manifest and
# the gate report, both named here by URI.
MAX_METADATA_VALUE: Final = 256

# SageMaker's ceiling on a model card document, which the manifest's facts go
# into whole. Room to spare: the card below is the manifest plus two sentences.
MAX_CARD: Final = 100_000

APPROVED: Final = "Approved"
REJECTED: Final = "Rejected"

# The card's own review state, which is about the document and not about the
# model. It is machine-written, complete when it is written and never edited, so
# it is final rather than a draft awaiting someone. The model's verdict is
# `ModelApprovalStatus`, and the two fields are deliberately not wired together:
# a rejected model still gets a finished card saying why it was rejected.
CARD_STATUS: Final = "Approved"


def approval_status(manifest: ModelManifest) -> str:
    """`Approved` or `Rejected`, from the gates the manifest carries.

    There is no third answer. `PendingManualApproval` is SageMaker's default and
    it is deliberately never used: a gate that ran produced a verdict, and a
    version sitting in the registry waiting for a human is a promotion decision
    made outside the machinery that is supposed to make it.
    """
    return APPROVED if manifest.gates_passed else REJECTED


def description(manifest: ModelManifest) -> str:
    """The one-line verdict a console listing shows, with its reasons.

    The reasons rather than a summary, because the whole claim of the gating
    plane is that a verdict is never recorded without one (design section 5), and
    this is the copy someone reads without opening an S3 object.
    """
    verdict = "passed" if manifest.gates_passed else "REJECTED"
    reasons = "; ".join(f"{gate.gate}: {gate.reason}" for gate in manifest.gates)
    line = f"cycle {manifest.cycle} {verdict} -- {reasons}"
    if len(line) <= MAX_DESCRIPTION:
        return line
    return line[: MAX_DESCRIPTION - len(DESCRIPTION_ELLIPSIS)] + DESCRIPTION_ELLIPSIS


def metadata(manifest: ModelManifest, buckets: Buckets) -> dict[str, str]:
    """The manifest attached to the version, as the flat map the API takes.

    A pointer and the fields anyone filters a listing on, not the document: the
    map holds at most 50 entries of at most 256 characters and the manifest has a
    gate reason in it. `manifest` is the URI of the whole thing, which is what
    makes this a summary rather than a second copy that can disagree with the
    first.

    The deployed seed's digest is here and the other seeds' are not. This is the
    one artifact that ships (design section 4.2); the rest are in the manifest,
    where the matched-seed comparison reads them.
    """
    return {
        "manifest": uri(buckets.artifacts, model_manifest_key(manifest.version)),
        "version": str(manifest.version),
        "run_id": str(manifest.run_id),
        "cycle": str(manifest.cycle),
        "partition_version": str(manifest.partition_version),
        "recipe_version": str(manifest.recipe_version),
        "git_commit": manifest.git_commit,
        "labels_spent": str(manifest.labels_spent),
        "deployed_seed": str(manifest.deployed_seed),
        "artifact_sha256": manifest.artifact_sha256[manifest.deployed_seed],
    }


def model_card(manifest: ModelManifest, buckets: Buckets) -> dict[str, Any]:
    """The manifest again, in the shape SageMaker documents a model in.

    The same facts as `manifest.json` and not a second set of them: the custom
    details are `metadata` verbatim, so there is one spelling of what this model
    is and the card cannot drift from the document it describes. The file still
    exists and is still the copy that matters -- a device verifies a digest
    before loading a model (design section 6) and cannot call an AWS API to read
    a card -- but a reviewer who knows SageMaker looks for a card, and a manifest
    only they can find in S3 is a manifest they do not find.

    Deliberately three fields and not the whole schema. Training details,
    intended uses and business context are sections this project answers
    elsewhere or not at all, and a card padded with restated defaults is a card
    nobody reads twice.
    """
    deployed = model_artifact_key(manifest.version, manifest.deployed_seed, ModelArtifact.ONNX)
    return {
        "model_overview": {
            "model_name": str(manifest.version),
            "model_description": description(manifest),
            "model_artifact": [uri(buckets.artifacts, deployed)],
        },
        "additional_information": {
            "caveats_and_recommendations": (
                f"The deployed artifact is seed {manifest.deployed_seed}, which is the lowest seed "
                f"the cycle trained and never the best-scoring one -- picking on the eval set "
                f"biases the number the gate reported. The manifest this card restates is at "
                f"{uri(buckets.artifacts, model_manifest_key(manifest.version))}, and it is the "
                f"copy the device verifies against."
            ),
            "custom_details": metadata(manifest, buckets),
        },
    }


def card_content(manifest: ModelManifest, buckets: Buckets) -> str:
    """The card as the API takes it: one JSON string, inside the size cap.

    Sorted keys for `manifest.to_document`'s reason -- two registrations of the
    same facts are the same string, so a diff of two cycles is about what
    changed.
    """
    content = json.dumps(model_card(manifest, buckets), sort_keys=True)
    if len(content) > MAX_CARD:
        raise ValueError(
            f"the model card for {manifest.version} is {len(content)} characters, over "
            f"SageMaker's {MAX_CARD}"
        )
    return content


def create_model_package(
    manifest: ModelManifest,
    buckets: Buckets,
    image: str,
    model_data_url: str,
) -> dict[str, Any]:
    """The whole `CreateModelPackage` request for one cycle's challenger.

    The group is the run (`conventions.model_package_group`), so the versions on
    one ladder are the versions one champion baseline covers.

    `ModelMetrics` points at the evaluation job's `metrics.json` rather than
    restating a number from it. That file is per seed and this request is per
    model, so a figure copied here would be one of several with nothing to say
    which -- and the registry would hold a metric nothing produced.

    **No `Tags`.** SageMaker refuses them on a package version and says to put
    them on the group, which is the one place a per-version fact cannot go: the
    group is the run, opened once by whichever cycle reaches it first. Nothing is
    lost either way. The tags were `project`, `run_id`, `cycle` and
    `recipe_version`, and `metadata` below carries all four of the per-model ones
    already -- so the request had two spellings of the same facts and the API
    rejected the redundant one.
    """
    return {
        "ModelPackageGroupName": model_package_group(manifest.run_id),
        "ModelPackageDescription": description(manifest),
        "ModelApprovalStatus": approval_status(manifest),
        "InferenceSpecification": {
            "Containers": [{"Image": image, "ModelDataUrl": model_data_url}],
            "SupportedContentTypes": [CONTENT_TYPE],
            "SupportedResponseMIMETypes": [RESPONSE_TYPE],
        },
        "ModelMetrics": {
            "ModelQuality": {
                "Statistics": {
                    "ContentType": RESPONSE_TYPE,
                    "S3Uri": uri(buckets.artifacts, eval_metrics_key(manifest.version)),
                }
            }
        },
        "CustomerMetadataProperties": metadata(manifest, buckets),
        "ModelCard": {
            "ModelCardContent": card_content(manifest, buckets),
            "ModelCardStatus": CARD_STATUS,
        },
    }
