"""The edge plane: what reaches a device, and what the device says back.

Five modules, split by where the code runs rather than by what it is about,
because that split is the one that decides what each may import.

- `component` builds the Greengrass recipe and deployment. Pure.
- `telemetry` is the record format and the reduction the canary gate reads. Pure.
- `detect` turns a raw ONNX output into confidences. Pure, numpy only.
- `replay` runs on the device and imports onnxruntime, Pillow and boto3.
- `deploy` runs on the operator's machine and imports boto3 and pyarrow.

**Nothing is imported here.** `gates` re-exports its four checks because every
caller wants one of them; this package has two callers that share no
dependencies, and a device that imported `deploy` through this file would need
pyarrow on a machine with half a gigabyte of memory. So a caller names the module
it means.

**Greengrass does what the design's agent loop describes** (design section 6),
so there is no daemon here. Verifying an artifact's digest, staging a rollout and
rolling back on a failed install are service behaviour configured by the
documents `component` writes. What this package adds is the part Greengrass has
no opinion about: which frames the device replays, what it reports about them,
and whether that report is good enough to leave the model deployed.
"""
