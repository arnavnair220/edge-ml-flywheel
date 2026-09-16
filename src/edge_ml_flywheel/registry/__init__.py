"""The registry plane: what a model claims about itself, and the record of it.

Three modules on `training`'s split, for the same reason. `manifest` is pure and
holds the document -- the encoding of `ModelManifest` and the reading of the gate
report it takes its verdict from, so the part that is written once and can never
be corrected is exercisable without credentials. `package` is pure and builds the
`CreateModelPackage` request. `launch` is the boto3 around both: read the report,
collect the digests, write the manifest, hand the request back.

**Nothing here decides anything.** The verdict was reached by the evaluation job
and written to `conventions.gate_report_key`; this reads it. What this plane adds
is the manifest -- the statement of what the model is, checked against what its
run declared -- and the registry entry that records the verdict where a decision
history can be read off it.

**Nothing here creates the model package either.** `launch.register` returns the
request and the state machine makes the call, the same split the three job
request steps make: the control function holds no SageMaker grant, so the
function that builds a registration cannot make one, and the call happens in the
open where the execution history records it.
"""
