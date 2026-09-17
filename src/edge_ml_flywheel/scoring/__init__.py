"""Running a model over the images a cycle cares about, once.

The front half of the evaluation plane, and the step the two after it are both
queries over. It covers `eval`, once per model, and the gates are queries over
what it wrote rather than passes of their own (design section 4.3).

**The pool is not here.** It is scored on the device by the deployed artifact, so
the ranking selection buys against is the fleet's own uncertainty. What this
package still owns for it is the draw: `cohorts` decides which images the device
is given and `launch` writes that list where the component recipe names it.

Four modules, split by where they run:

- `cohorts` decides which images each cohort contributes, pure, from the
  partition and what the run has bought, and draws the fleet's sample out of what
  is left
- `job` is what one `CreateProcessingJob` request is, pure, with no client
- `launch` is the caller -- it writes the manifests and resolves the request
  against the account
- `entrypoint` is the container: load the checkpoint, walk the channels, write
  the parquet

**Detections are facts, not verdicts.** Nothing here matches a box against
ground truth or forms an opinion about a model. That is the next step's work, and
keeping it there is what lets this job's role be denied every label prefix in the
account rather than trusted to leave them alone.
"""
