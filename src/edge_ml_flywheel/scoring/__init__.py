"""Running a model over the images a cycle cares about, once.

The front half of the evaluation plane, and the step the two after it are both
queries over. Scoring 67,000 frames is the expensive thing a cycle does that is
not training, and it is done once per model: the gates read the eval detections
and selection reads the pool's, rather than either running its own pass (design
section 4.3).

Four modules, split by where they run:

- `cohorts` decides which images each cohort contributes, pure, from the
  partition and what the run has bought
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
