"""Fine-tuning YOLO11n on the labels a cycle owns.

Four modules, split by where the code runs rather than by what it is about:

- `job` builds the `CreateTrainingJob` request and calls nothing. It is the one
  statement of what a training job *is*, so the CLI that starts the first job by
  hand and the Step Functions state machine that starts every later one are two
  callers of one definition rather than two definitions that drift.
- `labels` and `dataset` are pure and run on both sides: `labels` reads the
  parquet a cycle's labeled set is spread across, `dataset` turns it into the
  files Ultralytics reads.
- `entrypoint` is the container. It is the only module here that imports
  `ultralytics`, and it does the work the job exists for: seed, convert, train,
  hash, upload.
- `launch` is the operator's side -- packaging the tree, uploading it, starting
  the job, and reporting what the channel download cost.

**The training job is where the label wall has to hold under something that
genuinely needs the data.** Nothing in this package can name a withheld label:
the boxes come from the bootstrap cohort file and the run's own purchases, both
of which are labels this run already owns, and the role in `infra/training.tf`
is denied `raw/labels/` twice over.
"""
