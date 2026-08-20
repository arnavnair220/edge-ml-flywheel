# Edge ML Flywheel

Closed-loop retraining and deployment pipeline for an edge perception model.

A model runs on a simulated edge fleet, scores unlabeled imagery, and returns the frames it is
least certain about. The cloud buys ground truth for that batch against a hard label budget,
retrains, and checks the challenger against five pass/fail gates. A passing model is promoted,
converted to an edge format, deployed to one device, watched, then rolled out or rolled back. That
deployment produces new observations, and the cycle repeats.

Every promotion decision is recorded with its supporting evidence, and each cycle's cost is
denominated in labels, so the pipeline reports accuracy gained per label spent alongside the model.

## Design principles

- **Labels are a metered resource.** The pipeline has no read access to ground truth. A label is
  obtained only by spending from an audited budget, which makes improvement per label spent a
  measured quantity.
- **Promotion requires a significant gain.** The quality gate compares champion and challenger
  across five matched training seeds and requires a confidence interval that excludes zero. A gain
  inside the noise band does not promote.
- **The evaluation machinery is itself under test.** An A/A control trains a challenger on zero new
  information and asserts that the gate refuses it, measuring the gate's false-positive rate
  directly.
- **Regressions are caught per slice.** Every weather, time-of-day, class and object-scale slice is
  gated on its own noise band, so overall accuracy cannot rise while one condition degrades.

## Stack

| Concern | Choice |
|---|---|
| Data | BDD100K (Berkeley DeepDrive) — non-commercial research license |
| Cloud | AWS |
| Edge | Simulated fleet, ARM64 containers on ECS Fargate (Graviton) |
| Orchestration | Step Functions, single orchestrator |
| Training | Fargate CPU, then SageMaker spot GPU |
| Model | COCO-pretrained nano detector, frozen backbone, ONNX int8 |
| IaC | Terraform, S3 backend with DynamoDB lock |
| CI | GitHub Actions via OIDC, no long-lived keys |
| Running cost | Approximately $17/month |

## Scope

- The fleet is simulated: five ARM64 containers replaying unlabeled pool imagery on real ARM
  silicon. Latency and quantization numbers are measured, not estimated, but the tasks are not
  thermally constrained the way physical hardware would be.
- No new data is collected or annotated. BDD100K ships its own ground truth, and the pipeline is
  denied read access to it, so a frame can only be labeled by buying it from the oracle against a
  metered budget.
- Selection is validated by controls inside the run: an A/A test and a confidence-ordered cycle. The
  label-efficiency comparison against a random-sampling arm is deferred; see
  [planned additions](docs/00-overview.md#planned-additions).

## Documentation

Start with the [architecture overview](docs/00-overview.md) — definitions, the loop diagram, and
the nine planes the system is built from. Each plane has its own document under [docs/](docs/).

## Status

Ingest is deployed and has run: `raw/` holds the 80,000-image train and val pool with its derived
image manifest, and every partition and eval-sizing question is now a query against that manifest.
Partitioning is next.

## License

Code: TBD. Data: BDD100K is UC Berkeley, free for non-commercial research use only; commercial
use requires a license from Berkeley's Office of Technology Licensing.
