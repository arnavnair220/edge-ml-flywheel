# Edge ML Flywheel

Closed-loop retraining and deployment pipeline for an edge perception model.

A model runs on a simulated edge fleet, scores unlabeled imagery, and returns the frames it is
least certain about. The cloud buys ground truth for that batch against a hard label budget,
retrains, and checks the challenger against four pass/fail gates. A passing model is promoted,
converted to an edge format, deployed to one device, watched, then rolled out or rolled back. That
deployment produces new observations, and the cycle repeats.

Every promotion decision is recorded with its supporting evidence, and each cycle's cost is
denominated in labels, so the pipeline reports accuracy gained per label spent alongside the model.

## Stack

| Concern | Choice |
|---|---|
| Data | BDD100K (Berkeley DeepDrive) — non-commercial research license |
| Cloud | AWS |
| Edge | Simulated fleet, AWS IoT Greengrass on Graviton EC2 |
| Orchestration | Step Functions, single orchestrator |
| Training | SageMaker training jobs, on-demand GPU, prebuilt PyTorch container |
| Model | COCO-pretrained Ultralytics YOLO11n, frozen backbone, ONNX int8 |
| IaC | Terraform, S3 backend with native S3 locking |
| CI | GitHub Actions via OIDC, no long-lived keys |
| Running cost | Approximately $20/month |

## Scope

- The fleet is simulated: Greengrass devices on Graviton instances replaying unlabeled pool
  imagery on real ARM silicon. Latency and quantization numbers are measured, not estimated, but
  the instances are not thermally constrained the way physical hardware would be.
- No new data is collected or annotated. BDD100K ships its own ground truth, and the pipeline is
  denied read access to it, so a frame can only be labeled by buying it from the oracle against a
  metered budget.
- Selection is validated by controls inside the run: an A/A test and a confidence-ordered cycle. The
  label-efficiency comparison against a random-sampling arm is deferred; see
  [planned additions](docs/00-overview.md#planned-additions).

## Documentation

Start with the [architecture overview](docs/00-overview.md) — definitions, the loop diagram, and
the nine planes the system is built from. Each plane has its own document under [docs/](docs/).
