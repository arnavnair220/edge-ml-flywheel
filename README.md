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
| Edge | Simulated fleet, AWS IoT Greengrass on a Graviton EC2 instance |
| Orchestration | Step Functions, single orchestrator |
| Training | SageMaker training jobs, on-demand GPU, prebuilt PyTorch container |
| Model | COCO-pretrained Ultralytics YOLO11n, frozen backbone, ONNX int8 |
| IaC | Terraform, S3 backend with native S3 locking |
| CI | GitHub Actions via OIDC, no long-lived keys |
| Running cost | Approximately $20/month, with the device stopped between cycles |

## Scope

- The fleet is simulated and is one device: a Greengrass core on a Graviton instance replaying
  unlabeled pool imagery on real ARM silicon. Latency and quantization numbers are measured, not
  estimated, but the instance is not thermally constrained the way physical hardware would be.
- No new data is collected or annotated. BDD100K ships its own ground truth, and the pipeline is
  denied read access to it, so a frame can only be labeled by buying it from the oracle against a
  metered budget.

## Documentation

Start with the [architecture overview](docs/00-overview.md) — definitions, the loop diagram, and the
five stages the system is built from. Each stage has its own numbered document under
[docs/](docs/), alongside unnumbered ones for the cross-cutting concerns: [control](docs/control.md)
and [gates](docs/gates.md).
