# Edge ML Flywheel

Closed-loop retraining and deployment pipeline for an edge perception model.

A model runs on a simulated edge fleet, notices which examples confuse it, and sends only those
back to the cloud. The cloud buys ground truth for that small batch against a hard label budget,
retrains, and checks the challenger against five pass/fail gates. If it passes, the model is
promoted, converted to an edge format, deployed to one device, watched, then rolled out — or
rolled back. That deployment produces new observations, and the loop repeats.

Every promotion decision is recorded with the evidence behind it, and the cost of each cycle is
denominated in labels, so the pipeline reports accuracy gained per label spent alongside the model
itself.

## Design principles

- **Labels are a metered resource.** The pipeline has no read access to ground truth; the only way
  to obtain a label is to spend from an audited budget. That makes "improvement per label spent" a
  real number rather than a claim.
- **Promotion requires evidence, not a bump.** The quality gate compares champion and challenger
  across five matched training seeds and requires a confidence interval that excludes zero, so a
  metric gain sitting inside the noise band is correctly refused.
- **The evaluation machinery is itself under test.** An A/A control trains a challenger on zero
  new information and asserts that the gate does not promote, which measures the gate's
  false-positive rate directly.
- **Regressions are caught per slice.** Overall accuracy rising while one condition collapses is
  the failure mode that matters, so every weather, time-of-day, class and object-scale slice is
  gated on its own noise band.

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

- The fleet is simulated: five ARM64 containers replaying held-out imagery on real ARM silicon.
  Latency and quantization numbers are measured, not estimated, but the tasks are not thermally
  constrained the way physical hardware would be.
- No new data is collected or annotated. Distribution shift is simulated by releasing BDD100K in
  ordered waves, and ground truth is served from withheld annotations behind a budgeted oracle.
- Selection quality is evaluated by running the loop twice, once with uncertainty sampling and
  once with random sampling, and comparing the two label-efficiency curves.

## Documentation

Start with the [architecture overview](docs/00-overview.md) — definitions, the loop diagram, and
the nine planes the system is built from. Each plane has its own document under [docs/](docs/).

## Status

Design complete. Implementation not yet started.

## License

Code: TBD. Data: BDD100K is UC Berkeley, free for non-commercial research use only; commercial
use requires a license from Berkeley's Office of Technology Licensing.
