# Edge ML Flywheel

Closed-loop retraining and deployment pipeline for an edge perception model.

A model runs on a simulated edge fleet, notices which examples confuse it, and sends only those
back to the cloud. The cloud buys ground truth for that small batch against a hard label budget,
retrains, and checks the challenger against five pass/fail gates. If it passes, the model is
promoted, converted to an edge format, deployed to one device, watched, then rolled out — or
rolled back. That deployment produces new observations, and the loop repeats.

The point of the project is to *measure* model improvement per unit of labeling effort spent.
**The system is the deliverable, not the model.**

## Status

Design complete, key decisions locked. Not yet built.

## Stack

| Concern | Choice |
|---|---|
| Data | BDD100K (Berkeley DeepDrive) — non-commercial research license |
| Cloud | AWS |
| Edge | Simulated fleet, ARM64 containers on ECS Fargate (Graviton) |
| Orchestration | Step Functions (single orchestrator, no local execution path) |
| IaC | Terraform, S3 backend with DynamoDB lock |
| CI | GitHub Actions via OIDC, no long-lived keys |
| Model | COCO-pretrained nano detector, frozen backbone, ONNX int8 |
| Budget | ~$17/month steady state, ~$26 in the A/B month |

## Definition of done

All 7 waves run, the label-efficiency A/B shows a real gap, at least one gate rejection happened
honestly, and a rollback has been demonstrated. Ship then, regardless of how good the model is.

## Docs

See [docs/](docs/) — system overview plus one doc per major component.

## License

Code: TBD. Data: BDD100K is UC Berkeley, free for non-commercial research use only; commercial
use requires a license from Berkeley's Office of Technology Licensing.
