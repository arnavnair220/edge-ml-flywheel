"""The control plane's one Lambda.

Plane 5 is Step Functions, and the ASL in `infra/cycle.asl.json` is where the
control flow lives -- there is deliberately no second orchestrator here. What a
state machine cannot do is call a Python function, and two of a cycle's steps
are Python functions this project already has: writing the image manifest, and
building a `CreateTrainingJob` request. So this package is an adapter and
nothing else.

- `handler` -- one entry point, dispatching on a step name

**Nothing here decides anything about a cycle.** It reads no clock that matters,
holds no state, and makes no branch: which cycle is running comes from the
conditional update the state machine performs before this is ever invoked, and
whether to run another one is a `Choice` state. A Lambda that decided either
would be the second orchestrator the design says must not exist.
"""
