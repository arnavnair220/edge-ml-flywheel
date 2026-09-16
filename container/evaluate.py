"""The evaluation job's entry point, and nothing else.

`container/score.py`'s shape and its cause: a Processing job has no script mode,
so this file is here because `evaluation.job.container_entrypoint` names it. The
job is `edge_ml_flywheel.evaluation.entrypoint`, which sits in the package where
ruff and mypy see it and where the pure parts it calls have tests.

The archive is built by `edge_ml_flywheel.training.launch.archive`: this file,
`train.py`, `score.py` and `requirements.txt` beside it, and the package. One
archive for all three jobs, so the code that gated a model is the tree that
trained and scored it.
"""

from edge_ml_flywheel.evaluation.entrypoint import main

if __name__ == "__main__":
    main()
