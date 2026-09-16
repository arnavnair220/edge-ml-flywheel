"""The scoring job's entry point, and nothing else.

`container/train.py`'s shape, for a different cause. Script mode *requires* the
training entry point at the root of the source archive; a Processing job has no
script mode at all, so this file is here because `scoring.job.container_entrypoint`
names it. The job is `edge_ml_flywheel.scoring.entrypoint`, which sits in the
package where ruff and mypy see it and where the pure parts it calls have tests.

The archive is built by `edge_ml_flywheel.training.launch.archive`: this file,
`train.py` and `requirements.txt` beside it, and the package. One archive for
both jobs, so the code that scored a model is the tree that trained it.
"""

from edge_ml_flywheel.scoring.entrypoint import main

if __name__ == "__main__":
    main()
