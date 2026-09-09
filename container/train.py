"""SageMaker's entry point, and nothing else.

Script mode requires the entry point at the root of the source archive, so this
file exists to be that root. The job is `edge_ml_flywheel.training.entrypoint`,
which sits in the package where ruff and mypy see it and where the pure parts it
calls have tests.

The archive is built by `edge_ml_flywheel.training.launch.archive`: this file,
`requirements.txt` beside it, and the package.
"""

from edge_ml_flywheel.training.entrypoint import main

if __name__ == "__main__":
    main()
