# syntax=docker/dockerfile:1.7
# The training job. Build from the repo root: docker build -f images/train.Dockerfile .

FROM ghcr.io/astral-sh/uv:0.12.10@sha256:2bb3ebca0a796a155094a27773d290c4b074572e6107f171d88d086682fd2500 AS uv

# Dependencies only. Nothing from src is copied here, so this stage, and the
# multi-gigabyte layer the runtime stage copies from it, are reused across
# source-only changes.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS deps
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/opt/venv
WORKDIR /src

# Locked dependencies first, so a source-only change reuses this layer.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project --group training

# torch is not in uv.lock: the CUDA build is chosen here, and requirements.txt
# is resolved on top of it. 2.9.0 with CUDA 13.0 matches the SageMaker image
# the old stack trains in, so models stay comparable across the move.
RUN --mount=type=cache,target=/root/.cache/uv \
    uv pip install --python /opt/venv/bin/python \
      --index-url https://download.pytorch.org/whl/cu130 \
      torch==2.9.0 torchvision==0.24.0
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=container/requirements.txt,target=requirements.txt \
    uv pip install --python /opt/venv/bin/python -r requirements.txt

# The project as a wheel, built apart from the dependencies.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258 AS wheel
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_PYTHON_DOWNLOADS=never
WORKDIR /src
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv build --wheel --out-dir /dist

FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258
# OpenCV, which ultralytics imports, loads these shared libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
# ultralytics writes a settings file and checks the folder exists first.
RUN useradd --uid 10001 --create-home app \
    && install -d -o app -g app /home/app/.config/Ultralytics
COPY --from=deps /opt/venv /opt/venv
# The project goes on top in its own small layer. --no-deps because every
# dependency is already in /opt/venv, including torch, which uv.lock does not list.
RUN --mount=from=uv,source=/uv,target=/usr/local/bin/uv \
    --mount=from=wheel,source=/dist,target=/dist \
    uv pip install --python /opt/venv/bin/python --no-deps --no-cache --compile-bytecode /dist/*.whl
COPY container/train.py /app/train.py
ENV PATH=/opt/venv/bin:$PATH \
    PYTHONUNBUFFERED=1
USER 10001
WORKDIR /home/app
ENTRYPOINT ["python", "/app/train.py"]
