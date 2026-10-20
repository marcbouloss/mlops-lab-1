# ---- Stage 1: build the virtualenv with uv ----
FROM python:3.12 AS builder
COPY --from=ghcr.io/astral-sh/uv:0.11.3 /uv /bin/uv
ENV UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy
WORKDIR /app
# only the dependency manifests first: this layer stays cached until they change
COPY pyproject.toml uv.lock ./
# --no-install-project: src/ isn't here yet, install the dependencies only
RUN uv sync --frozen --no-dev --no-install-project

# ---- Stage 2: slim runtime, just the venv and the code ----
FROM python:3.12-slim
WORKDIR /app
COPY --from=builder /app/.venv /app/.venv
COPY src/ ./src/
ENV PATH="/app/.venv/bin:$PATH" PYTHONUNBUFFERED=1
EXPOSE 8000
ENTRYPOINT ["uvicorn", "src.food11.serve:app", "--host", "0.0.0.0", "--port", "8000"]
