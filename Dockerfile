# Minimal, dependency-free runtime image.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip && python -m pip install .

# Runtime data lives in a volume so the SQLite database is not baked in.
VOLUME ["/data"]
ENV RLR_DATA_DIR=/data

# The container is inert by default: it runs the deterministic offline demo.
# Point it at a workload by overriding the command.
CMD ["python", "-m", "reflective_loop_runtime", "demo", "--ticks", "8", "--data-dir", "/data"]