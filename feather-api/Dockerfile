# ──────────────────────────────────────────────
# Stage 1 — Build feather_db wheel
# ──────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /build

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY . /build/feather_source

WORKDIR /build/feather_source
RUN pip install --upgrade pip setuptools wheel pybind11 && \
    python setup.py bdist_wheel

# ──────────────────────────────────────────────
# Stage 2 — Runtime image
# ──────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install feather_db from wheel built in stage 1
COPY --from=builder /build/feather_source/dist/*.whl /tmp/wheels/
RUN pip install --no-cache-dir /tmp/wheels/*.whl && rm -rf /tmp/wheels

# Install API dependencies
COPY feather-api/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy API source
COPY feather-api/app /app/app

# Data directory (override with -v /host/data:/data)
RUN mkdir -p /data
VOLUME ["/data"]

# Environment defaults
ENV FEATHER_DATA_DIR=/data
ENV FEATHER_DB_DIM=768
ENV FEATHER_API_KEY=""

# Health-check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
