# Build Stage
FROM python:3.11-slim as builder

WORKDIR /build

# System dependencies for C++ compilation
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy Feather Source
COPY . /build/feather_source

# Build Wheel for Feather DB
WORKDIR /build/feather_source
RUN pip install --upgrade pip setuptools wheel pybind11
RUN python setup.py bdist_wheel

# --- Runtime Stage ---
FROM python:3.11-slim

WORKDIR /app

# Install Runtime Deps
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy Wheel from Builder
COPY --from=builder /build/feather_source/dist/*.whl /app/wheels/

# Install Feather DB & Fast API
RUN pip install /app/wheels/*.whl
RUN pip install fastapi uvicorn python-multipart pydantic

# Copy API Code
COPY feather-api/app /app/app

# Env Vars
ENV FEATHER_DB_PATH=/data/feather.db
ENV FEATHER_DB_DIM=768

# Run
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
