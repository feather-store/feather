# Feather DB Docker Image
# Usage: docker run -it yourusername/feather-db python

FROM python:3.11-slim

LABEL maintainer="your.email@example.com"
LABEL description="Feather DB - Fast, lightweight vector database"
LABEL version="0.1.0"

# Install build dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    make \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy source code
COPY src/ ./src/
COPY bindings/ ./bindings/
COPY include/ ./include/
COPY setup.py .
COPY pyproject.toml .
COPY README.md .
COPY LICENSE .

# Build C++ core
RUN g++ -O3 -std=c++17 -fPIC -c src/feather_core.cpp -o feather_core.o && \
    ar rcs libfeather.a feather_core.o

# Install Python dependencies and package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir pybind11 numpy && \
    pip install --no-cache-dir .

# Create data directory
WORKDIR /data

# Test installation
RUN python -c "import feather_db; print('✓ Feather DB installed successfully')"

# Default command
CMD ["python"]
