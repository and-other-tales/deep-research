FROM ubuntu:24.10

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DEBIAN_FRONTEND=noninteractive \
    PORT=2024 \
    HOST=0.0.0.0

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3.12-venv \
    build-essential \
    curl \
    wget \
    git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy Python package files first for better caching
COPY pyproject.toml /app/
COPY README.md /app/
COPY langgraph.json /app/
COPY src/ /app/src/

# Setup virtual environment and install Python dependencies
RUN python3 -m venv .venv && \
    . .venv/bin/activate && \
    pip install --no-cache-dir --upgrade pip setuptools wheel && \
    pip install --no-cache-dir "langgraph-cli[inmem]>=0.0.15" && \
    pip install --no-cache-dir -e .

# Copy the rest of the application
COPY tests/ /app/tests/
COPY *.py /app/
COPY *.md /app/

# Allow the application to access port 2024
EXPOSE 2024

# Command to run the application using langgraph CLI
CMD . .venv/bin/activate && langgraph dev --host 0.0.0.0 --port 2024 --allow-blocking