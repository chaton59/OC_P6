# syntax=docker/dockerfile:1

# Base image (lightweight Python 3.11)
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# EXPLICATION : Sous-étape 3 - création dossier logs pour persistance (évite erreurs permissions)
RUN mkdir -p /app/logs

# Install system dependencies required by LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy dependency manifests first for better caching
COPY pyproject.toml uv.lock ./

# Install uv and sync dependencies (without installing the project)
RUN pip install --no-cache-dir uv \
    && uv sync --frozen --no-install-project

# Copy application code
COPY . ./

# Install project (and any remaining dependencies)
RUN uv sync --frozen

# Expose Gradio default port
EXPOSE 7860

# Set PORT for compatibility
ENV PORT=7860

# Ensure Python output is not buffered (logs visible immediately)
ENV PYTHONUNBUFFERED=1

# EXPLICATION : Volume pour logs (bonne pratique Docker - permet docker cp ou mount externe)
VOLUME ["/app/logs"]

# Launch the Gradio app
CMD ["uv", "run", "app.py"]
