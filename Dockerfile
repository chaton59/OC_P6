# Minimal runtime image for Gradio inference (small + fast build)
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install system dependencies required by LightGBM (OpenMP)
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install only inference dependencies (better cache reuse)
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy application code
COPY . .

# Gradio default port
EXPOSE 7860

# Launch the app
CMD ["python", "app.py"]
