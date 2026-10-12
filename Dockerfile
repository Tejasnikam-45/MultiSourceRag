FROM python:3.10-slim

WORKDIR /app

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PORT=7860 \
    HOME=/app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create non-root user (Hugging Face Spaces runs as UID 1000)
RUN useradd -m -u 1000 user

# Copy application files
COPY . .

# Pre-populate ChromaDB with the research papers during build
RUN python ingest_pdfs.py

# Ensure user 1000 has full ownership and permissions
RUN chown -R user:user /app && chmod -R 777 /app

USER user

EXPOSE 7860

# Launch FastAPI on port 7860 (required by Hugging Face Spaces)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860"]
