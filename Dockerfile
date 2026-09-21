# Multi-stage build for CLDB
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Production stage
FROM python:3.11-slim

WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r cldb && useradd -r -g cldb cldb

# Copy installed packages from builder
COPY --from=builder /root/.local /home/cldb/.local

# Copy application source code
COPY cldb/ /app/cldb/
COPY main.py /app/main.py

# Set up environment
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    CLDB_ENVIRONMENT=production \
    LOG_LEVEL=INFO

# Create necessary directories
RUN mkdir -p /app/models/checkpoints /app/experiments /app/logs && \
    chown -R cldb:cldb /app

# Switch to non-root user
USER cldb

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:5000/health || exit 1

# Expose ports
EXPOSE 8080 9090 5000 8501

# Default command - can be overridden
CMD ["python", "main.py"]