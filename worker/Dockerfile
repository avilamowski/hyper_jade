FROM python:3.12-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy worker source code
COPY *.py ./
COPY job_queue/ ./job_queue/

# Environment variables
ENV JADE_SERVER_URL=http://server:8000
ENV REDIS_URL=redis://redis:6379

# Health check - check if worker can connect to Redis
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "from redis_conn import RedisConnection; RedisConnection.test_connection()" || exit 1

# Run the worker
CMD ["python", "main.py"]
