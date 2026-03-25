# Stage 1: Basic Image with Python
FROM python:3.12.12-slim as base

# Set environment variables
ENV PYTHONUNBUFFERED = 1 \
    PYTHONDONTWRITEBYTECODE = 1 \
    PIP_NO_CACHE_DIR = 1 \
    PIP_DISABLE_PIP_VERSION_CHECK = 1 \
    DEBIAN_FRONTEND = noninteractive

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    git \
    build-essential \
    libpq-dev \
    && rm -rf /var/lib/apt/lists/*

# Stage 2: Dependencies Installation
FROM base as dependencies

# Upgrade pip to latest version
RUN pip install --upgrade pip setuptools wheel

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies as application

# Copy application code
COPY . /app/

# Create necessary directories
RUN chmod -R 755 /app

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:5000/api/health || exit 1

# Run application
CMD ["python", "app/application.py"]