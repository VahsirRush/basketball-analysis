FROM python:3.9-slim-bullseye

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    pkg-config \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libglib2.0-0 \
    gfortran \
    && rm -rf /var/lib/apt/lists/*

# Configure pip to use pre-built wheels
RUN pip install --no-cache-dir --upgrade pip
RUN pip config set global.extra-index-url https://pypi.org/simple

# Install dependencies
COPY api/requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY api /app

# Expose port 5000
EXPOSE 5000

# Run the application
CMD ["python", "server.py"] 