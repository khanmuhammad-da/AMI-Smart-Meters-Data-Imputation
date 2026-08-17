FROM python:3.12-slim

# Prevent Python from creating .pyc files
# and ensure logs appear immediately.
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Application directory
WORKDIR /app

# System packages required by scientific Python
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first for Docker layer caching
COPY requirements.docker.txt .

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.docker.txt

# Copy only the application/runtime files
COPY src ./src
COPY outputs/model_tuned ./outputs/model_tuned
COPY assets ./assets
COPY config ./config

# Create runtime directories
RUN mkdir -p \
    /app/outputs/inference \
    /app/mlruns

# Streamlit port
EXPOSE 8501

# Streamlit configuration
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Start application
CMD ["streamlit", "run", "src/app3.py", "--server.address=0.0.0.0", "--server.port=8501"]