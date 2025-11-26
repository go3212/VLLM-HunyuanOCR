FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1

# Upgrade pip
RUN python -m pip install --upgrade pip

# Install vLLM nightly with HunyuanOCR support
RUN pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Install additional dependencies
RUN pip install transformers accelerate pillow

WORKDIR /app

# Create images directory for mounting
RUN mkdir -p /app/images

EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["vllm", "serve", "tencent/HunyuanOCR", "--host", "0.0.0.0", "--port", "8000"]

