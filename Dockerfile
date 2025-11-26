FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# Install minimal system dependencies (git needed for pip installs from git)
RUN apt-get update && apt-get install -y \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Create venv with Python 3.11 and install dependencies
RUN uv venv --python 3.11 /app/.venv
ENV PATH="/app/.venv/bin:$PATH"
ENV VIRTUAL_ENV="/app/.venv"

# Install vLLM nightly with HunyuanOCR support
RUN uv pip install -U vllm --pre --extra-index-url https://wheels.vllm.ai/nightly

# Install Tencent's fork of transformers with HunyuanVL support
# The standard transformers library doesn't include hunyuan_vl architecture yet
RUN uv pip install --force-reinstall --no-deps git+https://github.com/huggingface/transformers@82a06db03535c49aa987719ed0746a76093b1ec4 && \
    uv pip install accelerate pillow tiktoken

# Create images directory for mounting
RUN mkdir -p /app/images

EXPOSE 8000

# Default command (can be overridden in docker-compose)
CMD ["vllm", "serve", "tencent/HunyuanOCR", "--host", "0.0.0.0", "--port", "8000"]
