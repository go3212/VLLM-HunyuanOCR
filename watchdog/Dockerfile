FROM python:3.11-slim

WORKDIR /app

# Install docker CLI for container control
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && curl -fsSL https://get.docker.com -o get-docker.sh \
    && sh get-docker.sh \
    && rm get-docker.sh \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir httpx

COPY idle_shutdown.py .

CMD ["python", "-u", "idle_shutdown.py"]

