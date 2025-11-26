FROM python:3.11-slim

# Install poppler for PDF to image conversion and curl for uv
RUN apt-get update && apt-get install -y \
    poppler-utils \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN uv pip install --system --no-cache -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8520

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8520"]
