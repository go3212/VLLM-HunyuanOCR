FROM python:3.11-slim

# Install poppler for PDF to image conversion
RUN apt-get update && apt-get install -y \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

EXPOSE 8520

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8520"]

