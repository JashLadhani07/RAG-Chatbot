# Use full Python 3.11 image to avoid dependency build issues
FROM python:3.11

# Set working directory
WORKDIR /app

# Install system dependencies (needed for Chroma / LangChain / PDFs)
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    gcc \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python packages
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir --prefer-binary -r requirements.txt

# Copy all code
COPY . .

# Optional: expose default port (Render uses $PORT)
EXPOSE 8000

# Run FastAPI using Render's port
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "${PORT:-8000}"]
