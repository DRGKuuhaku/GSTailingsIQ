FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y     gcc     g++     libpq-dev     curl     libxml2-dev     libxslt1-dev     antiword     poppler-utils     tesseract-ocr     && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN groupadd -r tailingsiq && useradd -r -g tailingsiq tailingsiq

# Set work directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip
RUN pip install --no-cache-dir -r requirements.txt

# Download spaCy model
RUN python -m spacy download en_core_web_sm

# Copy application code
COPY . .

# Create necessary directories and set permissions
RUN mkdir -p uploads chroma_db logs &&     chown -R tailingsiq:tailingsiq /app

# Switch to non-root user
USER tailingsiq

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3     CMD curl -f http://localhost:8000/health || exit 1

# Expose port
EXPOSE 8000

# Run the application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2"]
