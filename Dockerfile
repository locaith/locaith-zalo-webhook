# --- Slim, nhanh build
FROM python:3.11-slim

# System deps (libmagic, build tools nếu cần)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Tối ưu layer
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code
COPY . .

# Uvicorn port
ENV PORT=8080
EXPOSE 8080

# Health check for container
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

# Run with optimized settings for Render.com
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--timeout-keep-alive", "30", "--timeout-graceful-shutdown", "30", "--workers", "1"]