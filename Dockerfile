# Use official Python runtime as a parent image
FROM python:3.11-slim

# Set work directory
WORKDIR /app

# System deps (optional, keep minimal)
RUN apt-get update -y && apt-get install -y --no-install-recommends \
    build-essential \
 && rm -rf /var/lib/apt/lists/*

# Copy requirement spec first (better layer caching)
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the source code
COPY . .

# Environment
ENV PORT=8080 \
    HOST=0.0.0.0 \
    PYTHONUNBUFFERED=1

# Expose the port Fly.io will route to
EXPOSE 8080

# Healthcheck: use a portable one-liner (no heredoc) for broader builder compatibility
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import os,sys,urllib.request,urllib.error; url=f'http://127.0.0.1:{os.environ.get("PORT","8080")}/health';\
try:\
    r=urllib.request.urlopen(url, timeout=3); sys.exit(0 if getattr(r,'status',getattr(r,'code',200))==200 else 1)\
except Exception:\
    sys.exit(1)" 

# Run FastAPI via uvicorn
# App is defined in web_app/main.py as `app`
CMD ["python", "-m", "uvicorn", "web_app.main:app", "--host", "0.0.0.0", "--port", "8080", "--app-dir", "/app"]
