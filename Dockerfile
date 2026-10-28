# PolyCompIE — frontend image
#
# Build:   docker build -t polycompie .
# Run:     docker run -p 8080:8080 polycompie
# The container reads $PORT (default 8080) and binds 0.0.0.0.

FROM python:3.12-slim

# Native libs required by torch/EasyOCR/Docling at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgl1 \
        libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Force HOME and cache locations to /app so weight lookups resolve to the
# same path during build (root) AND runtime (e.g. HF Spaces' uid 1000).
# Without this, EasyOCR/Docling re-download model weights on first request
# because ~/.EasyOCR and ~/.cache/huggingface point to different dirs.
ENV HOST=0.0.0.0 \
    HOME=/app \
    HF_HOME=/app/.cache/huggingface \
    XDG_CACHE_HOME=/app/.cache

# 1) Install deps. This layer caches as long as pyproject.toml + src/ don't change.
COPY pyproject.toml README.md LICENSE ./
COPY src/ ./src/
RUN pip install --no-cache-dir -e ".[app]"

# 2) Pre-download all model weights into /app caches.
#    EasyOCR caches to $HOME/.EasyOCR (= /app/.EasyOCR).
#    Docling uses HF Hub which respects HF_HOME (= /app/.cache/huggingface).
RUN python -c "import easyocr; easyocr.Reader(['en'], download_enabled=True)" \
 && docling-tools models download

# 3) Make the cached weights world-readable so any UID can use them at runtime.
RUN chmod -R a+rX /app

# 4) Copy the frontend last so app/* edits don't bust the deps + weights cache.
COPY app/ ./app/

EXPOSE 8080

CMD ["python", "-m", "app.main"]
