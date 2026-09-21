# ── Build stage — install dependencies ───────────────────────────────────────
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Build tools needed for faiss-cpu, numpy, sentence-transformers
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        gcc \
        g++ \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# ── Runtime stage ─────────────────────────────────────────────────────────────
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    # Ollama runs on the host machine — use host-gateway bridge
    OLLAMA_HOST=http://host.docker.internal:11434 \
    OLLAMA_MODEL=llama3:8b \
    EMBEDDING_MODEL=all-MiniLM-L6-v2 \
    VECTOR_BACKEND=faiss \
    INCLUDE_CODE_INDEX=true \
    EVAL_DISABLE_ARXIV=true \
    # Context window large enough for comparison prompts
    OLLAMA_NUM_CTX=8192 \
    OLLAMA_NUM_PREDICT=512 \
    # Rate limiting
    RATE_LIMIT_ENABLED=true \
    RATE_LIMIT_REQUESTS=30 \
    RATE_LIMIT_WINDOW_SECONDS=60

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /usr/local/lib/python3.12 /usr/local/lib/python3.12
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy application source
COPY . .

# Create required data directories (volumes will mount over these)
RUN mkdir -p \
        /app/data/pdfs \
        /app/data/indexes/chroma \
        /app/data/indexes/eval \
        /app/eval/results

# API port
EXPOSE 8000
# Streamlit UI port
EXPOSE 8501

# Default: run the FastAPI backend
# Override CMD in docker-compose for the UI service
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
