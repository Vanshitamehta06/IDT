# Autonomous Multi-Agent Academic Research Assistant & Local RAG

Local, space-conscious research engine for academic PDFs and live arXiv discovery. A planner agent classifies intent, then an orchestrator runs RAG, hybrid re-ranking, specialized analysis tools, and citation-grounded synthesis against a 7B/8B Ollama model.

## What it does

- Multi-PDF ingestion from `data/pdfs/` (filename, page, section, chunk id metadata)
- Hybrid retrieval: dense FAISS or Chroma + BM25 fused with Reciprocal Rank Fusion
- Intent planning: `research_qa`, `discovery`, `analysis`, `comparison`, `gap_analysis`, `recommendation`
- arXiv search with timeouts, User-Agent, XML parsing, and local RAG fallback
- Inline evidence tags `[Source N]` and `[Page X | Section Y]`
- Streamlit dark dashboard with conversational memory, traces, citations, and telemetry

## Hardware notes

Optimized for local 7B-class models (`llama3:8b`, `codellama:7b`) via Ollama. Embeddings default to `all-MiniLM-L6-v2`. Combined LLM context is truncated (~3.5k–4k characters) to reduce OOM risk. Vector indexes live under `data/indexes/`.

## Prerequisites

1. Python 3.11+
2. [Ollama](https://ollama.com) running locally
3. A pulled model, for example:

```bash
ollama pull llama3:8b
```

## Local setup

```bash
cd research-assistant-rag
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
python scripts/download_seed_pdf.py
```

Place additional PDFs in `data/pdfs/`. Then start the API and UI (from this directory so `config.py` imports resolve):

```bash
set PYTHONPATH=.
uvicorn api.main:app --reload --host 0.0.0.0 --port 8000
```

In a second terminal:

```bash
set PYTHONPATH=.
set RESEARCH_API_URL=http://localhost:8000
streamlit run ui/app.py
```

On macOS/Linux use `export PYTHONPATH=.` and `export RESEARCH_API_URL=http://localhost:8000`.

## Docker

Ollama stays on the host. Compose maps `host.docker.internal` to the host gateway:

```bash
docker compose up --build
```

- API: http://localhost:8000/docs
- UI: http://localhost:8501
- Health: http://localhost:8000/health

Environment defaults:

| Variable | Default |
| --- | --- |
| `OLLAMA_HOST` | `http://localhost:11434` (Docker: `http://host.docker.internal:11434`) |
| `OLLAMA_MODEL` | `llama3:8b` |
| `EMBEDDING_MODEL` | `all-MiniLM-L6-v2` |
| `VECTOR_BACKEND` | `faiss` (`chroma` also supported) |
| `CHUNK_SIZE` / `CHUNK_OVERLAP` | `512` / `100` |
| `USE_LLM_PLANNER` | `false` (heuristic intent; set `true` for an extra Ollama planner call) |
| `OLLAMA_TIMEOUT_SECONDS` | `600` |
| `OLLAMA_NUM_PREDICT` | `384` |

## Troubleshooting: Streamlit “Query failed: timed out”

That message came from the UI giving up after **180 seconds** while the backend was still waiting on Ollama (planner + answer + synthesizer, plus first-time model load). The UI now waits up to **15 minutes**, and the pipeline makes **one** local LLM generation by default.

Restart both processes after pulling these changes:

1. Confirm Ollama is up: `ollama list` and `ollama run llama3:8b`
2. Sidebar health should be green (`http://localhost:8000/health`)
3. Restart `uvicorn` and `streamlit`, then run the query again

The first request after a reboot can still take several minutes while the model loads into memory.

## Multi-model evaluation (assignment lab)

Held constant: application, `llm/prompt_builder.py` prompts, `eval/dataset.json` (26 tasks), shared PDF+source index, chunking, embeddings, retrieval k, temperature/`num_ctx`/`num_predict`. arXiv is disabled during eval.

Default models: **llama3:8b**, **codellama:7b**, **starcoder2:3b**.

```bash
ollama pull llama3:8b
ollama pull codellama:7b
ollama pull starcoder2:3b
```

Rebuild the index once so Python sources are chunked (needed for repo questions):

```bash
# POST /ingest {"rebuild": true} or use the Streamlit sidebar
```

Then either:

```bash
set PYTHONPATH=.
python -m eval.runner --limit 3
python -m eval.runner
```

or use **Model Evaluation Lab** in Streamlit (`POST /eval/run`). Results: `eval/results/latest.json` and `eval/results/analysis.md`.

**Backend Inspector** (after any query) shows chunking parameters, total chunks, retrieved chunk text/length/metadata, embedding model + dimension + query-embed latency, L2/RRF scores, the exact raw prompt, and tokens/sec.

## API

- `GET /health` — Ollama reachability, PDF count, indexed chunks
- `POST /ingest` — `{ "rebuild": true }` rescan of `data/pdfs/`
- `POST /query` — `{ "query": "...", "history": [{"role":"user","content":"..."}] }`

## Project layout

See `api/`, `services/`, `vector_db/`, `llm/`, and `ui/` for the modular pipeline. The orchestrator records step traces and a unified citation registry across local chunks and arXiv papers.
