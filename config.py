"""Runtime configuration for the local multi-agent research assistant."""

from __future__ import annotations

import os
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"
PDF_DIR = DATA_DIR / "pdfs"
INDEX_DIR = DATA_DIR / "indexes"
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
FAISS_META_PATH = INDEX_DIR / "faiss_meta.json"
CHROMA_DIR = INDEX_DIR / "chroma"

OLLAMA_HOST = os.getenv("OLLAMA_HOST", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3:8b")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
VECTOR_BACKEND = os.getenv("VECTOR_BACKEND", "faiss").lower()
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "academic_pdfs")

# Sliding-window chunking (token-approximate via whitespace split)
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "512"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# Retrieval
RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "8"))
RERANK_TOP_K = int(os.getenv("RERANK_TOP_K", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))

# Defensive context bounds for 7B/8B local models
# Increased to ensure abstract/intro chunks are not truncated in evidence blocks.
MAX_PROMPT_CHARS = int(os.getenv("MAX_PROMPT_CHARS", "6000"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "5000"))
MAX_ABSTRACT_CHARS = int(os.getenv("MAX_ABSTRACT_CHARS", "1200"))
MAX_COMBINED_TOOL_CHARS = int(os.getenv("MAX_COMBINED_TOOL_CHARS", "5000"))
MAX_HISTORY_TURNS = int(os.getenv("MAX_HISTORY_TURNS", "6"))
MAX_HISTORY_CHARS = int(os.getenv("MAX_HISTORY_CHARS", "1800"))

ARXIV_BASE_URL = os.getenv("ARXIV_BASE_URL", "http://export.arxiv.org/api/query")
ARXIV_TIMEOUT_SECONDS = float(os.getenv("ARXIV_TIMEOUT_SECONDS", "12"))
ARXIV_MAX_RESULTS = int(os.getenv("ARXIV_MAX_RESULTS", "8"))
USER_AGENT = os.getenv(
    "USER_AGENT",
    "AcademicResearchAssistant/1.0 (local; mailto:research@localhost)",
)

API_HOST = os.getenv("API_HOST", "0.0.0.0")
API_PORT = int(os.getenv("API_PORT", "8000"))

OLLAMA_TIMEOUT_SECONDS = float(os.getenv("OLLAMA_TIMEOUT_SECONDS", "600"))
# 8192 tokens gives room for long comparison prompts (both papers' evidence = ~3500 tok)
OLLAMA_NUM_CTX = int(os.getenv("OLLAMA_NUM_CTX", "8192"))
OLLAMA_NUM_PREDICT = int(os.getenv("OLLAMA_NUM_PREDICT", "512"))
# Heuristic planner avoids a full extra local-LLM round trip (main cause of UI timeouts).
USE_LLM_PLANNER = os.getenv("USE_LLM_PLANNER", "false").lower() in {"1", "true", "yes"}
PLANNER_TIMEOUT_SECONDS = float(os.getenv("PLANNER_TIMEOUT_SECONDS", "25"))
QUERY_READ_TIMEOUT_SECONDS = float(os.getenv("QUERY_READ_TIMEOUT_SECONDS", "900"))

EVAL_MODELS = [
    m.strip()
    for m in os.getenv("EVAL_MODELS", "llama3:8b,codellama:7b,starcoder2:3b").split(",")
    if m.strip()
]
EVAL_INDEX_DIR = INDEX_DIR / "eval"
EVAL_FAISS_INDEX_PATH = EVAL_INDEX_DIR / "faiss.index"
EVAL_FAISS_META_PATH = EVAL_INDEX_DIR / "faiss_meta.json"
INCLUDE_CODE_INDEX = os.getenv("INCLUDE_CODE_INDEX", "true").lower() in {"1", "true", "yes"}
EVAL_DISABLE_ARXIV = os.getenv("EVAL_DISABLE_ARXIV", "true").lower() in {"1", "true", "yes"}
EVAL_RESULTS_DIR = ROOT_DIR / "eval" / "results"

# ---------------------------------------------------------------------------
# Rate limiting
# ---------------------------------------------------------------------------
RATE_LIMIT_ENABLED = os.getenv("RATE_LIMIT_ENABLED", "true").lower() in {"1", "true", "yes"}
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))  # per window
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

# ---------------------------------------------------------------------------
# Guardrails — scope keywords define what the assistant is intended to help with
# ---------------------------------------------------------------------------
GUARDRAIL_SCOPE_KEYWORDS: list[str] = [
    kw.strip()
    for kw in os.getenv(
        "GUARDRAIL_SCOPE_KEYWORDS",
        "research,paper,study,literature,retrieval,rag,embedding,vector,language model,"
        "llm,nlp,machine learning,deep learning,neural,dataset,benchmark,method,model,"
        "algorithm,evaluation,experiment,finding,limitation,gap,citation,abstract,"
        "methodology,contribution,summary,summarize,compare,analysis,recommend,discover,"
        "code,function,implementation,architecture,repository,service,file,module",
    ).split(",")
    if kw.strip()
]
# Minimum score (0–1) for a query to be considered in-scope
GUARDRAIL_SCOPE_THRESHOLD = float(os.getenv("GUARDRAIL_SCOPE_THRESHOLD", "0.0"))

# ---------------------------------------------------------------------------
# Intelligent model orchestration — complexity routing
# ---------------------------------------------------------------------------
# Tokens threshold above which a query is considered "complex"
COMPLEXITY_TOKEN_THRESHOLD = int(os.getenv("COMPLEXITY_TOKEN_THRESHOLD", "15"))
# Keywords that immediately mark a query as complex regardless of length
COMPLEXITY_KEYWORDS: list[str] = [
    kw.strip()
    for kw in os.getenv(
        "COMPLEXITY_KEYWORDS",
        "compare,analyze,contrast,summarize,explain,critique,methodology,limitation,"
        "gap,multi-document,cross-paper,refactor,implement,architecture,dependency,"
        "impact,affected,trace,pipeline",
    ).split(",")
    if kw.strip()
]
# Default model tier mapping (overridden by EVAL_MODELS ordering at runtime)
MODEL_TIER_SIMPLE = os.getenv("MODEL_TIER_SIMPLE", "")     # empty = use whatever is set
MODEL_TIER_COMPLEX = os.getenv("MODEL_TIER_COMPLEX", "")   # empty = use whatever is set
# Grounding score below which an answer is flagged as potentially unsupported
GROUNDING_THRESHOLD = float(os.getenv("GROUNDING_THRESHOLD", "0.35"))


def ensure_data_dirs() -> None:
    PDF_DIR.mkdir(parents=True, exist_ok=True)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_INDEX_DIR.mkdir(parents=True, exist_ok=True)
    EVAL_RESULTS_DIR.mkdir(parents=True, exist_ok=True)
