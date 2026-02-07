"""
Configuration module for Document Q&A AI Agent.
Loads settings from environment variables with sensible defaults.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ── Project Paths ────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = Path(os.getenv("DOCUMENTS_DIR", str(BASE_DIR / "documents")))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")))
IMAGES_DIR = BASE_DIR / "extracted_images"

# ── LLM Provider ────────────────────────────────────────────────
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "gemini").lower()

# OpenAI
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o")

# Google Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")

# Ollama (local open-source models)
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")

# ── Embeddings ───────────────────────────────────────────────────
EMBEDDING_PROVIDER = os.getenv("EMBEDDING_PROVIDER", "gemini").lower()
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

# ── Chunking ─────────────────────────────────────────────────────
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))

# ── Enterprise Settings ─────────────────────────────────────────
MAX_CONTEXT_CHUNKS = int(os.getenv("MAX_CONTEXT_CHUNKS", "8"))
RESPONSE_MAX_TOKENS = int(os.getenv("RESPONSE_MAX_TOKENS", "2048"))
RATE_LIMIT_RPM = int(os.getenv("RATE_LIMIT_RPM", "60"))
ENABLE_CACHE = os.getenv("ENABLE_CACHE", "true").lower() == "true"

# ── Auto-Ingest & Arxiv Settings ────────────────────────────────
AUTO_INGEST_ON_START = os.getenv("AUTO_INGEST_ON_START", "true").lower() == "true"
ARXIV_AUTO_INGEST_TOPN = int(os.getenv("ARXIV_AUTO_INGEST_TOPN", "5"))
INGEST_BATCH_SIZE = int(os.getenv("INGEST_BATCH_SIZE", "50"))

# ── Ensure required directories exist ────────────────────────────
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
CHROMA_PERSIST_DIR.mkdir(parents=True, exist_ok=True)
IMAGES_DIR.mkdir(parents=True, exist_ok=True)
