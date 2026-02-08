"""
Streamlit Chat UI for Document Q&A AI Agent
============================================
Beautiful, interactive web interface for document ingestion,
Arxiv search + auto-ingest, and conversational Q&A.

Run with:
    streamlit run ui/streamlit_app.py
"""

import sys
import time
import logging
import threading
from pathlib import Path

# ── Ensure project root is importable ────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st

# ── Page Config (MUST be first Streamlit call) ───────────────────
st.set_page_config(
    page_title="Document Q&A Agent",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

from config import (
    DOCUMENTS_DIR,
    LLM_PROVIDER,
    AUTO_INGEST_ON_START,
    ARXIV_AUTO_INGEST_TOPN,
    OPENAI_MODEL,
    GEMINI_MODEL,
    OLLAMA_MODEL,
)
from utils.helpers import setup_logging
from knowledge_base.vector_store import VectorStore
from agent.qa_agent import QAAgent
from ingestion.pdf_extractor import PDFExtractor
from ingestion.chunker import DocumentChunker
from arxiv_integration.arxiv_client import ArxivClient

setup_logging("INFO")
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# Cached Singletons (survive Streamlit reruns)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initializing vector store…")
def _get_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner="Loading AI agent…")
def _get_agent(_vs: VectorStore, provider: str = None, model: str = None, api_key: str = None) -> QAAgent:
    return QAAgent(vector_store=_vs, provider=provider, model=model, api_key=api_key)


# ═══════════════════════════════════════════════════════════════
# Background Ingest Helpers
# ═══════════════════════════════════════════════════════════════

def _run_pdf_ingest(vs: VectorStore, pdf_paths: list[str], status: dict):
    """Ingest PDFs in a background thread, updating shared status dict."""
    try:
        extractor = PDFExtractor(save_images=True)
        chunker = DocumentChunker()
        total = 0
        details = []

        for i, p in enumerate(pdf_paths):
            fname = Path(p).name
            status["msg"] = f"Extracting {fname}… ({i+1}/{len(pdf_paths)})"
            try:
                doc = extractor.extract(p)
                chunks = chunker.chunk_document(doc)
                count = vs.add_chunks(chunks)
                total += count
                details.append(
                    f"✅ {fname} — {doc.total_pages} pages, "
                    f"{len(doc.tables)} tables, {count} chunks"
                )
            except Exception as e:
                logger.error(f"PDF ingest error for {fname}: {e}")
                details.append(f"❌ {fname} — {e}")

        status["msg"] = "done"
        status["result"] = {"total": total, "details": details}
    except Exception as e:
        logger.error(f"PDF ingest thread error: {e}")
        status["msg"] = f"error: {e}"


def _run_arxiv_ingest(vs: VectorStore, papers: list, status: dict):
    """Download + ingest Arxiv papers in a background thread."""
    try:
        client = ArxivClient()
        extractor = PDFExtractor(save_images=True)
        chunker = DocumentChunker()
        total = 0
        details = []

        for i, paper in enumerate(papers):
            short = paper.title[:55]
            status["msg"] = f"Downloading {short}… ({i+1}/{len(papers)})"
            logger.info(f"Arxiv ingest: downloading {short}")
            pdf_path = client.download_pdf(paper, str(DOCUMENTS_DIR))
            if not pdf_path:
                details.append(f"❌ {short} — download failed")
                continue
            try:
                doc = extractor.extract(pdf_path)
                chunks = chunker.chunk_document(doc)
                count = vs.add_chunks(chunks)
                total += count
                details.append(f"✅ {short} — {doc.total_pages}p, {count} chunks")
            except Exception as e:
                logger.error(f"Arxiv ingest error for {short}: {e}")
                details.append(f"❌ {short} — {e}")

        status["msg"] = "done"
        status["result"] = {"total": total, "details": details}
        logger.info(f"Arxiv ingest complete: {total} chunks from {len(papers)} papers")
    except Exception as e:
        logger.error(f"Arxiv ingest thread error: {e}")
        status["msg"] = f"error: {e}"


def _launch_job(target, args, key):
    """Start a daemon thread for background work using a shared status dict."""
    status = {"msg": "starting…", "result": None}
    st.session_state[f"{key}_status"] = status
    t = threading.Thread(target=target, args=(*args, status), daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════
# Session State Defaults
# ═══════════════════════════════════════════════════════════════

_PROVIDER_MODELS = {
    "gemini": GEMINI_MODEL,
    "openai": OPENAI_MODEL,
    "ollama": OLLAMA_MODEL,
}

_DEFAULTS = {
    "messages": [],
    "arxiv_results": [],
    "auto_ingested": False,
    "llm_provider": LLM_PROVIDER,
    "llm_model": _PROVIDER_MODELS.get(LLM_PROVIDER, ""),
    "llm_api_key": "",
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# Initialize Backend
# ═══════════════════════════════════════════════════════════════

vs = _get_vector_store()
agent = _get_agent(
    vs,
    provider=st.session_state.llm_provider,
    model=st.session_state.llm_model or None,
    api_key=st.session_state.llm_api_key or None,
)


# ═══════════════════════════════════════════════════════════════
# Auto-Ingest on First Load
# ═══════════════════════════════════════════════════════════════

if AUTO_INGEST_ON_START and not st.session_state.auto_ingested:
    pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
    if pdfs:
        existing = set(vs.list_sources())
        new_pdfs = [str(p) for p in pdfs if p.name not in existing]
        if new_pdfs:
            st.session_state.auto_ingested = True
            _launch_job(_run_pdf_ingest, (vs, new_pdfs), "auto_status")
        else:
            st.session_state.auto_ingested = True
    else:
        st.session_state.auto_ingested = True


# ═══════════════════════════════════════════════════════════════
# Custom CSS
# ═══════════════════════════════════════════════════════════════

st.markdown(
    """
    <style>
    /* Sidebar width */
    [data-testid="stSidebar"] { min-width: 340px; max-width: 420px; }

    /* Chat messages */
    .stChatMessage { border-radius: 12px; margin-bottom: 8px; }

    /* Source badges */
    .source-badge {
        display: inline-block;
        background: #e8f4f8;
        color: #1a5276;
        padding: 2px 8px;
        border-radius: 4px;
        font-size: 0.82em;
        margin: 2px 4px 2px 0;
    }

    /* Status banner */
    .ingest-status {
        padding: 10px 14px;
        border-radius: 8px;
        margin-bottom: 12px;
        font-size: 0.9em;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ═══════════════════════════════════════════════════════════════
# Status Banners (background jobs) — runs as a fragment so the
# rest of the page stays interactive during ingestion.
# ═══════════════════════════════════════════════════════════════

@st.fragment(run_every=2)
def _status_fragment():
    """Re-runs every 2 s *independently* of the main script."""
    _any_active = False
    for key, label in [
        ("auto_status", "Auto-Ingest"),
        ("upload_status", "Upload Ingest"),
        ("folder_status", "Folder Ingest"),
        ("arxiv_status", "Arxiv Download & Ingest"),
    ]:
        status_dict = st.session_state.get(f"{key}_status")
        if not status_dict:
            continue
        msg = status_dict.get("msg", "")
        if not msg:
            continue
        if msg == "done":
            result = status_dict.get("result") or {}
            total = result.get("total", 0)
            details = result.get("details", [])
            with st.expander(f"✅ {label} complete — {total} chunks added", expanded=True):
                for d in details:
                    st.markdown(d)
                if st.button("Dismiss", key=f"dismiss_{key}"):
                    st.session_state.pop(f"{key}_status", None)
                    st.rerun()
        elif msg.startswith("error"):
            st.error(f"{label}: {msg}")
            if st.button("Dismiss", key=f"dismiss_err_{key}"):
                st.session_state.pop(f"{key}_status", None)
                st.rerun()
        else:
            st.info(f"⏳ **{label}:** {msg}")
            _any_active = True

_status_fragment()


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📄 Document Q&A Agent")
    st.caption(
        f"Provider: **{st.session_state.llm_provider}** "
        f"({st.session_state.llm_model}) "
        f"&nbsp;|&nbsp; Chunks: **{vs.count}**"
    )

    # ── LLM Configuration ───────────────────────────────────────
    st.markdown("### 🤖 LLM Settings")
    _providers = ["gemini", "openai", "ollama"]
    _cur_idx = _providers.index(st.session_state.llm_provider) if st.session_state.llm_provider in _providers else 0
    sel_provider = st.selectbox(
        "Provider",
        _providers,
        index=_cur_idx,
        key="_sel_provider",
    )
    sel_model = st.text_input(
        "Model",
        value=st.session_state.llm_model,
        placeholder=_PROVIDER_MODELS.get(sel_provider, ""),
        key="_sel_model",
    )
    sel_api_key = st.text_input(
        "API Key (session only)",
        value=st.session_state.llm_api_key,
        type="password",
        placeholder="Leave blank to use .env key",
        key="_sel_api_key",
    )

    _changed = (
        sel_provider != st.session_state.llm_provider
        or sel_model != st.session_state.llm_model
        or sel_api_key != st.session_state.llm_api_key
    )
    if _changed:
        if st.button("🔄 Apply LLM Settings", type="primary", use_container_width=True):
            st.session_state.llm_provider = sel_provider
            st.session_state.llm_model = sel_model or _PROVIDER_MODELS.get(sel_provider, "")
            st.session_state.llm_api_key = sel_api_key
            # Clear the cached agent so it gets re-created with new settings
            _get_agent.clear()
            st.success(f"Switched to **{sel_provider}** / **{st.session_state.llm_model}**")
            time.sleep(0.8)
            st.rerun()

    st.divider()

    # ── Knowledge Base ──────────────────────────────────────────
    st.markdown("### 📚 Knowledge Base")
    sources = vs.list_sources()
    if sources:
        for s in sources:
            st.markdown(f"- `{s}`")
    else:
        st.info("No documents loaded yet. Upload PDFs or use Arxiv search below.")

    st.divider()

    # ── Upload PDFs ─────────────────────────────────────────────
    st.markdown("### ⬆️ Upload PDFs")
    uploaded_files = st.file_uploader(
        "Drop PDF files here",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )
    if uploaded_files:
        if st.button("📥 Ingest uploaded files", type="primary", use_container_width=True):
            # Save all uploaded files to disk first
            DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)
            paths = []
            for f in uploaded_files:
                dest = DOCUMENTS_DIR / f.name
                dest.write_bytes(f.getvalue())
                paths.append(str(dest))
                logger.info(f"Saved uploaded file: {dest}")
            # Launch background ingest
            _launch_job(_run_pdf_ingest, (vs, paths), "upload_status")
            st.rerun()

    # ── Ingest documents/ folder ────────────────────────────────
    st.markdown("### 📂 documents/ Folder")
    if st.button("Ingest all PDFs in documents/", use_container_width=True):
        pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
        if pdfs:
            _launch_job(
                _run_pdf_ingest,
                (vs, [str(p) for p in pdfs]),
                "folder_status",
            )
            st.rerun()
        else:
            st.warning("No PDFs found in documents/ folder.")

    st.divider()

    # ── Arxiv Search + Auto-Ingest ──────────────────────────────
    st.markdown("### 🔍 Arxiv Search")
    arxiv_query = st.text_input(
        "Search query",
        placeholder="e.g. large language models 2024",
        label_visibility="collapsed",
    )
    col1, col2 = st.columns([2, 1])
    with col2:
        arxiv_topn = st.number_input("Top N", 1, 20, ARXIV_AUTO_INGEST_TOPN, label_visibility="collapsed")
    with col1:
        do_search = st.button("🔎 Search", use_container_width=True)

    if do_search and arxiv_query:
        with st.spinner("Searching Arxiv…"):
            client = ArxivClient(max_results=int(arxiv_topn))
            papers = client.search(arxiv_query)
            st.session_state.arxiv_results = papers
            if not papers:
                st.warning("No papers found.")
            st.rerun()

    # ── Display Arxiv results ───────────────────────────────────
    if st.session_state.arxiv_results:
        papers = st.session_state.arxiv_results
        st.success(f"Found **{len(papers)}** papers")
        for i, p in enumerate(papers):
            with st.expander(f"{i+1}. {p.title[:75]}", expanded=False):
                st.markdown(f"**Authors:** {', '.join(p.authors[:4])}")
                st.markdown(f"**Published:** {p.published} &nbsp;|&nbsp; **ID:** `{p.arxiv_id}`")
                st.markdown(f"[📥 PDF]({p.pdf_url})")
                st.caption(
                    p.abstract[:280] + "…" if len(p.abstract) > 280 else p.abstract
                )

        if st.button(
            f"⬇️ Download & Ingest All {len(papers)} Papers",
            type="primary",
            use_container_width=True,
        ):
            _launch_job(
                _run_arxiv_ingest,
                (vs, papers),
                "arxiv_status",
            )
            st.session_state.arxiv_results = []
            st.rerun()

        if st.button("Clear results", use_container_width=True):
            st.session_state.arxiv_results = []
            st.rerun()

    st.divider()

    # ── Controls ────────────────────────────────────────────────
    st.markdown("### ⚙️ Controls")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            agent.clear_history()
            st.rerun()
    with c2:
        if st.button("🗄️ Reset KB", use_container_width=True):
            vs.clear()
            st.session_state.messages = []
            agent.clear_history()
            st.rerun()

    # ── Stats ───────────────────────────────────────────────────
    with st.expander("📊 Agent Stats"):
        stats = agent.get_stats()
        for k, v in stats.items():
            st.markdown(f"**{k}:** `{v}`")


# ═══════════════════════════════════════════════════════════════
# Main Chat Area
# ═══════════════════════════════════════════════════════════════

st.header("💬 Ask your Documents")

if not sources:
    st.info(
        "👈 **Get started:** Upload PDFs in the sidebar, click "
        "\"Ingest all PDFs\", or search Arxiv to download papers."
    )

# ── Render conversation history ─────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"], avatar="🧑‍💻" if msg["role"] == "user" else "🤖"):
        st.markdown(msg["content"])
        if msg.get("sources"):
            src_html = " ".join(
                f'<span class="source-badge">📄 {s}</span>' for s in msg["sources"]
            )
            st.markdown(src_html, unsafe_allow_html=True)

# ── Chat input ──────────────────────────────────────────────────

if prompt := st.chat_input("Ask a question about your documents…"):
    # Append & render user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Get agent response
    with st.chat_message("assistant", avatar="🤖"):
        with st.spinner("Thinking…"):
            response = agent.ask(prompt)

        answer = response.get("answer", "I couldn't find an answer.")
        st.markdown(answer)

        srcs = response.get("sources", [])
        if srcs:
            src_html = " ".join(
                f'<span class="source-badge">📄 {s}</span>' for s in srcs
            )
            st.markdown(src_html, unsafe_allow_html=True)

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": srcs,
    })
