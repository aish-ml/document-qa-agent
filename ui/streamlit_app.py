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
)
from utils.helpers import setup_logging
from knowledge_base.vector_store import VectorStore
from agent.qa_agent import QAAgent
from ingestion.pdf_extractor import PDFExtractor
from ingestion.chunker import DocumentChunker
from arxiv_integration.arxiv_client import ArxivClient

setup_logging("INFO")


# ═══════════════════════════════════════════════════════════════
# Cached Singletons (survive Streamlit reruns)
# ═══════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner="Initializing vector store…")
def _get_vector_store() -> VectorStore:
    return VectorStore()


@st.cache_resource(show_spinner="Loading AI agent…")
def _get_agent(_vs: VectorStore) -> QAAgent:
    return QAAgent(vector_store=_vs)


# ═══════════════════════════════════════════════════════════════
# Background Ingest Helpers
# ═══════════════════════════════════════════════════════════════

def _run_pdf_ingest(vs: VectorStore, pdf_paths: list[str], key: str):
    """Ingest PDFs in a background thread, updating session_state."""
    try:
        extractor = PDFExtractor(save_images=True)
        chunker = DocumentChunker()
        total = 0
        details = []

        for i, p in enumerate(pdf_paths):
            fname = Path(p).name
            st.session_state[key] = f"Extracting {fname}… ({i+1}/{len(pdf_paths)})"
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
                details.append(f"❌ {fname} — {e}")

        st.session_state[key] = "done"
        st.session_state[f"{key}_result"] = {"total": total, "details": details}
    except Exception as e:
        st.session_state[key] = f"error: {e}"


def _run_arxiv_ingest(vs: VectorStore, papers: list, key: str):
    """Download + ingest Arxiv papers in a background thread."""
    try:
        client = ArxivClient()
        extractor = PDFExtractor(save_images=True)
        chunker = DocumentChunker()
        total = 0
        details = []

        for i, paper in enumerate(papers):
            short = paper.title[:55]
            st.session_state[key] = f"Downloading {short}… ({i+1}/{len(papers)})"
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
                details.append(f"❌ {short} — {e}")

        st.session_state[key] = "done"
        st.session_state[f"{key}_result"] = {"total": total, "details": details}
    except Exception as e:
        st.session_state[key] = f"error: {e}"


def _launch_job(target, args, key):
    """Start a daemon thread for background work."""
    st.session_state[key] = "starting…"
    t = threading.Thread(target=target, args=args, daemon=True)
    t.start()


# ═══════════════════════════════════════════════════════════════
# Session State Defaults
# ═══════════════════════════════════════════════════════════════

_DEFAULTS = {
    "messages": [],
    "arxiv_results": [],
    "auto_ingested": False,
}
for k, v in _DEFAULTS.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ═══════════════════════════════════════════════════════════════
# Initialize Backend
# ═══════════════════════════════════════════════════════════════

vs = _get_vector_store()
agent = _get_agent(vs)


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
            _launch_job(_run_pdf_ingest, (vs, new_pdfs, "auto_status"), "auto_status")
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
# Status Banners (background jobs)
# ═══════════════════════════════════════════════════════════════

def _show_status(key: str, label: str):
    """Render a status banner and auto-rerun while busy."""
    status = st.session_state.get(key)
    if not status:
        return
    if status == "done":
        result = st.session_state.get(f"{key}_result", {})
        total = result.get("total", 0)
        details = result.get("details", [])
        with st.expander(f"✅ {label} complete — {total} chunks added", expanded=True):
            for d in details:
                st.markdown(d)
            if st.button("Dismiss", key=f"dismiss_{key}"):
                for k2 in (key, f"{key}_result"):
                    st.session_state.pop(k2, None)
                st.rerun()
    elif status.startswith("error"):
        st.error(f"{label}: {status}")
    else:
        st.info(f"⏳ **{label}:** {status}")
        time.sleep(1.5)
        st.rerun()


_show_status("auto_status", "Auto-Ingest")
_show_status("upload_status", "Upload Ingest")
_show_status("folder_status", "Folder Ingest")
_show_status("arxiv_status", "Arxiv Download & Ingest")


# ═══════════════════════════════════════════════════════════════
# Sidebar
# ═══════════════════════════════════════════════════════════════

with st.sidebar:
    st.markdown("## 📄 Document Q&A Agent")
    st.caption(f"Provider: **{LLM_PROVIDER}** &nbsp;|&nbsp; Chunks: **{vs.count}**")

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
            paths = []
            for f in uploaded_files:
                dest = DOCUMENTS_DIR / f.name
                dest.write_bytes(f.getvalue())
                paths.append(str(dest))
            _launch_job(_run_pdf_ingest, (vs, paths, "upload_status"), "upload_status")
            st.rerun()

    # ── Ingest documents/ folder ────────────────────────────────
    st.markdown("### 📂 documents/ Folder")
    if st.button("Ingest all PDFs in documents/", use_container_width=True):
        pdfs = sorted(DOCUMENTS_DIR.glob("*.pdf"))
        if pdfs:
            _launch_job(
                _run_pdf_ingest,
                (vs, [str(p) for p in pdfs], "folder_status"),
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
                (vs, papers, "arxiv_status"),
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
