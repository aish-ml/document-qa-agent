"""
Agent Tools
------------
LangChain tools for the Document Q&A agent:
- Document lookup / search
- Summarization
- Evaluation result extraction
- Arxiv paper search (bonus)
"""

import logging

from langchain_core.tools import tool

from knowledge_base.vector_store import VectorStore
from arxiv_integration.arxiv_client import ArxivClient, search_arxiv

logger = logging.getLogger(__name__)

# Shared vector store instance — set by QAAgent on init
_vector_store: VectorStore | None = None


def set_vector_store(vs: VectorStore):
    global _vector_store
    _vector_store = vs


# ── Tool 1: Document Search / Lookup ─────────────────────────────

@tool
def search_documents(query: str) -> str:
    """
    Search ingested documents for content relevant to the query.
    Use this tool to look up specific information, facts, sections,
    conclusions, or data from the uploaded PDFs.

    Args:
        query: The search query describing what to find.
    """
    if not _vector_store:
        return "Error: No documents loaded. Please ingest PDFs first."

    results = _vector_store.search(query, k=6)
    if not results:
        return f"No relevant content found for: '{query}'"

    context_parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        section = doc.metadata.get("section_title", "")
        content_type = doc.metadata.get("content_type", "text")
        page = doc.metadata.get("page_start", "?")

        header = f"[Result {i} | {source} | Page {page}"
        if section:
            header += f" | {section}"
        header += f" | Type: {content_type}]"

        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


# ── Tool 2: Summarization ────────────────────────────────────────

@tool
def summarize_document_section(query: str) -> str:
    """
    Retrieve content from documents to summarize a specific topic,
    methodology, section, or key insights. Returns relevant chunks
    that can be summarized by the agent.

    Args:
        query: Describe what you want summarized, e.g.,
               'methodology of Paper X' or 'key findings from Paper Y'.
    """
    if not _vector_store:
        return "Error: No documents loaded. Please ingest PDFs first."

    # Retrieve more chunks for summarization
    results = _vector_store.search(query, k=10)
    if not results:
        return f"No content found to summarize for: '{query}'"

    context_parts = []
    for i, doc in enumerate(results, 1):
        source = doc.metadata.get("source", "Unknown")
        section = doc.metadata.get("section_title", "")
        page = doc.metadata.get("page_start", "?")

        header = f"[Source: {source} | Page {page}"
        if section:
            header += f" | Section: {section}"
        header += "]"

        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


# ── Tool 3: Extract Evaluation Results ──────────────────────────

@tool
def extract_evaluation_results(query: str) -> str:
    """
    Search for evaluation metrics, experimental results, accuracy,
    F1-scores, precision, recall, benchmarks, or statistical data
    reported in the documents.

    Args:
        query: Describe what evaluation results to extract, e.g.,
               'accuracy and F1-score from Paper D' or
               'benchmark results comparison'.
    """
    if not _vector_store:
        return "Error: No documents loaded. Please ingest PDFs first."

    # Search with keywords biased toward results/evaluation sections
    enhanced_query = (
        f"{query} evaluation results metrics accuracy F1 precision recall "
        f"benchmark performance score"
    )
    results = _vector_store.search(enhanced_query, k=8)

    if not results:
        return f"No evaluation results found for: '{query}'"

    # Also search for tables which often contain results
    table_results = _vector_store.search(
        query, k=4, filter_metadata={"content_type": "table"}
    )

    all_results = results + [
        r for r in table_results if r not in results
    ]

    context_parts = []
    for i, doc in enumerate(all_results, 1):
        source = doc.metadata.get("source", "Unknown")
        content_type = doc.metadata.get("content_type", "text")
        page = doc.metadata.get("page_start", "?")

        header = f"[Result {i} | {source} | Page {page} | Type: {content_type}]"
        context_parts.append(f"{header}\n{doc.page_content}")

    return "\n\n---\n\n".join(context_parts)


# ── Tool 4: List Available Documents ─────────────────────────────

@tool
def list_loaded_documents() -> str:
    """
    List all documents currently loaded in the knowledge base.
    Use this to see which papers/PDFs are available for querying.
    """
    if not _vector_store:
        return "No documents loaded."

    sources = _vector_store.list_sources()
    if not sources:
        return "No documents loaded in the knowledge base."

    lines = ["Documents in knowledge base:"]
    for i, src in enumerate(sources, 1):
        lines.append(f"  {i}. {src}")
    lines.append(f"\nTotal: {_vector_store.count} chunks indexed")
    return "\n".join(lines)


# ── Tool 5: Arxiv Search (Bonus) ────────────────────────────────

@tool
def search_arxiv_papers(query: str) -> str:
    """
    Search Arxiv for academic papers based on a description.
    Use this when the user wants to find papers not already loaded,
    or wants to look up a paper by topic, author, or description.

    Args:
        query: Description of the paper or topic to search for.
    """
    return search_arxiv(query, max_results=5)


# ── Tool 6: Download & Ingest Arxiv Paper ────────────────────────

@tool
def download_arxiv_paper(arxiv_id: str) -> str:
    """
    Download a paper from Arxiv by its ID and add it to the
    knowledge base for querying. Use after searching Arxiv to
    ingest a specific paper.

    Args:
        arxiv_id: The Arxiv paper ID (e.g., '2301.12345').
    """
    from config import DOCUMENTS_DIR
    from ingestion.pdf_extractor import PDFExtractor
    from ingestion.chunker import DocumentChunker

    client = ArxivClient()
    paper = client.get_paper_by_id(arxiv_id)
    if not paper:
        return f"Could not find paper with Arxiv ID: {arxiv_id}"

    # Download PDF
    pdf_path = client.download_pdf(paper, str(DOCUMENTS_DIR))
    if not pdf_path:
        return f"Failed to download PDF for: {paper.title}"

    # Extract and ingest
    try:
        extractor = PDFExtractor()
        doc = extractor.extract(pdf_path)

        chunker = DocumentChunker()
        chunks = chunker.chunk_document(doc)

        if _vector_store:
            _vector_store.add_chunks(chunks)
            return (
                f"Successfully downloaded and ingested:\n"
                f"  Title: {paper.title}\n"
                f"  Pages: {doc.total_pages}\n"
                f"  Chunks: {len(chunks)}\n"
                f"  Tables: {len(doc.tables)}\n"
                f"  Images: {len(doc.images)}\n"
                f"Paper is now available for Q&A."
            )
        else:
            return "Paper downloaded but vector store not initialized."
    except Exception as e:
        return f"Error ingesting paper: {e}"


# ── Tool 7: Search & Auto-Ingest Arxiv Papers ───────────────────

@tool
def search_and_ingest_arxiv(query: str) -> str:
    """
    Search Arxiv for papers AND automatically download + ingest
    the top results into the knowledge base. Use when the user
    wants to find and immediately load papers for Q&A.

    Args:
        query: Description of papers or topic to search and ingest.
    """
    from config import DOCUMENTS_DIR, ARXIV_AUTO_INGEST_TOPN
    from ingestion.pdf_extractor import PDFExtractor
    from ingestion.chunker import DocumentChunker

    client = ArxivClient(max_results=ARXIV_AUTO_INGEST_TOPN)
    papers = client.search(query)
    if not papers:
        return f"No papers found on Arxiv for: '{query}'"

    extractor = PDFExtractor()
    chunker = DocumentChunker()
    results = [f"Found {len(papers)} papers. Downloading & ingesting...\n"]

    for i, paper in enumerate(papers, 1):
        try:
            pdf_path = client.download_pdf(paper, str(DOCUMENTS_DIR))
            if not pdf_path:
                results.append(f"  {i}. ❌ {paper.title[:60]} — download failed")
                continue
            doc = extractor.extract(pdf_path)
            chunks = chunker.chunk_document(doc)
            if _vector_store:
                count = _vector_store.add_chunks(chunks)
                results.append(
                    f"  {i}. ✅ {paper.title[:60]} — "
                    f"{doc.total_pages} pages, {count} chunks"
                )
            else:
                results.append(f"  {i}. ⚠️ {paper.title[:60]} — no vector store")
        except Exception as e:
            results.append(f"  {i}. ❌ {paper.title[:60]} — {e}")

    results.append("\nPapers are now available for Q&A.")
    return "\n".join(results)


def get_all_tools() -> list:
    """Return all available agent tools."""
    return [
        search_documents,
        summarize_document_section,
        extract_evaluation_results,
        list_loaded_documents,
        search_arxiv_papers,
        download_arxiv_paper,
        search_and_ingest_arxiv,
    ]
