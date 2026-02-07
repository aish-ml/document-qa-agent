"""
Background Ingest Worker
-------------------------
Lightweight job queue & worker thread for non-blocking PDF
ingestion and Arxiv download+ingest operations.

Usage:
    worker = IngestWorker(vector_store)
    worker.start()
    worker.enqueue_pdfs(["/path/to/file.pdf", ...])
    worker.enqueue_arxiv_papers([ArxivPaper(...), ...])
    # check status:
    worker.status   -> "idle" | "working: ..." | "done" | "error: ..."
    worker.results  -> list of per-file result strings
"""

import logging
import threading
from pathlib import Path
from queue import Queue, Empty
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class IngestJob:
    """A single ingestion job."""
    job_type: str  # "pdf" or "arxiv"
    payload: object = None  # str path for pdf, ArxivPaper for arxiv
    label: str = ""


@dataclass
class IngestResult:
    """Result of a completed ingest batch."""
    total_chunks: int = 0
    details: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class IngestWorker:
    """
    Background worker that processes ingestion jobs from a queue.
    Thread-safe: can be queried for status from the main thread.
    """

    def __init__(self, vector_store):
        from knowledge_base.vector_store import VectorStore
        self.vector_store: VectorStore = vector_store
        self._queue: Queue = Queue()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._status: str = "idle"
        self._results: IngestResult = IngestResult()
        self._running = False

    # ── Public API ──────────────────────────────────────────────

    def start(self):
        """Start the worker thread (idempotent)."""
        with self._lock:
            if self._running:
                return
            self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True, name="ingest-worker")
        self._thread.start()
        logger.info("Ingest worker started")

    def stop(self):
        """Signal the worker to stop after finishing current jobs."""
        with self._lock:
            self._running = False
        self._queue.put(None)  # sentinel to wake up
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)

    def enqueue_pdfs(self, pdf_paths: list[str]) -> None:
        """Queue PDF files for ingestion."""
        self._reset_results()
        for p in pdf_paths:
            job = IngestJob(job_type="pdf", payload=p, label=Path(p).name)
            self._queue.put(job)
        logger.info(f"Enqueued {len(pdf_paths)} PDF(s) for ingestion")

    def enqueue_arxiv_papers(self, papers: list) -> None:
        """Queue ArxivPaper objects for download + ingestion."""
        self._reset_results()
        for paper in papers:
            job = IngestJob(job_type="arxiv", payload=paper, label=paper.title[:60])
            self._queue.put(job)
        logger.info(f"Enqueued {len(papers)} Arxiv paper(s) for download+ingest")

    @property
    def status(self) -> str:
        with self._lock:
            return self._status

    @property
    def results(self) -> IngestResult:
        with self._lock:
            return IngestResult(
                total_chunks=self._results.total_chunks,
                details=list(self._results.details),
                errors=list(self._results.errors),
            )

    @property
    def is_busy(self) -> bool:
        return not self._queue.empty() or self._status.startswith("working")

    def _reset_results(self):
        with self._lock:
            self._results = IngestResult()

    # ── Internal Worker Loop ────────────────────────────────────

    def _run(self):
        """Main worker loop — processes jobs sequentially."""
        from ingestion.pdf_extractor import PDFExtractor
        from ingestion.chunker import DocumentChunker
        from arxiv_integration.arxiv_client import ArxivClient
        from config import DOCUMENTS_DIR

        extractor = PDFExtractor(save_images=True)
        chunker = DocumentChunker()
        arxiv_client = ArxivClient()

        while True:
            with self._lock:
                if not self._running:
                    break

            try:
                job: IngestJob | None = self._queue.get(timeout=1.0)
            except Empty:
                with self._lock:
                    if self._status.startswith("working"):
                        self._status = "idle"
                continue

            if job is None:  # sentinel
                break

            with self._lock:
                self._status = f"working: {job.label}"

            try:
                if job.job_type == "pdf":
                    self._process_pdf(job.payload, extractor, chunker)
                elif job.job_type == "arxiv":
                    self._process_arxiv(job.payload, arxiv_client, extractor, chunker, str(DOCUMENTS_DIR))
            except Exception as e:
                logger.error(f"Ingest job failed ({job.label}): {e}")
                with self._lock:
                    self._results.errors.append(f"❌ {job.label} — {e}")

            self._queue.task_done()

            # If queue is now empty, mark idle
            if self._queue.empty():
                with self._lock:
                    self._status = "idle"

        with self._lock:
            self._status = "stopped"

    def _process_pdf(self, pdf_path: str, extractor, chunker):
        """Extract, chunk, and add a single PDF."""
        name = Path(pdf_path).name
        try:
            doc = extractor.extract(pdf_path)
            chunks = chunker.chunk_document(doc)
            count = self.vector_store.add_chunks(chunks)
            with self._lock:
                self._results.total_chunks += count
                self._results.details.append(
                    f"✅ {name} — {doc.total_pages} pages, "
                    f"{len(doc.tables)} tables, {count} chunks"
                )
            logger.info(f"Ingested {name}: {count} chunks")
        except Exception as e:
            with self._lock:
                self._results.errors.append(f"❌ {name} — {e}")
            logger.error(f"Failed to ingest {name}: {e}")

    def _process_arxiv(self, paper, arxiv_client, extractor, chunker, dest_dir: str):
        """Download an Arxiv paper then extract+chunk+add."""
        title_short = paper.title[:60]
        pdf_path = arxiv_client.download_pdf(paper, dest_dir)
        if not pdf_path:
            with self._lock:
                self._results.errors.append(f"❌ {title_short} — download failed")
            return
        self._process_pdf(pdf_path, extractor, chunker)
