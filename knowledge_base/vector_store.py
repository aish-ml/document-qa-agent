"""
Vector Store Module
-------------------
Manages Chroma vector store for document chunk embeddings.
Supports multiple embedding providers (OpenAI, Gemini, Ollama).
"""

import logging
import time
import threading
from pathlib import Path

from langchain_chroma import Chroma
from langchain_core.documents import Document

from config import (
    CHROMA_PERSIST_DIR,
    EMBEDDING_PROVIDER,
    EMBEDDING_MODEL,
    OPENAI_API_KEY,
    GEMINI_API_KEY,
    OLLAMA_BASE_URL,
    MAX_CONTEXT_CHUNKS,
)
from ingestion.chunker import DocumentChunk

logger = logging.getLogger(__name__)

COLLECTION_NAME = "document_qa"


def get_embedding_function():
    """Create embedding function based on configured provider."""
    if EMBEDDING_PROVIDER == "openai":
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            openai_api_key=OPENAI_API_KEY,
            model="text-embedding-3-small",
        )
    elif EMBEDDING_PROVIDER == "gemini":
        from langchain_google_genai import GoogleGenerativeAIEmbeddings

        return GoogleGenerativeAIEmbeddings(
            google_api_key=GEMINI_API_KEY,
            model=EMBEDDING_MODEL,
        )
    elif EMBEDDING_PROVIDER == "ollama":
        from langchain_community.embeddings import OllamaEmbeddings

        return OllamaEmbeddings(
            base_url=OLLAMA_BASE_URL,
            model="nomic-embed-text",
        )
    else:
        raise ValueError(f"Unknown embedding provider: {EMBEDDING_PROVIDER}")


class VectorStore:
    """
    Chroma-backed vector store for document retrieval.
    Enterprise features: configurable top-k, metadata filtering,
    persistence to disk.
    """

    def __init__(self, persist_dir: str | Path | None = None):
        self.persist_dir = str(persist_dir or CHROMA_PERSIST_DIR)
        self.embeddings = get_embedding_function()
        self.store: Chroma | None = None
        self._lock = threading.Lock()
        self._initialize()

    def _initialize(self):
        """Initialize or load existing Chroma collection."""
        self.store = Chroma(
            collection_name=COLLECTION_NAME,
            embedding_function=self.embeddings,
            persist_directory=self.persist_dir,
        )
        count = self.store._collection.count()
        logger.info(
            f"Vector store initialized: {count} existing documents "
            f"in '{self.persist_dir}'"
        )

    def add_chunks(self, chunks: list[DocumentChunk]) -> int:
        """Add document chunks to the vector store with rate-limit retry (thread-safe)."""
        if not chunks:
            return 0

        documents = []
        ids = []
        seen_ids = set()
        for chunk in chunks:
            cid = chunk.chunk_id
            # Deduplicate IDs within the batch
            if cid in seen_ids:
                cid = f"{cid}_{len(seen_ids)}"
            seen_ids.add(cid)
            doc = Document(
                page_content=chunk.text,
                metadata=chunk.metadata,
            )
            documents.append(doc)
            ids.append(cid)

        # Batch to avoid hitting rate limits (embed max BATCH_SIZE at a time)
        BATCH_SIZE = 20
        total_added = 0
        for start in range(0, len(documents), BATCH_SIZE):
            batch_docs = documents[start:start + BATCH_SIZE]
            batch_ids = ids[start:start + BATCH_SIZE]
            max_retries = 5
            for attempt in range(max_retries):
                try:
                    with self._lock:
                        self.store.add_documents(documents=batch_docs, ids=batch_ids)
                    total_added += len(batch_docs)
                    break
                except Exception as e:
                    err_str = str(e)
                    if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str:
                        wait = min(60, 10 * (attempt + 1))
                        logger.warning(
                            f"Rate limited on embed batch {start//BATCH_SIZE+1}, "
                            f"retrying in {wait}s (attempt {attempt+1}/{max_retries})"
                        )
                        time.sleep(wait)
                    else:
                        logger.error(f"Failed to add batch: {e}")
                        raise
            else:
                logger.error(f"Exhausted retries for batch starting at index {start}")

        logger.info(f"Added {total_added} chunks to vector store")
        return total_added

    def search(
        self,
        query: str,
        k: int | None = None,
        filter_metadata: dict | None = None,
    ) -> list[Document]:
        """
        Retrieve relevant document chunks for a query.

        Args:
            query: Natural language query
            k: Number of results (defaults to MAX_CONTEXT_CHUNKS)
            filter_metadata: Optional Chroma metadata filter

        Returns:
            List of matching LangChain Documents
        """
        k = k or MAX_CONTEXT_CHUNKS
        kwargs = {"k": k}
        if filter_metadata:
            kwargs["filter"] = filter_metadata

        with self._lock:
            results = self.store.similarity_search(query, **kwargs)
        logger.debug(f"Search returned {len(results)} results for: {query[:60]}...")
        return results

    def search_with_scores(
        self,
        query: str,
        k: int | None = None,
    ) -> list[tuple[Document, float]]:
        """Search and return results with similarity scores."""
        k = k or MAX_CONTEXT_CHUNKS
        return self.store.similarity_search_with_score(query, k=k)

    def get_retriever(self, k: int | None = None):
        """Return a LangChain retriever interface."""
        k = k or MAX_CONTEXT_CHUNKS
        return self.store.as_retriever(
            search_type="similarity",
            search_kwargs={"k": k},
        )

    def list_sources(self) -> list[str]:
        """List all unique source documents in the store."""
        try:
            all_docs = self.store.get()
            sources = set()
            if all_docs and "metadatas" in all_docs:
                for meta in all_docs["metadatas"]:
                    if meta and "source" in meta:
                        sources.add(meta["source"])
            return sorted(sources)
        except Exception:
            return []

    def clear(self):
        """Delete all documents from the vector store."""
        self.store.delete_collection()
        self._initialize()
        logger.info("Vector store cleared")

    @property
    def count(self) -> int:
        return self.store._collection.count()
