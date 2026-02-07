"""
Text Chunking Module
--------------------
Splits extracted document text into overlapping chunks for
embedding and retrieval, preserving metadata and context.
"""

import logging
from dataclasses import dataclass, field

from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_SIZE, CHUNK_OVERLAP
from ingestion.pdf_extractor import ExtractedDocument

logger = logging.getLogger(__name__)


@dataclass
class DocumentChunk:
    """A single chunk of document text with rich metadata."""
    chunk_id: str
    text: str
    metadata: dict = field(default_factory=dict)


class DocumentChunker:
    """
    Splits extracted documents into overlapping chunks suitable for
    embedding & vector store insertion. Preserves source metadata
    (file, page, section) for accurate retrieval attribution.
    """

    def __init__(
        self,
        chunk_size: int = CHUNK_SIZE,
        chunk_overlap: int = CHUNK_OVERLAP,
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""],
        )

    def chunk_document(self, doc: ExtractedDocument) -> list[DocumentChunk]:
        """
        Convert an ExtractedDocument into a list of DocumentChunks.
        Processes: body text by section, tables, and image captions.
        """
        chunks: list[DocumentChunk] = []

        # ── 1. Chunk sections ────────────────────────────────────
        for section in doc.sections:
            if not section.content.strip():
                continue

            section_chunks = self.splitter.split_text(section.content)
            for i, text in enumerate(section_chunks):
                chunk = DocumentChunk(
                    chunk_id=f"{doc.doc_id}_sec_{section.section_type}_{i}",
                    text=text,
                    metadata={
                        "source": doc.file_name,
                        "file_path": doc.file_path,
                        "doc_id": doc.doc_id,
                        "section_title": section.title,
                        "section_type": section.section_type,
                        "page_start": section.page_start,
                        "page_end": section.page_end,
                        "chunk_index": i,
                        "content_type": "text",
                    },
                )
                chunks.append(chunk)

        # ── 2. Chunk tables (as Markdown) ────────────────────────
        for table in doc.tables:
            table_text = (
                f"[Table from page {table.page_number}]\n"
                f"{table.to_markdown()}"
            )
            chunk = DocumentChunk(
                chunk_id=(
                    f"{doc.doc_id}_table_p{table.page_number}"
                    f"_t{table.table_index}"
                ),
                text=table_text,
                metadata={
                    "source": doc.file_name,
                    "file_path": doc.file_path,
                    "doc_id": doc.doc_id,
                    "page_start": table.page_number,
                    "page_end": table.page_number,
                    "content_type": "table",
                    "table_headers": ", ".join(table.headers),
                },
            )
            chunks.append(chunk)

        # ── 3. Image metadata chunks ─────────────────────────────
        for img in doc.images:
            img_text = (
                f"[Image/Figure from page {img.page_number}, "
                f"size {img.width}x{img.height}]"
            )
            if img.caption:
                img_text += f"\nCaption: {img.caption}"

            chunk = DocumentChunk(
                chunk_id=(
                    f"{doc.doc_id}_img_p{img.page_number}"
                    f"_i{img.image_index}"
                ),
                text=img_text,
                metadata={
                    "source": doc.file_name,
                    "file_path": doc.file_path,
                    "doc_id": doc.doc_id,
                    "page_start": img.page_number,
                    "page_end": img.page_number,
                    "content_type": "image",
                    "image_path": img.image_path,
                },
            )
            chunks.append(chunk)

        # ── 4. Fallback: if no sections detected, chunk full text
        if not doc.sections:
            text_chunks = self.splitter.split_text(doc.full_text)
            for i, text in enumerate(text_chunks):
                chunk = DocumentChunk(
                    chunk_id=f"{doc.doc_id}_full_{i}",
                    text=text,
                    metadata={
                        "source": doc.file_name,
                        "file_path": doc.file_path,
                        "doc_id": doc.doc_id,
                        "chunk_index": i,
                        "content_type": "text",
                    },
                )
                chunks.append(chunk)

        logger.info(
            f"Chunked {doc.file_name}: {len(chunks)} chunks "
            f"({len(doc.sections)} sections, {len(doc.tables)} tables, "
            f"{len(doc.images)} images)"
        )
        return chunks

    def chunk_documents(
        self, docs: list[ExtractedDocument]
    ) -> list[DocumentChunk]:
        """Chunk multiple documents."""
        all_chunks = []
        for doc in docs:
            all_chunks.extend(self.chunk_document(doc))
        logger.info(f"Total chunks across {len(docs)} documents: {len(all_chunks)}")
        return all_chunks
