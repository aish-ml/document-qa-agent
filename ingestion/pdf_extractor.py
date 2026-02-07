"""
PDF Extraction Pipeline
-----------------------
Extracts text, tables, images, graphs, charts, and structured sections
from PDF documents using pdfplumber + PyMuPDF.
"""

import io
import base64
import hashlib
import logging
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

import pdfplumber
import fitz  # PyMuPDF
from PIL import Image

from config import IMAGES_DIR

logger = logging.getLogger(__name__)


# ── Data Models ──────────────────────────────────────────────────

@dataclass
class ExtractedTable:
    """Represents a table extracted from a PDF page."""
    page_number: int
    table_index: int
    headers: list[str]
    rows: list[list[str]]
    raw_text: str  # Markdown-formatted for LLM consumption

    def to_markdown(self) -> str:
        return self.raw_text


@dataclass
class ExtractedImage:
    """Represents an image/figure extracted from a PDF page."""
    page_number: int
    image_index: int
    image_path: str          # Path to saved image file
    base64_data: str         # Base64-encoded image for LLM APIs
    width: int
    height: int
    caption: str = ""


@dataclass
class ExtractedSection:
    """Represents a logical section of a document."""
    title: str
    content: str
    page_start: int
    page_end: int
    section_type: str = "body"  # title, abstract, section, references, etc.


@dataclass
class ExtractedDocument:
    """Complete extraction result for a single PDF document."""
    file_name: str
    file_path: str
    total_pages: int
    full_text: str
    sections: list[ExtractedSection] = field(default_factory=list)
    tables: list[ExtractedTable] = field(default_factory=list)
    images: list[ExtractedImage] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)

    @property
    def doc_id(self) -> str:
        return hashlib.md5(self.file_path.encode()).hexdigest()[:12]


# ── PDF Extractor ────────────────────────────────────────────────

class PDFExtractor:
    """
    Multi-modal PDF extraction engine.
    Combines pdfplumber (text/tables) and PyMuPDF (images/figures).
    """

    # Heuristics for section detection in academic papers
    SECTION_KEYWORDS = [
        "abstract", "introduction", "background", "related work",
        "methodology", "method", "methods", "approach", "proposed",
        "experiment", "experiments", "results", "evaluation",
        "discussion", "conclusion", "conclusions", "future work",
        "references", "bibliography", "appendix", "acknowledgment",
        "acknowledgements",
    ]

    def __init__(self, save_images: bool = True):
        self.save_images = save_images

    def extract(self, pdf_path: str | Path) -> ExtractedDocument:
        """Extract all content from a PDF file."""
        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {pdf_path}")

        logger.info(f"Extracting content from: {pdf_path.name}")

        full_text = ""
        tables: list[ExtractedTable] = []
        images: list[ExtractedImage] = []
        page_texts: list[str] = []
        metadata: dict = {}

        # ── Pass 1: Text + Tables via pdfplumber ─────────────────
        with pdfplumber.open(str(pdf_path)) as pdf:
            metadata = pdf.metadata or {}
            for page_num, page in enumerate(pdf.pages, start=1):
                # Extract text
                text = page.extract_text() or ""
                page_texts.append(text)

                # Extract tables
                page_tables = page.extract_tables() or []
                for t_idx, table_data in enumerate(page_tables):
                    if table_data and len(table_data) > 0:
                        extracted = self._process_table(
                            table_data, page_num, t_idx
                        )
                        if extracted:
                            tables.append(extracted)

        full_text = "\n\n".join(page_texts)

        # ── Pass 2: Images via PyMuPDF ───────────────────────────
        images = self._extract_images(pdf_path)

        # ── Pass 3: Section detection ────────────────────────────
        sections = self._detect_sections(page_texts)

        doc = ExtractedDocument(
            file_name=pdf_path.name,
            file_path=str(pdf_path.resolve()),
            total_pages=len(page_texts),
            full_text=full_text,
            sections=sections,
            tables=tables,
            images=images,
            metadata=metadata,
        )

        logger.info(
            f"Extracted from {pdf_path.name}: "
            f"{len(sections)} sections, {len(tables)} tables, "
            f"{len(images)} images, {doc.total_pages} pages"
        )
        return doc

    # ── Table Processing ─────────────────────────────────────────

    def _process_table(
        self,
        table_data: list[list],
        page_num: int,
        table_idx: int,
    ) -> Optional[ExtractedTable]:
        """Convert raw table data to structured ExtractedTable."""
        if not table_data or len(table_data) < 2:
            return None

        # Clean cells
        cleaned = []
        for row in table_data:
            cleaned_row = [
                str(cell).strip() if cell else "" for cell in row
            ]
            cleaned.append(cleaned_row)

        headers = cleaned[0]
        rows = cleaned[1:]

        # Build Markdown representation
        md = self._table_to_markdown(headers, rows)

        return ExtractedTable(
            page_number=page_num,
            table_index=table_idx,
            headers=headers,
            rows=rows,
            raw_text=md,
        )

    @staticmethod
    def _table_to_markdown(headers: list[str], rows: list[list[str]]) -> str:
        """Convert table to Markdown format for LLM consumption."""
        if not headers:
            return ""

        col_widths = [max(len(h), 3) for h in headers]
        for row in rows:
            for i, cell in enumerate(row):
                if i < len(col_widths):
                    col_widths[i] = max(col_widths[i], len(cell))

        def fmt_row(cells):
            padded = []
            for i, cell in enumerate(cells):
                w = col_widths[i] if i < len(col_widths) else len(cell)
                padded.append(cell.ljust(w))
            return "| " + " | ".join(padded) + " |"

        lines = [fmt_row(headers)]
        lines.append("| " + " | ".join("-" * w for w in col_widths) + " |")
        for row in rows:
            # Pad row to match header length
            padded_row = row + [""] * (len(headers) - len(row))
            lines.append(fmt_row(padded_row[:len(headers)]))

        return "\n".join(lines)

    # ── Image Extraction ─────────────────────────────────────────

    def _extract_images(self, pdf_path: Path) -> list[ExtractedImage]:
        """Extract images from PDF using PyMuPDF."""
        images = []
        try:
            doc = fitz.open(str(pdf_path))
            for page_num in range(len(doc)):
                page = doc[page_num]
                image_list = page.get_images(full=True)

                for img_idx, img_info in enumerate(image_list):
                    try:
                        xref = img_info[0]
                        base_image = doc.extract_image(xref)
                        if not base_image:
                            continue

                        image_bytes = base_image["image"]
                        image_ext = base_image.get("ext", "png")
                        width = base_image.get("width", 0)
                        height = base_image.get("height", 0)

                        # Skip tiny images (likely icons/bullets)
                        if width < 50 or height < 50:
                            continue

                        # Save image to disk
                        img_filename = (
                            f"{pdf_path.stem}_p{page_num + 1}_img{img_idx + 1}"
                            f".{image_ext}"
                        )
                        img_path = IMAGES_DIR / img_filename

                        if self.save_images:
                            with open(img_path, "wb") as f:
                                f.write(image_bytes)

                        # Base64 encode for LLM APIs
                        b64_data = base64.b64encode(image_bytes).decode("utf-8")

                        images.append(
                            ExtractedImage(
                                page_number=page_num + 1,
                                image_index=img_idx,
                                image_path=str(img_path),
                                base64_data=b64_data,
                                width=width,
                                height=height,
                            )
                        )
                    except Exception as e:
                        logger.warning(
                            f"Failed to extract image {img_idx} from "
                            f"page {page_num + 1}: {e}"
                        )
            doc.close()
        except Exception as e:
            logger.error(f"Image extraction failed for {pdf_path}: {e}")

        return images

    # ── Section Detection ────────────────────────────────────────

    def _detect_sections(
        self, page_texts: list[str]
    ) -> list[ExtractedSection]:
        """Detect logical sections based on heading heuristics."""
        sections: list[ExtractedSection] = []
        current_section = ExtractedSection(
            title="Document Start",
            content="",
            page_start=1,
            page_end=1,
            section_type="preamble",
        )

        for page_num, text in enumerate(page_texts, start=1):
            lines = text.split("\n")
            for line in lines:
                stripped = line.strip()
                if not stripped:
                    current_section.content += "\n"
                    continue

                section_type = self._classify_heading(stripped)
                if section_type:
                    # Save previous section
                    if current_section.content.strip():
                        current_section.page_end = page_num
                        sections.append(current_section)

                    # Start new section
                    current_section = ExtractedSection(
                        title=stripped,
                        content="",
                        page_start=page_num,
                        page_end=page_num,
                        section_type=section_type,
                    )
                else:
                    current_section.content += stripped + "\n"

            current_section.page_end = page_num

        # Save final section
        if current_section.content.strip():
            sections.append(current_section)

        return sections

    def _classify_heading(self, line: str) -> Optional[str]:
        """
        Classify a line as a section heading.
        Returns section type or None if not a heading.
        """
        clean = line.lower().strip()

        # Remove numbering patterns: "1.", "1.1", "I.", "A.", etc.
        import re
        clean_no_num = re.sub(
            r"^[\d]+[\.\)]\s*|^[IVXLC]+[\.\)]\s*|^[a-zA-Z][\.\)]\s*",
            "",
            clean,
        )

        for keyword in self.SECTION_KEYWORDS:
            if clean_no_num.startswith(keyword) or clean == keyword:
                if keyword in ("abstract",):
                    return "abstract"
                elif keyword in ("references", "bibliography"):
                    return "references"
                elif keyword in ("conclusion", "conclusions", "future work"):
                    return "conclusion"
                elif keyword in ("introduction", "background"):
                    return "introduction"
                elif keyword in ("appendix",):
                    return "appendix"
                else:
                    return "section"

        # Heuristic: short uppercase lines are likely headings
        if (
            len(line) < 80
            and line == line.upper()
            and len(line.split()) <= 8
            and len(line) > 3
        ):
            return "section"

        return None

    # ── Batch Processing ─────────────────────────────────────────

    def extract_directory(
        self, directory: str | Path
    ) -> list[ExtractedDocument]:
        """Extract all PDFs from a directory."""
        directory = Path(directory)
        pdf_files = sorted(directory.glob("*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return []

        logger.info(f"Found {len(pdf_files)} PDF files to process")
        documents = []
        for pdf_path in pdf_files:
            try:
                doc = self.extract(pdf_path)
                documents.append(doc)
            except Exception as e:
                logger.error(f"Failed to process {pdf_path.name}: {e}")

        return documents
