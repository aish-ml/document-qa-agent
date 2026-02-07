"""
Arxiv API Integration
---------------------
Search and fetch academic papers from Arxiv based on user queries.
Supports ingestion into the document knowledge base.
"""

import logging
from dataclasses import dataclass

import arxiv

logger = logging.getLogger(__name__)


@dataclass
class ArxivPaper:
    """Represents an Arxiv paper search result."""
    arxiv_id: str
    title: str
    authors: list[str]
    abstract: str
    published: str
    updated: str
    pdf_url: str
    categories: list[str]
    primary_category: str

    def summary_text(self) -> str:
        """Format paper info for LLM context."""
        authors = ", ".join(self.authors[:5])
        if len(self.authors) > 5:
            authors += f" et al. ({len(self.authors)} authors)"

        return (
            f"Title: {self.title}\n"
            f"Authors: {authors}\n"
            f"Published: {self.published}\n"
            f"Categories: {', '.join(self.categories)}\n"
            f"Arxiv ID: {self.arxiv_id}\n"
            f"PDF: {self.pdf_url}\n\n"
            f"Abstract:\n{self.abstract}"
        )


class ArxivClient:
    """
    Client for searching and retrieving papers from Arxiv.
    Used as a function-calling tool by the AI agent.
    """

    def __init__(self, max_results: int = 5):
        self.max_results = max_results

    def search(
        self,
        query: str,
        max_results: int | None = None,
        sort_by: str = "relevance",
    ) -> list[ArxivPaper]:
        """
        Search Arxiv for papers matching the query.

        Args:
            query: Natural language search query
            max_results: Override default max results
            sort_by: "relevance" or "submitted_date"

        Returns:
            List of ArxivPaper results
        """
        max_results = max_results or self.max_results

        sort_criterion = (
            arxiv.SortCriterion.Relevance
            if sort_by == "relevance"
            else arxiv.SortCriterion.SubmittedDate
        )

        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=sort_criterion,
        )

        papers = []
        try:
            for result in client.results(search):
                paper = ArxivPaper(
                    arxiv_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary,
                    published=result.published.strftime("%Y-%m-%d"),
                    updated=result.updated.strftime("%Y-%m-%d"),
                    pdf_url=result.pdf_url,
                    categories=result.categories,
                    primary_category=result.primary_category,
                )
                papers.append(paper)

            logger.info(f"Arxiv search '{query[:50]}...' returned {len(papers)} results")
        except Exception as e:
            logger.error(f"Arxiv search failed: {e}")

        return papers

    def get_paper_by_id(self, arxiv_id: str) -> ArxivPaper | None:
        """Fetch a specific paper by its Arxiv ID."""
        client = arxiv.Client()
        search = arxiv.Search(id_list=[arxiv_id])

        try:
            for result in client.results(search):
                return ArxivPaper(
                    arxiv_id=result.entry_id.split("/")[-1],
                    title=result.title,
                    authors=[a.name for a in result.authors],
                    abstract=result.summary,
                    published=result.published.strftime("%Y-%m-%d"),
                    updated=result.updated.strftime("%Y-%m-%d"),
                    pdf_url=result.pdf_url,
                    categories=result.categories,
                    primary_category=result.primary_category,
                )
        except Exception as e:
            logger.error(f"Failed to fetch paper {arxiv_id}: {e}")
        return None

    def download_pdf(self, paper: ArxivPaper, dest_dir: str) -> str | None:
        """
        Download a paper's PDF to the specified directory.

        Returns:
            Path to downloaded PDF or None on failure.
        """
        from pathlib import Path
        import urllib.request

        dest = Path(dest_dir)
        dest.mkdir(parents=True, exist_ok=True)

        safe_title = "".join(
            c if c.isalnum() or c in " -_" else "_"
            for c in paper.title[:60]
        ).strip()
        filename = f"{paper.arxiv_id}_{safe_title}.pdf"
        filepath = dest / filename

        try:
            urllib.request.urlretrieve(paper.pdf_url, str(filepath))
            logger.info(f"Downloaded: {filepath}")
            return str(filepath)
        except Exception as e:
            logger.error(f"Download failed for {paper.arxiv_id}: {e}")
            return None


def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Standalone function for agent tool integration.
    Returns formatted search results as a string.
    """
    client = ArxivClient(max_results=max_results)
    papers = client.search(query)

    if not papers:
        return f"No papers found on Arxiv for: '{query}'"

    results = [f"Found {len(papers)} papers on Arxiv:\n"]
    for i, paper in enumerate(papers, 1):
        results.append(
            f"--- Paper {i} ---\n{paper.summary_text()}\n"
        )
    return "\n".join(results)
