"""
Document Q&A AI Agent — Main Entry Point
==========================================
Interactive CLI for ingesting PDFs and querying with AI.

Usage:
    python main.py                    # Interactive mode
    python main.py ingest             # Ingest PDFs from ./documents/
    python main.py ask "question"     # Ask a single question
"""

import sys
import argparse
import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
from rich.prompt import Prompt

from config import DOCUMENTS_DIR, LLM_PROVIDER, AUTO_INGEST_ON_START, ARXIV_AUTO_INGEST_TOPN
from utils.helpers import setup_logging
from utils.ingest_worker import IngestWorker
from ingestion.pdf_extractor import PDFExtractor
from ingestion.chunker import DocumentChunker
from knowledge_base.vector_store import VectorStore
from agent.qa_agent import QAAgent

console = Console()
logger = logging.getLogger(__name__)


# ── Document Ingestion ───────────────────────────────────────────

def ingest_documents(
    vector_store: VectorStore,
    docs_dir: str | Path | None = None,
    pdf_paths: list[str] | None = None,
) -> int:
    """
    Ingest PDF documents into the vector store.

    Args:
        vector_store: Target vector store
        docs_dir: Directory containing PDFs (default: ./documents/)
        pdf_paths: Specific PDF file paths to ingest

    Returns:
        Number of chunks ingested
    """
    extractor = PDFExtractor(save_images=True)
    chunker = DocumentChunker()

    documents = []

    if pdf_paths:
        for path in pdf_paths:
            try:
                doc = extractor.extract(path)
                documents.append(doc)
            except Exception as e:
                console.print(f"[red]Failed to extract {path}: {e}[/red]")
    else:
        docs_dir = Path(docs_dir or DOCUMENTS_DIR)
        if not any(docs_dir.glob("*.pdf")):
            console.print(
                f"[yellow]No PDF files found in {docs_dir}[/yellow]\n"
                f"Place PDF files in the '{docs_dir}' directory and run again."
            )
            return 0
        documents = extractor.extract_directory(docs_dir)

    if not documents:
        console.print("[yellow]No documents to ingest.[/yellow]")
        return 0

    # Display extraction summary
    table = Table(title="Extraction Summary")
    table.add_column("Document", style="cyan")
    table.add_column("Pages", justify="right")
    table.add_column("Sections", justify="right")
    table.add_column("Tables", justify="right")
    table.add_column("Images", justify="right")

    for doc in documents:
        table.add_row(
            doc.file_name,
            str(doc.total_pages),
            str(len(doc.sections)),
            str(len(doc.tables)),
            str(len(doc.images)),
        )
    console.print(table)

    # Chunk and index
    all_chunks = chunker.chunk_documents(documents)
    count = vector_store.add_chunks(all_chunks)

    console.print(
        f"\n[green]✓ Ingested {len(documents)} document(s) → "
        f"{count} chunks indexed[/green]"
    )
    return count


# ── Interactive CLI ──────────────────────────────────────────────

def interactive_mode(agent: QAAgent, vector_store: VectorStore):
    """Run the interactive Q&A loop."""
    console.print(
        Panel(
            "[bold cyan]Document Q&A AI Agent[/bold cyan]\n"
            "Ask questions about your ingested PDFs, or use commands below.\n\n"
            "Commands:\n"
            "  [green]/ingest[/green]              — Ingest PDFs from documents/ folder\n"
            "  [green]/ingest <path>[/green]       — Ingest a specific PDF file\n"
            "  [green]/sources[/green]             — List loaded documents\n"
            "  [green]/stats[/green]               — Show agent statistics\n"
            "  [green]/arxiv <query>[/green]       — Search Arxiv for papers\n"
            "  [green]/arxiv-ingest <query>[/green]— Search Arxiv & auto-ingest top N\n"
            "  [green]/download <arxiv_id>[/green] — Download & ingest an Arxiv paper\n"
            "  [green]/clear[/green]               — Clear conversation history\n"
            "  [green]/verbose[/green]             — Toggle verbose mode (show tool calls)\n"
            "  [green]/help[/green]                — Show this help message\n"
            "  [green]/quit[/green]                — Exit\n",
            title="Welcome",
            border_style="blue",
        )
    )

    verbose = False

    while True:
        try:
            user_input = Prompt.ask("\n[bold blue]You[/bold blue]").strip()
        except (KeyboardInterrupt, EOFError):
            console.print("\n[dim]Goodbye![/dim]")
            break

        if not user_input:
            continue

        # ── Handle commands ─────────────────────────────────────
        if user_input.startswith("/"):
            cmd_parts = user_input.split(maxsplit=1)
            cmd = cmd_parts[0].lower()
            cmd_arg = cmd_parts[1] if len(cmd_parts) > 1 else ""

            if cmd in ("/quit", "/exit", "/q"):
                console.print("[dim]Goodbye![/dim]")
                break

            elif cmd == "/help":
                interactive_mode.__doc__  # Re-show panel
                console.print("[dim]Type a question or command.[/dim]")

            elif cmd == "/ingest":
                if cmd_arg:
                    ingest_documents(vector_store, pdf_paths=[cmd_arg])
                else:
                    ingest_documents(vector_store)

            elif cmd == "/sources":
                sources = vector_store.list_sources()
                if sources:
                    for i, s in enumerate(sources, 1):
                        console.print(f"  {i}. [cyan]{s}[/cyan]")
                    console.print(
                        f"\n[dim]{vector_store.count} total chunks[/dim]"
                    )
                else:
                    console.print("[yellow]No documents loaded.[/yellow]")

            elif cmd == "/stats":
                stats = agent.get_stats()
                stats_table = Table(title="Agent Statistics")
                stats_table.add_column("Metric", style="cyan")
                stats_table.add_column("Value", style="green")
                for k, v in stats.items():
                    stats_table.add_row(k, str(v))
                console.print(stats_table)

            elif cmd == "/arxiv":
                if not cmd_arg:
                    console.print("[yellow]Usage: /arxiv <search query>[/yellow]")
                else:
                    from arxiv_integration.arxiv_client import search_arxiv
                    console.print("[dim]Searching Arxiv...[/dim]")
                    result = search_arxiv(cmd_arg)
                    console.print(Markdown(result))

            elif cmd == "/arxiv-ingest":
                if not cmd_arg:
                    console.print("[yellow]Usage: /arxiv-ingest <search query>[/yellow]")
                else:
                    from arxiv_integration.arxiv_client import ArxivClient as _AC
                    console.print(
                        f"[dim]Searching Arxiv & ingesting top "
                        f"{ARXIV_AUTO_INGEST_TOPN} papers...[/dim]"
                    )
                    _client = _AC(max_results=ARXIV_AUTO_INGEST_TOPN)
                    papers = _client.search(cmd_arg)
                    if not papers:
                        console.print("[yellow]No papers found.[/yellow]")
                    else:
                        worker = IngestWorker(vector_store)
                        worker.start()
                        worker.enqueue_arxiv_papers(papers)
                        console.print(
                            f"[green]✓ Queued {len(papers)} paper(s) for "
                            f"background download & ingest[/green]"
                        )
                        for p in papers:
                            console.print(f"  • {p.title[:70]}")
                        console.print(
                            "[dim]Ingestion runs in background. "
                            "Use /sources to check progress.[/dim]"
                        )

            elif cmd == "/download":
                if not cmd_arg:
                    console.print("[yellow]Usage: /download <arxiv_id>[/yellow]")
                else:
                    from agent.tools import download_arxiv_paper as _dl
                    console.print("[dim]Downloading and ingesting paper...[/dim]")
                    res = _dl.invoke(cmd_arg.strip())
                    console.print(Markdown(res))

            elif cmd == "/clear":
                agent.clear_history()
                console.print("[dim]Conversation history cleared.[/dim]")

            elif cmd == "/verbose":
                verbose = not verbose
                state = "ON" if verbose else "OFF"
                console.print(f"[dim]Verbose mode: {state}[/dim]")

            else:
                console.print(f"[yellow]Unknown command: {cmd}[/yellow]")

            continue

        # ── Ask the agent ───────────────────────────────────────
        with console.status("[bold green]Thinking...[/bold green]"):
            response = agent.ask(user_input, verbose=verbose)

        # Display answer
        console.print(
            Panel(
                Markdown(response["answer"]),
                title="[bold green]Agent[/bold green]",
                border_style="green",
            )
        )

        # Display sources
        if response["sources"]:
            console.print("[dim]Sources:[/dim]")
            for src in response["sources"]:
                console.print(f"  [dim]📄 {src}[/dim]")

        # Display tool steps (verbose)
        if verbose and "steps" in response:
            console.print("\n[dim]Tool calls:[/dim]")
            for step in response["steps"]:
                console.print(
                    f"  [dim]🔧 {step['tool']}({step['input']}) → "
                    f"{step['output'][:200]}...[/dim]"
                )


# ── CLI Entry Point ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Document Q&A AI Agent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="interactive",
        choices=["interactive", "ingest", "ask"],
        help="Command to run (default: interactive)",
    )
    parser.add_argument(
        "query",
        nargs="?",
        help="Question for 'ask' command",
    )
    parser.add_argument(
        "--provider",
        default=None,
        help=f"LLM provider: openai, gemini, ollama (default: {LLM_PROVIDER})",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="LLM model name override",
    )
    parser.add_argument(
        "--docs-dir",
        default=None,
        help=f"Documents directory (default: {DOCUMENTS_DIR})",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--auto-ingest",
        action="store_true",
        default=None,
        help="Auto-ingest PDFs from documents/ on startup (overrides env)",
    )
    parser.add_argument(
        "--no-auto-ingest",
        action="store_true",
        default=False,
        help="Disable auto-ingest even if enabled in .env",
    )

    args = parser.parse_args()

    # Setup logging
    setup_logging("DEBUG" if args.verbose else "INFO")

    # Initialize components
    console.print("[dim]Initializing vector store...[/dim]")
    vector_store = VectorStore()

    console.print("[dim]Initializing AI agent...[/dim]")
    agent = QAAgent(
        vector_store=vector_store,
        provider=args.provider,
        model=args.model,
    )

    # ── Auto-ingest documents/ on startup ─────────────────────────
    should_auto_ingest = AUTO_INGEST_ON_START
    if args.auto_ingest:
        should_auto_ingest = True
    if args.no_auto_ingest:
        should_auto_ingest = False

    if should_auto_ingest and args.command != "ingest":
        pdfs = sorted(Path(args.docs_dir or DOCUMENTS_DIR).glob("*.pdf"))
        if pdfs:
            existing = set(vector_store.list_sources())
            new_pdfs = [str(p) for p in pdfs if p.name not in existing]
            if new_pdfs:
                console.print(
                    f"[dim]Auto-ingesting {len(new_pdfs)} new PDF(s) "
                    f"from documents/ in background...[/dim]"
                )
                worker = IngestWorker(vector_store)
                worker.start()
                worker.enqueue_pdfs(new_pdfs)
            else:
                console.print(
                    f"[dim]All {len(pdfs)} PDF(s) in documents/ "
                    f"already indexed.[/dim]"
                )

    # Execute command
    if args.command == "ingest":
        ingest_documents(vector_store, docs_dir=args.docs_dir)

    elif args.command == "ask":
        if not args.query:
            console.print("[red]Please provide a question: python main.py ask \"your question\"[/red]")
            sys.exit(1)
        response = agent.ask(args.query, verbose=args.verbose)
        console.print(
            Panel(Markdown(response["answer"]), border_style="green")
        )
        if response["sources"]:
            for src in response["sources"]:
                console.print(f"  📄 {src}")

    else:  # interactive
        interactive_mode(agent, vector_store)


if __name__ == "__main__":
    main()
