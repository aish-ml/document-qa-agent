"""
Utility Helpers
---------------
Shared utility functions for the Document Q&A Agent.
"""

import logging
import sys


def setup_logging(level: str = "INFO") -> None:
    """Configure structured logging for the application."""
    log_level = getattr(logging, level.upper(), logging.INFO)
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s │ %(levelname)-7s │ %(name)-25s │ %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stderr,
    )
    # Quiet noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("googleapiclient").setLevel(logging.WARNING)


def format_answer(response: dict) -> str:
    """Format agent response for CLI display."""
    answer = response.get("answer", "No answer.")
    sources = response.get("sources", [])

    output = f"\n{answer}\n"

    if sources:
        output += "\n📄 Sources:\n"
        for src in sources:
            output += f"   • {src}\n"

    return output
