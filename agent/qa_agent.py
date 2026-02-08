"""
Q&A Agent
---------
Main AI agent that orchestrates document Q&A using LangGraph's
ReAct agent with function-calling tools, context-aware retrieval,
and enterprise response optimization.
"""

import re
import logging

from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from agent.llm_provider import ManagedLLM, create_llm
from agent.tools import get_all_tools, set_vector_store
from knowledge_base.vector_store import VectorStore

logger = logging.getLogger(__name__)

# ── System Prompt ────────────────────────────────────────────────

SYSTEM_PROMPT = """\
You are an expert Document Q&A AI Agent designed for enterprise use.
You have access to a knowledge base of ingested PDF documents and can
search Arxiv for additional papers.

Your capabilities:
1. **Direct Content Lookup** — Find specific information, conclusions,
   definitions, or data from documents.
2. **Summarization** — Summarize methodologies, key insights, findings,
   or entire sections of papers.
3. **Evaluation Extraction** — Extract accuracy, F1-scores, precision,
   recall, benchmarks, and other metrics from results sections and tables.
4. **Arxiv Search** — Search for and download academic papers from Arxiv
   based on user descriptions.

Guidelines:
- Always use the available tools to search for information before answering.
- Cite sources with document name and page numbers when available.
- If information is not found in the knowledge base, say so clearly.
- For summarization, provide concise but comprehensive summaries.
- When extracting metrics, present them in a clear, structured format.
- Be precise and factual — do not fabricate information.
- If the user asks about a paper not in the knowledge base, offer to
  search Arxiv for it.
"""


class QAAgent:
    """
    Document Q&A Agent with function-calling, conversation memory,
    and enterprise-grade context management.
    Uses LangGraph's create_react_agent for tool-calling orchestration.
    """

    def __init__(
        self,
        vector_store: VectorStore,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ):
        self.vector_store = vector_store
        self.managed_llm = ManagedLLM(provider=provider, model=model, api_key=api_key)
        self.llm = self.managed_llm.langchain_llm
        self.conversation_history: list = []

        # Register vector store with tools
        set_vector_store(vector_store)

        # Build agent
        self.tools = get_all_tools()
        self.agent = self._build_agent()

        logger.info("Q&A Agent initialized successfully")

    def _build_agent(self):
        """Build a LangGraph ReAct agent with function-calling tools."""
        agent = create_react_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SYSTEM_PROMPT,
        )
        return agent

    def ask(self, question: str, verbose: bool = False) -> dict:
        """
        Ask a question to the agent.

        Args:
            question: User's natural language question
            verbose: If True, include intermediate tool calls in response

        Returns:
            Dict with 'answer', 'sources', and optionally 'steps'
        """
        logger.info(f"Question: {question[:80]}...")

        try:
            # Build message list with history + new question
            messages = list(self.conversation_history)
            messages.append(HumanMessage(content=question))

            result = self.agent.invoke({"messages": messages})

            # Extract the final AI response from messages
            output_messages = result.get("messages", [])
            answer = "I couldn't find an answer."
            tool_calls_made = []

            for msg in reversed(output_messages):
                if isinstance(msg, AIMessage) and msg.content:
                    answer = msg.content
                    break

            # Collect tool call info for verbose mode and source extraction
            for msg in output_messages:
                if isinstance(msg, AIMessage) and msg.tool_calls:
                    for tc in msg.tool_calls:
                        tool_calls_made.append({
                            "tool": tc["name"],
                            "input": tc["args"],
                        })
                # Collect tool outputs for source extraction
                if hasattr(msg, "name") and hasattr(msg, "content") and msg.type == "tool":
                    for tc in tool_calls_made:
                        if tc.get("tool") == msg.name and "output" not in tc:
                            tc["output"] = str(msg.content)[:500]
                            break

            # Extract sources from tool outputs
            sources = self._extract_sources(tool_calls_made)

            # Update conversation history
            self.conversation_history.append(HumanMessage(content=question))
            self.conversation_history.append(AIMessage(content=answer))

            # Keep history manageable (last 10 exchanges)
            if len(self.conversation_history) > 20:
                self.conversation_history = self.conversation_history[-20:]

            response = {
                "answer": answer,
                "sources": sources,
            }

            if verbose:
                response["steps"] = tool_calls_made

            return response

        except Exception as e:
            logger.error(f"Agent error: {e}")
            return {
                "answer": f"An error occurred while processing your question: {e}",
                "sources": [],
            }

    def _extract_sources(self, tool_calls: list[dict]) -> list[str]:
        """Extract unique source document names from tool results."""
        sources = set()
        for tc in tool_calls:
            output = tc.get("output", "")
            for match in re.finditer(r"Source:\s*([^\]|]+)", output):
                sources.add(match.group(1).strip())
            for match in re.finditer(r"\|\s*([^\s|]+\.pdf)", output):
                sources.add(match.group(1).strip())
        return sorted(sources)

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history.clear()
        logger.info("Conversation history cleared")

    def get_stats(self) -> dict:
        """Get agent statistics."""
        return {
            "documents_loaded": len(self.vector_store.list_sources()),
            "total_chunks": self.vector_store.count,
            "conversation_turns": len(self.conversation_history) // 2,
            "llm_provider": self.managed_llm.provider,
            "tools_available": [t.name for t in self.tools],
        }
