"""
Unit tests for the LLM provider api_key pass-through.
"""

import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from agent.llm_provider import create_llm, ManagedLLM


def test_create_llm_passes_api_key_to_openai():
    """create_llm should forward the api_key kwarg to ChatOpenAI."""
    MockChat = MagicMock()
    mock_module = MagicMock()
    mock_module.ChatOpenAI = MockChat
    with patch.dict("sys.modules", {"langchain_openai": mock_module}):
        create_llm(provider="openai", model="gpt-test", api_key="sk-test-123")

    MockChat.assert_called_once()
    call_kwargs = MockChat.call_args[1]
    assert call_kwargs["api_key"] == "sk-test-123", (
        f"Expected api_key='sk-test-123', got {call_kwargs.get('api_key')}"
    )
    assert call_kwargs["model"] == "gpt-test"


def test_create_llm_passes_api_key_to_gemini():
    """create_llm should forward the api_key kwarg to ChatGoogleGenerativeAI."""
    MockGemini = MagicMock()
    mock_module = MagicMock()
    mock_module.ChatGoogleGenerativeAI = MockGemini
    with patch.dict("sys.modules", {"langchain_google_genai": mock_module}):
        create_llm(provider="gemini", model="gemini-test", api_key="gem-key-456")

    MockGemini.assert_called_once()
    call_kwargs = MockGemini.call_args[1]
    assert call_kwargs["google_api_key"] == "gem-key-456"
    assert call_kwargs["model"] == "gemini-test"


def test_create_llm_falls_back_to_config_key():
    """When api_key is None, config default should be used."""
    MockChat = MagicMock()
    mock_module = MagicMock()
    mock_module.ChatOpenAI = MockChat
    with patch.dict("sys.modules", {"langchain_openai": mock_module}):
        from config import OPENAI_API_KEY
        create_llm(provider="openai", model="gpt-test", api_key=None)

    call_kwargs = MockChat.call_args[1]
    assert call_kwargs["api_key"] == OPENAI_API_KEY, (
        "Should fall back to OPENAI_API_KEY from config when api_key is None"
    )


def test_managed_llm_accepts_api_key():
    """ManagedLLM.__init__ should accept and pass api_key to create_llm."""
    with patch("agent.llm_provider.create_llm") as mock_create:
        mock_create.return_value = MagicMock()
        m = ManagedLLM(provider="openai", model="gpt-test", api_key="sk-managed")

    mock_create.assert_called_once_with("openai", "gpt-test", 0.2, api_key="sk-managed")


if __name__ == "__main__":
    test_create_llm_passes_api_key_to_openai()
    test_create_llm_passes_api_key_to_gemini()
    test_create_llm_falls_back_to_config_key()
    test_managed_llm_accepts_api_key()
    print("All LLM provider tests passed ✓")
