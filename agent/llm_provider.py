"""
Multi-LLM Provider
------------------
Configurable LLM backend supporting OpenAI, Google Gemini,
and Ollama (local open-source models).
Enterprise features: response caching, token-aware context management.
"""

import logging
import time
import hashlib
from functools import lru_cache

from langchain_core.language_models import BaseChatModel

from config import (
    LLM_PROVIDER,
    OPENAI_API_KEY,
    OPENAI_MODEL,
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OLLAMA_BASE_URL,
    OLLAMA_MODEL,
    RESPONSE_MAX_TOKENS,
    RATE_LIMIT_RPM,
    ENABLE_CACHE,
)

logger = logging.getLogger(__name__)


# ── Response Cache ───────────────────────────────────────────────

class ResponseCache:
    """Simple in-memory LRU cache for LLM responses."""

    def __init__(self, max_size: int = 256):
        self._cache: dict[str, tuple[str, float]] = {}
        self._max_size = max_size

    @staticmethod
    def _key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()

    def get(self, prompt: str) -> str | None:
        key = self._key(prompt)
        if key in self._cache:
            value, ts = self._cache[key]
            # Expire after 1 hour
            if time.time() - ts < 3600:
                logger.debug("Cache hit")
                return value
            del self._cache[key]
        return None

    def set(self, prompt: str, response: str):
        if len(self._cache) >= self._max_size:
            oldest = min(self._cache, key=lambda k: self._cache[k][1])
            del self._cache[oldest]
        self._cache[self._key(prompt)] = (response, time.time())


# ── Rate Limiter ─────────────────────────────────────────────────

class RateLimiter:
    """Token-bucket rate limiter for API calls."""

    def __init__(self, rpm: int = RATE_LIMIT_RPM):
        self.min_interval = 60.0 / rpm if rpm > 0 else 0
        self._last_call = 0.0

    def wait(self):
        now = time.time()
        elapsed = now - self._last_call
        if elapsed < self.min_interval:
            sleep_time = self.min_interval - elapsed
            logger.debug(f"Rate limit: sleeping {sleep_time:.2f}s")
            time.sleep(sleep_time)
        self._last_call = time.time()


# ── LLM Factory ──────────────────────────────────────────────────

def create_llm(
    provider: str | None = None,
    model: str | None = None,
    temperature: float = 0.2,
    max_tokens: int | None = None,
    api_key: str | None = None,
) -> BaseChatModel:
    """
    Create a LangChain chat model for the specified provider.

    Args:
        provider: "openai", "gemini", or "ollama" (defaults to config)
        model: Model name override
        temperature: Sampling temperature
        max_tokens: Max response tokens
        api_key: Optional API key override (session-only, not persisted)

    Returns:
        Configured LangChain ChatModel
    """
    provider = (provider or LLM_PROVIDER).lower()
    max_tokens = max_tokens or RESPONSE_MAX_TOKENS

    if provider == "openai":
        from langchain_openai import ChatOpenAI

        model = model or OPENAI_MODEL
        llm = ChatOpenAI(
            api_key=api_key or OPENAI_API_KEY,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        logger.info(f"LLM initialized: OpenAI {model}")

    elif provider == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        model = model or GEMINI_MODEL
        llm = ChatGoogleGenerativeAI(
            google_api_key=api_key or GEMINI_API_KEY,
            model=model,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        logger.info(f"LLM initialized: Gemini {model}")

    elif provider == "ollama":
        from langchain_community.chat_models import ChatOllama

        model = model or OLLAMA_MODEL
        llm = ChatOllama(
            base_url=OLLAMA_BASE_URL,
            model=model,
            temperature=temperature,
            num_predict=max_tokens,
        )
        logger.info(f"LLM initialized: Ollama {model}")

    else:
        raise ValueError(
            f"Unknown LLM provider: {provider}. "
            "Choose from: openai, gemini, ollama"
        )

    return llm


# ── Managed LLM Wrapper ─────────────────────────────────────────

class ManagedLLM:
    """
    Enterprise wrapper around LangChain LLM with:
    - Response caching
    - Rate limiting
    - Error handling with retries
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        temperature: float = 0.2,
        api_key: str | None = None,
    ):
        self.llm = create_llm(provider, model, temperature, api_key=api_key)
        self.cache = ResponseCache() if ENABLE_CACHE else None
        self.rate_limiter = RateLimiter()
        self.provider = provider or LLM_PROVIDER

    def invoke(self, prompt: str, use_cache: bool = True) -> str:
        """Send a prompt to the LLM with caching and rate limiting."""
        # Check cache
        if use_cache and self.cache:
            cached = self.cache.get(prompt)
            if cached is not None:
                return cached

        # Rate limit
        self.rate_limiter.wait()

        # Call LLM with retry
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self.llm.invoke(prompt)
                result = response.content
                # Cache response
                if self.cache:
                    self.cache.set(prompt, result)
                return result
            except Exception as e:
                if attempt < max_retries - 1:
                    wait = 2 ** attempt
                    logger.warning(
                        f"LLM call failed (attempt {attempt + 1}): {e}. "
                        f"Retrying in {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    logger.error(f"LLM call failed after {max_retries} attempts: {e}")
                    raise

    @property
    def langchain_llm(self) -> BaseChatModel:
        """Access the underlying LangChain LLM for agent integration."""
        return self.llm
