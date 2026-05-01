"""
LLM provider abstraction via LiteLLM.

Supports any model LiteLLM supports (OpenAI, Anthropic, Gemini, Ollama, etc.)
with unified interface, retry logic, and token counting.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional

try:
    import litellm
    litellm.set_verbose = False
    HAS_LITELLM = True
except ImportError:
    HAS_LITELLM = False


@dataclass
class LLMResponse:
    """Standardized response from any LLM provider."""
    text: str
    input_tokens: int = 0
    output_tokens: int = 0
    model: str = ""
    cost: float = 0.0


@dataclass
class LLMConfig:
    """Configuration for the LLM provider."""
    model: str = "gemini/gemini-2.0-flash"
    temperature: float = 0.1
    max_tokens: int = 200
    timeout: int = 30
    max_retries: int = 2
    api_key: Optional[str] = None
    api_base: Optional[str] = None

    def __post_init__(self):
        # If no explicit api_key, try to pick up from env vars
        if not self.api_key:
            if self.model.startswith("gemini"):
                self.api_key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
            elif self.model.startswith("gpt") or self.model.startswith("o"):
                self.api_key = os.environ.get("OPENAI_API_KEY")
            elif self.model.startswith("claude"):
                self.api_key = os.environ.get("ANTHROPIC_API_KEY")
        if self.api_key:
            # Force-set env var (not setdefault, so it always wins)
            if self.model.startswith("gemini"):
                os.environ["GEMINI_API_KEY"] = self.api_key
                os.environ["GOOGLE_API_KEY"] = self.api_key
            elif self.model.startswith("gpt") or self.model.startswith("o"):
                os.environ["OPENAI_API_KEY"] = self.api_key
            elif self.model.startswith("claude"):
                os.environ["ANTHROPIC_API_KEY"] = self.api_key


def complete(
    prompt: str,
    system: str = "",
    config: Optional[LLMConfig] = None,
) -> LLMResponse:
    """Send a completion request to the configured LLM.

    Args:
        prompt: User message content.
        system: System message content.
        config: LLM configuration. Uses defaults if None.

    Returns:
        LLMResponse with text and usage stats.

    Raises:
        RuntimeError: If litellm is not installed.
    """
    if not HAS_LITELLM:
        raise RuntimeError(
            "litellm is required for LLM features. "
            "Install with: pip install skeletongraph[llm]"
        )

    cfg = config or LLMConfig()

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    kwargs: Dict = dict(
        model=cfg.model,
        messages=messages,
        temperature=cfg.temperature,
        max_tokens=cfg.max_tokens,
        timeout=cfg.timeout,
        num_retries=cfg.max_retries,
    )
    if cfg.api_key:
        kwargs["api_key"] = cfg.api_key
    if cfg.api_base:
        kwargs["api_base"] = cfg.api_base
    response = litellm.completion(**kwargs)

    text = response.choices[0].message.content or ""
    usage = response.usage

    return LLMResponse(
        text=text.strip(),
        input_tokens=getattr(usage, "prompt_tokens", 0),
        output_tokens=getattr(usage, "completion_tokens", 0),
        model=cfg.model,
        cost=getattr(response, "_hidden_params", {}).get("response_cost", 0.0),
    )


def batch_complete(
    prompts: List[str],
    system: str = "",
    config: Optional[LLMConfig] = None,
    batch_size: int = 5,
) -> List[LLMResponse]:
    """Send multiple completion requests sequentially.

    For true parallelism, use litellm.batch_completion directly.
    This wrapper provides simpler error handling per-item.
    """
    results = []
    for prompt in prompts:
        try:
            resp = complete(prompt, system=system, config=config)
            results.append(resp)
        except Exception as e:
            results.append(LLMResponse(text=f"[ERROR] {e}", model="error"))
    return results
