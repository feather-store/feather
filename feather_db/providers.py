"""
feather_db.providers — LLM Provider Abstraction
================================================
Thin wrappers that expose a single `complete(messages) -> str` interface
for every major LLM backend — closed-source and open-source alike.

Supported backends
------------------
  ClaudeProvider   — Anthropic Claude (claude-haiku-4-5, claude-opus-4-6, …)
  OpenAIProvider   — OpenAI, Azure OpenAI, Groq, Mistral, Together AI, vLLM
  OllamaProvider   — Ollama local server  (subclass of OpenAIProvider)
  GeminiProvider   — Google Gemini (gemini-2.0-flash, …)

All providers default to temperature=0.0 — deterministic output is critical
for the JSON classification task in ContextEngine.ingest().

Usage
-----
  from feather_db.providers import ClaudeProvider, OllamaProvider

  # Closed-source
  p = ClaudeProvider(model="claude-haiku-4-5", api_key="sk-ant-...")
  # Open-source local
  p = OllamaProvider(model="llama3.1:8b")

  reply = p.complete([
      {"role": "system", "content": "You are helpful."},
      {"role": "user",   "content": "Say hi."},
  ])
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from typing import Optional


# ── Base ──────────────────────────────────────────────────────────────────────

class LLMProvider(ABC):
    """
    Minimal LLM abstraction for ContextEngine.
    Subclass and implement `complete()`.
    """

    @abstractmethod
    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        """
        Send `messages` to the model and return the text reply.

        Parameters
        ----------
        messages    : OpenAI-style list of {"role": ..., "content": ...} dicts
        max_tokens  : upper bound on response length
        temperature : 0.0 = fully deterministic (recommended for JSON tasks)
        """
        ...

    def name(self) -> str:
        return self.__class__.__name__

    def __repr__(self) -> str:
        return f"<{self.name()}>"


# ── Claude ────────────────────────────────────────────────────────────────────

class ClaudeProvider(LLMProvider):
    """
    Anthropic Claude via the `anthropic` SDK.

    Default model: claude-haiku-4-5 — fast and cheap for ingestion pipelines.
    Swap to claude-opus-4-6 for highest-quality classification.

    pip install anthropic
    """

    def __init__(
        self,
        model: str = "claude-haiku-4-5-20251001",
        api_key: Optional[str] = None,
        max_retries: int = 2,
    ):
        try:
            import anthropic as _ant
        except ImportError:
            raise ImportError("pip install anthropic  # required for ClaudeProvider")

        self._model = model
        self._client = _ant.Anthropic(
            api_key=api_key or os.environ.get("ANTHROPIC_API_KEY"),
            max_retries=max_retries,
        )

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        # Anthropic separates system from conversation messages
        system = ""
        conv: list[dict] = []
        for m in messages:
            if m["role"] == "system":
                system = m["content"]
            else:
                conv.append(m)

        kwargs: dict = {
            "model":      self._model,
            "max_tokens": max_tokens,
            "messages":   conv,
        }
        if system:
            kwargs["system"] = system
        # Anthropic doesn't accept temperature=0.0 via kwarg the same way;
        # pass only if non-default to avoid API version issues
        if temperature != 1.0:
            kwargs["temperature"] = temperature

        response = self._client.messages.create(**kwargs)
        return response.content[0].text

    def name(self) -> str:
        return f"ClaudeProvider({self._model})"


# ── OpenAI-compatible ─────────────────────────────────────────────────────────

class OpenAIProvider(LLMProvider):
    """
    OpenAI Chat Completions API — and any OpenAI-compatible endpoint:
      OpenAI, Azure OpenAI, Groq, Mistral, Together AI, vLLM, LM Studio, …

    pip install openai

    Examples
    --------
      OpenAIProvider()                                   # OpenAI default
      OpenAIProvider(model="gpt-4o")
      OpenAIProvider(                                    # Groq
          model="llama-3.3-70b-versatile",
          api_key=GROQ_KEY,
          base_url="https://api.groq.com/openai/v1",
      )
      OpenAIProvider(                                    # vLLM local
          model="mistralai/Mistral-7B-Instruct-v0.3",
          api_key="EMPTY",
          base_url="http://localhost:8000/v1",
      )
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 2,
        json_mode: bool = True,          # enable response_format=json_object when True
    ):
        try:
            from openai import OpenAI
        except ImportError:
            raise ImportError("pip install openai  # required for OpenAIProvider")

        self._model      = model
        self._json_mode  = json_mode
        self._client     = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY", ""),
            base_url=base_url,
            max_retries=max_retries,
        )

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        kwargs: dict = {
            "model":       self._model,
            "messages":    messages,
            "max_tokens":  max_tokens,
            "temperature": temperature,
        }
        if self._json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = self._client.chat.completions.create(**kwargs)
        return response.choices[0].message.content or ""

    def name(self) -> str:
        return f"OpenAIProvider({self._model})"


# ── Ollama ────────────────────────────────────────────────────────────────────

class OllamaProvider(OpenAIProvider):
    """
    Ollama local server — zero-config wrapper around OpenAIProvider.

    Ollama exposes an OpenAI-compatible endpoint at /v1/chat/completions.
    No extra SDK needed.

    Usage
    -----
      OllamaProvider()                          # llama3.1:8b at localhost:11434
      OllamaProvider(model="mistral:7b")
      OllamaProvider(model="phi3", base_url="http://remote-host:11434/v1")

    Ensure the model is pulled first:
      ollama pull llama3.1:8b
    """

    def __init__(
        self,
        model: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434/v1",
        json_mode: bool = True,
    ):
        # Ollama accepts any string as the API key
        super().__init__(
            model=model,
            api_key="ollama",
            base_url=base_url,
            max_retries=1,     # local server — one retry is enough
            json_mode=json_mode,
        )

    def name(self) -> str:
        return f"OllamaProvider({self._model})"


# ── Gemini ────────────────────────────────────────────────────────────────────

class GeminiProvider(LLMProvider):
    """
    Google Gemini via the `google-genai` SDK.

    pip install google-genai

    Usage
    -----
      GeminiProvider(api_key="AIza...")
      GeminiProvider(model="gemini-2.0-flash-lite")   # cheapest Gemini
    """

    def __init__(
        self,
        model: str = "gemini-2.0-flash",
        api_key: Optional[str] = None,
    ):
        try:
            from google import genai
            from google.genai import types as _gt
        except ImportError:
            raise ImportError("pip install google-genai  # required for GeminiProvider")

        self._model = model
        key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise ValueError(
                "Provide api_key= or set GOOGLE_API_KEY env var. "
                "Use GeminiProvider(api_key='AIza...')"
            )
        self._client = genai.Client(api_key=key)
        self._types  = _gt

    def complete(
        self,
        messages: list[dict],
        max_tokens: int = 512,
        temperature: float = 0.0,
    ) -> str:
        # Convert OpenAI-style messages to a Gemini-compatible prompt
        system_parts = [m["content"] for m in messages if m["role"] == "system"]
        user_parts   = [m["content"] for m in messages if m["role"] != "system"]

        system_text = "\n\n".join(system_parts)
        user_text   = "\n\n".join(user_parts)

        config_kwargs: dict = {
            "max_output_tokens": max_tokens,
            "temperature":       temperature,
            "response_mime_type": "application/json",  # JSON mode
        }
        if system_text:
            config_kwargs["system_instruction"] = system_text

        config   = self._types.GenerateContentConfig(**config_kwargs)
        response = self._client.models.generate_content(
            model=self._model,
            contents=user_text,
            config=config,
        )
        return response.text or ""

    def name(self) -> str:
        return f"GeminiProvider({self._model})"
