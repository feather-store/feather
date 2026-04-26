"""Azure OpenAI chat provider for the benchmark harness.

Implements the LLMProvider.complete() interface so it drops into
LLMJudge's existing answerer/judge slots. Keeps Azure-specific logic
out of the core feather_db.providers module.

Env vars (chat-specific so they don't collide with the embedder's vars):
    AZURE_OPENAI_CHAT_ENDPOINT      defaults to AZURE_OPENAI_ENDPOINT
    AZURE_OPENAI_CHAT_API_KEY       defaults to AZURE_OPENAI_API_KEY
    AZURE_OPENAI_CHAT_DEPLOYMENT    defaults to the constructor's model arg
    AZURE_OPENAI_CHAT_API_VERSION   defaults to "2025-01-01-preview"
"""
from __future__ import annotations
import os
import time
from typing import Optional


class AzureChatProvider:
    """Minimal LLMProvider — just complete()."""

    def __init__(self, model: Optional[str] = None,
                 api_key: Optional[str] = None,
                 endpoint: Optional[str] = None,
                 api_version: Optional[str] = None,
                 deployment: Optional[str] = None,
                 max_retries: int = 3):
        try:
            from openai import AzureOpenAI
        except ImportError as e:
            raise ImportError("pip install openai>=1.0  # required") from e

        endpoint = (endpoint
                    or os.environ.get("AZURE_OPENAI_CHAT_ENDPOINT")
                    or os.environ.get("AZURE_OPENAI_ENDPOINT"))
        if not endpoint:
            raise ValueError(
                "Provide endpoint= or set AZURE_OPENAI_CHAT_ENDPOINT / "
                "AZURE_OPENAI_ENDPOINT."
            )

        key = (api_key
               or os.environ.get("AZURE_OPENAI_CHAT_API_KEY")
               or os.environ.get("AZURE_OPENAI_API_KEY"))
        if not key:
            raise ValueError(
                "Provide api_key= or set AZURE_OPENAI_CHAT_API_KEY / "
                "AZURE_OPENAI_API_KEY."
            )

        api_version = (api_version
                       or os.environ.get("AZURE_OPENAI_CHAT_API_VERSION")
                       or "2025-01-01-preview")

        self._client = AzureOpenAI(
            api_key=key,
            api_version=api_version,
            azure_endpoint=endpoint,
        )
        # In Azure the `model` arg of chat.completions.create is the
        # deployment name. Resolution priority: explicit deployment kwarg →
        # env var → constructor model arg.
        self._deployment = (deployment
                            or os.environ.get("AZURE_OPENAI_CHAT_DEPLOYMENT")
                            or model)
        if not self._deployment:
            raise ValueError(
                "No Azure deployment name. Pass deployment= or set "
                "AZURE_OPENAI_CHAT_DEPLOYMENT, or pass model=<deployment>."
            )
        self._max_retries = max_retries

    def complete(self, messages: list[dict],
                 max_tokens: int = 512,
                 temperature: float = 0.0) -> str:
        kwargs = {
            "model": self._deployment,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        last = None
        for i in range(self._max_retries):
            try:
                resp = self._client.chat.completions.create(**kwargs)
                return (resp.choices[0].message.content or "").strip()
            except Exception as e:
                last = e
                if i + 1 < self._max_retries:
                    time.sleep(1.5 ** (i + 1))
        raise RuntimeError(f"Azure chat completion failed: {last}")

    def __repr__(self) -> str:
        return f"AzureChatProvider(deployment={self._deployment!r})"
