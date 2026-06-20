"""
Feather DB — client-side chat-completion helper
===============================================
A tiny, dependency-free (urllib only) chat factory so benchmarks and connectors
can call an LLM for answer generation / judging without pulling vendor SDKs.

    from feather_db.integrations.llm import make_chat
    chat = make_chat("gemini", api_key=KEY)
    answer = chat("You are concise.", "What is 2+2?")   # -> str

Providers: gemini, openai, ollama, anthropic.
"""
from __future__ import annotations
import os, json, urllib.request, urllib.error
from typing import Callable, Optional

_DEFAULT_MODELS = {
    "gemini":    "gemini-2.0-flash",
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-5-haiku-latest",
    "ollama":    "llama3.1",
}


def _post(url: str, headers: dict, body: dict, timeout: float = 60.0) -> dict:
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, headers={**headers, "Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        raise RuntimeError(f"{url} -> HTTP {e.code}: {e.read().decode('utf-8','replace')[:300]}")
    except urllib.error.URLError as e:
        raise RuntimeError(f"{url} -> {e.reason}")


def _gemini(system, user, model, key, base_url):
    base = (base_url or "https://generativelanguage.googleapis.com").rstrip("/")
    url = f"{base}/v1beta/models/{model}:generateContent"
    body = {"contents": [{"parts": [{"text": user}]}],
            "generationConfig": {"temperature": 0.0}}
    if system:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    j = _post(url, {"x-goog-api-key": key}, body)
    try:
        return j["candidates"][0]["content"]["parts"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"gemini: unexpected response: {json.dumps(j)[:300]}")


def _openai(system, user, model, key, base_url):
    base = (base_url or "https://api.openai.com").rstrip("/")
    msgs = ([{"role": "system", "content": system}] if system else []) + \
           [{"role": "user", "content": user}]
    body = {"model": model, "messages": msgs, "temperature": 0.0}
    j = _post(f"{base}/v1/chat/completions", {"Authorization": f"Bearer {key}"}, body)
    try:
        return j["choices"][0]["message"]["content"]
    except (KeyError, IndexError):
        raise RuntimeError(f"openai: unexpected response: {json.dumps(j)[:300]}")


def _anthropic(system, user, model, key, base_url):
    base = (base_url or "https://api.anthropic.com").rstrip("/")
    body = {"model": model, "max_tokens": 1024, "temperature": 0.0,
            "messages": [{"role": "user", "content": user}]}
    if system:
        body["system"] = system
    j = _post(f"{base}/v1/messages",
              {"x-api-key": key, "anthropic-version": "2023-06-01"}, body)
    try:
        return j["content"][0]["text"]
    except (KeyError, IndexError):
        raise RuntimeError(f"anthropic: unexpected response: {json.dumps(j)[:300]}")


def _ollama(system, user, model, key, base_url):
    base = (base_url or "http://localhost:11434").rstrip("/")
    body = {"model": model, "system": system or "", "prompt": user, "stream": False,
            "options": {"temperature": 0.0}}
    j = _post(f"{base}/api/generate", {}, body)
    return j.get("response", "")


_DRIVERS = {"gemini": _gemini, "openai": _openai,
            "anthropic": _anthropic, "ollama": _ollama}


def make_chat(provider: str, model: Optional[str] = None,
              api_key: Optional[str] = None,
              base_url: Optional[str] = None) -> Callable[[str, str], str]:
    """Return `chat(system, user) -> str` for the provider.
    api_key falls back to the provider's standard env var."""
    provider = (provider or "gemini").lower()
    if provider not in _DRIVERS:
        raise ValueError(f"unknown chat provider: {provider} "
                         f"(choose from {', '.join(_DRIVERS)})")
    model = model or _DEFAULT_MODELS[provider]
    key = (api_key or os.getenv("FEATHER_LLM_API_KEY") or {
        "gemini":    os.getenv("GOOGLE_API_KEY") or os.getenv("GEMINI_API_KEY"),
        "openai":    os.getenv("OPENAI_API_KEY"),
        "anthropic": os.getenv("ANTHROPIC_API_KEY"),
        "ollama":    "",
    }.get(provider, "") or "")
    if provider not in ("ollama",) and not key:
        raise RuntimeError(f"{provider} chat needs an API key "
                           f"(pass api_key= or set the provider env var).")
    driver = _DRIVERS[provider]

    def chat(system: str, user: str) -> str:
        return driver(system, user, model, key, base_url)
    return chat
