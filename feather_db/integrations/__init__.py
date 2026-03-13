"""
Feather DB — LLM Agent Connectors
===================================
Ready-made connectors that expose Feather DB as tool-use / function-calling
tools for every major LLM provider.

Supported providers
-------------------
  Claude (Anthropic)         → ClaudeConnector
  OpenAI + compatible APIs   → OpenAIConnector
    (Azure OpenAI, Groq, Mistral, Together AI, Ollama …)
  Google Gemini              → GeminiConnector + GeminiEmbedder

Quick start
-----------
  # --- Claude ---
  import anthropic
  from feather_db.integrations import ClaudeConnector

  conn   = ClaudeConnector(db_path="my.feather", dim=3072, embedder=embed_fn)
  client = anthropic.Anthropic()
  result = conn.run_loop(client,
                         messages=[{"role":"user","content":"Why is CTR dropping?"}],
                         model="claude-opus-4-6")

  # --- OpenAI / Groq / Mistral ---
  from openai import OpenAI
  from feather_db.integrations import OpenAIConnector

  conn   = OpenAIConnector(db_path="my.feather", dim=3072, embedder=embed_fn)
  client = OpenAI()
  result = conn.run_loop(client,
                         messages=[{"role":"user","content":"Why is CTR dropping?"}],
                         model="gpt-4o")

  # --- Gemini ---
  from google import genai
  from feather_db.integrations import GeminiConnector, GeminiEmbedder

  emb    = GeminiEmbedder(api_key="AIza...")
  conn   = GeminiConnector(db_path="my.feather", dim=3072, embedder=emb.embed_text)
  client = genai.Client(api_key="AIza...")
  chat   = client.chats.create(model="gemini-2.0-flash", config=conn.chat_config())
  result = conn.run_loop(chat, "Why is CTR dropping?")
"""

from .base          import FeatherTools, TOOL_SPECS
from .claude        import ClaudeConnector
from .openai_compat import OpenAIConnector
from .gemini        import GeminiConnector, GeminiEmbedder

# LangChain / LlamaIndex adapters — optional deps, import gracefully
try:
    from .langchain_compat  import FeatherVectorStore, FeatherMemory, FeatherRetriever
    _LANGCHAIN_LOADED = True
except Exception:
    _LANGCHAIN_LOADED = False

try:
    from .llamaindex_compat import FeatherVectorStoreIndex, FeatherReader
    _LLAMAINDEX_LOADED = True
except Exception:
    _LLAMAINDEX_LOADED = False

__all__ = [
    "FeatherTools",
    "TOOL_SPECS",
    "ClaudeConnector",
    "OpenAIConnector",
    "GeminiConnector",
    "GeminiEmbedder",
    "FeatherVectorStore",
    "FeatherMemory",
    "FeatherRetriever",
    "FeatherVectorStoreIndex",
    "FeatherReader",
]
