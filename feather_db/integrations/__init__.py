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
# Typed agent memory. Pure Python, no optional deps of its own — it needs a
# store instance, not a store library, so it imports eagerly.
from .agent_memory  import AgentMemory, KINDS

# LangChain / LlamaIndex adapters — optional deps, import gracefully
try:
    from .langchain_compat  import FeatherVectorStore, FeatherMemory, FeatherRetriever
    _LANGCHAIN_LOADED = True
except Exception:
    _LANGCHAIN_LOADED = False

try:
    # Exported under an alias: the LangChain adapter already owns the name
    # FeatherVectorStore. (This imported a non-existent class before, and the
    # except below silently hid it — the LlamaIndex adapters were never exported.)
    from .llamaindex_compat import FeatherVectorStore as FeatherVectorStoreIndex, FeatherReader
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
    "AgentMemory",
    "KINDS",
]


# LangGraph long-term memory. Imported lazily — langgraph is optional, and the
# module must not break `import feather_db.integrations` when it is absent.
def __getattr__(name):
    if name == "FeatherStore":
        from .langgraph_store import FeatherStore
        return FeatherStore
    raise AttributeError(name)
