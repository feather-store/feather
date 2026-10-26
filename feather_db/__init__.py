from .core import (DB, ContextType, Metadata, ScoringConfig,
                   Edge, IncomingEdge,
                   ContextNode, ContextEdge, ContextChainResult)
from .filter import FilterBuilder
from .domain_profiles import DomainProfile, MarketingProfile
from .graph import visualize, export_graph, RelType

# v0.6.0: Memory, Triggers, Episodes, Merge
from .memory   import MemoryManager
from .triggers import WatchManager, ContradictionDetector
from .episodes import EpisodeManager
from .merge    import merge

# LLM agent connectors (lazy-import safe — heavy deps are optional)
from .integrations import (
    ClaudeConnector,
    OpenAIConnector,
    GeminiConnector,
    GeminiEmbedder,
)

# v0.7.0: Self-Aligned Context Engine
from .providers import (
    LLMProvider,
    ClaudeProvider,
    OpenAIProvider,
    OllamaProvider,
    GeminiProvider,
)
from .engine import ContextEngine

__all__ = [
    "Pocket", "PocketItem", "pocket", "scoped_id",
    "PacketBuilder", "ContextPacket", "Ref", "Omission",
    "RequiredRule", "RequiredContextUnavailable",
    "DB", "ContextType", "Metadata", "ScoringConfig",
    "Edge", "IncomingEdge",
    "ContextNode", "ContextEdge", "ContextChainResult",
    "FilterBuilder",
    "DomainProfile", "MarketingProfile",
    "visualize", "export_graph", "RelType",
    # v0.6.0 memory layer
    "MemoryManager", "WatchManager", "ContradictionDetector",
    "EpisodeManager", "merge",
    # Integrations
    "ClaudeConnector", "OpenAIConnector",
    "GeminiConnector", "GeminiEmbedder",
    # v0.7.0 Self-Aligned Context Engine
    "LLMProvider", "ClaudeProvider", "OpenAIProvider",
    "OllamaProvider", "GeminiProvider",
    "ContextEngine",
]
from .pocket import Pocket, PocketItem, pocket  # noqa: E402

__version__ = "0.21.0"


# ── Context packets (Phase 9) ────────────────────────────────────────────
# Lazy so a plain `import feather_db` does not pull the packet machinery.
# `importlib` rather than `from feather_db import pocket`, because the name
# `pocket` in this namespace is already the FACTORY FUNCTION exported above, so
# that import returns the function and attribute lookup on it fails.
def __getattr__(name):
    import importlib
    if name in ("PacketBuilder", "ContextPacket", "Ref", "Omission",
                "RequiredRule", "RequiredContextUnavailable"):
        return getattr(importlib.import_module("feather_db.packet"), name)
    if name == "scoped_id":
        return getattr(importlib.import_module("feather_db.pocket"), name)
    raise AttributeError(f"module 'feather_db' has no attribute {name!r}")
