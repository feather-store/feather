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
__version__ = "0.10.8"
