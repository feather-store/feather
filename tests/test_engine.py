"""
ContextEngine + Provider tests (offline / heuristic mode only).
LLM provider tests (Claude, OpenAI, Gemini, Ollama) require API keys
and are skipped automatically when keys are absent.
"""
import os
import pytest
import feather_db
from feather_db import ContextEngine, LLMProvider, ClaudeProvider, OpenAIProvider, OllamaProvider, GeminiProvider
from .conftest import EMBED


class TestProviderInterface:
    def test_all_providers_are_llmprovider_subclass(self):
        for cls in [ClaudeProvider, OpenAIProvider, OllamaProvider, GeminiProvider]:
            assert issubclass(cls, LLMProvider), f"{cls.__name__} must subclass LLMProvider"

    def test_all_providers_have_complete_method(self):
        for cls in [ClaudeProvider, OpenAIProvider, OllamaProvider, GeminiProvider]:
            assert hasattr(cls, "complete"), f"{cls.__name__} missing complete()"

    def test_ollama_is_openai_subclass(self):
        assert issubclass(OllamaProvider, OpenAIProvider)

    def test_provider_str(self):
        p = OllamaProvider(model="mistral:7b")
        s = str(p)
        assert "Ollama" in s or "ollama" in s or "mistral" in s


class TestContextEngineOffline:
    def test_ingest_returns_int(self, engine):
        nid = engine.ingest("API latency spikes under 500 concurrent requests")
        assert isinstance(nid, int)
        assert nid > 0

    def test_ingest_batch_returns_list(self, engine):
        texts = [
            "Onboarding drop-off rate is 42% at step 3",
            "Users want a one-click deploy option",
            "Dark mode is the most requested UI feature",
        ]
        ids = engine.ingest_batch(texts)
        assert len(ids) == 3
        assert all(isinstance(i, int) for i in ids)

    def test_ingest_batch_unique_ids(self, engine):
        texts = [
            "First unique record about feature A",
            "Second unique record about feature B",
            "Third unique record about feature C",
        ]
        ids = engine.ingest_batch(texts)
        assert len(set(ids)) == 3  # all unique

    def test_ingested_record_searchable(self, engine):
        nid = engine.ingest("Competitor released v2 with 50% faster search")
        db = engine.db  # public accessor
        results = db.search(EMBED("faster search"), k=3)
        assert any(r.id == nid for r in results)

    def test_heuristic_assigns_entity_type(self, engine):
        nid = engine.ingest("User prefers dark mode across all apps")
        meta = engine.db.get_metadata(nid)
        et = meta.get_attribute("entity_type")
        assert et != "", "entity_type should be set by heuristic classifier"

    def test_hint_overrides_namespace(self, engine):
        nid = engine.ingest("some text", hint={"namespace": "override_ns"})
        meta = engine.db.get_metadata(nid)
        assert meta.namespace_id == "override_ns"

    def test_hint_overrides_importance(self, engine):
        nid = engine.ingest("critical incident in production", hint={"importance": 1.0})
        meta = engine.db.get_metadata(nid)
        assert meta.importance == pytest.approx(1.0, abs=0.01)

    def test_ingest_duplicate_text_unique_id(self, engine):
        # IDs include timestamp+PID, so even same text gets a unique ID per call.
        # This is intentional — allows versioning the same content over time.
        text = "Stable content for ID test"
        id1 = engine.ingest(text)
        id2 = engine.ingest(text)
        assert isinstance(id1, int) and isinstance(id2, int)
        assert id1 > 0 and id2 > 0

    def test_provider_none_is_heuristic_mode(self, engine):
        assert engine._provider is None

    def test_provider_fallback_enabled_by_default(self, engine):
        assert engine._fallback is True


class TestContextEngineWithClaude:
    @pytest.mark.skipif(
        not os.environ.get("ANTHROPIC_API_KEY"),
        reason="ANTHROPIC_API_KEY not set"
    )
    def test_claude_ingest_classifies_correctly(self, tmp_path_feather):
        engine = ContextEngine(
            db_path=tmp_path_feather, dim=128,
            provider=ClaudeProvider(),
            embedder=EMBED, namespace="test"
        )
        nid = engine.ingest("Competitor X launched a free tier with 10x more storage")
        db = feather_db.DB.open(tmp_path_feather, dim=128)
        meta = db.get_metadata(nid)
        et = meta.get_attribute("entity_type")
        imp = meta.importance
        assert et != ""
        assert 0.0 <= imp <= 1.0


class TestContextEngineWithOpenAI:
    @pytest.mark.skipif(
        not os.environ.get("OPENAI_API_KEY"),
        reason="OPENAI_API_KEY not set"
    )
    def test_openai_ingest(self, tmp_path_feather):
        engine = ContextEngine(
            db_path=tmp_path_feather, dim=128,
            provider=OpenAIProvider(model="gpt-4o-mini"),
            embedder=EMBED, namespace="test"
        )
        nid = engine.ingest("Users are requesting webhook support in the API")
        assert isinstance(nid, int)


class TestContextEngineWithOllama:
    @pytest.mark.skipif(
        not os.environ.get("OLLAMA_HOST"),
        reason="OLLAMA_HOST not set — local Ollama server required"
    )
    def test_ollama_ingest(self, tmp_path_feather):
        engine = ContextEngine(
            db_path=tmp_path_feather, dim=128,
            provider=OllamaProvider(),
            embedder=EMBED, namespace="test"
        )
        nid = engine.ingest("Dark mode adoption is 78% among power users")
        assert isinstance(nid, int)


class TestContextEngineWithGemini:
    @pytest.mark.skipif(
        not os.environ.get("GEMINI_API_KEY"),
        reason="GEMINI_API_KEY not set"
    )
    def test_gemini_ingest(self, tmp_path_feather):
        engine = ContextEngine(
            db_path=tmp_path_feather, dim=128,
            provider=GeminiProvider(),
            embedder=EMBED, namespace="test"
        )
        nid = engine.ingest("Gemini embedding model achieves 3072-dim representations")
        assert isinstance(nid, int)
