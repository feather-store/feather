"""Client-side embedders cache provider calls (feather_db.integrations.embedders)."""
import pytest

from feather_db.integrations import embedders


@pytest.fixture
def counting(monkeypatch):
    calls = []

    def fake(text, model, key, dim, base_url):
        calls.append(text)
        if text == "boom":
            raise RuntimeError("provider down")
        return [float(len(text)), 1.0]

    monkeypatch.setitem(embedders._DRIVERS, "openai", fake)
    return calls


def test_repeat_text_hits_cache(counting):
    embed = embedders.make_embedder("openai", api_key="k", dim=4)
    a = embed("same text")
    b = embed("same text")
    assert a == b == [9.0, 1.0, 0.0, 0.0]
    assert counting == ["same text"]
    assert embed.stats == {"calls": 2, "cache_hits": 1}


def test_hit_returns_copy(counting):
    embed = embedders.make_embedder("openai", api_key="k", dim=2)
    embed("x")[0] = 99.0
    assert embed("x") == [1.0, 1.0]


def test_lru_evicts_oldest(counting, monkeypatch):
    monkeypatch.setenv("FEATHER_EMBED_CACHE", "2")
    embed = embedders.make_embedder("openai", api_key="k", dim=2)
    for t in ["a", "b", "a", "c", "b", "a"]:
        embed(t)
    # a,b miss; a hit; c miss evicts b; b miss evicts a; a miss
    assert counting == ["a", "b", "c", "b", "a"]


def test_failures_not_cached(counting):
    embed = embedders.make_embedder("openai", api_key="k", dim=2)
    for _ in range(2):
        with pytest.raises(RuntimeError):
            embed("boom")
    assert counting == ["boom", "boom"]


def test_zero_disables(counting, monkeypatch):
    monkeypatch.setenv("FEATHER_EMBED_CACHE", "0")
    embed = embedders.make_embedder("openai", api_key="k", dim=2)
    embed("t"); embed("t")
    assert counting == ["t", "t"]
    assert not hasattr(embed, "stats")
