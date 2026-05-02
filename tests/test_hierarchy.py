"""Tests for feather_db.hierarchy."""
from __future__ import annotations
import pytest

from feather_db.hierarchy import (
    Hierarchy, HierarchyNode, MARKETING_HIERARCHY,
)


@pytest.fixture
def acme_h():
    """A small Marketing hierarchy:

        Brand:Acme
          ├─ Channel:meta
          │   └─ Campaign:summer_sale_2026
          │       ├─ AdSet:runners
          │       │   ├─ Ad:carousel_a
          │       │   │   └─ Creative:hero_v1
          │       │   └─ Ad:carousel_b
          │       └─ AdSet:walkers
          │           └─ Ad:single_a
          │               └─ Creative:hero_v2
          └─ Channel:google  (empty)
    """
    h = Hierarchy()
    h.add_many([
        HierarchyNode("Brand",    "brand::acme",          name="Acme"),
        HierarchyNode("Channel",  "channel::meta",
                      parent_id="brand::acme"),
        HierarchyNode("Channel",  "channel::google",
                      parent_id="brand::acme"),
        HierarchyNode("Campaign", "camp::summer_sale_2026",
                      parent_id="channel::meta"),
        HierarchyNode("AdSet",    "adset::runners",
                      parent_id="camp::summer_sale_2026"),
        HierarchyNode("AdSet",    "adset::walkers",
                      parent_id="camp::summer_sale_2026"),
        HierarchyNode("Ad",       "ad::carousel_a",
                      parent_id="adset::runners"),
        HierarchyNode("Ad",       "ad::carousel_b",
                      parent_id="adset::runners"),
        HierarchyNode("Ad",       "ad::single_a",
                      parent_id="adset::walkers"),
        HierarchyNode("Creative", "cre::hero_v1",
                      parent_id="ad::carousel_a"),
        HierarchyNode("Creative", "cre::hero_v2",
                      parent_id="ad::single_a"),
    ])
    return h


def test_default_levels_marketing():
    h = Hierarchy()
    assert h.levels == MARKETING_HIERARCHY


def test_custom_levels():
    h = Hierarchy(levels=["Org", "Team", "Person"])
    assert h.levels == ["Org", "Team", "Person"]


def test_duplicate_levels_rejected():
    with pytest.raises(ValueError):
        Hierarchy(levels=["A", "B", "A"])


def test_add_unknown_kind_rejected():
    h = Hierarchy()
    with pytest.raises(ValueError, match="unknown level"):
        h.add(HierarchyNode("BogusKind", "x::1"))


def test_add_inverted_parent_rejected():
    """Creative cannot be the parent of an Ad."""
    h = Hierarchy()
    h.add(HierarchyNode("Creative", "cre::1"))
    with pytest.raises(ValueError, match="not strictly above"):
        h.add(HierarchyNode("Ad", "ad::1", parent_id="cre::1"))


def test_parent_and_ancestors(acme_h):
    p = acme_h.parent("ad::carousel_a")
    assert p is not None and p.canonical_id == "adset::runners"
    chain = [n.canonical_id for n in acme_h.ancestors("cre::hero_v1")]
    assert chain == [
        "ad::carousel_a", "adset::runners",
        "camp::summer_sale_2026", "channel::meta", "brand::acme",
    ]


def test_ancestors_of_root_is_empty(acme_h):
    assert acme_h.ancestors("brand::acme") == []


def test_ancestors_of_unknown_is_empty(acme_h):
    assert acme_h.ancestors("nope::x") == []


def test_children(acme_h):
    kids = [n.canonical_id for n in acme_h.children("adset::runners")]
    assert sorted(kids) == ["ad::carousel_a", "ad::carousel_b"]


def test_descendants(acme_h):
    """All transitive descendants of the Meta channel."""
    dids = [n.canonical_id
            for n in acme_h.descendants("channel::meta")]
    assert "cre::hero_v1" in dids
    assert "cre::hero_v2" in dids
    assert "ad::single_a" in dids
    # google channel is a sibling, not in here
    assert "channel::google" not in dids


def test_is_descendant_of(acme_h):
    assert acme_h.is_descendant_of("cre::hero_v1", "brand::acme")
    assert acme_h.is_descendant_of("cre::hero_v1", "adset::runners")
    assert not acme_h.is_descendant_of("cre::hero_v1", "channel::google")


def test_common_ancestor(acme_h):
    # Two creatives under different adsets → common ancestor is the
    # campaign.
    ca = acme_h.common_ancestor("cre::hero_v1", "cre::hero_v2")
    assert ca is not None
    assert ca.canonical_id == "camp::summer_sale_2026"


def test_common_ancestor_disjoint():
    h = Hierarchy()
    h.add(HierarchyNode("Brand", "brand::a"))
    h.add(HierarchyNode("Brand", "brand::b"))
    assert h.common_ancestor("brand::a", "brand::b") is None


def test_level_of(acme_h):
    assert acme_h.level_of("brand::acme") == "Brand"
    assert acme_h.level_of("ad::carousel_a") == "Ad"
    assert acme_h.level_of("nope::x") is None


def test_level_index(acme_h):
    assert acme_h.level_index("Brand") == 0
    assert acme_h.level_index("Creative") == 5
    assert acme_h.level_index("BogusKind") is None


def test_len_and_contains(acme_h):
    assert len(acme_h) == 11
    assert "brand::acme" in acme_h
    assert "nope::x" not in acme_h
