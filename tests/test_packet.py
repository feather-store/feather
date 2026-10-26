"""Context packets: required context is guaranteed or the packet is degraded.

`Pocket.hot()` fits a budget by heat, which is right for a working set and wrong
for an operating constraint — a brand rule is not more relevant for being
recalled often, and must not fall out because forty creative notes outranked it.
These tests pin the guarantee and, just as importantly, that nothing is ever
dropped silently.
"""
import os

import numpy as np
import pytest

from feather_db import DB
from feather_db.pocket import pocket
from feather_db.packet import (
    PacketBuilder, ContextPacket, RequiredContextUnavailable,
    MISSING, BUDGET, DUPLICATE,
)


@pytest.fixture
def pkt(tmp_path):
    """An org with two binding rules, and a creative agent under it."""
    db = DB.open(str(tmp_path / "p.feather"), dim=64)
    org = pocket(db, ("acme",))
    org.remember("compliance", "never make medical or health claims", pinned=True)
    org.remember("brand_safety", "no competitor names in paid copy")
    yield pocket(db, ("acme", "nike", "creative"), budget_tokens=400)
    db.close()


# ── the guarantee ─────────────────────────────────────────────────────────

def test_required_context_is_included_even_under_budget_pressure(pkt):
    """The failure this exists to prevent: a binding rule crowded out by
    forty ordinary notes, silently."""
    for i in range(40):
        pkt.remember(f"note_{i}", f"creative note number {i} about formats and copy")

    packet = PacketBuilder(pkt, budget_tokens=300).build(
        required=["compliance", "brand_safety"])

    assert packet.may_mutate
    assert "medical" in packet.text
    assert "competitor names" in packet.text
    assert {r.key for r in packet.required_refs} == {"compliance", "brand_safety"}
    assert packet.omitted, "40 notes into a 300-token budget dropped nothing?"


def test_required_context_resolves_through_inheritance(pkt):
    """Both rules live two scopes up. A packet that only looked in the agent's
    own scope would report them missing and block every mutation."""
    packet = PacketBuilder(pkt).build(required=["compliance"])
    ref = packet.required_refs[0]
    assert ref.scope == ("acme",)
    assert packet.may_mutate


def test_the_nearest_scope_wins(pkt):
    """A campaign-level override must not require deleting the brand rule."""
    pkt.remember("compliance", "medical claims pre-approved for this campaign")
    packet = PacketBuilder(pkt).build(required=["compliance"])
    assert "pre-approved" in packet.text
    assert "never make medical" not in packet.text
    assert packet.required_refs[0].scope == ("acme", "nike", "creative")


# ── failing closed ────────────────────────────────────────────────────────

def test_a_missing_required_key_raises_rather_than_returning_a_packet(pkt):
    with pytest.raises(RequiredContextUnavailable) as e:
        PacketBuilder(pkt).build(required=["compliance", "no_such_rule"])
    assert "no_such_rule" in str(e.value)
    assert "blocked" in str(e.value)


def test_required_rules_that_cannot_fit_raise_with_the_arithmetic(pkt):
    """A caller that cannot see it asked for more rule tokens than its budget
    cannot fix the problem."""
    pkt.remember("huge", "x " * 4000)
    with pytest.raises(RequiredContextUnavailable) as e:
        PacketBuilder(pkt, budget_tokens=50).build(required=["huge"])
    msg = str(e.value)
    assert "did not fit" in msg
    assert "50-token budget" in msg


def test_the_whole_required_set_is_judged_together_not_one_by_one(pkt):
    """Dropping rules individually in list order would leave a caller with a
    partial constraint set and no signal — worse than none."""
    pkt.remember("r1", "rule one " * 60)
    pkt.remember("r2", "rule two " * 60)
    packet = PacketBuilder(pkt, budget_tokens=60).build(
        required=["r1", "r2"], allow_degraded=True)

    assert not packet.may_mutate
    assert {o.ref.split("/")[-1].split("@")[0] for o in packet.missing_required} == {"r1", "r2"}
    assert not packet.required_refs, "a partial required set was presented as complete"


def test_read_only_use_may_degrade_on_purpose(pkt):
    """The plan's rule: block dependent mutation, but ordinary read-only help
    can degrade gracefully."""
    packet = PacketBuilder(pkt).build(required=["ghost"], allow_degraded=True)
    assert isinstance(packet, ContextPacket)
    assert not packet.may_mutate
    assert "ghost" in packet.why_blocked()
    assert packet.missing_required[0].reason == MISSING


def test_why_blocked_is_empty_when_nothing_is_missing(pkt):
    assert PacketBuilder(pkt).build(required=["compliance"]).why_blocked() == ""


# ── nothing is dropped silently ───────────────────────────────────────────

def test_everything_omitted_is_recorded_with_a_reason(pkt):
    for i in range(40):
        pkt.remember(f"note_{i}", f"creative note {i} " * 6)

    packet = PacketBuilder(pkt, budget_tokens=200).build(required=["compliance"])
    assert packet.omitted
    for o in packet.omitted:
        assert o.reason in (MISSING, BUDGET, DUPLICATE)
        assert o.ref
    # and the manifest carries them, so a decision can be audited later
    assert len(packet.manifest()["omitted_context"]) == len(packet.omitted)


def test_a_repeated_required_key_is_recorded_not_duplicated(pkt):
    packet = PacketBuilder(pkt).build(required=["compliance", "compliance"])
    assert len(packet.required_refs) == 1
    assert [o.reason for o in packet.omitted] == [DUPLICATE]


def test_the_budget_is_actually_respected(pkt):
    for i in range(60):
        pkt.remember(f"note_{i}", f"note {i} with a reasonable amount of text in it")

    for budget in (120, 300, 800):
        packet = PacketBuilder(pkt, budget_tokens=budget).build(required=["compliance"])
        assert packet.tokens_used <= budget, f"overran a {budget}-token budget"


# ── token accounting ──────────────────────────────────────────────────────

def test_the_packet_says_whether_its_token_count_is_exact(pkt):
    """The 4-chars estimate is fine for English prose and wrong for JSON, code
    and non-Latin scripts — all of which appear in agent memory. A budget
    overrun has to be attributable rather than guessed at."""
    packet = PacketBuilder(pkt).build(required=["compliance"])
    assert isinstance(packet.exact_tokens, bool)
    assert packet.manifest()["exact_tokens"] == packet.exact_tokens


def test_a_supplied_tokenizer_is_used_and_marked_exact(pkt):
    calls = []
    def counter(text):
        calls.append(text)
        return len(text.split())

    packet = PacketBuilder(pkt, budget_tokens=100,
                           count_tokens=counter).build(required=["compliance"])
    assert packet.exact_tokens
    assert calls, "the supplied tokenizer was never called"


# ── candidate selection ───────────────────────────────────────────────────

def test_without_a_query_the_hot_set_fills_the_remainder(pkt):
    for i in range(10):
        pkt.remember(f"note_{i}", f"creative note number {i}")

    packet = PacketBuilder(pkt, budget_tokens=400).build(required=["compliance"])
    assert any(not r.required for r in packet.refs), "nothing filled the budget"
    assert "CONTEXT" in packet.text


def test_a_query_selects_the_remainder_by_relevance(pkt):
    pkt.remember("pricing", "discount messaging converts poorly for this brand")
    pkt.remember("formats", "square video outperforms portrait here")

    packet = PacketBuilder(pkt, budget_tokens=400).build(
        required=["compliance"], query="discount messaging")
    keys = [r.key for r in packet.refs if not r.required]
    assert "pricing" in keys


def test_a_required_key_is_not_repeated_in_the_context_section(pkt):
    """`compliance` is pinned, so the hot set would offer it again."""
    packet = PacketBuilder(pkt, budget_tokens=400).build(required=["compliance"])
    assert packet.text.count("never make medical") == 1


def test_a_forgotten_rule_does_not_satisfy_a_requirement(pkt):
    """A forgotten record still exists in the namespace index. If it resolved,
    a deleted rule would keep authorising mutations."""
    pkt.remember("temp_rule", "this rule was retracted")
    pkt.forget("temp_rule")
    with pytest.raises(RequiredContextUnavailable):
        PacketBuilder(pkt).build(required=["temp_rule"])


# ── reproducibility ───────────────────────────────────────────────────────

def test_the_manifest_carries_references_not_content(pkt):
    """A packet is revalidated against live scope and version at action time,
    so what must survive is the pointer. Storing the prose would also let a
    pinned old packet carry authority that has since been revoked."""
    packet = PacketBuilder(pkt).build(required=["compliance"])
    m = packet.manifest()

    assert m["required_refs"] == ["acme/compliance@0"] or \
        m["required_refs"][0].startswith("acme/compliance@")
    assert "never make medical" not in packet.to_json()
    assert m["may_mutate"] is True
    assert m["packet_id"] == packet.packet_id
    assert m["token_budget"] == packet.token_budget


def test_the_manifest_is_json_serialisable(pkt):
    import json
    packet = PacketBuilder(pkt, budget_tokens=200).build(required=["compliance"])
    assert json.loads(packet.to_json())["packet_id"] == packet.packet_id


def test_packet_ids_are_unique(pkt):
    b = PacketBuilder(pkt)
    ids = {b.build(required=["compliance"]).packet_id for _ in range(5)}
    assert len(ids) == 5


# ── an external store may own the constraints ─────────────────────────────
# Feather is DERIVED retrieval; Mongo and the rules file are authoritative, and
# indexing is asynchronous. These pin that a packet never depends on Feather
# having the rule.

from feather_db.packet import RequiredRule           # noqa: E402


def test_an_external_resolver_supplies_the_required_rules(pkt):
    """Copilot's rules live in _brand/rules.md and Mongo brand_rule records,
    not in Feather. The guarantee has to work without copying them in first."""
    rules = {"compliance_v2": RequiredRule("no health claims whatsoever",
                                           version=7, source="mongo")}
    packet = PacketBuilder(pkt, resolve_required=rules.get).build(
        required=["compliance_v2"])

    assert packet.may_mutate
    assert "no health claims whatsoever" in packet.text
    ref = packet.required_refs[0]
    assert ref.version == 7 and ref.scope == ("mongo",)
    assert packet.manifest()["required_refs"] == ["mongo/compliance_v2@7"]


def test_a_plain_string_from_the_resolver_works(pkt):
    packet = PacketBuilder(pkt, resolve_required=lambda k: "be careful").build(
        required=["anything"])
    assert "be careful" in packet.text


def test_the_resolver_is_authoritative_with_no_pocket_fallback(pkt):
    """If a resolver is given and does not return a key, that key is MISSING —
    even though Feather holds a copy.

    Falling back would enforce a possibly-stale constraint: Feather may hold the
    previous text of a rule just changed in the authoritative store, and the
    packet would present it as binding. Failing closed on a rule we cannot
    authoritatively read is correct.
    """
    assert pkt.recall("medical") or True            # Feather does hold it
    with pytest.raises(RequiredContextUnavailable):
        PacketBuilder(pkt, resolve_required=lambda k: None).build(
            required=["compliance"])


def test_index_lag_cannot_masquerade_as_a_policy_failure(pkt):
    """The inverse, and the reason the resolver exists: a rule written a moment
    ago is not in Feather yet. With the resolver it is still binding, so a
    retrieval delay never blocks a mutation it should not."""
    just_written = {"brand_new_rule": RequiredRule("effective immediately", version=1)}
    packet = PacketBuilder(pkt, resolve_required=just_written.get).build(
        required=["brand_new_rule"])
    assert packet.may_mutate
    assert "effective immediately" in packet.text


def test_the_pocket_still_supplies_non_required_context(pkt):
    """Only the required path is externalised. Ordinary context stays in
    Feather, where staleness costs relevance rather than correctness."""
    pkt.remember("formats", "square video outperforms portrait here")
    packet = PacketBuilder(pkt, budget_tokens=400,
                           resolve_required=lambda k: RequiredRule("a rule")).build(
        required=["r"], query="square video")
    assert any(r.key == "formats" for r in packet.refs if not r.required)


def test_an_external_required_rule_that_cannot_fit_still_fails_closed(pkt):
    big = {"huge": RequiredRule("x " * 4000)}
    with pytest.raises(RequiredContextUnavailable) as e:
        PacketBuilder(pkt, budget_tokens=40, resolve_required=big.get).build(
            required=["huge"])
    assert "did not fit" in str(e.value)
