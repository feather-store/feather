"""Context packets — required context guaranteed, everything else budgeted.

`Pocket.hot()` answers "what is this agent most likely to want", fits it to a
token budget by heat, and drops the rest. For a working set that is right. For
an operating constraint it is exactly wrong: a brand rule that says *never make
medical claims* is not more relevant because it was recalled often, and it must
not fall out of the prompt because forty creative notes outranked it. It will,
today, silently.

That is the gap Singularity's memory plan names (§1.4): *required constraints
need a separate guaranteed path and actual token accounting*. A packet is that
path. Required refs go in first and are never dropped; if they cannot be
resolved or cannot fit, the packet is **degraded** and the caller must not
perform a dependent mutation. Everything else competes for what is left, and
anything dropped is listed in `omitted` with a reason — the one thing a context
assembler must never do is lose content quietly.

    builder = PacketBuilder(pkt, budget_tokens=4000)
    packet  = builder.build(required=["compliance", "brand_safety"],
                            query="which hook should I use")

    if not packet.may_mutate:
        raise RuntimeError(packet.why_blocked())      # fail closed
    prompt = packet.text

Read-only help may degrade on purpose:

    packet = builder.build(required=[...], allow_degraded=True)
    # packet.may_mutate is False, packet.omitted says what is missing

`manifest()` is a reproducible record of what went in — refs, versions, budget,
omissions — so a decision can be replayed later. It deliberately stores
references rather than content: per the plan, a packet is revalidated against
authoritative scope and version at action time, and a stale packet must not be
able to extend expired authority.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable, Optional, Sequence

from feather_db import keys
from feather_db.pocket import (
    Pocket, SCOPE_SEP, estimate_tokens, _CACHE_KIND,
)

# Reasons a ref did not make it into the packet. Stable strings — callers and
# dashboards key off them.
MISSING = "missing"          # no such key in scope or any ancestor
BUDGET  = "budget"           # resolved, but did not fit
DUPLICATE = "duplicate"      # already included from a nearer scope


@dataclass(frozen=True)
class Ref:
    """A pointer to one memory, with the version it was read at."""
    key: str
    scope: tuple[str, ...] = ()
    # The record's version (Metadata.version) when the packet read it: moves on
    # every write to the record and never on a read. It used to be
    # recall_count, which every search changed and an edit of the rule did not,
    # so a replay could neither trust an unchanged ref nor see a changed one.
    version: Optional[int] = None
    tokens: int = 0
    required: bool = False

    def __str__(self) -> str:
        base = f"{SCOPE_SEP.join(self.scope)}/{self.key}" if self.scope else self.key
        return f"{base}@{self.version}" if self.version is not None else base


@dataclass(frozen=True)
class RequiredRule:
    """A required constraint supplied by an authoritative store.

    Returned by a `resolve_required` callable. `version` and `source` are
    recorded in the manifest so a decision can be revalidated against the store
    that owns the rule rather than against Feather's copy of it.
    """
    text: str
    version: Optional[int] = None
    source: str = "external"


@dataclass(frozen=True)
class Omission:
    ref: str
    reason: str
    tokens: int = 0
    required: bool = False

    def __str__(self) -> str:
        return f"{self.ref} ({self.reason})"


class RequiredContextUnavailable(RuntimeError):
    """Required context could not be included. Do not mutate.

    Raised by `build()` unless `allow_degraded=True`. The message names every
    missing ref and, when the cause is budget, the arithmetic — a caller that
    cannot see whether it asked for 9000 tokens of rules in a 4000-token budget
    cannot fix it.
    """

    def __init__(self, omissions: Sequence[Omission], budget: int, required_tokens: int):
        self.omissions = list(omissions)
        self.budget = budget
        self.required_tokens = required_tokens
        missing = [o for o in self.omissions if o.reason == MISSING]
        overrun = [o for o in self.omissions if o.reason == BUDGET]
        parts = []
        if missing:
            parts.append("not found: " + ", ".join(o.ref for o in missing))
        if overrun:
            parts.append(
                f"did not fit: {', '.join(o.ref for o in overrun)} "
                f"({required_tokens} required tokens in a {budget}-token budget)")
        if not parts:                      # should be unreachable; never raise blind
            parts.append(f"{len(self.omissions)} required ref(s) unavailable")
        super().__init__(
            "required context unavailable — "
            + "; ".join(parts)
            + ". Dependent mutations must be blocked; raise the budget, shorten "
              "the rules, or pass allow_degraded=True for read-only use.")


@dataclass
class ContextPacket:
    """What one execution was given, and what it was not."""
    packet_id: str
    as_of: float
    scope: tuple[str, ...]
    token_budget: int
    tokens_used: int
    refs: list[Ref] = field(default_factory=list)
    omitted: list[Omission] = field(default_factory=list)
    text: str = ""
    exact_tokens: bool = False     # False ⇒ tokens are a 4-chars-each estimate

    # ── the fail-closed contract ──────────────────────────────────────────
    @property
    def required_refs(self) -> list[Ref]:
        return [r for r in self.refs if r.required]

    @property
    def missing_required(self) -> list[Omission]:
        """Required context that is genuinely absent.

        A DUPLICATE omission is recorded against the required set — it tells you
        the caller listed a rule twice, which is worth seeing — but the rule was
        included once, so it is not missing. Counting it blocked every mutation
        for a caller whose only mistake was a repeated key, and produced an
        error message with nothing in it.
        """
        return [o for o in self.omitted
                if o.required and o.reason in (MISSING, BUDGET)]

    @property
    def may_mutate(self) -> bool:
        """False when any required context is absent.

        The plan's rule: if required instructions cannot fit or load, block the
        dependent mutation and surface the reason. Read-only help may proceed.
        """
        return not self.missing_required

    def why_blocked(self) -> str:
        if self.may_mutate:
            return ""
        return ("packet " + self.packet_id + " is missing required context: "
                + "; ".join(str(o) for o in self.missing_required))

    # ── reproducibility ───────────────────────────────────────────────────
    def manifest(self) -> dict:
        """References and versions, not content.

        A packet is revalidated against live scope, status and version at action
        time, so what has to survive is the pointer, not the prose. Storing the
        content would also let a pinned old packet carry authority that has
        since been revoked.
        """
        return {
            "packet_id": self.packet_id,
            "as_of": self.as_of,
            "scope": list(self.scope),
            "token_budget": self.token_budget,
            "tokens_used": self.tokens_used,
            "exact_tokens": self.exact_tokens,
            "required_refs": [str(r) for r in self.refs if r.required],
            "memory_refs": [str(r) for r in self.refs if not r.required],
            # The same refs, structured, for revalidate().
            "refs": [{"key": r.key, "scope": list(r.scope), "version": r.version,
                      "required": r.required} for r in self.refs],
            "omitted_context": [
                {"ref": o.ref, "reason": o.reason, "tokens": o.tokens,
                 "required": o.required} for o in self.omitted
            ],
            "may_mutate": self.may_mutate,
        }

    def to_json(self, **kw) -> str:
        return json.dumps(self.manifest(), **kw)

    def __repr__(self) -> str:
        state = "ok" if self.may_mutate else "DEGRADED"
        return (f"<ContextPacket {self.packet_id} {state} "
                f"{self.tokens_used}/{self.token_budget}tok "
                f"{len(self.refs)} refs, {len(self.omitted)} omitted>")


def _default_counter() -> tuple[Callable[[str], int], bool]:
    """Real token counting when tiktoken is installed, estimate otherwise.

    The estimate is 4 characters per token, which is close enough for English
    prose and wrong for JSON, code and non-Latin scripts — all of which appear
    in agent memory. `ContextPacket.exact_tokens` records which one was used, so
    a budget overrun can be attributed rather than guessed at.
    """
    try:
        import tiktoken
        enc = tiktoken.get_encoding("cl100k_base")
        return (lambda text: len(enc.encode(text))), True
    except Exception:
        return estimate_tokens, False


class PacketBuilder:
    """Assembles packets from a `Pocket`.

    Assembly order follows the plan: required policies first, then scoped
    candidates by relevance or heat, then fit the budget, then persist the
    manifest. The order is not cosmetic — required-first is what makes the
    guarantee possible at all.
    """

    def __init__(self, pocket: Pocket, *, budget_tokens: Optional[int] = None,
                 count_tokens: Optional[Callable[[str], int]] = None,
                 resolve_required: Optional[Callable[[str], Any]] = None,
                 required_header: str = "OPERATING CONSTRAINTS (binding)",
                 context_header: str = "CONTEXT"):
        """`resolve_required(key)` makes an external store authoritative for
        constraints, returning a `RequiredRule`, a plain string, or None.

        This matters more than it looks. Feather is *derived* retrieval — the
        authoritative memory and its revisions live elsewhere (Mongo, a rules
        file), and indexing is asynchronous. If required rules had to be found
        in Feather, two things would follow, both bad:

        1. Index lag becomes an outage. A rule written a second ago is not in
           Feather yet, so it reads as missing, so every dependent mutation is
           blocked. A retrieval delay would masquerade as a policy failure.
        2. A stale copy becomes enforceable. Feather might hold the *previous*
           text of a rule that was just changed, and a packet would present it
           as binding.

        So when a resolver is given it is authoritative and there is NO fallback
        to the pocket for required refs: a key the resolver does not return is
        missing, full stop. Failing closed on a rule we cannot authoritatively
        read is correct; silently substituting a possibly-stale copy is not.
        The pocket still supplies the non-required context, where staleness
        costs relevance rather than correctness.
        """
        self.pocket = pocket
        self.resolve_required = resolve_required
        self.budget = budget_tokens if budget_tokens is not None else pocket.budget
        if count_tokens is not None:
            self._count, self.exact = count_tokens, True
        else:
            self._count, self.exact = _default_counter()
        self.required_header = required_header
        self.context_header = context_header

    # ── resolution ────────────────────────────────────────────────────────
    def _resolve(self, key: str):
        """Find `key` in this scope or the nearest ancestor that has it.

        Nearest wins, which is what lets a campaign-level rule override a
        brand-level one without deleting the brand rule.

        O(levels) key lookups rather than a scan. By native key, not by an id
        hashed from it: under a collision the hashed id held ANOTHER key's
        record, and a required rule resolved to whatever that was.
        """
        for depth, scope in enumerate(self.pocket.scopes()):
            rid, meta = keys.get(self.pocket.db, SCOPE_SEP.join(scope), key)
            if meta is None:
                continue
            # A forgotten record and a cache entry both exist in the index but
            # are not memory. A required rule must never resolve to either.
            if meta.source in ("_forgotten", _CACHE_KIND):
                continue
            return rid, meta, scope, depth
        return None, None, (), 0

    def _resolve_required_one(self, key: str):
        """(text, version, scope) for a required key, or (None, None, ()).

        The external resolver wins outright when present — see __init__ for why
        there is deliberately no pocket fallback.
        """
        if self.resolve_required is not None:
            found = self.resolve_required(key)
            if found is None:
                return None, None, ()
            if isinstance(found, RequiredRule):
                return found.text, found.version, (found.source,)
            return str(found), None, ("external",)

        rid, meta, scope, _depth = self._resolve(key)
        if meta is None:
            return None, None, ()
        return meta.content, meta.version, tuple(scope)

    def build(self, *, required: Iterable[str] = (), query: Optional[str] = None,
              k: int = 10, budget_tokens: Optional[int] = None,
              allow_degraded: bool = False) -> ContextPacket:
        """Assemble a packet.

        `required` keys are included in order and never dropped. `query` selects
        the rest by relevance; without one, the pocket's hot set fills the
        remainder. Raises `RequiredContextUnavailable` unless `allow_degraded`.
        """
        budget = budget_tokens if budget_tokens is not None else self.budget
        now = time.time()
        packet = ContextPacket(
            packet_id="pkt_" + uuid.uuid4().hex[:12],
            as_of=now,
            scope=tuple(self.pocket.scopes()[0]),
            token_budget=budget,
            tokens_used=0,
            exact_tokens=self.exact,
        )

        # ── 1. required, first and unconditionally ────────────────────────
        seen: set[str] = set()
        required_blocks: list[str] = []
        required_tokens = 0
        for key in required:
            if key in seen:
                packet.omitted.append(Omission(key, DUPLICATE, 0, True))
                continue
            text, version, scope = self._resolve_required_one(key)
            if text is None:
                packet.omitted.append(Omission(key, MISSING, 0, True))
                continue
            block = f"- {text}"
            cost = self._count(block)
            required_tokens += cost
            seen.add(key)
            required_blocks.append(block)
            packet.refs.append(Ref(key, scope, version, cost, True))

        # Budget check on the WHOLE required set, not item by item: a caller
        # needs to know its rules do not fit, not watch them disappear one at a
        # time in whatever order it happened to list them.
        header_cost = self._count(self.required_header) + self._count(self.context_header)
        if required_tokens + header_cost > budget:
            for ref in list(packet.refs):
                if ref.required:
                    packet.omitted.append(Omission(str(ref), BUDGET, ref.tokens, True))
            packet.refs = [r for r in packet.refs if not r.required]
            required_blocks = []
            required_tokens = 0

        if packet.missing_required and not allow_degraded:
            raise RequiredContextUnavailable(
                packet.missing_required, budget,
                sum(o.tokens for o in packet.missing_required) or required_tokens)

        packet.tokens_used = required_tokens + (header_cost if required_blocks else 0)

        # ── 2. everything else competes for the remainder ─────────────────
        candidates = (self.pocket.recall(query, k=k) if query
                      else self.pocket.hot(budget))
        chosen: list[str] = []
        for item in candidates:
            if item.key in seen:
                continue
            block = f"- {item.content}"
            cost = self._count(block)
            if packet.tokens_used + cost > budget:
                packet.omitted.append(Omission(
                    str(Ref(item.key, item.scope, item.version)), BUDGET, cost, False))
                continue
            seen.add(item.key)
            chosen.append(block)
            packet.tokens_used += cost
            packet.refs.append(Ref(item.key, item.scope, item.version, cost, False))

        # ── 3. render ─────────────────────────────────────────────────────
        sections = []
        if required_blocks:
            sections.append(self.required_header + "\n" + "\n".join(required_blocks))
        if chosen:
            sections.append(self.context_header + "\n" + "\n".join(chosen))
        packet.text = "\n\n".join(sections)
        return packet

    # ── replay ────────────────────────────────────────────────────────────
    def revalidate(self, manifest: dict) -> dict:
        """Is what a packet was built from still what it would be built from?

        For each ref in `manifest` (a `ContextPacket.manifest()`, e.g. one that
        was persisted with a decision): the record must still be there, at the
        version the packet read. Versions move on every write to a record and
        never on a read, so an unchanged ref is one nobody has edited, however
        often it has been recalled since — and an edited rule is caught, which
        a recall counter could not do.

        Required refs are checked against the resolver when this builder has
        one (that is where build() read them), else against the pocket. Returns
        {"current", "may_mutate", "stale", "unverified"}: `stale` lists refs
        that are missing or changed ({ref, required, reason, was, now});
        `may_mutate` is False if the packet could not mutate then, or a
        required ref is stale now. A required rule the resolver returns without
        a version can only be checked for presence: it is listed in
        `unverified`.
        """
        stale, unverified = [], []
        for r in manifest.get("refs", []):
            scope, key, was, required = tuple(r["scope"]), r["key"], r["version"], r["required"]
            name = str(Ref(key, scope, was))
            if required and self.resolve_required is not None:
                _text, now, _ = self._resolve_required_one(key)
                if _text is None:
                    stale.append({"ref": name, "required": True, "reason": MISSING, "was": was, "now": None})
                elif was is None or now is None:
                    unverified.append(name)
                elif now != was:
                    stale.append({"ref": name, "required": True, "reason": "changed", "was": was, "now": now})
                continue
            _rid, meta = keys.get(self.pocket.db, SCOPE_SEP.join(scope), key)
            if meta is None or meta.source in ("_forgotten", _CACHE_KIND):
                stale.append({"ref": name, "required": required, "reason": MISSING, "was": was, "now": None})
            elif meta.version != was:
                stale.append({"ref": name, "required": required, "reason": "changed",
                              "was": was, "now": meta.version})
        blocked = any(x["required"] for x in stale)
        return {"current": not stale, "may_mutate": bool(manifest.get("may_mutate")) and not blocked,
                "stale": stale, "unverified": unverified}
