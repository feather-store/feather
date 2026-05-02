"""ContradictionResolver — detect-only conflict finder for incoming facts.

Phase 9.1 Week 3. Designed for ingest-time triggering: a new Fact arrives,
the resolver checks it against a candidate set of existing facts (typically
fetched by subject+predicate match from the DB) and surfaces conflicts as
ContradictionFinding instances. **Never auto-resolves.** The marketing-team
moat is provenance + audit trail — silent merges break that.

Two-layer detection:
1. Rule pre-filter (cheap, no LLM): groups by canonical (subject,
   predicate); same SP + different object = candidate finding. Numeric
   tolerance + temporal awareness eliminates obvious non-conflicts
   (4.50% vs 4.51%, or "old → new" supersedure).
2. LLM severity (optional): when a provider is supplied, classifies
   each rule-flagged finding as definite/probable/possible and adds
   rationale. Cheap because rule pre-filter has already pruned 99% of
   pairs.
"""
from __future__ import annotations
import re
import time
from typing import Optional

from .base import ContradictionFinding, Fact
from ._jsonparse import extract_json


_NUMERIC_RE = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)")
_CURRENCY_PREFIX_RE = re.compile(r"^[$€£¥₹]\s*")


CONTRADICTION_SCORE_SYSTEM_PROMPT = """\
You classify candidate fact contradictions for severity.

Each candidate is a pair: a NEW fact and an EXISTING fact with the same
subject+predicate but a different object. Decide whether they truly
contradict.

Severity:
- definite  — same factual claim, different answer; both can't be true.
- probable  — likely contradict but ambiguous (e.g. units, scope).
- possible  — could be temporal supersedure, different scope, or
              two true measurements over different windows.

Suggested resolution:
- supersedes — new fact replaces old (clear ordering, e.g. an updated
               metric or a corrected value).
- review     — surface to human reviewer.
- merge      — both refer to the same value within tolerance.

Output JSON array, one entry per input pair, in the SAME ORDER as input:
[
  {"severity": "definite|probable|possible",
   "suggested_resolution": "supersedes|review|merge",
   "rationale": "one short sentence"},
  ...
]
"""


class ContradictionResolver:
    """Detect-only conflict finder.

    Args:
        provider:           optional LLMProvider for severity scoring. If
                            None, severity defaults from rule pre-filter
                            and rationale is auto-generated.
        numeric_tolerance:  relative tolerance for numeric object
                            comparison (default 0.02 = 2%). Within
                            tolerance → no contradiction.
        ignore_supersedure: if True, drop findings where new_fact has a
                            later valid_at than the candidate (treat as
                            supersedure, not contradiction). Default
                            False — surface as 'possible' so the audit
                            trail keeps both.
        max_tokens:         LLM completion budget.
        max_retries:        retries on provider exception.

    Example:
        >>> from feather_db.extractors import ContradictionResolver, Fact
        >>> resolver = ContradictionResolver()  # rule-only
        >>> new = Fact("Acme Summer Sale", "had_CTR", "4.5%")
        >>> existing = [
        ...     Fact("Acme Summer Sale", "had_CTR", "4.2%",
        ...          valid_at=1712000000),
        ... ]
        >>> findings = resolver.detect(new, existing)
        >>> findings[0].severity   # 'probable'  (numeric, outside tol)
        >>> findings[0].rationale  # auto-generated from rule path
    """

    name = "contradiction_resolver"
    version = "0.1.0"

    def __init__(self, provider=None, *,
                 numeric_tolerance: float = 0.02,
                 ignore_supersedure: bool = False,
                 max_tokens: int = 800,
                 max_retries: int = 2):
        self._provider = provider
        self._tolerance = numeric_tolerance
        self._ignore_supersedure = ignore_supersedure
        self._max_tokens = max_tokens
        self._max_retries = max_retries

    # ── Public API ──────────────────────────────────────────────────

    def detect(self, new_fact: Fact,
               candidates: list[Fact],
               *,
               context: Optional[dict] = None) -> list[ContradictionFinding]:
        """Return contradiction findings for `new_fact` vs `candidates`.

        Empty list when there are no rule-flagged conflicts. Always-empty
        when candidates is empty.
        """
        if not candidates:
            return []
        ctx = context or {}
        on_error = ctx.get("on_error")
        now_ts = int(time.time())

        rule_findings = self._rule_filter(new_fact, candidates, now_ts)
        if not rule_findings:
            return []
        if self._ignore_supersedure:
            rule_findings = [f for f in rule_findings
                             if f.suggested_resolution != "supersedes"]
        if not rule_findings or self._provider is None:
            return rule_findings

        # LLM severity scoring
        scored = self._llm_score(rule_findings, ctx, on_error)
        return scored if scored else rule_findings

    # ── Rule pre-filter ─────────────────────────────────────────────

    def _rule_filter(self, new_fact: Fact, candidates: list[Fact],
                     now_ts: int) -> list[ContradictionFinding]:
        out: list[ContradictionFinding] = []
        nsubj = _norm(new_fact.subject)
        npred = _norm(new_fact.predicate)
        nobj = (new_fact.object or "").strip()
        nobj_lower = nobj.lower()
        nobj_num = _maybe_num(nobj)

        for c in candidates:
            if c is new_fact:
                continue
            if _norm(c.subject) != nsubj:
                continue
            if _norm(c.predicate) != npred:
                continue
            cobj = (c.object or "").strip()
            if not cobj:
                continue
            if cobj.lower() == nobj_lower:
                continue  # same object → not a conflict

            cobj_num = _maybe_num(cobj)
            # Numeric tolerance escape
            if nobj_num is not None and cobj_num is not None:
                base = max(abs(nobj_num), abs(cobj_num), 1e-9)
                rel = abs(nobj_num - cobj_num) / base
                if rel <= self._tolerance:
                    out.append(ContradictionFinding(
                        new_fact=new_fact,
                        conflicting_with=c,
                        severity="possible",
                        rationale=(f"numeric values within {self._tolerance*100:.1f}% "
                                   f"({nobj_num} vs {cobj_num})"),
                        suggested_resolution="merge",
                        detected_at=now_ts,
                    ))
                    continue
                # Outside tolerance → numeric conflict
                severity, suggestion, rationale = self._classify_temporal(
                    new_fact, c,
                    base_severity="probable",
                    base_rationale=(f"numeric values differ beyond tolerance "
                                    f"({nobj_num} vs {cobj_num})"),
                )
            else:
                # Non-numeric: stronger contradiction signal
                severity, suggestion, rationale = self._classify_temporal(
                    new_fact, c,
                    base_severity="definite",
                    base_rationale=("same subject+predicate; different "
                                    "non-numeric object"),
                )

            out.append(ContradictionFinding(
                new_fact=new_fact,
                conflicting_with=c,
                severity=severity,
                rationale=rationale,
                suggested_resolution=suggestion,
                detected_at=now_ts,
            ))
        return out

    @staticmethod
    def _classify_temporal(new_fact: Fact, candidate: Fact, *,
                           base_severity: str,
                           base_rationale: str
                           ) -> tuple[str, str, str]:
        """Apply temporal interpretation to a candidate finding.

        Returns (severity, suggested_resolution, rationale).
        """
        nv = new_fact.valid_at
        cv = candidate.valid_at
        if nv and cv and nv > cv:
            return ("possible", "supersedes",
                    f"{base_rationale}; new fact has later valid_at "
                    f"({nv} > {cv})")
        if nv and cv and nv < cv:
            # New is older than candidate — odd; treat as 'possible' too.
            return ("possible", "review",
                    f"{base_rationale}; new fact has EARLIER valid_at "
                    f"({nv} < {cv}) — verify ordering")
        return (base_severity, "review", base_rationale)

    # ── LLM severity scoring ────────────────────────────────────────

    def _llm_score(self, findings: list[ContradictionFinding],
                   ctx: dict, on_error
                   ) -> Optional[list[ContradictionFinding]]:
        messages = [
            {"role": "system",
             "content": CONTRADICTION_SCORE_SYSTEM_PROMPT},
            {"role": "user",
             "content": self._user_message(findings, ctx)},
        ]

        raw = None
        last_err: Optional[Exception] = None
        for attempt in range(self._max_retries + 1):
            try:
                raw = self._provider.complete(
                    messages,
                    max_tokens=self._max_tokens,
                    temperature=0.0,
                )
                break
            except Exception as e:
                last_err = e
                if attempt < self._max_retries:
                    time.sleep(1.5 ** (attempt + 1))
        if raw is None:
            if on_error:
                on_error(
                    f"ContradictionResolver LLM call failed: {last_err}"
                )
            return None

        parsed, ok, _ = extract_json(raw)
        if not ok or not isinstance(parsed, list):
            if on_error:
                on_error(
                    f"ContradictionResolver JSON parse failed; "
                    f"raw[:200]={raw[:200]!r}"
                )
            return None

        out: list[ContradictionFinding] = []
        for f, item in zip(findings, parsed):
            if not isinstance(item, dict):
                out.append(f)
                continue
            sev = (item.get("severity") or f.severity).strip().lower()
            if sev not in {"definite", "probable", "possible"}:
                sev = f.severity
            res = (item.get("suggested_resolution")
                   or f.suggested_resolution).strip().lower()
            if res not in {"supersedes", "review", "merge"}:
                res = f.suggested_resolution
            rationale = item.get("rationale")
            if isinstance(rationale, str):
                rationale = rationale.strip() or f.rationale
            else:
                rationale = f.rationale
            out.append(ContradictionFinding(
                new_fact=f.new_fact,
                conflicting_with=f.conflicting_with,
                severity=sev,
                rationale=rationale,
                suggested_resolution=res,
                detected_at=f.detected_at,
            ))
        # If LLM returned fewer items than findings, pass through
        # the un-scored remainder unchanged.
        if len(out) < len(findings):
            out.extend(findings[len(out):])
        return out

    def _user_message(self, findings: list[ContradictionFinding],
                      ctx: dict) -> str:
        lines: list[str] = []
        if ctx.get("namespace"):
            lines.append(f"Namespace: {ctx['namespace']}")
            lines.append("")
        lines.append("CANDIDATE PAIRS:")
        for i, f in enumerate(findings):
            new_v = (f"valid_at={f.new_fact.valid_at}"
                     if f.new_fact.valid_at else "valid_at=?")
            old_v = (f"valid_at={f.conflicting_with.valid_at}"
                     if f.conflicting_with.valid_at else "valid_at=?")
            lines.append(f"[{i}]")
            lines.append(f"  NEW:      {f.new_fact.subject} | "
                         f"{f.new_fact.predicate} | "
                         f"{f.new_fact.object} ({new_v})")
            lines.append(f"  EXISTING: {f.conflicting_with.subject} | "
                         f"{f.conflicting_with.predicate} | "
                         f"{f.conflicting_with.object} ({old_v})")
        return "\n".join(lines)


# ── Helpers ─────────────────────────────────────────────────────────

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()


def _maybe_num(s: str) -> Optional[float]:
    """Parse leading number from a string ('4.5%', '$1,200', '3.2x' → 4.5,
    1200, 3.2). Returns None if no leading number.

    Strips a single optional currency-symbol prefix ($€£¥₹). Does NOT
    strip arbitrary leading text — a date like 'March 15, 2024' must
    not parse as 15.0 and quietly trigger a numeric comparison.
    """
    if not s:
        return None
    cleaned = s.replace(",", "").strip()
    cleaned = _CURRENCY_PREFIX_RE.sub("", cleaned)
    m = _NUMERIC_RE.match(cleaned)
    if not m:
        return None
    try:
        return float(m.group(0))
    except (TypeError, ValueError):
        return None
