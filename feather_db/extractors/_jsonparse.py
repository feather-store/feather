"""Robust JSON extraction from LLM responses.

LLMs return JSON wrapped in many ways:
    - clean JSON
    - JSON in ```json``` fences
    - JSON in ``` fences without language hint
    - JSON preceded by prose ("Here is the JSON: ...")
    - JSON followed by prose ("[...] — let me know if you need more")
    - JSON cut off mid-output (token-budget exhaustion)

This module finds the first balanced JSON value (object or array) in a
string and parses it. Returns (parsed, ok, raw_str). If parsing fails
returns (None, False, the_substring_we_tried).
"""
from __future__ import annotations
import json
import re
from typing import Any


_FENCE_RE = re.compile(r"```(?:json)?\s*(.+?)```", re.DOTALL)


def extract_json(raw: str) -> tuple[Any, bool, str]:
    """Find and parse the first JSON value in `raw`.

    Strategy:
        1. If a fenced block contains JSON, prefer that.
        2. Otherwise, find the first '{' or '[' and walk to the matching
           closer, balancing braces/brackets and respecting strings.
        3. Try json.loads on the candidate substring.
        4. On failure, try common repairs (remove trailing commas).

    Returns:
        (parsed, ok, candidate_substring)
    """
    if not raw:
        return None, False, ""

    # Step 1: fenced block?
    m = _FENCE_RE.search(raw)
    candidates: list[str] = []
    if m:
        candidates.append(m.group(1).strip())

    # Step 2: scan for first balanced JSON
    candidates.extend(_balanced_candidates(raw))

    for cand in candidates:
        parsed = _try_parse(cand)
        if parsed is not None:
            return parsed, True, cand

    return None, False, candidates[0] if candidates else raw[:200]


def _balanced_candidates(raw: str) -> list[str]:
    """Yield the first balanced top-level JSON value (array OR object).

    Whichever opener appears earliest in the string wins — otherwise
    `[{"k":"v"}]` would pick the inner object before the outer array.
    """
    first_brace = raw.find("{")
    first_bracket = raw.find("[")
    if first_brace < 0 and first_bracket < 0:
        return []
    if first_brace < 0:
        first, opener, closer = first_bracket, "[", "]"
    elif first_bracket < 0:
        first, opener, closer = first_brace, "{", "}"
    elif first_brace < first_bracket:
        first, opener, closer = first_brace, "{", "}"
    else:
        first, opener, closer = first_bracket, "[", "]"

    out: list[str] = []
    end = _find_balanced(raw, first, opener, closer)
    if end > first:
        out.append(raw[first : end + 1])

    # Also expose the *other* opener as a fallback candidate, in case
    # the chosen one fails to parse (defensive).
    other_idx = first_bracket if opener == "{" else first_brace
    if other_idx >= 0 and other_idx != first:
        other_open, other_close = (("[", "]") if opener == "{" else ("{", "}"))
        other_end = _find_balanced(raw, other_idx, other_open, other_close)
        if other_end > other_idx:
            out.append(raw[other_idx : other_end + 1])
    return out


def _find_balanced(s: str, start: int, opener: str, closer: str) -> int:
    """Index of the closer that balances `s[start]`. Respects strings.
    Returns -1 if unbalanced."""
    depth = 0
    in_str = False
    escape = False
    for i in range(start, len(s)):
        c = s[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            escape = True
            continue
        if c == '"':
            in_str = not in_str
            continue
        if in_str:
            continue
        if c == opener:
            depth += 1
        elif c == closer:
            depth -= 1
            if depth == 0:
                return i
    return -1


def _try_parse(cand: str) -> Any:
    """Try json.loads with simple repairs."""
    try:
        return json.loads(cand)
    except json.JSONDecodeError:
        pass
    # Remove trailing commas before } or ]
    repaired = re.sub(r",(\s*[}\]])", r"\1", cand)
    try:
        return json.loads(repaired)
    except json.JSONDecodeError:
        return None
