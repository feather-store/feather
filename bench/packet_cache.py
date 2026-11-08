"""Prompt-cache reuse of context packets: how much of each request a provider
prefix cache serves, by where the packet goes in the prompt.

    python -m bench.packet_cache                         # offline, no keys
    python -m bench.packet_cache --anthropic claude-opus-5-5 --openai gpt-4.1-mini

A session: one agent pocket (org rules, pinned notes, a few hundred notes) and
`--turns` questions, each with its own packet,
`PacketBuilder.build(required=RULES, query=question)`. The same packets, three
layouts:

  whole       system = instructions + packet.text; history; the question.
              The packet in the system prompt, where agents usually put memory.
  split-keep  system = instructions + packet.stable_text; the question turn =
              packet.volatile_text + the question, kept in the history.
  split       the same, but earlier turns keep only their question: what was
              recalled for a turn is sent with that turn alone.

Offline, the reusable part of a request is its longest common prefix with an
earlier request of the session: the most any prefix cache could serve.
`--anthropic MODEL` sends the requests with max_tokens=0 (prefill only, no
output billed), breakpoints after the system prompt, on the last assistant
turn and on the newest message, and reports usage.cache_read_input_tokens.
`--openai MODEL` reports usage.prompt_tokens_details.cached_tokens (automatic
caching, prompts of 1024+ tokens); `--gemini MODEL` Gemini's implicit cache,
usage_metadata.cached_content_token_count. Assistant replies are stand-in text of
--reply-words words: what is measured is the prompt, not the answers.

`billed` is input in uncached-token equivalents: a cache read at --read-price
(0.1; Claude Opus 5.5 is 0.05) and, for Anthropic, a write at 1.25. A higher
hit rate is not a saving by itself: split-keep caches more than whole but
sends more, because every turn's recalled memory stays in the history.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import tempfile
import time
from pathlib import Path

from feather_db import DB
from feather_db.packet import PacketBuilder, _default_counter
from feather_db.pocket import pocket

RULES = {
    "claims":      "Never state or imply medical, financial or legal outcomes.",
    "pricing":     "Quote prices only from the current price sheet; never estimate.",
    "privacy":     "Do not repeat a customer's personal data back to them unasked.",
    "tone":        "Plain, direct sentences. No exclamation marks, no emoji.",
    "escalation":  "Hand refunds above $500 and every legal threat to a human.",
}
PINNED = {
    "brand":   "Brand voice: confident, warm, specific; we say 'members', never 'users'.",
    "persona": "The account owner is Dana Ruiz, head of growth, who prefers short answers.",
    "goal":    "Q4 goal: lift trial-to-paid conversion from 9% to 12% without discounts.",
}
TOPICS = ["onboarding", "pricing page", "churn", "email campaign", "referral",
          "trial length", "checkout", "mobile app", "support backlog", "webinar",
          "case study", "landing page", "retention", "upsell", "survey"]
VERBS = ["tested", "reviewed", "shipped", "paused", "measured", "redesigned",
         "discussed", "rolled back", "expanded", "audited"]
QUESTIONS = [
    "What did we learn from the last {t} test?",
    "Draft two options for the next {t} change.",
    "Which {t} ideas were rejected and why?",
    "Summarise where the {t} work stands.",
    "What should I tell Dana about the {t} numbers?",
]
INSTRUCTIONS = "\n".join([
    "You are the growth assistant for a subscription software company.",
    "You help the growth team plan, run and review experiments.",
    "Work from the memory provided. When memory does not cover something, say so.",
    "Answer in plain prose; use a short list only for options or steps.",
    "When you recommend an experiment, give the hypothesis, the metric, the",
    "minimum sample size and the decision rule before anything else.",
    "Prefer reversible changes. Flag anything that touches pricing or billing.",
    "Never invent numbers; cite the memory line a number came from.",
    "If two memories conflict, prefer the newer one and say that they conflict.",
    "Keep answers under 200 words unless asked for more.",
] * 6)
REPLY_WORDS = ("the test ran for two weeks and conversion moved by about one point "
               "which is inside the noise for this sample so I would keep it running "
               "and decide after the next cohort closes while watching churn and refunds").split()


def reply(turn: int, words: int) -> str:
    """A stand-in answer of `words` words, different every turn."""
    rng = random.Random(turn)
    return f"Answer {turn}: " + " ".join(rng.choice(REPLY_WORDS) for _ in range(words)) + "."


def build_memory(path: str, n_notes: int, seed: int):
    rng = random.Random(seed)
    db = DB.open(path, dim=8)
    org = pocket(db, "acme", budget_tokens=4000)
    for k, v in RULES.items():
        org.remember(k, v)
    for k, v in PINNED.items():
        org.remember(k, v, pinned=True)
    agent = pocket(db, "acme.growth", budget_tokens=4000)
    for i in range(n_notes):
        t, v = rng.choice(TOPICS), rng.choice(VERBS)
        agent.remember(f"n{i}", f"We {v} the {t} on week {rng.randint(1, 52)}: "
                                f"conversion moved {rng.uniform(-3, 4):+.1f} points; "
                                f"{rng.choice(['keep it', 'revert', 'needs more data'])}.")
    return db, agent


def session(agent, turns: int, seed: int) -> list:
    rng = random.Random(seed + 1)
    builder = PacketBuilder(agent, budget_tokens=2500)
    out = []
    for _ in range(turns):
        q = rng.choice(QUESTIONS).format(t=rng.choice(TOPICS))
        out.append((q, builder.build(required=list(RULES), query=q, k=25)))
    return out


def requests(turns: list, layout: str, reply_words: int) -> list:
    """Each request as (system blocks, messages), in the provider-neutral shape
    both clients are built from."""
    out, history = [], []
    for n, (q, pkt) in enumerate(turns):
        if layout == "whole":
            system, ask = [INSTRUCTIONS + "\n\n" + pkt.text], q
        else:
            system = [INSTRUCTIONS + "\n\n" + pkt.stable_text]
            ask = f"{pkt.volatile_text}\n\n{q}" if pkt.volatile_text else q
        msgs = history + [{"role": "user", "content": ask}]
        out.append((system, msgs))
        kept = q if layout == "split" else ask
        history = history + [{"role": "user", "content": kept},
                             {"role": "assistant", "content": reply(n, reply_words)}]
    return out


def flatten(system: list, msgs: list) -> str:
    return "\n\n".join(system) + "".join(f"\n\n<{m['role']}>\n{m['content']}" for m in msgs)


def offline(reqs: list, count) -> dict:
    texts = [flatten(*r) for r in reqs]
    total = cached = 0
    for i, t in enumerate(texts):
        lcp = max((len(os.path.commonprefix([t, p])) for p in texts[:i]), default=0)
        total += count(t)
        cached += count(t[:lcp]) if lcp else 0
    return {"tokens": total, "cached": cached}


def anthropic_run(reqs: list, model: str) -> dict:
    import anthropic
    client = anthropic.Anthropic()
    bp = {"type": "ephemeral"}

    def block(text: str, cached: bool) -> list:
        b = {"type": "text", "text": text}
        return [{**b, "cache_control": bp} if cached else b]

    tot = {"tokens": 0, "cached": 0, "written": 0}
    for system, msgs in reqs:
        # Breakpoints: end of the system prompt, the last assistant turn (the
        # history every later request repeats) and the newest message.
        last_reply = max((i for i, m in enumerate(msgs) if m["role"] == "assistant"),
                         default=-1)
        messages = [{"role": m["role"],
                     "content": block(m["content"], i == last_reply or i == len(msgs) - 1)}
                    for i, m in enumerate(msgs)]
        r = client.messages.create(model=model, max_tokens=0,
                                   system=block(system[0], True), messages=messages)
        u = r.usage
        read, wrote = u.cache_read_input_tokens or 0, u.cache_creation_input_tokens or 0
        tot["tokens"] += u.input_tokens + read + wrote
        tot["cached"] += read
        tot["written"] += wrote
    return tot


def openai_run(reqs: list, model: str, key: str) -> dict:
    from openai import OpenAI
    client = OpenAI()
    tot = {"tokens": 0, "cached": 0}
    for system, msgs in reqs:
        r = client.chat.completions.create(
            model=model, max_completion_tokens=16, prompt_cache_key=key,
            messages=[{"role": "system", "content": "\n\n".join(s for s in system if s)}] + msgs)
        tot["tokens"] += r.usage.prompt_tokens
        tot["cached"] += (r.usage.prompt_tokens_details.cached_tokens or 0)
        time.sleep(0.2)
    return tot


def gemini_run(reqs: list, model: str) -> dict:
    """Gemini's implicit cache (2.5 models, on by default, prompts of 1024+
    tokens): usage_metadata.cached_content_token_count."""
    from google import genai
    from google.genai import types
    client = genai.Client(api_key=os.environ.get("GOOGLE_API_KEY")
                          or os.environ.get("GEMINI_API_KEY"))
    tot = {"tokens": 0, "cached": 0}
    for system, msgs in reqs:
        contents = [types.Content(role="model" if m["role"] == "assistant" else "user",
                                  parts=[types.Part(text=m["content"])]) for m in msgs]
        r = client.models.generate_content(
            model=model, contents=contents,
            config=types.GenerateContentConfig(
                system_instruction=system[0], max_output_tokens=1,
                thinking_config=types.ThinkingConfig(thinking_budget=0)))
        u = r.usage_metadata
        tot["tokens"] += u.prompt_token_count or 0
        tot["cached"] += u.cached_content_token_count or 0
        time.sleep(0.5)
    return tot


# A cached token's price relative to an uncached one, per provider.
READ_PRICE = {"offline": None, "anthropic": None, "openai": 0.25, "gemini": 0.25}


def billed(t: dict, read_price: float, write_price: float) -> float:
    """Input in uncached-token equivalents."""
    written = t.get("written", 0)
    uncached = t["tokens"] - t["cached"] - written
    return uncached + read_price * t["cached"] + write_price * written


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--turns", type=int, default=12)
    ap.add_argument("--notes", type=int, default=400)
    ap.add_argument("--reply-words", type=int, default=180,
                    help="length of each stand-in assistant answer")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--anthropic", metavar="MODEL", help="also measure on the Claude API")
    ap.add_argument("--openai", metavar="MODEL", help="also measure on the OpenAI API")
    ap.add_argument("--gemini", metavar="MODEL",
                    help="also measure Gemini's implicit cache (e.g. gemini-2.5-flash)")
    ap.add_argument("--read-price", type=float, default=0.1,
                    help="cache read price relative to input, offline and Claude "
                         "(Claude Opus 5.5: 0.05); OpenAI and Gemini use 0.25")
    ap.add_argument("--out", help="write the results as JSON")
    a = ap.parse_args(argv)

    d = tempfile.mkdtemp()
    db, agent = build_memory(str(Path(d) / "m.feather"), a.notes, a.seed)
    turns = session(agent, a.turns, a.seed)
    stable = [p.stable_tokens for _, p in turns]
    print(f"{a.turns} turns; packet {sum(p.tokens_used for _, p in turns) // a.turns} tokens "
          f"on average, of which stable {sum(stable) // a.turns}; "
          f"stable text identical across turns: {len({p.stable_text for _, p in turns}) == 1}")
    count, _exact = _default_counter()
    results = {}
    for layout in ("whole", "split-keep", "split"):
        reqs = requests(turns, layout, a.reply_words)
        results[f"offline/{layout}"] = offline(reqs, count)
        if a.anthropic:
            results[f"anthropic/{layout}"] = anthropic_run(reqs, a.anthropic)
        if a.openai:
            results[f"openai/{layout}"] = openai_run(
                reqs, a.openai, f"feather-bench-{layout}-{a.seed}-{int(time.time())}")
        if a.gemini:
            results[f"gemini/{layout}"] = gemini_run(reqs, a.gemini)
    db.close()

    print("\n| measured | layout | input tokens | cached | cached % | billed | vs whole |")
    print("|---|---|---:|---:|---:|---:|---:|")
    base = {}
    for name, t in results.items():
        where, layout = name.split("/")
        b = billed(t, READ_PRICE[where] or a.read_price,
                   1.25 if where == "anthropic" else 0.0)
        base.setdefault(where, b)
        t["billed"] = round(b)
        print(f"| {where} | {layout} | {t['tokens']:,} | {t['cached']:,} | "
              f"{100 * t['cached'] / t['tokens']:.0f}% | {b:,.0f} | "
              f"{b / base[where]:.2f} |")
    if a.out:
        Path(a.out).write_text(json.dumps({"args": vars(a), "results": results}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
