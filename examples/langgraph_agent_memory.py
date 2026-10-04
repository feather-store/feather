"""Feather as long-term memory for a LangGraph agent.

Runs offline: no API key, no model download. The "LLM" is a stub so the thing
being demonstrated is the memory, not the generation.

The point of this file is the one line that adds memory to an existing graph:

    graph = builder.compile(store=FeatherStore("agent_memory.feather", ...))

Everything else here is an ordinary LangGraph agent. Run it twice — the second
run starts with a brand new thread, and the agent still knows who you are,
because `BaseStore` is cross-thread by definition and Feather persists it to one
file on disk.

    python examples/langgraph_agent_memory.py
"""
import os
import shutil
import tempfile

import numpy as np
from langgraph.graph import StateGraph, START, END
from typing_extensions import TypedDict

from feather_db.integrations import AgentMemory
from feather_db.integrations.langgraph_store import FeatherStore

ORG, USER = "hawky", "user_842"


def embedder(dim=64):
    """A bag-of-words stand-in so the demo needs no API key. In production pass
    a real embedder — `feather_db.integrations.embedders.make_embedder`."""
    def embed(text: str):
        v = np.zeros(dim, dtype=np.float32)
        for tok in str(text).lower().split():
            v[hash(tok) % dim] += 1.0
        n = np.linalg.norm(v)
        return v / n if n else v
    return embed


class State(TypedDict):
    message: str
    reply: str


# ── the agent ─────────────────────────────────────────────────────────────

def remember(state: State, *, store) -> dict:
    """Capture anything durable the user just said."""
    mem = AgentMemory(store, org=ORG, user=USER)
    text = state["message"]

    if "prefer" in text.lower():
        mem.preferences.set("comms", text)
    mem.episodes.record(text)
    return {}


def respond(state: State, *, store) -> dict:
    """Answer using what is already known. A real agent puts this in the prompt."""
    mem = AgentMemory(store, org=ORG, user=USER)

    known = mem.preferences.get("comms")
    history = mem.episodes.recent(3)

    if known:
        reply = (f"I remember: {known['text']!r}. "
                 f"We have spoken {len(history)} time(s) recently.")
    else:
        reply = "I do not know anything about you yet."
    return {"reply": reply}


def build_graph(store):
    builder = StateGraph(State)
    builder.add_node("remember", remember)
    builder.add_node("respond", respond)
    builder.add_edge(START, "remember")
    builder.add_edge("remember", "respond")
    builder.add_edge("respond", END)
    #                        ↓ the only line that adds memory
    return builder.compile(store=store)


def main() -> None:
    workdir = tempfile.mkdtemp()
    path = os.path.join(workdir, "agent_memory.feather")
    try:
        store = FeatherStore(path, dim=64, embed=embedder())
        graph = build_graph(store)

        print("── session 1 " + "─" * 50)
        out = graph.invoke({"message": "I prefer written async updates, not calls"},
                           config={"configurable": {"thread_id": "conv-1"}})
        print("  user  : I prefer written async updates, not calls")
        print("  agent :", out["reply"])

        # A completely separate conversation. A checkpointer would remember
        # nothing here — thread state does not cross threads. The store does.
        print("\n── session 2, new thread " + "─" * 37)
        out = graph.invoke({"message": "what do you know about me?"},
                           config={"configurable": {"thread_id": "conv-2"}})
        print("  user  : what do you know about me?")
        print("  agent :", out["reply"])

        # And a new process entirely: the file is the memory.
        store.close()
        print("\n── after restart, new store object " + "─" * 27)
        reopened = FeatherStore(path, dim=64, embed=embedder())
        mem = AgentMemory(reopened, org=ORG, user=USER)
        print("  held  :", mem.summary())
        print("  recall:", [v["text"] for v in mem.recall("how to contact", limit=1)])
        reopened.close()

        print(f"\n  one file, {os.path.getsize(path):,} bytes: {os.path.basename(path)}")
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


if __name__ == "__main__":
    main()
