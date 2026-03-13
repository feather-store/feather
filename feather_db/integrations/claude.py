"""
Claude Connector — Feather DB × Anthropic
==========================================
Exposes Feather DB tools in Anthropic's tool_use format.

Quick start:
    import anthropic
    from feather_db.integrations.claude import ClaudeConnector

    conn  = ClaudeConnector(db_path="my.feather", dim=3072, embedder=my_embed_fn)
    client = anthropic.Anthropic()

    response = client.messages.create(
        model="claude-opus-4-6",
        max_tokens=4096,
        tools=conn.tools(),
        messages=[{"role": "user", "content": "Why is our FD CTR dropping?"}],
    )

    # Agent loop — run until model stops using tools
    messages = [{"role": "user", "content": "Why is our FD CTR dropping?"}]
    final    = conn.run_loop(client, messages, model="claude-opus-4-6")
    print(final)
"""

from __future__ import annotations
from typing import Any, Callable, Optional

from .base import FeatherTools, TOOL_SPECS


class ClaudeConnector(FeatherTools):
    """Feather DB tools formatted for Anthropic's tool_use API."""

    def tools(self) -> list[dict]:
        """
        Returns a list of tool definitions ready to pass to
        `client.messages.create(tools=...)`.
        """
        result = []
        for spec in TOOL_SPECS:
            properties = {}
            for pname, pdef in spec["parameters"].items():
                prop: dict[str, Any] = {"type": pdef["type"]}
                if "description" in pdef:
                    prop["description"] = pdef["description"]
                if "enum" in pdef:
                    prop["enum"] = pdef["enum"]
                properties[pname] = prop

            result.append({
                "name":        spec["name"],
                "description": spec["description"],
                "input_schema": {
                    "type":       "object",
                    "properties": properties,
                    "required":   spec.get("required", []),
                },
            })
        return result

    def process_response(self, response) -> tuple[bool, list[dict]]:
        """
        Process a claude response object.
        Returns (done, tool_result_messages).

        done=True  → model finished (stop_reason != "tool_use")
        done=False → tool_result_messages should be appended to the conversation
        """
        if response.stop_reason != "tool_use":
            return True, []

        tool_results = []
        for block in response.content:
            if block.type != "tool_use":
                continue
            result_content = self.handle(block.name, block.input)
            tool_results.append({
                "type":        "tool_result",
                "tool_use_id": block.id,
                "content":     result_content,
            })

        return False, [{
            "role":    "user",
            "content": tool_results,
        }]

    def run_loop(
        self,
        client,
        messages: list[dict],
        model: str = "claude-opus-4-6",
        max_tokens: int = 4096,
        max_rounds: int = 10,
        system: Optional[str] = None,
        verbose: bool = True,
    ) -> str:
        """
        Run the full agent loop until the model stops calling tools.
        Returns the final text response.

        Parameters
        ----------
        client      anthropic.Anthropic() instance
        messages    Initial message list (user turn)
        model       Claude model ID
        max_tokens  Max tokens per response
        max_rounds  Safety limit on tool call rounds
        system      Optional system prompt
        verbose     Print tool calls and results as they happen
        """
        kwargs: dict[str, Any] = {
            "model":      model,
            "max_tokens": max_tokens,
            "tools":      self.tools(),
            "messages":   messages,
        }
        if system:
            kwargs["system"] = system

        for round_n in range(max_rounds):
            response = client.messages.create(**kwargs)

            if verbose:
                for block in response.content:
                    if hasattr(block, "type") and block.type == "tool_use":
                        print(f"  [tool_use] {block.name}({json_compact(block.input)})")

            done, tool_messages = self.process_response(response)

            if done:
                # Extract final text
                for block in response.content:
                    if hasattr(block, "text"):
                        return block.text
                return ""

            if verbose:
                for tm in tool_messages:
                    for tr in tm.get("content", []):
                        preview = str(tr.get("content", ""))[:120]
                        print(f"  [tool_result] {preview}...")

            # Append assistant response + tool results
            kwargs["messages"] = (
                kwargs["messages"]
                + [{"role": "assistant", "content": response.content}]
                + tool_messages
            )

        return "[max_rounds reached]"


def json_compact(d: dict) -> str:
    import json
    return json.dumps(d, separators=(",", ":"))
