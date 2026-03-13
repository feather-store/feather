"""
OpenAI-Compatible Connector — Feather DB
=========================================
Exposes Feather DB tools in OpenAI's function-calling format.
Works with: OpenAI, Azure OpenAI, Groq, Mistral, Together AI,
            Ollama (OpenAI-compat mode), and any OpenAI-spec API.

Quick start:
    from openai import OpenAI
    from feather_db.integrations.openai_compat import OpenAIConnector

    conn   = OpenAIConnector(db_path="my.feather", dim=3072, embedder=my_embed_fn)
    client = OpenAI()

    # Single call
    response = client.chat.completions.create(
        model="gpt-4o",
        tools=conn.tools(),
        messages=[{"role": "user", "content": "Why is FD CTR dropping?"}],
    )

    # Full agent loop
    messages = [{"role": "user", "content": "Why is FD CTR dropping?"}]
    final    = conn.run_loop(client, messages, model="gpt-4o")
    print(final)

Alternate base_url (e.g. Groq):
    client = OpenAI(api_key=GROQ_KEY, base_url="https://api.groq.com/openai/v1")
    conn.run_loop(client, messages, model="llama-3.3-70b-versatile")
"""

from __future__ import annotations
import json
from typing import Any, Callable, Optional

from .base import FeatherTools, TOOL_SPECS


class OpenAIConnector(FeatherTools):
    """Feather DB tools in OpenAI function-calling format."""

    def tools(self) -> list[dict]:
        """
        Returns tool definitions for `client.chat.completions.create(tools=...)`.
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
                "type": "function",
                "function": {
                    "name":        spec["name"],
                    "description": spec["description"],
                    "parameters": {
                        "type":       "object",
                        "properties": properties,
                        "required":   spec.get("required", []),
                    },
                },
            })
        return result

    def process_response(self, response) -> tuple[bool, list[dict]]:
        """
        Process an OpenAI ChatCompletion response.
        Returns (done, tool_result_messages).
        """
        choice = response.choices[0]
        if choice.finish_reason != "tool_calls" or not choice.message.tool_calls:
            return True, []

        tool_results = []
        for tc in choice.message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            content = self.handle(tc.function.name, args)
            tool_results.append({
                "role":         "tool",
                "tool_call_id": tc.id,
                "name":         tc.function.name,
                "content":      content,
            })

        # Must include the assistant message with tool_calls before results
        assistant_msg = {
            "role":       "assistant",
            "content":    choice.message.content,
            "tool_calls": [
                {
                    "id":       tc.id,
                    "type":     "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments},
                }
                for tc in choice.message.tool_calls
            ],
        }
        return False, [assistant_msg] + tool_results

    def run_loop(
        self,
        client,
        messages: list[dict],
        model: str = "gpt-4o",
        max_tokens: int = 4096,
        max_rounds: int = 10,
        system: Optional[str] = None,
        verbose: bool = True,
    ) -> str:
        """Run the full agent loop. Returns the final text response."""
        all_messages = list(messages)
        if system:
            all_messages = [{"role": "system", "content": system}] + all_messages

        for round_n in range(max_rounds):
            response = client.chat.completions.create(
                model=model,
                max_tokens=max_tokens,
                tools=self.tools(),
                tool_choice="auto",
                messages=all_messages,
            )

            choice = response.choices[0]

            if verbose and choice.message.tool_calls:
                for tc in choice.message.tool_calls:
                    print(f"  [tool_call] {tc.function.name}({tc.function.arguments[:80]})")

            done, new_messages = self.process_response(response)

            if done:
                return choice.message.content or ""

            if verbose:
                for nm in new_messages:
                    if nm.get("role") == "tool":
                        print(f"  [tool_result] {nm['content'][:120]}...")

            all_messages.extend(new_messages)

        return "[max_rounds reached]"
