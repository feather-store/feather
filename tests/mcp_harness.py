"""Drive an MCP server the way a client actually drives one.

The tests in `test_mcp_agent.py` call `server.call_tool(...)` directly. That
tests the Python functions; it does not test the server. Everything between the
function and the model — the JSON Schema the SDK derives from the type hints,
argument validation, the initialize handshake, result envelopes, error
signalling — is skipped, and that layer is exactly where an MCP server breaks
for a user while every unit test stays green. The previous `mcp_server` module
is the proof: it passed its own tests and raised on import against SDK 2.0.

So these run a real `ClientSession` against the server over an in-process
transport. Same protocol messages as Claude Desktop over stdio, no subprocess.

`raise_exceptions` stays False on purpose: a real client is not handed the
server's traceback, it is handed `isError` with text. Tests assert on what the
model would actually see.

Usage — write the test as a coroutine whose first parameter is `mcp`:

    @over_mcp
    async def test_something(mcp):
        assert "policy" in await mcp.call("context")

The decorator turns it back into an ordinary sync test taking the `server`
fixture, so no pytest-asyncio dependency and no event loop shared between tests.
"""
from __future__ import annotations

import asyncio
import functools
import inspect
import json
from contextlib import asynccontextmanager

import pytest

pytest.importorskip("mcp", reason="needs the MCP SDK")

from mcp import ClientSession                            # noqa: E402
from mcp.client._memory import InMemoryTransport         # noqa: E402


def _text(result) -> str:
    """Pull the text payload out of a tool result / resource / prompt message."""
    for attr in ("content", "contents", "messages"):
        part = getattr(result, attr, None)
        if part:
            first = part[0]
            inner = getattr(first, "content", first)
            return getattr(inner, "text", str(inner))
    return str(result)


class Client:
    """A thin, assertion-friendly view over a live `ClientSession`."""

    def __init__(self, session: ClientSession, init):
        self.session = session
        self.init = init

    # ── discovery ────────────────────────────────────────────────────────
    async def tools(self) -> dict:
        return {t.name: t for t in (await self.session.list_tools()).tools}

    async def prompts(self) -> dict:
        return {p.name: p for p in (await self.session.list_prompts()).prompts}

    async def resources(self) -> dict:
        return {str(r.uri): r for r in (await self.session.list_resources()).resources}

    # ── invocation ───────────────────────────────────────────────────────
    async def call_raw(self, name: str, **kw):
        """Call a tool and return the raw result, error or not."""
        return await self.session.call_tool(name, kw)

    async def call(self, name: str, **kw) -> str:
        """Call a tool that is expected to succeed; return its text."""
        result = await self.session.call_tool(name, kw)
        assert not result.is_error, f"{name} failed: {_text(result)}"
        return _text(result)

    async def fails(self, name: str, **kw) -> str:
        """Call a tool that is expected to fail; return the error text."""
        result = await self.session.call_tool(name, kw)
        assert result.is_error, f"{name} unexpectedly succeeded: {_text(result)}"
        return _text(result)

    async def call_json(self, name: str, **kw):
        return json.loads(await self.call(name, **kw))

    async def read(self, uri: str) -> str:
        return _text(await self.session.read_resource(uri))

    async def prompt(self, name: str, **kw) -> str:
        return _text(await self.session.get_prompt(name, kw))


@asynccontextmanager
async def connect(server, *, raise_exceptions: bool = False):
    """Open a real client session against `server` and complete the handshake."""
    async with InMemoryTransport(server, raise_exceptions=raise_exceptions) as (read, write):
        async with ClientSession(read, write) as session:
            init = await session.initialize()
            yield Client(session, init)


def over_mcp(fn):
    """Turn `async def test(mcp, ...)` into a sync test taking `server` plus fixtures.

    The first parameter is replaced by a connected `Client`; every other
    parameter stays a normal pytest fixture request.
    """
    signature = inspect.signature(fn)
    params = list(signature.parameters.values())
    if not params or params[0].name != "mcp":
        raise TypeError(f"{fn.__name__}: first parameter must be `mcp`")

    @functools.wraps(fn)
    def wrapper(server, **kw):
        async def go():
            async with connect(server) as mcp:
                return await fn(mcp, **kw)
        return asyncio.run(go())

    server_param = inspect.Parameter("server", inspect.Parameter.POSITIONAL_OR_KEYWORD)
    wrapper.__signature__ = signature.replace(parameters=[server_param, *params[1:]])
    return wrapper
