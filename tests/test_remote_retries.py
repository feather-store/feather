"""mcp_remote's HTTP client retries only what is safe to repeat.

Retried: 429 / 503 (the server says `retryable`) on an idempotent request —
GET, a read-only POST (search), or a write carrying an Idempotency-Key (the
same key on every attempt) — and a request that never reached the server.
Not retried: a write without a key, any other 4xx (the request is wrong), a
500 (repeating a request that broke the server makes an outage), a 421 (only
another shard can serve it).
"""
import json
import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from feather_db.integrations.mcp_remote import (
    ConnectionFailed, OverloadedError, RemoteFeatherTools, RequestError, ServerError,
    UnavailableError, WrongShardError)


class Script:
    """Answers each request with the next (status, body, headers) in line."""

    def __init__(self, *responses):
        self.responses = list(responses)
        self.seen = []                  # (method, path, Idempotency-Key)


@pytest.fixture
def server():
    state = {}

    class H(BaseHTTPRequestHandler):
        def _answer(self):
            s = state["script"]
            n = int(self.headers.get("Content-Length") or 0)
            self.rfile.read(n)
            s.seen.append((self.command, self.path, self.headers.get("Idempotency-Key")))
            status, body, headers = s.responses.pop(0) if s.responses else (200, {}, {})
            raw = json.dumps(body).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            for k, v in headers.items():
                self.send_header(k, v)
            self.end_headers()
            self.wfile.write(raw)
        do_GET = do_POST = do_PUT = do_DELETE = _answer

        def log_message(self, *a):
            pass

    httpd = HTTPServer(("127.0.0.1", 0), H)
    t = threading.Thread(target=httpd.serve_forever, daemon=True)
    t.start()

    def use(*responses):
        state["script"] = Script(*responses)
        tools = RemoteFeatherTools(f"http://127.0.0.1:{httpd.server_port}", dim=4,
                                   backoff_s=0.001, max_retries=3)
        return tools, state["script"]
    yield use
    httpd.shutdown()


BUSY = (429, {"detail": "busy", "error": "overloaded", "retryable": True}, {"Retry-After": "0"})
DOWN = (503, {"detail": "loading", "error": "namespace_loading", "retryable": True}, {"Retry-After": "0"})
OK = (200, {"id": 1, "content": "x"}, {})


def test_get_is_retried_through_429_and_503(server):
    tools, s = server(BUSY, DOWN, OK)
    assert tools._req("GET", "/v1/ns/records/1")["id"] == 1
    assert len(s.seen) == 3


def test_a_read_only_post_is_retried(server):
    tools, s = server(BUSY, (200, {"results": []}, {}))
    assert tools._t_recall({"query": "x"})["results"] == []
    assert len(s.seen) == 2


def test_a_write_without_a_key_is_not_retried(server):
    tools, s = server(DOWN, OK)
    with pytest.raises(UnavailableError):
        tools._req("POST", "/v1/ns/vectors", {"id": 1})
    assert len(s.seen) == 1


def test_a_keyed_write_is_retried_with_the_same_key(server):
    tools, s = server(BUSY, DOWN, (201, {"id": 1}, {}))
    assert tools._t_ingest({"text": "remember this"})["ok"]
    keys = {k for _, _, k in s.seen}
    assert len(s.seen) == 3 and len(keys) == 1 and None not in keys


def test_retries_are_bounded(server):
    tools, s = server(BUSY, BUSY, BUSY, BUSY, BUSY)
    with pytest.raises(OverloadedError) as e:
        tools._req("GET", "/v1/ns/records/1")
    assert len(s.seen) == 4 and e.value.retryable and e.value.error == "overloaded"


@pytest.mark.parametrize("status, body, exc", [
    (421, {"detail": "elsewhere", "error": "wrong_shard", "retryable": False, "owner_shard": 2}, WrongShardError),
    (404, {"detail": "no such record", "error": "not_found", "retryable": False}, RequestError),
    (500, {"detail": "boom", "error": "internal", "retryable": False}, ServerError),
])
def test_go_elsewhere_your_fault_and_our_fault_are_not_retried(server, status, body, exc):
    tools, s = server((status, body, {}), OK)
    with pytest.raises(exc) as e:
        tools._req("GET", "/v1/ns/records/1")
    assert len(s.seen) == 1
    if exc is WrongShardError:
        assert e.value.owner_shard == 2


def test_an_old_servers_untyped_429_is_still_retried(server):
    tools, s = server((429, {"detail": "busy"}, {}), OK)
    assert tools._req("GET", "/v1/ns/records/1")["id"] == 1


def test_a_server_without_keys_gets_the_write_once_without_one(server):
    tools, s = server((501, {"detail": "Idempotency-Key needs feather-db >= 0.22"}, {}),
                      (201, {"id": 1}, {}))
    assert tools._t_ingest({"text": "x"})["ok"]
    assert s.seen[0][2] is not None and s.seen[1][2] is None


def test_a_refused_connection_is_retried_even_for_a_write():
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]          # nothing listens here
    tools = RemoteFeatherTools(f"http://127.0.0.1:{port}", dim=4, backoff_s=0.001, max_retries=2,
                               timeout=1)
    with pytest.raises(ConnectionFailed) as e:
        tools._req("POST", "/v1/ns/vectors", {"id": 1})
    assert e.value.sent is False


def test_tool_errors_carry_the_type(server):
    tools, _ = server((421, {"detail": "elsewhere", "error": "wrong_shard", "retryable": False}, {}))
    out = json.loads(tools.handle("feather_get_record", {"id": 1}))
    assert out["error_type"] == "wrong_shard" and out["status"] == 421 and out["retryable"] is False
