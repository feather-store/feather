"""Provider embedders: kept-alive connections and native Gemini dims."""
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from feather_db.integrations import embedders


@pytest.fixture
def server():
    """A local stand-in for the Gemini API: records each request's body and
    the client port it came from (one port = one connection)."""
    seen = []

    class H(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"          # keep-alive
        close_next = False

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            seen.append((self.client_address[1], self.path, body))
            dim = body.get("outputDimensionality", 3072)
            out = json.dumps({"embedding": {"values": [0.5] * dim}}).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(out)))
            if H.close_next:
                self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(out)

        def log_message(self, *a):
            pass

    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", seen, H
    srv.shutdown()


def test_default_gemini_model_is_one_google_still_serves():
    assert embedders._DEFAULT_MODELS["gemini"] == "gemini-embedding-001"


def test_gemini_is_asked_for_dim_and_reuses_one_connection(server, monkeypatch):
    url, seen, _ = server
    monkeypatch.setenv("FEATHER_EMBED_CACHE", "0")
    embed = embedders.make_embedder("gemini", api_key="k", dim=768, base_url=url)
    for t in ("a", "b", "c", "d"):
        assert len(embed(t)) == 768
    assert {port for port, _, _ in seen} == {seen[0][0]}          # one connection
    assert all(b["outputDimensionality"] == 768 for _, _, b in seen)
    assert seen[0][1] == "/v1beta/models/gemini-embedding-001:embedContent"
    embedders.make_embedder("gemini", api_key="k", dim=3072, base_url=url)("e")
    assert "outputDimensionality" not in seen[-1][2]               # native size


def test_a_connection_the_server_closed_is_replaced(server, monkeypatch):
    url, seen, H = server
    monkeypatch.setenv("FEATHER_EMBED_CACHE", "0")
    embed = embedders.make_embedder("gemini", api_key="k", dim=768, base_url=url)
    H.close_next = True
    embed("a")
    H.close_next = False
    embed("b"); embed("c")
    ports = [port for port, _, _ in seen]
    assert len(ports) == 3 and ports[0] != ports[1] == ports[2]


def test_http_errors_keep_their_message(monkeypatch):
    with pytest.raises(RuntimeError, match="127.0.0.1:9"):
        embedders._http_post_json("http://127.0.0.1:9/x", {}, {}, timeout=1)
