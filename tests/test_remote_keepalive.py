"""RemoteDB's default transport keeps connections open between calls.

A new connection per request costs a TCP handshake (plus TLS over https) on
every call. These run a real HTTP/1.1 server and count accepted connections.
"""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from feather_db.remote import RemoteDB


class _Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"          # keep-alive
    close_next = False
    close_all = False
    closed = threading.Event()

    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", 0)))
        self.do_GET()

    def do_GET(self):
        body = json.dumps({"size": 7}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        if type(self).close_next or type(self).close_all:   # closes, unannounced
            type(self).close_next = False
            self.close_connection = True

    def finish(self):
        super().finish()
        if self.close_connection:
            type(self).closed.set()

    def log_message(self, *a):
        pass


@pytest.fixture
def server(monkeypatch):
    for v in ("http_proxy", "HTTP_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(v, raising=False)
    accepted = []

    class Server(ThreadingHTTPServer):
        daemon_threads = True

        def get_request(self):
            conn = super().get_request()
            accepted.append(conn)
            return conn

    srv = Server(("127.0.0.1", 0), _Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}", accepted
    srv.shutdown()
    srv.server_close()


def _call(db):
    return db._call("GET", "/stats")["size"]


def test_one_connection_for_many_calls(server):
    url, accepted = server
    db = RemoteDB(url, "ns")
    assert [_call(db) for _ in range(20)] == [7] * 20
    assert len(accepted) == 1


def test_server_closed_idle_connection_is_replaced(server):
    url, accepted = server
    db = RemoteDB(url, "ns")
    _Handler.closed.clear()
    _Handler.close_next = True
    assert _call(db) == 7                 # the server closes after answering
    assert _Handler.closed.wait(5)        # ...and the close lands while idle
    time.sleep(0.05)
    assert _call(db) == 7                 # no error: a fresh connection is used
    assert _call(db) == 7
    assert len(accepted) == 2


def test_close_racing_the_send(server, monkeypatch):
    """The server closes between the liveness check and the send."""
    url, accepted = server

    def landed():                         # the close has reached the socket
        assert _Handler.closed.wait(5)
        _Handler.closed.clear()
        time.sleep(0.05)

    _Handler.close_all = True             # every connection dies after one answer
    try:
        _Handler.closed.clear()
        db = RemoteDB(url, "ns")
        assert _call(db) == 7
        landed()
        monkeypatch.setattr("feather_db.remote.select.select", lambda *a: ([], [], []))
        assert db._call("GET", "/stats")["size"] == 7     # idempotent: sent again
        landed()
        with pytest.raises((ConnectionError, OSError)):   # may have been applied
            db._call("POST", "/stats", body={})
        assert len(accepted) == 2
    finally:
        _Handler.close_all = False


def test_close_releases_connection(server):
    url, accepted = server
    db = RemoteDB(url, "ns")
    _call(db)
    conns = list(db._session._all)
    assert len(conns) == 1 and conns[0].sock is not None
    db.close(save=False)
    assert conns[0].sock is None
