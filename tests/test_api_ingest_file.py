"""POST /v1/{ns}/ingest_file: a file is stored by what is IN it, not its name.

Gemini is replaced by an httpx.MockTransport that answers like the real REST
API (generateContent, and the resumable File API for large files), so the
request building and response parsing in app/describe.py are what run.
"""
import base64
import json
import os
import sys

import pytest

pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
pytest.importorskip("multipart", reason="file uploads need python-multipart")
httpx = pytest.importorskip("httpx")

from fastapi.testclient import TestClient  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")
NS = "creatives"
DIM = 16

# What the fake model "sees" in each file, keyed by the file's bytes.
SEEN = {
    b"\x89PNG car": {
        "summary": "A blue hatchback car parked on a city street at dusk.",
        "visual": "Blue hatchback car, wet asphalt, street lights.",
        "colors": ["blue", "grey"],
        "text_in_media": "DRIVE MORE. PAY LESS.",
        "tags": ["car", "automobile", "hatchback", "Blue"],
    },
    b"\x00\x00\x00 ftypmp4 video": {
        "summary": "A young woman in a red Bandhani saree dances in a village courtyard.",
        "visual": "Courtyard, marigold garlands.",
        "people": "Woman, 20s, wearing a traditional red Bandhani saree and silver jewellery.",
        "speech": "Navratri ki shubhkamnayein [Happy Navratri]",
        "scenes": ["0:00-0:03 courtyard wide shot", "0:03-0:08 close-up dance"],
        "tags": ["saree", "traditional dress", "festival"],
    },
}


class FakeGemini:
    def __init__(self):
        self.calls = []
        self.uploaded = {}
        self.deleted = []

    def __call__(self, request: httpx.Request) -> httpx.Response:
        self.calls.append((request.method, request.url.path))
        path = request.url.path
        if path.endswith(":generateContent"):
            assert request.headers["x-goog-api-key"] == "test-key"
            body = json.loads(request.content)
            part = body["contents"][0]["parts"][0]
            if "inlineData" in part:
                data = base64.b64decode(part["inlineData"]["data"])
            else:
                data = self.uploaded[part["fileData"]["fileUri"]]
            assert body["generationConfig"]["responseMimeType"] == "application/json"
            d = SEEN[data]
            return httpx.Response(200, json={"candidates": [{
                "content": {"parts": [{"text": json.dumps(d)}]}, "finishReason": "STOP"}]})
        if path == "/upload/v1beta/files":
            assert request.headers["X-Goog-Upload-Command"] == "start"
            return httpx.Response(200, headers={"x-goog-upload-url": "https://upload.test/u1"}, json={})
        if request.url.host == "upload.test":
            self.uploaded["https://files.test/abc"] = request.read()
            return httpx.Response(200, json={"file": {
                "name": "files/abc", "uri": "https://files.test/abc", "state": "PROCESSING"}})
        if path == "/v1beta/files/abc" and request.method == "GET":
            return httpx.Response(200, json={"name": "files/abc", "uri": "https://files.test/abc",
                                             "state": "ACTIVE"})
        if path == "/v1beta/files/abc" and request.method == "DELETE":
            self.deleted.append("files/abc")
            return httpx.Response(200, json={})
        return httpx.Response(404, json={"error": path})


def _client(tmp_path, monkeypatch, **env):
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DB_DIM", str(DIM))
    monkeypatch.delenv("FEATHER_API_KEY", raising=False)
    for k in ("GEMINI_API_KEY", "GOOGLE_API_KEY", "FEATHER_DESCRIBE_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    monkeypatch.setenv("FEATHER_EMBED_PROVIDER", "mock")
    monkeypatch.setenv("FEATHER_EMBED_DIM", str(DIM))
    for k, v in env.items():
        monkeypatch.setenv(k, str(v))
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.main import app  # noqa: WPS433
    return TestClient(app)


@pytest.fixture
def env(tmp_path, monkeypatch):
    with _client(tmp_path, monkeypatch, FEATHER_DESCRIBE_API_KEY="test-key") as c:
        describe = sys.modules["app.describe"]
        fake = FakeGemini()
        describe.DESCRIBER._transport = httpx.MockTransport(fake)
        monkeypatch.setattr(describe, "_POLL_S", 0.0)
        yield c, fake, describe


def _upload(c, name, data, mime, **form):
    if "metadata" in form:
        form["metadata"] = json.dumps(form["metadata"])
    return c.post(f"/v1/{NS}/ingest_file", files={"file": (name, data, mime)}, data=form)


def _keyword(c, q):
    r = c.post(f"/v1/{NS}/keyword_search", json={"query": q, "k": 5})
    assert r.status_code == 200, r.text
    return r.json()["results"]


def test_image_is_found_by_what_it_shows_not_its_name(env):
    c, fake, _ = env
    r = _upload(c, "IMG_0042.png", b"\x89PNG car", "image/png")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["kind"] == "image" and body["described"] and body["described_by"] == "gemini-2.5-flash"
    assert "blue hatchback" in body["content"].lower()
    assert "DRIVE MORE. PAY LESS." in body["content"]          # visible text, verbatim

    assert [h["id"] for h in _keyword(c, "blue car")] == [body["id"]]
    assert [h["id"] for h in _keyword(c, "drive more pay less")] == [body["id"]]
    assert _keyword(c, "IMG_0042") == []                      # the name is not searched

    meta = c.get(f"/v1/{NS}/records/{body['id']}").json()
    attrs = meta["attributes"]
    assert attrs["file_name"] == "IMG_0042.png" and attrs["media_kind"] == "image"
    assert attrs["colors"] == "blue, grey"
    assert json.loads(meta["tags_json"]) == ["car", "automobile", "hatchback", "blue"]
    assert fake.calls == [("POST", "/v1beta/models/gemini-2.5-flash:generateContent")]


def test_large_video_goes_through_the_file_api(env, monkeypatch):
    c, fake, describe = env
    monkeypatch.setattr(describe.DESCRIBER, "inline_max", 4)
    r = _upload(c, "final_v3.mp4", b"\x00\x00\x00 ftypmp4 video", "video/mp4")
    assert r.status_code == 200, r.text
    rid = r.json()["id"]
    assert [h["id"] for h in _keyword(c, "saree")] == [rid]
    assert [h["id"] for h in _keyword(c, "navratri")] == [rid]  # spoken words
    assert ("GET", "/v1beta/files/abc") in fake.calls          # waited for PROCESSING
    assert fake.deleted == ["files/abc"]                       # not left on Google's side


def test_same_file_twice_updates_one_record_and_reuses_its_description(env):
    c, fake, _ = env
    first = _upload(c, "a.png", b"\x89PNG car", "image/png",
                    metadata={"content": "Q3 launch", "attributes": {"url": "s3://b/a.png"}}).json()
    second = _upload(c, "renamed.png", b"\x89PNG car", "image/png",
                     metadata={"content": "Q4 relaunch"})
    assert second.status_code == 200, second.text
    second = second.json()
    assert second["id"] == first["id"] and not second["described"]
    assert len([x for x in fake.calls if x[1].endswith(":generateContent")]) == 1
    assert second["content"].startswith("Q4 relaunch\n\nA blue hatchback")
    assert "Q3 launch" not in second["content"]
    meta = c.get(f"/v1/{NS}/records/{first['id']}").json()
    assert meta["attributes"]["file_name"] == "renamed.png"
    assert meta["attributes"]["colors"] == "blue, grey"
    assert "car" in json.loads(meta["tags_json"])

    third = _upload(c, "a.png", b"\x89PNG car", "image/png", refresh="true").json()
    assert third["described"] and third["id"] == first["id"]


def test_text_file_is_read_without_a_model(env):
    c, fake, _ = env
    r = _upload(c, "brief.txt", "Hook: monsoon sale, 40% off umbrellas".encode(), "text/plain")
    assert r.status_code == 200, r.text
    assert r.json()["described_by"] == "text"
    assert [h["id"] for h in _keyword(c, "umbrellas")] == [r.json()["id"]]
    assert fake.calls == []


def test_hybrid_search_embeds_the_query_when_no_vector_is_sent(env):
    c, _, _ = env
    rid = _upload(c, "x.png", b"\x89PNG car", "image/png").json()["id"]
    r = c.post(f"/v1/{NS}/hybrid_search", json={"query": "blue car", "k": 3})
    assert r.status_code == 200, r.text
    assert r.json()["results"][0]["id"] == rid
    r = c.post(f"/v1/{NS}/hybrid_search", json={"query": "  "})
    assert r.status_code == 422


def test_idempotency_key_replays_without_describing_again(env):
    c, fake, _ = env
    h = {"Idempotency-Key": "up-1"}
    a = c.post(f"/v1/{NS}/ingest_file", headers=h, files={"file": ("a.png", b"\x89PNG car", "image/png")})
    b = c.post(f"/v1/{NS}/ingest_file", headers=h, files={"file": ("a.png", b"\x89PNG car", "image/png")})
    assert a.status_code == b.status_code == 200
    assert b.headers.get("Idempotent-Replayed") == "true"
    assert a.json()["id"] == b.json()["id"]
    assert len(fake.calls) == 1


def test_refusals(env, tmp_path, monkeypatch):
    c, _, describe = env
    r = _upload(c, "deck.pptx", b"PK\x03\x04", "application/vnd.openxmlformats-officedocument.presentationml.presentation")
    assert r.status_code == 415 and r.json()["error"] == "unsupported_media_type"
    assert _upload(c, "e.png", b"", "image/png").status_code == 400
    r = _upload(c, "a.png", b"\x89PNG car", "image/png", metadata="{not json")
    assert r.status_code == 422
    r = _upload(c, "a.png", b"\x89PNG car", "image/png", metadata={"bogus": 1})
    assert r.status_code == 422

    monkeypatch.setattr(describe.DESCRIBER, "_provider", "none")
    r = _upload(c, "a.png", b"\x89PNG car", "image/png")
    assert r.status_code == 400 and r.json()["error"] == "describe_not_configured"


def test_model_failures_map_to_retryable_or_not(env):
    c, _, describe = env
    describe.DESCRIBER._transport = httpx.MockTransport(lambda req: httpx.Response(429, text="slow down"))
    r = _upload(c, "a.png", b"\x89PNG car", "image/png")
    assert r.status_code == 503 and r.json()["retryable"]
    describe.DESCRIBER._transport = httpx.MockTransport(lambda req: httpx.Response(403, text="bad key"))
    r = _upload(c, "a.png", b"\x89PNG car", "image/png")
    assert r.status_code == 502 and not r.json()["retryable"]
    describe.DESCRIBER._transport = httpx.MockTransport(
        lambda req: httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}}))
    r = _upload(c, "a.png", b"\x89PNG car", "image/png")
    assert r.status_code == 502 and "SAFETY" in r.json()["detail"]


def test_file_ingests_have_their_own_admission_class():
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    from app.admission import classify
    assert classify("POST", f"/v1/{NS}/ingest_file") == "file"
    assert classify("POST", f"/v1/{NS}/ingest_text") == "write"
