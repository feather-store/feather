"""Read an uploaded file and turn what is IN it into searchable text.

A creative is only findable by the text stored with it. Stored by name, a
search for "blue car" finds `IMG_0042.jpg` only if someone typed that into
the name. So /ingest_file never indexes the name: it hands the file to a
multimodal model, which writes down what is actually in the file (objects,
colours, people and their clothing, the setting, every word of visible text,
every spoken word). That text becomes the record's content, so BM25 can find
exact words, and it is embedded, so a paraphrase like "a car in blue" or
"woman in a saree" lands near it.

What reads what:
  text/* , json, csv, xml   decoded as is; no model call
  image/*, video/*, audio/* the describe model (Gemini, which takes all three)
  application/pdf           the describe model, plus pypdf's full text when
                            installed (a model's output tokens cannot carry a
                            long document)

Config (env):
  FEATHER_DESCRIBE_PROVIDER   gemini | none. Default: gemini when a key is
                              found below, else none.
  FEATHER_DESCRIBE_MODEL      default gemini-2.5-flash
  FEATHER_DESCRIBE_API_KEY    else GEMINI_API_KEY, else GOOGLE_API_KEY, else
                              the embedding config's key when that is Gemini
  FEATHER_DESCRIBE_INLINE_MB  files up to this go inline (default 14: base64
                              grows it by a third, and a request is capped at
                              20 MB); larger ones go through the File API
  FEATHER_DESCRIBE_TIMEOUT_S  whole describe call, upload + processing (300)
"""
from __future__ import annotations

import asyncio
import base64
import json
import mimetypes
import os
import time
from typing import List, Optional

import httpx

from .embedding import EMBEDDING, UpstreamError

_DEFAULT_MODEL = "gemini-2.5-flash"
_BASE = "https://generativelanguage.googleapis.com"
_CHUNK = 1 << 20
_POLL_S = 2.0                  # File API: how often to ask whether a video is processed

TEXT_MIMES = {"application/json", "application/xml", "application/x-ndjson",
              "application/csv", "application/yaml", "application/x-yaml"}


def kind_of(mime: str) -> Optional[str]:
    """'text' | 'image' | 'video' | 'audio' | 'document', or None if unsupported."""
    major = mime.split("/", 1)[0]
    if major == "text" or mime in TEXT_MIMES:
        return "text"
    if mime == "image/svg+xml":
        return None                 # markup, not pixels; the model cannot see it
    if major in ("image", "video", "audio"):
        return major
    if mime == "application/pdf":
        return "document"
    return None


def resolve_mime(declared: Optional[str], filename: str) -> str:
    """The upload's type. Browsers send application/octet-stream for anything
    they do not recognise, so that (or nothing) falls back to the extension."""
    m = (declared or "").split(";", 1)[0].strip().lower()
    if m and m != "application/octet-stream":
        return m
    return (mimetypes.guess_type(filename or "")[0] or "application/octet-stream").lower()


PROMPT = """You are indexing a file (a marketing creative, photo, video, audio clip or \
document) so that people can find it later by describing what is IN it. Describe only \
what is actually present. Do not guess from a file name and do not invent details.

Be concrete and specific:
- objects, products and vehicles with their colours ("blue hatchback car", "red sneakers")
- people: apparent gender, age range, skin tone only if relevant, expression, pose, and \
clothing. Name traditional or regional attire precisely (saree, lehenga, salwar kameez, \
kurta, sherwani, hijab, kimono, hanbok, dashiki...)
- the setting and location type, time of day, actions taking place
- the format (static image, carousel frame, UGC video, product shot, talking head, \
animation, screenshot, poster...) and overall mood and style
- text: transcribe EVERY piece of visible text exactly as written (headlines, captions, \
CTA buttons, prices, logos, labels), in the original language
- speech: transcribe every spoken word, voiceover and lyric, in the original language; \
add an English translation in [brackets] when it is not English
- for video, a timeline of scenes ("0:00-0:03 ...")
- tags: 10 to 30 short lowercase search terms a person might type, including synonyms \
(car, automobile, sedan) and the language of any text or speech.

Leave a field empty when it does not apply."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "summary":       {"type": "STRING", "description": "one or two sentences"},
        "visual":        {"type": "STRING"},
        "people":        {"type": "STRING"},
        "setting":       {"type": "STRING"},
        "colors":        {"type": "ARRAY", "items": {"type": "STRING"}},
        "format":        {"type": "STRING"},
        "mood":          {"type": "STRING"},
        "brands":        {"type": "ARRAY", "items": {"type": "STRING"}},
        "text_in_media": {"type": "STRING"},
        "scenes":        {"type": "ARRAY", "items": {"type": "STRING"}},
        "speech":        {"type": "STRING"},
        "audio":         {"type": "STRING"},
        "tags":          {"type": "ARRAY", "items": {"type": "STRING"}},
    },
    "required": ["summary", "visual", "tags"],
}


def compose(d: dict) -> str:
    """The description as one text, most descriptive fields first: an
    embedding model reads only the start of a long input, and a transcript is
    what BM25 is for."""
    def lst(k):
        return ", ".join(str(x).strip() for x in (d.get(k) or []) if str(x).strip())

    parts = [
        (d.get("summary") or "").strip(),
        ("Visual: ", (d.get("visual") or "").strip()),
        ("People: ", (d.get("people") or "").strip()),
        ("Setting: ", (d.get("setting") or "").strip()),
        ("Colors: ", lst("colors")),
        ("Format: ", (d.get("format") or "").strip()),
        ("Mood: ", (d.get("mood") or "").strip()),
        ("Brands: ", lst("brands")),
        ("Tags: ", lst("tags")),
        ("Text in creative: ", (d.get("text_in_media") or "").strip()),
        ("Scenes:\n", "\n".join(f"- {s}" for s in (d.get("scenes") or []) if str(s).strip())),
        ("Speech: ", (d.get("speech") or "").strip()),
        ("Audio: ", (d.get("audio") or "").strip()),
    ]
    out = []
    for p in parts:
        if isinstance(p, str):
            if p:
                out.append(p)
        elif p[1]:
            out.append(p[0] + p[1])
    return "\n".join(out)


class Description:
    __slots__ = ("text", "tags", "colors", "model")

    def __init__(self, text: str, tags: List[str] = (), colors: List[str] = (), model: str = ""):
        self.text, self.tags, self.colors, self.model = text, list(tags), list(colors), model


class Describer:
    def __init__(self):
        self.model = os.getenv("FEATHER_DESCRIBE_MODEL", _DEFAULT_MODEL)
        self.base_url = os.getenv("FEATHER_DESCRIBE_BASE_URL", _BASE).rstrip("/")
        self.inline_max = int(float(os.getenv("FEATHER_DESCRIBE_INLINE_MB", "14")) * 1024 * 1024)
        self.timeout_s = float(os.getenv("FEATHER_DESCRIBE_TIMEOUT_S", "300"))
        self._provider = os.getenv("FEATHER_DESCRIBE_PROVIDER", "").strip().lower()
        self._transport: Optional[httpx.AsyncBaseTransport] = None   # tests inject one

    def _key(self) -> str:
        for name in ("FEATHER_DESCRIBE_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY"):
            if os.getenv(name):
                return os.environ[name]
        return EMBEDDING.key_for("gemini")

    @property
    def provider(self) -> str:
        if self._provider:
            return self._provider
        return "gemini" if self._key() else "none"

    def snapshot(self) -> dict:
        return {"provider": self.provider, "model": self.model}

    async def describe(self, f, size: int, mime: str, filename: str) -> Description:
        """Describe the open binary file `f` (`size` bytes of type `mime`).
        Raises RuntimeError for a configuration problem, UpstreamError when the
        model fails."""
        if self.provider == "none":
            raise RuntimeError(
                "No describe model configured, so images, video, audio and PDFs cannot be "
                "read. Set FEATHER_DESCRIBE_API_KEY (or GEMINI_API_KEY / GOOGLE_API_KEY), "
                "or configure Gemini as the embedding provider.")
        if self.provider != "gemini":
            raise RuntimeError(f"Unknown FEATHER_DESCRIBE_PROVIDER {self.provider!r} (gemini | none)")
        key = self._key()
        if not key:
            raise RuntimeError("FEATHER_DESCRIBE_PROVIDER=gemini needs a Gemini API key.")
        headers = {"x-goog-api-key": key}
        async with httpx.AsyncClient(timeout=self.timeout_s, transport=self._transport) as c:
            deadline = time.monotonic() + self.timeout_s
            uploaded = None
            try:
                if size <= self.inline_max:
                    data = await asyncio.to_thread(_read_all, f)
                    part = {"inlineData": {"mimeType": mime,
                                           "data": base64.b64encode(data).decode("ascii")}}
                else:
                    uploaded = await self._upload(c, headers, f, size, mime, filename, deadline)
                    part = {"fileData": {"mimeType": mime, "fileUri": uploaded["uri"]}}
                d = await self._generate(c, headers, part)
            finally:
                if uploaded is not None:     # the File API keeps files 48 h otherwise
                    try:
                        await c.delete(f"{self.base_url}/v1beta/{uploaded['name']}", headers=headers)
                    except httpx.HTTPError:
                        pass
        return Description(compose(d), d.get("tags") or [], d.get("colors") or [], self.model)

    async def _generate(self, c: httpx.AsyncClient, headers: dict, part: dict) -> dict:
        cfg = {"responseMimeType": "application/json", "responseSchema": SCHEMA,
               "temperature": 0.2, "maxOutputTokens": 8192}
        if "flash" in self.model:       # thinking adds latency, not detail, to a description
            cfg["thinkingConfig"] = {"thinkingBudget": 0}
        body = {"contents": [{"role": "user", "parts": [part, {"text": PROMPT}]}],
                "generationConfig": cfg}
        url = f"{self.base_url}/v1beta/models/{self.model}:generateContent"
        j = await _send(c, "POST", url, headers=headers, json=body)
        cands = j.get("candidates") or []
        if not cands:
            why = (j.get("promptFeedback") or {}).get("blockReason") or "no candidates"
            raise UpstreamError(f"gemini declined to describe the file: {why}", status=422)
        text = "".join(p.get("text", "") for p in (cands[0].get("content") or {}).get("parts") or [])
        try:
            d = json.loads(text)
        except ValueError:
            reason = cands[0].get("finishReason", "")
            raise UpstreamError(f"gemini returned an unreadable description (finishReason={reason})")
        if not isinstance(d, dict):
            raise UpstreamError("gemini returned a description that is not an object")
        return d

    async def _upload(self, c, headers, f, size, mime, filename, deadline) -> dict:
        """Gemini File API: resumable upload, then wait until it is processed
        (a video is not usable until its state is ACTIVE)."""
        start = await c.post(
            f"{self.base_url}/upload/v1beta/files",
            headers={**headers, "X-Goog-Upload-Protocol": "resumable",
                     "X-Goog-Upload-Command": "start",
                     "X-Goog-Upload-Header-Content-Length": str(size),
                     "X-Goog-Upload-Header-Content-Type": mime},
            json={"file": {"display_name": (filename or "upload")[:200]}})
        _raise_for(start)
        upload_url = start.headers.get("x-goog-upload-url")
        if not upload_url:
            raise UpstreamError("gemini file upload: no upload URL returned")
        f.seek(0)
        j = await _send(c, "POST", upload_url,
                        headers={"Content-Length": str(size), "X-Goog-Upload-Offset": "0",
                                 "X-Goog-Upload-Command": "upload, finalize"},
                        content=_chunks(f))
        info = j.get("file") or {}
        while info.get("state") == "PROCESSING":
            if time.monotonic() > deadline:
                raise UpstreamError("gemini is still processing the file; try again", status=503)
            await asyncio.sleep(_POLL_S)
            info = await _send(c, "GET", f"{self.base_url}/v1beta/{info['name']}", headers=headers)
        if info.get("state") not in (None, "ACTIVE") or not info.get("uri"):
            raise UpstreamError(f"gemini could not process the file (state={info.get('state')})")
        return info


def _read_all(f) -> bytes:
    f.seek(0)
    return f.read()


async def _chunks(f):
    while True:
        b = await asyncio.to_thread(f.read, _CHUNK)
        if not b:
            return
        yield b


def _raise_for(r: httpx.Response) -> None:
    if r.status_code >= 400:
        raise UpstreamError(f"gemini HTTP {r.status_code}: {r.text[:300]}", status=r.status_code)


async def _send(c: httpx.AsyncClient, method: str, url: str, **kw) -> dict:
    try:
        r = await c.request(method, url, **kw)
    except httpx.TimeoutException as e:
        raise UpstreamError(f"gemini timed out: {e}", status=504)
    except httpx.HTTPError as e:
        raise UpstreamError(f"gemini unreachable: {e}", status=503)
    _raise_for(r)
    try:
        return r.json()
    except ValueError:
        raise UpstreamError(f"gemini returned non-JSON (HTTP {r.status_code})")


def decode_text(f, limit: int) -> str:
    """A text upload's contents, at most `limit` characters."""
    f.seek(0)
    raw = f.read(limit * 4)          # utf-8 is at most 4 bytes a character
    for enc in ("utf-8-sig", "utf-16"):
        try:
            return raw.decode(enc)[:limit]
        except UnicodeDecodeError:
            continue
    return raw.decode("latin-1")[:limit]


def pdf_text(f, limit: int) -> str:
    """The text layer of a PDF via pypdf, or "" when pypdf is not installed or
    the PDF has none (a scan: only the describe model can read that)."""
    try:
        from pypdf import PdfReader
    except ImportError:
        return ""
    try:
        f.seek(0)
        out, n = [], 0
        for page in PdfReader(f).pages:
            t = page.extract_text() or ""
            out.append(t)
            n += len(t)
            if n >= limit:
                break
        return "\n".join(out).strip()[:limit]
    except Exception:  # noqa: BLE001 — a malformed PDF still gets described
        return ""


DESCRIBER = Describer()
