"""
Gemini Connector — Feather DB × Google Gemini
==============================================
Two things in one module:

  1. GeminiEmbedder   — wraps gemini-embedding-2-preview (3072-dim)
                        for ingesting text / image / video into Feather DB
  2. GeminiConnector  — exposes Feather DB tools as Gemini FunctionDeclarations
                        for use with the google-genai SDK chat API

Quick start — embedder:
    from feather_db.integrations.gemini import GeminiEmbedder
    emb = GeminiEmbedder(api_key="AIza...")
    vec = emb.embed_text("FD monthly payout creative brief")
    vec = emb.embed_image(image_description="Black/gold palette. Senior testimonial.")
    vec = emb.embed_video_transcript("0:00 Are your savings safe? 0:12 8.5% guaranteed.")

Quick start — agent connector:
    from google import genai
    from feather_db.integrations.gemini import GeminiConnector, GeminiEmbedder

    emb  = GeminiEmbedder(api_key="AIza...")
    conn = GeminiConnector(db_path="my.feather", dim=3072, embedder=emb.embed_text)

    client = genai.Client(api_key="AIza...")
    chat   = client.chats.create(model="gemini-2.0-flash", config=conn.chat_config())

    result = conn.run_loop(chat, "Why is our FD CTR dropping?")
    print(result)
"""

from __future__ import annotations

import base64
import hashlib
import json
import math
import mimetypes
import os
from typing import Callable, Optional

import numpy as np

from .base import FeatherTools, TOOL_SPECS

# ── Embedding model constants ─────────────────────────────────────────────────
GEMINI_EMBED_MODEL = "models/gemini-embedding-2-preview"
EMBED_DIM          = 3072
MOCK_DIM           = 3072

# ── Shared mock vocabulary ────────────────────────────────────────────────────
_VOCAB = [
    "fd","fixeddeposit","bond","creditcard","mutualfund","roas","ctr","cpm",
    "invest","return","rate","interest","savings","growth","yield","tax",
    "campaign","ad","creative","static","video","hook","cta","click",
    "retargeting","acquisition","retention","attribution","spend",
    "competitor","budget","rbi","india","viral","sentiment","instagram",
    "image","visual","color","palette","dialogue","voiceover",
    "context","memory","recall","decay","sticky","graph","edge","chain",
]


def _mock_embed(text: str, salt: str = "") -> np.ndarray:
    vec = np.zeros(MOCK_DIM, dtype=np.float32)
    tokens = text.lower().replace(",", " ").replace(".", " ").split()
    for tok in tokens:
        for i, kw in enumerate(_VOCAB):
            if kw in tok or tok in kw:
                idx = (i * 11 + len(tok) * 7) % MOCK_DIM
                vec[idx] += 1.0
                vec[(idx + 37) % MOCK_DIM] += 0.3
    if salt:
        h = int(hashlib.md5(salt.encode()).hexdigest(), 16)
        vec = np.roll(vec, h % 12)
        vec += np.random.default_rng(h % 10000).random(MOCK_DIM).astype(np.float32) * 0.08
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


# ═══════════════════════════════════════════════════════════════════════════════
# GeminiEmbedder
# ═══════════════════════════════════════════════════════════════════════════════

class GeminiEmbedder:
    """
    Unified multimodal embedder — text, image, video → 3072-dim vector.

    Real mode  : calls models/gemini-embedding-2-preview via google-genai SDK
    Mock mode  : deterministic offline simulation (no API key needed)

    Both modes return np.ndarray shape=(3072,) dtype=float32, L2-normalized.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        mock: bool = False,
    ):
        self.mock = mock
        self.dim  = EMBED_DIM

        if not mock:
            key = api_key or os.environ.get("GOOGLE_API_KEY")
            if not key:
                raise ValueError(
                    "Provide api_key= or set GOOGLE_API_KEY. "
                    "Use GeminiEmbedder(mock=True) for offline mode."
                )
            try:
                from google import genai
                self._client = genai.Client(api_key=key)
            except ImportError:
                raise ImportError("pip install google-genai")

    def embed_text(self, text: str) -> np.ndarray:
        if self.mock:
            return _mock_embed(text, salt="text")
        return self._call_api(text)

    def embed_image(
        self,
        image_path: Optional[str] = None,
        image_description: Optional[str] = None,
    ) -> np.ndarray:
        if self.mock:
            desc = image_description or image_path or "image"
            return _mock_embed(desc, salt="image")
        if image_path:
            mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            from google.genai import types
            content = types.Content(parts=[
                types.Part(inline_data=types.Blob(mime_type=mime, data=data))
            ])
            return self._call_api(content)
        elif image_description:
            return self.embed_text(image_description)
        raise ValueError("Provide image_path or image_description")

    def embed_video_transcript(
        self,
        transcript: str,
        duration_seconds: Optional[float] = None,
    ) -> np.ndarray:
        if self.mock:
            return _mock_embed(transcript, salt="video")
        return self._call_api(transcript)

    def embed_any(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        image_description: Optional[str] = None,
        video_transcript: Optional[str] = None,
    ) -> np.ndarray:
        vecs = []
        if text:
            vecs.append(self.embed_text(text))
        if image_path or image_description:
            vecs.append(self.embed_image(image_path, image_description))
        if video_transcript:
            vecs.append(self.embed_video_transcript(video_transcript))
        if not vecs:
            raise ValueError("Provide at least one of text / image_path / video_transcript")
        combined = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(combined)
        return (combined / norm) if norm > 0 else combined

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))

    def _call_api(self, content) -> np.ndarray:
        from google.genai import types
        response = self._client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=content,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        vec = np.array(response.embeddings[0].values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec


# ═══════════════════════════════════════════════════════════════════════════════
# GeminiConnector
# ═══════════════════════════════════════════════════════════════════════════════

# Gemini schema type mapping
_TYPE_MAP = {
    "string":  "STRING",
    "integer": "INTEGER",
    "number":  "NUMBER",
    "boolean": "BOOLEAN",
    "array":   "ARRAY",
    "object":  "OBJECT",
}


class GeminiConnector(FeatherTools):
    """Feather DB tools as Gemini FunctionDeclarations."""

    def tools(self) -> list:
        """
        Returns a list of google.genai.types.Tool objects for
        `client.chats.create(config=genai.types.GenerateContentConfig(tools=...))`.
        """
        try:
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai")

        declarations = []
        for spec in TOOL_SPECS:
            properties = {}
            for pname, pdef in spec["parameters"].items():
                gtype = _TYPE_MAP.get(pdef["type"], "STRING")
                prop = types.Schema(type=gtype, description=pdef.get("description", ""))
                if "enum" in pdef:
                    prop = types.Schema(type="STRING", enum=pdef["enum"], description=pdef.get("description", ""))
                properties[pname] = prop

            declarations.append(types.FunctionDeclaration(
                name=spec["name"],
                description=spec["description"],
                parameters=types.Schema(
                    type="OBJECT",
                    properties=properties,
                    required=spec.get("required", []),
                ),
            ))

        return [types.Tool(function_declarations=declarations)]

    def chat_config(self, system: Optional[str] = None):
        """Returns a GenerateContentConfig with tools attached."""
        try:
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai")

        kwargs: dict = {"tools": self.tools()}
        if system:
            kwargs["system_instruction"] = system
        return types.GenerateContentConfig(**kwargs)

    def process_response(self, response) -> tuple[bool, list]:
        """
        Process a Gemini response.
        Returns (done, function_response_parts).
        """
        try:
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai")

        parts = response.candidates[0].content.parts if response.candidates else []
        fn_calls = [p for p in parts if hasattr(p, "function_call") and p.function_call]

        if not fn_calls:
            return True, []

        result_parts = []
        for part in fn_calls:
            fc     = part.function_call
            args   = dict(fc.args) if fc.args else {}
            result = self.handle(fc.name, args)
            result_parts.append(
                types.Part.from_function_response(
                    name=fc.name,
                    response={"result": result},
                )
            )

        return False, result_parts

    def run_loop(
        self,
        chat,
        user_message: str,
        max_rounds: int = 10,
        verbose: bool = True,
    ) -> str:
        """
        Run the Gemini agent loop on an existing chat session.
        Returns the final text response.

        chat = client.chats.create(model="gemini-2.0-flash", config=conn.chat_config())
        """
        try:
            from google.genai import types
        except ImportError:
            raise ImportError("pip install google-genai")

        response = chat.send_message(user_message)

        for round_n in range(max_rounds):
            done, fn_parts = self.process_response(response)

            if done:
                text = ""
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "text") and part.text:
                        text += part.text
                return text

            if verbose:
                for part in response.candidates[0].content.parts:
                    if hasattr(part, "function_call") and part.function_call:
                        print(f"  [fn_call] {part.function_call.name}({dict(part.function_call.args)})")
                for fp in fn_parts:
                    preview = str(fp)[:120]
                    print(f"  [fn_result] {preview}...")

            response = chat.send_message(fn_parts)

        return "[max_rounds reached]"
