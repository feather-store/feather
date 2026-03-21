"""
GeminiEmbedder — Feather DB × Gemini Embedding 2
=================================================
Wraps google-genai SDK to produce unified multimodal embeddings
(text, image, video) in a single shared vector space.

Two modes:
  REAL   — requires GOOGLE_API_KEY, calls gemini-embedding-exp-03-07
  MOCK   — no API key needed, simulates a 768-dim unified embedding space
           suitable for local experiments and CI

Usage:
    from embedder import GeminiEmbedder

    emb = GeminiEmbedder(api_key="AIza...")   # real
    emb = GeminiEmbedder(mock=True)           # mock / offline

    vec = emb.embed_text("FD interest rate budget announcement")
    vec = emb.embed_image("path/to/ad_creative.jpg")
    vec = emb.embed_video_transcript("0:00 Invest in gold. 0:03 Download now.")
    vec = emb.embed_any(text="...", image_path="...", video_transcript="...")
"""

import os
import math
import hashlib
import numpy as np
from typing import Optional

# ── Model constants ──────────────────────────────────────────────────────────
GEMINI_EMBED_MODEL = "models/gemini-embedding-2-preview"
EMBED_DIM          = 3072      # Gemini Embedding 2 output dimension
MOCK_DIM           = 3072      # same dim in mock mode for drop-in compatibility


# ── Shared vocabulary for mock embeddings ────────────────────────────────────
_VOCAB = [
    # Finance
    "fd","fixeddeposit","bond","creditcard","mutualfund","roas","ctr","cpm","roi",
    "invest","return","rate","interest","savings","growth","yield","inflation","tax",
    # Marketing
    "campaign","ad","creative","static","video","carousel","reel","story","banner",
    "hook","hold","cta","calltoaction","click","impression","conversion","install",
    "retargeting","lookalike","acquisition","retention","attribution","spend",
    # Competitor
    "groww","bajaj","hdfc","paytm","zerodha","sbi","navi","angel","finflex",
    # Social
    "budget2026","rbi","india","viral","trending","sentiment","twitter","instagram",
    # Visual / Creative
    "image","video","audio","visual","color","palette","motion","text","overlay",
    "portrait","landscape","thumbnail","frame","scene","dialogue","voiceover",
    # Multimodal
    "multimodal","unified","cross","modal","embedding","semantic","similarity",
    # Product
    "fdrate","loungeaccess","monthlypayout","sip","elss","nps","gold","silver",
    # Context
    "context","memory","recall","decay","sticky","graph","edge","hop","chain",
]


def _mock_embed(text: str, salt: str = "") -> np.ndarray:
    """
    Deterministic mock embedding.
    Maps tokens → positions in a 3072-dim vector via a stable hash.
    Adds a small unique salt so different modalities of the same concept
    land near each other but not identical — simulating a unified space.
    """
    vec = np.zeros(MOCK_DIM, dtype=np.float32)
    tokens = text.lower().replace(",", " ").replace(".", " ").replace("_", " ").split()
    for tok in tokens:
        for i, kw in enumerate(_VOCAB):
            if kw in tok or tok in kw:
                # Primary position
                idx = (i * 11 + len(tok) * 7) % MOCK_DIM
                vec[idx] += 1.0 + len(kw) * 0.1
                # Secondary spread (simulate dense embedding)
                for j in range(1, 4):
                    vec[(idx + j * 37) % MOCK_DIM] += 0.3 / j
    # Salt shifts (cross-modal proximity, not identity)
    if salt:
        h = int(hashlib.md5(salt.encode()).hexdigest(), 16)
        shift = h % 12
        vec = np.roll(vec, shift)
        vec += np.random.default_rng(h % 10000).random(MOCK_DIM).astype(np.float32) * 0.08
    norm = np.linalg.norm(vec)
    return (vec / norm) if norm > 0 else vec


class GeminiEmbedder:
    """
    Unified multimodal embedder wrapping Gemini Embedding 2.

    In REAL mode:  calls google-genai SDK → gemini-embedding-exp-03-07
    In MOCK mode:  deterministic 768-dim simulation, no API key required

    Both modes produce np.ndarray of shape (768,) dtype float32.
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

    # ── Public API ────────────────────────────────────────────────────────────

    def embed_text(self, text: str) -> np.ndarray:
        """Embed a text string into the unified space."""
        if self.mock:
            return _mock_embed(text, salt="text")
        return self._call_api(text, modality="text")

    def embed_image(self, image_path: Optional[str] = None,
                    image_description: Optional[str] = None) -> np.ndarray:
        """
        Embed an image.
        REAL mode: loads image file, sends inline_data to Gemini.
        MOCK mode: embeds the description text with an image-space salt.
        """
        if self.mock:
            desc = image_description or image_path or "image"
            return _mock_embed(desc, salt="image")

        if image_path:
            import base64, mimetypes
            mime = mimetypes.guess_type(image_path)[0] or "image/jpeg"
            with open(image_path, "rb") as f:
                data = base64.b64encode(f.read()).decode()
            from google.genai import types
            content = types.Content(parts=[
                types.Part(inline_data=types.Blob(mime_type=mime, data=data))
            ])
            return self._call_api(content, modality="image")
        elif image_description:
            return self.embed_text(image_description)
        else:
            raise ValueError("Provide image_path or image_description")

    def embed_video_transcript(self, transcript: str,
                               duration_seconds: Optional[float] = None) -> np.ndarray:
        """
        Embed a video via its transcript / frame descriptions.
        REAL mode: sends as text (Gemini Emb 2 unifies the space).
        MOCK mode: embeds with video-space salt to simulate cross-modal proximity.
        """
        if self.mock:
            return _mock_embed(transcript, salt="video")
        return self._call_api(transcript, modality="video")

    def embed_any(
        self,
        text: Optional[str] = None,
        image_path: Optional[str] = None,
        image_description: Optional[str] = None,
        video_transcript: Optional[str] = None,
    ) -> np.ndarray:
        """
        Embed the richest available representation.
        Combines signals if multiple are present (average of unit vectors).
        """
        vecs = []
        if text:
            vecs.append(self.embed_text(text))
        if image_path or image_description:
            vecs.append(self.embed_image(image_path, image_description))
        if video_transcript:
            vecs.append(self.embed_video_transcript(video_transcript))
        if not vecs:
            raise ValueError("Provide at least one of text, image_path, video_transcript")
        combined = np.mean(vecs, axis=0).astype(np.float32)
        norm = np.linalg.norm(combined)
        return (combined / norm) if norm > 0 else combined

    # ── Internal ──────────────────────────────────────────────────────────────

    def _call_api(self, content, modality: str = "text") -> np.ndarray:
        from google.genai import types
        response = self._client.models.embed_content(
            model=GEMINI_EMBED_MODEL,
            contents=content,
            config=types.EmbedContentConfig(task_type="SEMANTIC_SIMILARITY"),
        )
        vec = np.array(response.embeddings[0].values, dtype=np.float32)
        norm = np.linalg.norm(vec)
        return (vec / norm) if norm > 0 else vec

    def cosine_similarity(self, a: np.ndarray, b: np.ndarray) -> float:
        return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))
