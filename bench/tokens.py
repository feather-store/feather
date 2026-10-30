"""Token counting for context budgets, with the answer model's tokenizer.

Gemini's tokenizer runs locally through google-genai's LocalTokenizer
(needs `sentencepiece` and `protobuf`; the tokenizer model is downloaded
once and cached). There is deliberately no estimate fallback: a budget
reported in tokens that was really chars/4 is the error this replaces.
"""
from __future__ import annotations
import threading
import warnings
from typing import Callable

# Raised on every load AND every count; catch_warnings is not thread-safe.
warnings.filterwarnings("ignore", message=".*local tokenizer.*")


def gemini_counter(model: str = "gemini-2.5-flash") -> Callable[[str], int]:
    """count(text) -> tokens. One tokenizer per thread (~1 s to load):
    questions run on worker threads, and a shared one behind a lock would
    serialise them."""
    from google.genai.local_tokenizer import LocalTokenizer
    LocalTokenizer(model_name=model)                 # fail here, not mid-run
    local = threading.local()

    def count(text: str) -> int:
        tok = getattr(local, "tok", None)
        if tok is None:
            tok = local.tok = LocalTokenizer(model_name=model)
        return tok.count_tokens(text).total_tokens
    return count
