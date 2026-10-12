"""Porter stemming in the BM25 path.

Before this, keyword search matched only exact surface forms: a memory whose
content said "creative" was not found by searching "creatives". For an agent
memory that is worse than it looks — the caller cannot distinguish "no such
memory" from "stored, but you spelled it differently", so a spelling mismatch
reads as the agent having forgotten.

Indexing and querying both go through `tokenize()`, so they cannot disagree
about how a word is folded. The index is never serialised — it is rebuilt from
`content` on every load and after compact — so this needed no format change and
existing files re-index under the new rules the next time they open.
"""
import os

import numpy as np
import pytest

import feather_db
from feather_db import DB

DIM = 16


@pytest.fixture
def db(tmp_path):
    d = DB.open(str(tmp_path / "s.feather"), dim=DIM)
    yield d
    d.close()


def store(db, rid, text):
    m = feather_db.Metadata()
    m.content = text
    db.add(id=rid, vec=np.random.rand(DIM).astype(np.float32), meta=m)


def finds(db, query):
    return len(db.keyword_search(query, k=10)) > 0


# ── what it fixes ─────────────────────────────────────────────────────────

@pytest.mark.parametrize("stored,query", [
    ("the ugc creative won",        "creatives"),     # plural
    ("lookalike audiences",         "audience"),      # singular
    ("running the campaign",        "run"),           # verb form
    ("we approve the budget",       "approval"),      # -al
    ("organise the assets",         "organisation"),  # -ation
    ("the policy forbids it",       "policies"),      # y -> ies
    ("spend pacing ran hot",        "pace"),          # -ing
    ("creative fatigue appears",    "appear"),        # -s
])
def test_an_inflected_query_finds_the_stored_form(db, stored, query):
    store(db, 1, stored)
    assert finds(db, query), f"{query!r} did not find {stored!r}"


def test_it_works_in_both_directions(db):
    """Indexing and querying share one tokenizer, so folding cannot be
    asymmetric — storing the plural and searching the singular must work too."""
    store(db, 1, "several creatives were tested")
    assert finds(db, "creative")


def test_a_word_that_is_already_a_stem_is_untouched(db):
    store(db, 1, "the budget is confirmed")
    assert finds(db, "budget")


def test_short_words_are_not_mangled(db):
    store(db, 1, "the ads ran in may")
    assert finds(db, "ads")


# ── what it does NOT fix, recorded honestly ───────────────────────────────

def test_porter_does_not_unify_allergy_with_allergic(db):
    """Characterisation, not an aspiration.

    Porter stems "allergic" to "allerg" (step 4 strips -ic at m>1) and
    "allergy" to "allergi" (step 1c maps y to i). Different stems, so they do
    not match. This is a known limit of the algorithm, not a defect in the
    implementation, and it is the reason the docs still say that anything
    relying on paraphrase needs a real embedder.

    If this starts passing because the stemmer was improved, update the
    expectation — do not restore the old behaviour.
    """
    store(db, 1, "severely allergic to shellfish")
    assert finds(db, "allergic")
    assert not finds(db, "allergy")
    assert not finds(db, "allergies")


def test_stemming_is_not_a_substitute_for_semantics(db):
    """A paraphrase with no shared root still finds nothing. Stemming widens
    matching on word FORM; it does nothing for meaning."""
    store(db, 1, "severely allergic to shellfish")
    assert not finds(db, "what food should be avoided")


# ── the escape hatch ──────────────────────────────────────────────────────

def test_feather_stem_0_restores_exact_matching(tmp_path):
    """Read once per process, so this runs in a child."""
    import subprocess, sys, textwrap
    repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    src = textwrap.dedent(f"""
        import sys; sys.path.insert(0, {repo!r})
        import numpy as np, feather_db
        db = feather_db.DB.open({str(tmp_path / 'off.feather')!r}, dim=16)
        m = feather_db.Metadata(); m.content = "the ugc creative won"
        db.add(id=1, vec=np.random.rand(16).astype(np.float32), meta=m)
        print("PLURAL", len(db.keyword_search("creatives", k=5)))
        print("EXACT", len(db.keyword_search("creative", k=5)))
        db.close()
    """)
    env = {**os.environ, "FEATHER_STEM": "0"}
    out = subprocess.run([sys.executable, "-c", src], capture_output=True,
                         text=True, timeout=120, env=env)
    assert out.returncode == 0, out.stderr[-500:]
    assert "PLURAL 0" in out.stdout, "stemming was still on"
    assert "EXACT 1" in out.stdout


# ── it survives the rebuild path ──────────────────────────────────────────

def test_the_index_rebuilds_with_stemming_after_a_reload(tmp_path):
    """bm25_index_ is derived, not stored. An existing file must come back
    searchable under the new folding without any migration."""
    path = str(tmp_path / "r.feather")
    first = DB.open(path, dim=DIM)
    store(first, 1, "the ugc creative delivered the best roas")
    first.close()

    reopened = DB.open(path, dim=DIM)
    try:
        assert finds(reopened, "creatives")
        assert finds(reopened, "delivering")
    finally:
        reopened.close()
