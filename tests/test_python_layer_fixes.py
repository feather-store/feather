"""Python-layer fixes shipped with the concurrency work (0.19).

* IngestPipeline continues id sequences after what is already stored (a new
  pipeline on an existing DB used to restart at base+1 and overwrite records)
  and re-learns stored entities.
* Internal self-queries (contradiction checks, MMR over-fetch, tool
  over-fetch) no longer count as recalls.
"""
import numpy as np

import feather_db
from feather_db import DB, MemoryManager
from feather_db.extractors import FactExtractor, EntityResolver
from feather_db.pipelines import IngestPipeline, IngestRecord
from feather_db.pipelines.ingest import SOURCE_ID_BASE, FACT_ID_BASE, ENTITY_ID_BASE
from feather_db.triggers import ContradictionDetector


class _Provider:
    def __init__(self, responses):
        self._r = list(responses)

    def complete(self, messages, max_tokens=512, temperature=0.0):
        return self._r.pop(0) if self._r else "[]"


class _Embed:
    def embed(self, text):
        rng = np.random.default_rng(abs(hash(text)) % (2 ** 32))
        v = rng.standard_normal(32).astype(np.float32)
        return v / np.linalg.norm(v)


_FACTS = '[{"subject":"Acme","predicate":"launched","object":"Sale","confidence":0.9,"valid_at":null}]'
_ENTS = ('[{"surface_form":"Acme","canonical_id":"brand::acme","kind":"Brand","confidence":0.9,"aliases":[]},'
         '{"surface_form":"Sale","canonical_id":"campaign::sale","kind":"Campaign","confidence":0.9,"aliases":[]}]')


def _pipeline(db):
    return IngestPipeline(db=db, embedder=_Embed(),
                          fact_extractor=FactExtractor(_Provider([_FACTS])),
                          entity_resolver=EntityResolver(_Provider([_ENTS])),
                          namespace="acme")


def test_new_pipeline_does_not_overwrite_earlier_records(tmp_path):
    p = str(tmp_path / "pipe.feather")
    db = DB.open(p, dim=32)
    _pipeline(db).ingest([IngestRecord(content="Acme launched the Sale.", source_id="m1")])
    first_source = db.get_metadata(SOURCE_ID_BASE + 1).content
    db.close()

    db = DB.open(p, dim=32)            # "server restart": brand-new pipeline instance
    n_before = db.size()
    _pipeline(db).ingest([IngestRecord(content="Acme launched the Sale again.", source_id="m2")])
    assert db.get_metadata(SOURCE_ID_BASE + 1).content == first_source, "record overwritten"
    assert db.get_metadata(SOURCE_ID_BASE + 2) is not None
    assert db.get_metadata(FACT_ID_BASE + 2) is not None
    # entities were recognised, not re-created
    entities = db.ids_with_attribute("kind", "entity")
    assert len(entities) == 2, f"expected the 2 known entities, got {sorted(entities)}"
    assert db.size() == n_before + 2    # one new source + one new fact
    db.close()


def test_contradiction_check_does_not_count_as_recall(tmp_path):
    db = DB.open(str(tmp_path / "cd.feather"), dim=8)
    v = np.ones(8, dtype=np.float32)
    for i, src in enumerate(["a", "b", "c"]):
        m = feather_db.Metadata()
        m.source = src
        db.add(i, v + i * 1e-4, m)
    ContradictionDetector().check(db, 0, new_vec=v)
    assert all(db.get_metadata(i).recall_count == 0 for i in range(3))
    db.close()


def test_mmr_counts_only_returned_records(tmp_path):
    db = DB.open(str(tmp_path / "mmr.feather"), dim=8)
    rng = np.random.default_rng(1)
    for i in range(50):
        db.add(i, rng.random(8).astype(np.float32), feather_db.Metadata())
    out = MemoryManager.search_mmr(db, rng.random(8).astype(np.float32), k=3, fetch_k=30)
    touched = [i for i in range(50) if db.get_metadata(i).recall_count > 0]
    assert sorted(touched) == sorted(r.id for r in out)
    db.close()
