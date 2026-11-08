"""The evaluation protocol: paired statistics, frozen splits, the committed-tree
rule for test runs, and the LoCoMo answer-key audit. All offline: no model,
no network (the audit and dataset loaders are replaced where needed)."""
import json
import random

import numpy as np
import pytest

from bench import stats


# ── statistics ────────────────────────────────────────────────────────────

def _runs(n_q, acc, n_runs, flip=0.0, seed=0, base=None):
    """n_runs runs over n_q questions: a base pattern at accuracy `acc`, each
    run flipping a `flip` share of it (rerun noise)."""
    rng = np.random.default_rng(seed)
    base = base if base is not None else (rng.random(n_q) < acc)
    runs = []
    for _ in range(n_runs):
        r = base ^ (rng.random(n_q) < flip)
        runs.append({f"q{i}": float(v) for i, v in enumerate(r)})
    return runs, base


def test_mcnemar_exact_values():
    assert stats.mcnemar_p(0, 0) == 1.0
    assert abs(stats.mcnemar_p(0, 10) - 2 / 1024) < 1e-12
    assert stats.mcnemar_p(3, 9) == stats.mcnemar_p(9, 3)


def test_identical_configurations_are_not_distinguishable():
    a, base = _runs(800, 0.8, 3, flip=0.03, seed=1)
    b, _ = _runs(800, 0.8, 3, flip=0.03, seed=2, base=base)
    c = stats.compare(a, b)
    assert c["verdict"] == "not distinguishable from noise"
    assert c["ci95"][0] < 0 < c["ci95"][1]
    assert c["warnings"] == []
    # the measured rerun noise is the flip rate that was put in, ~2 x 3%
    assert 0.04 < c["noise_a"]["flip_rate"] < 0.08


def test_a_real_effect_is_found_and_its_ci_excludes_zero():
    a, base = _runs(1000, 0.7, 3, flip=0.03, seed=3)
    better = base.copy()
    better[np.where(~base)[0][:100]] = True          # +10 points on the same questions
    b, _ = _runs(1000, 0.0, 3, flip=0.03, seed=4, base=better)
    c = stats.compare(a, b)
    assert c["verdict"] == "B better" and c["ci95"][0] > 0.05
    assert c["mcnemar_p"] < 1e-6 and c["b_only"] > c["a_only"]


def test_one_run_per_side_warns_and_a_deterministic_metric_does_not():
    a, base = _runs(200, 0.5, 1, seed=5)
    b, _ = _runs(200, 0.5, 1, seed=6)
    assert stats.compare(a, b)["warnings"]
    c = stats.compare(a, b, deterministic=True)
    assert not c["warnings"] and "deterministic" in c["note"]


def test_a_tied_majority_is_not_counted_as_a_flip():
    a = [{"q": 1.0}, {"q": 0.0}]                      # 2 runs, split evenly
    b = [{"q": 0.0}, {"q": 0.0}]
    c = stats.compare(a, b)
    assert c["a_only"] == 0 and c["b_only"] == 0 and c["diff"] == -0.5


def test_only_questions_every_run_answered_are_compared():
    a = [{"q1": 1.0, "q2": 1.0}, {"q1": 1.0}]
    b = [{"q1": 0.0, "q2": 0.0}]
    c = stats.compare(a, b)
    assert c["n"] == 1 and c["unpaired"]["only_some_runs_of_a_or_b"] == 1


def test_bootstrap_ci_covers_the_mean_and_narrows_with_n():
    rng = np.random.default_rng(0)
    small = rng.random(100) < 0.6
    large = rng.random(10000) < 0.6
    lo1, hi1 = stats.bootstrap_ci(small.astype(float))
    lo2, hi2 = stats.bootstrap_ci(large.astype(float))
    assert lo1 < small.mean() < hi1 and lo2 < large.mean() < hi2
    assert (hi2 - lo2) < (hi1 - lo1) / 5


# ── the committed-tree rule ───────────────────────────────────────────────

@pytest.fixture
def dirty(monkeypatch):
    import bench.runner as runner
    monkeypatch.setattr(runner, "_git_dirty", lambda: True)


def test_a_test_run_needs_a_committed_tree(dirty):
    from bench.protocol import require_committed
    with pytest.raises(SystemExit, match="TEST questions"):
        require_committed("something")


def test_lme_guard_refuses_test_questions_from_a_dirty_tree(dirty):
    from types import SimpleNamespace
    from bench.__main__ import _held_out_guard
    from bench.datasets.lme_split import dev_ids
    dev = sorted(dev_ids())
    llm = SimpleNamespace(judge="llm")
    _held_out_guard(llm, [{"question_id": q} for q in dev[:5]])          # dev only: fine
    _held_out_guard(SimpleNamespace(judge="exact"), [{"question_id": "x_test"}])  # no model
    with pytest.raises(SystemExit):
        _held_out_guard(llm, [{"question_id": dev[0]}, {"question_id": "not_a_dev_id"}])


# ── LoCoMo ────────────────────────────────────────────────────────────────

def test_the_locomo_split_is_frozen_by_conversation():
    from bench.datasets import locomo_split as ls
    dev = ls.dev_conversations()
    assert dev == {"conv-43", "conv-44", "conv-50"}
    every = json.loads(ls.SPLIT_FILE.read_text())
    assert set(every["test"]).isdisjoint(dev) and len(every["test"]) == 7
    assert set(ls.make_split(every["dev"] + every["test"])) == dev    # the rule reproduces it


def _fake_data():
    def conv(sid, qs):
        return {"sample_id": sid, "qa": [{"question": q, "answer": a, "category": c}
                                         for q, a, c in qs]}
    return [conv("conv-44", [("When did A paint?", "2022", 2), ("What does A make?", "art", 4)]),
            conv("conv-26", [("Where does B live?", "Austin", 4)])]


def test_a_test_conversation_is_refused_from_a_dirty_tree(dirty, monkeypatch):
    from bench import locomo
    monkeypatch.setattr(locomo, "load", _fake_data)
    with pytest.raises(SystemExit, match="conv-26"):
        locomo.run("conv-26")


def test_the_audit_mapping_is_checked_against_the_data(monkeypatch):
    from bench import locomo
    good = [{"question_id": "locomo_0_qa1", "question": "What does A make?",
             "golden_answer": "art", "category": 4, "error_type": "AMBIGUOUS",
             "correct_answer": "paintings"},
            {"question_id": "locomo_0_qa0", "question": "When did A paint?",
             "golden_answer": "2022", "category": 2, "error_type": "WRONG_CITATION",
             "correct_answer": "2022"}]
    monkeypatch.setattr(locomo, "load_audit", lambda: good)
    corr = locomo.audit_corrections(_fake_data())
    assert corr == {("conv-44", 1): {"error_type": "AMBIGUOUS", "correct_answer": "paintings"}}
    bad = [dict(good[0], question_id="locomo_1_qa0")]           # points at another question
    monkeypatch.setattr(locomo, "load_audit", lambda: bad)
    with pytest.raises(SystemExit, match="does not match"):
        locomo.audit_corrections(_fake_data())


def _rec(sid, qi, score, cat=4, **kw):
    return {"sample_id": sid, "qa_index": qi, "category": cat, "category_name": "single-hop",
            "question": f"q{qi}", "gold": "g", "response": "r", "llm_score": score,
            "f1": 0.0, "bleu1": 0.0, "evidence_recall": 1.0, **kw}


def test_the_three_answer_keys():
    from bench import locomo
    corr = {("conv-44", 1): {"error_type": "AMBIGUOUS", "correct_answer": "paintings"}}
    recs = [_rec("conv-44", 0, 1), _rec("conv-44", 1, 0)]
    assert [r["llm_score"] for r in locomo.scored(recs, "original", corr)] == [1, 0]
    assert len(locomo.scored(recs, "sound", corr)) == 1
    with pytest.raises(SystemExit, match="rekey"):
        locomo.scored(recs, "audited", corr)
    rekeyed = [recs[0], dict(recs[1], gold_audited="paintings", llm_score_audited=1)]
    out = locomo.scored(rekeyed, "audited", corr)
    assert [r["llm_score"] for r in out] == [1, 1] and out[1]["gold"] == "paintings"


def test_compare_locomo_end_to_end(tmp_path, monkeypatch, capsys):
    from bench import compare, locomo
    monkeypatch.setattr(locomo, "audit_corrections",
                        lambda data=None: {("conv-44", 0): {"error_type": "HALLUCINATION",
                                                            "correct_answer": "x"}})
    rng = random.Random(0)
    paths = {}
    for side, acc in (("a", 0.5), ("b", 0.9)):
        paths[side] = []
        for rep in range(3):
            p = tmp_path / f"{side}{rep}.jsonl"
            with open(p, "w") as fh:
                for sid in ("conv-44", "conv-26"):
                    for qi in range(150):
                        fh.write(json.dumps(_rec(sid, qi, int(rng.random() < acc))) + "\n")
            paths[side].append(str(p))
    compare.main(["locomo", "--a", *paths["a"], "--b", *paths["b"], "--split", "dev",
                  "--key", "sound"])
    out = capsys.readouterr().out
    assert "n=149" in out and "B better" in out and "rerun noise A: 3 runs" in out


# ── contradiction ─────────────────────────────────────────────────────────

def test_the_contradiction_test_lexicon_shares_nothing_with_dev():
    from bench import contradiction as C
    dn, do, da, dd = C.LEXICON["dev"]
    tn, to, ta, td = C.LEXICON["test"]
    assert not set(dn) & set(tn) and not set(do) & set(to) and not set(da) & set(ta)
    dev_words = {v for a in da for v in C.ATTRS[a]["values"]} | {
        p for a in da for p in C.ATTRS[a]["say"]}
    test_words = {v for a in ta for v in C.ATTRS[a]["values"]} | {
        p for a in ta for p in C.ATTRS[a]["say"]}
    assert not dev_words & test_words
    assert not {t for t, _ in dd} & {t for t, _ in td}
    facts, _ = C.make_world(random.Random(1), 4, 5, 0.0, "test")
    assert {a for _, a, _ in facts} == set(ta)


def test_contradiction_records_feed_a_paired_comparison(tmp_path, capsys):
    from bench import compare, contradiction as C
    rep = C.run(trials=2, variants=("notes", "notes + keys"))
    assert len(rep["records"]) == 2 * 8 * 5 * 2
    p = tmp_path / "r.json"
    p.write_text(json.dumps(rep))
    compare.main(["contradiction", "--a", str(p), "--variant-a", "notes", "--b", str(p),
                  "--variant-b", "notes + keys", "--metric", "current_first"])
    out = capsys.readouterr().out
    assert "deterministic metric" in out and "n=80" in out


def test_contradiction_test_split_is_refused_from_a_dirty_tree(dirty):
    from bench import contradiction as C
    with pytest.raises(SystemExit, match="TEST"):
        C.run(trials=1, split="test", variants=("notes",))


# ── clustered questions ───────────────────────────────────────────────────

def _clustered(n_clusters, per, effect_in, seed=0):
    """A and B equal everywhere except B fixes every question of the first
    `effect_in` clusters: an effect that does not generalise across clusters."""
    rng = np.random.default_rng(seed)
    a, b, cl = {}, {}, {}
    for c in range(n_clusters):
        for i in range(per):
            q = f"c{c}#{i}"
            a[q] = float(rng.random() < 0.6)
            b[q] = 1.0 if c < effect_in else a[q]
            cl[q] = f"c{c}"
    return [a], [b], cl


def test_a_cluster_interval_is_wider_and_must_also_exclude_zero():
    a, b, cl = _clustered(10, 60, effect_in=1)
    plain = stats.compare(a, b, deterministic=True)
    clustered = stats.compare(a, b, clusters=cl, deterministic=True)
    assert plain["verdict"] == "B better"                      # questions look independent
    lo, hi = clustered["ci95_cluster"]
    assert hi - lo > plain["ci95"][1] - plain["ci95"][0]
    assert lo <= 0 and clustered["verdict"] == "not distinguishable from noise"


def test_too_few_clusters_warns_and_gives_no_cluster_interval():
    a, b, cl = _clustered(3, 60, effect_in=3)
    c = stats.compare(a, b, clusters=cl, deterministic=True)
    assert c["ci95_cluster"] is None and any("too few" in w for w in c["warnings"])


# ── the strict judge, rejudge and leniency (stub models) ─────────────────

class _StubChat:
    """mem0 prompt (json mode): CORRECT whenever the answer shares a word
    with the gold (lenient). strict prompt: yes only if the gold is in it."""
    calls = 0

    def __init__(self, spec, role):
        pass

    def __call__(self, messages, max_tokens=None, temperature=0.0, json_mode=False):
        type(self).calls += 1
        text = messages[-1]["content"]
        if json_mode:
            gold = text.rsplit("Gold answer:", 1)[1].split("\n")[0].strip().lower()
            resp = text.rsplit("Generated answer:", 1)[1].split("\n")[0].strip().lower()
            ok = bool(set(gold.split()) & set(resp.split()))
            return json.dumps({"label": "CORRECT" if ok else "WRONG"})
        gold = text.split("Correct Answer:")[1].split("Model Response:")[0].strip().lower()
        resp = text.split("Model Response:")[1].split("Is the model")[0].strip().lower()
        return "yes" if gold in resp else "no"


@pytest.fixture
def stub_models(monkeypatch, tmp_path):
    from bench import locomo
    monkeypatch.setattr(locomo, "_Chat", _StubChat)
    monkeypatch.setattr(locomo, "CACHE", tmp_path)
    _StubChat.calls = 0
    return locomo


def test_the_strict_judge_uses_the_official_prompts(stub_models):
    L = stub_models
    seen = []

    def chat(messages, **kw):
        seen.append(messages[0]["content"])
        return "Yes."
    assert L.judge_strict(chat, 2, "When?", "7 May 2023", "on 7 May 2023") == (1, "Yes.")
    L.judge_strict(chat, 4, "What?", "art", "art")
    assert "off-by-one" in seen[0] and "off-by-one" not in seen[1]


def test_rejudge_adds_strict_scores_and_caches_them(stub_models, monkeypatch, tmp_path):
    L = stub_models
    monkeypatch.setattr(L, "audit_corrections", lambda data=None: {
        ("conv-44", 1): {"error_type": "AMBIGUOUS", "correct_answer": "oil paintings"}})
    recs = [_rec("conv-44", 0, 1, question="What car?", gold="red Ferrari",
                 response="a red sports car"),
            _rec("conv-44", 1, 0, question="What art?", gold="abstract art",
                 response="oil paintings")]
    src = tmp_path / "run__questions.jsonl"
    src.write_text("".join(json.dumps(r) + "\n" for r in recs))
    out = L.rejudge(str(src))
    got = L.read_records(out)
    assert [r["strict_score"] for r in got] == [0, 0]          # "red sports car" is not it
    assert got[1]["strict_score_audited"] == 1 and "strict_score_audited" not in got[0]
    calls = _StubChat.calls
    L.rejudge(str(src))
    assert _StubChat.calls == calls                            # all from the cache
    assert [r["llm_score"] for r in L.scored(got, "original", judge="strict")] == [0, 0]
    with pytest.raises(SystemExit, match="rejudge"):
        L.scored(recs, "original", judge="strict")


def test_leniency_counts_accepted_wrong_answers_on_sound_questions(stub_models, monkeypatch):
    L = stub_models
    monkeypatch.setattr(L, "load", _fake_data)
    monkeypatch.setattr(L, "audit_corrections", lambda data=None: {
        ("conv-44", 1): {"error_type": "AMBIGUOUS", "correct_answer": "paintings"}})
    monkeypatch.setattr(L, "load_leniency", lambda: [
        {"question_id": "locomo_0_qa0", "question": "When did A paint?",
         "golden_answer": "7 May 2022", "generated_answer": "sometime in 2022", "category": "2"},
        {"question_id": "locomo_0_qa1", "question": "What does A make?",
         "golden_answer": "art", "generated_answer": "crafts", "category": "4"},
        {"question_id": "locomo_1_qa0", "question": "Where does B live?",
         "golden_answer": "Austin", "generated_answer": "Dallas", "category": "4"}])
    out = L.leniency()
    assert out["n"] == 2                                       # the wrong-key one left out
    assert out["mem0"]["accepts_wrong"] == 0.5 and out["strict"]["accepts_wrong"] == 0.0


# ── housekeeping ──────────────────────────────────────────────────────────

def test_an_unused_claim_is_released_and_a_used_one_kept(tmp_path):
    from bench import runner
    empty = runner.reserve("x", ".jsonl", tmp_path)
    used = runner.reserve("x", ".jsonl", tmp_path)
    used.write_text("{}\n")
    assert runner.release_unused_claims() >= 1
    assert not empty.exists() and used.exists()


def test_compare_refuses_another_benchs_judge(tmp_path):
    from bench import compare
    p = tmp_path / "a.jsonl"
    p.write_text("")
    with pytest.raises(SystemExit):
        compare.main(["lme", "--a", str(p), "--b", str(p), "--judge", "strict"])
