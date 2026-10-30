"""The bench CLI must not measure something other than what was asked
for, or report it as if it had: `--limit` once defaulted to 5, and a
"final" LongMemEval run scored 5 questions at 1.000."""
import json
from types import SimpleNamespace

import pytest

from bench import __main__ as cli


def _args(**kw):
    base = dict(limit=None, split="all", dataset="s", judge="llm",
                embedder="gemini")
    return SimpleNamespace(**{**base, **kw})


GOOD = {"official_judge": "official=openrouter/openai/gpt-4o-2024-08-06",
        "n_questions": 500, "n_scored": 500, "run_errors": 0,
        "judge_errors": 0}


def test_limit_defaults_to_every_question(monkeypatch):
    seen = {}
    def load(variant, limit):
        seen["limit"] = limit
        return []
    monkeypatch.setattr(cli, "load_longmemeval", load)
    monkeypatch.setattr(cli, "lme_scenario", SimpleNamespace(
        run=lambda *a, **k: (_ for _ in ()).throw(SystemExit(0))))
    for argv, want in ((["run", "longmemeval", "--dataset", "s"], None),
                       (["run", "longmemeval", "--dataset", "s", "--limit", "0"], None),
                       (["run", "longmemeval", "--dataset", "s", "--limit", "7"], 7)):
        seen.clear()
        with pytest.raises(SystemExit):
            cli.main(argv)
        assert seen["limit"] == want, argv


def test_split_and_limit_together_are_refused(capsys):
    assert cli.main(["run", "longmemeval", "--dataset", "s", "--split", "dev",
                     "--limit", "5"]) == 2
    assert "--split needs" in capsys.readouterr().err


def test_a_full_official_committed_run_is_reportable():
    assert cli.lme_reportability(_args(), GOOD, dirty=False) == []


@pytest.mark.parametrize("args,metrics,dirty,reason", [
    (dict(limit=5), {}, False, "--limit 5"),
    (dict(split="dev"), {}, False, "--split dev"),
    (dict(dataset="oracle"), {}, False, "--dataset oracle"),
    (dict(judge="substring"), {}, False, "--judge substring"),
    (dict(embedder="deterministic"), {}, False, "--embedder deterministic"),
    ({}, {"official_judge": None}, False, "no official judge"),
    ({}, {"official_judge": "official=openrouter/openai/gpt-4o-mini"}, False,
     "not gpt-4o-2024-08-06"),
    ({}, {"n_scored": 480}, False, "480 of 500 questions scored"),
    ({}, {"run_errors": 3}, False, "3 questions failed to run"),
    ({}, {"judge_errors": 2}, False, "2 questions were not judged"),
    ({}, {}, True, "uncommitted changes"),
])
def test_each_way_a_result_is_not_the_benchmark(args, metrics, dirty, reason):
    why = cli.lme_reportability(_args(**args), {**GOOD, **metrics}, dirty)
    assert any(reason in w for w in why), why
