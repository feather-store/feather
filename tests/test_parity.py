"""One feature set in every mode: the Cloud API cannot drift from the engine.

`feather_db.remote.PARITY` names, for every public method of the embedded
`DB`, the route that serves it over HTTP (or why it has none). This fails when:
  * the engine gains a public method that PARITY does not mention;
  * PARITY names a route the Cloud API does not have;
  * RemoteDB lacks the method, or takes different arguments than DB does.
Behaviour is held equal by tests/test_contracts.py, which runs the same tests
against both.
"""
import inspect
import os
import re
import sys

import pytest

import feather_db
from feather_db.remote import PARITY, RemoteDB

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")


def public_db_methods():
    return sorted(n for n in dir(feather_db.DB) if not n.startswith("_"))


def test_every_engine_method_is_in_parity():
    missing = [m for m in public_db_methods() if m not in PARITY]
    assert not missing, (f"DB methods with no Cloud API route and no stated reason: {missing}. "
                         "Add a route and map it in feather_db/remote.py PARITY "
                         "(or state why it cannot have one).")
    stale = [m for m in PARITY if m not in public_db_methods()]
    assert not stale, f"PARITY names methods DB no longer has: {stale}"


def test_every_mapped_route_exists(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    monkeypatch.setenv("FEATHER_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FEATHER_DEV_MODE", "1")
    if API_DIR not in sys.path:
        sys.path.insert(0, API_DIR)
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app.main import app
    routes = {(m, r.path) for r in app.routes for m in getattr(r, "methods", ()) or ()}
    for method, spec in PARITY.items():
        if spec.startswith("-:"):
            continue
        verb, path = spec.split(" ", 1)
        assert (verb, "/v1/{namespace}" + path) in routes, f"{method}: no route {spec}"


def _db_params(name):
    """Argument names from pybind11's signature line: name(self: DB, a: T = d, ...)."""
    doc = getattr(feather_db.DB, name).__doc__ or ""
    m = re.match(rf"{name}\((.*?)\)\s*->", doc.splitlines()[0] if doc else "")
    if not m:
        return None
    return [p.split(":")[0].strip() for p in re.split(r",\s*(?![^\[]*\])", m.group(1))
            if p.strip() and not p.strip().startswith("self")]


def test_remotedb_has_every_method_with_the_same_arguments():
    for name, spec in PARITY.items():
        if spec.startswith("-:") and name in ("open", "write_error", "idempotent_result"):
            continue
        assert hasattr(RemoteDB, name), f"RemoteDB has no {name}()"
        engine = _db_params(name)
        attr = getattr(RemoteDB, name)
        if engine is None or isinstance(attr, property):
            continue
        remote = [p for p in inspect.signature(attr).parameters if p != "self"]
        assert remote[:len(engine)] == engine, f"{name}: DB takes {engine}, RemoteDB takes {remote}"
