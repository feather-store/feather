"""Fair admission: one namespace's bulk load cannot take every slot of a
class. Another namespace's request is admitted at once, and the two share.
Driven through the ASGI middleware itself, with a handler that takes 20 ms."""
import asyncio
import os
import sys
import time

import pytest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
API_DIR = os.path.join(REPO, "feather-api")
pytest.importorskip("fastapi", reason="Cloud API tests need fastapi")
if API_DIR not in sys.path:
    sys.path.insert(0, API_DIR)


def _admission(monkeypatch, fair: bool, limit: int = 16):
    for mod in [m for m in sys.modules if m == "app" or m.startswith("app.")]:
        del sys.modules[mod]
    from app import admission as A
    monkeypatch.setattr(A, "FAIR", fair)
    monkeypatch.setitem(A.LIMITS, "write", limit)
    for d in (A.Admission.inflight, A.Admission.rejected, A.Admission.rejected_fair):
        d["write"] = 0
    A.Admission.held["write"] = {}

    async def handler(scope, receive, send):
        await asyncio.sleep(0.02)
        await send({"type": "http.response.start", "status": 200, "headers": []})
        await send({"type": "http.response.body", "body": b"{}"})
    return A, A.Admission(handler)


async def _call(app, ns):
    status = {}

    async def send(msg):
        if msg["type"] == "http.response.start":
            status["s"] = msg["status"]

    async def receive():
        return {"type": "http.request", "body": b"", "more_body": False}
    await app({"type": "http", "method": "POST", "path": f"/v1/{ns}/vectors",
               "headers": []}, receive, send)
    return status["s"]


async def _scenario(app, retry_after=1.0):
    """A bulk loader keeps 64 writes in flight against `bulk`; a small
    tenant writes 20 times, one at a time, retrying a 429 after
    Retry-After. Returns the small tenant's latencies and 429 count."""
    stop = False

    async def loader():
        # Keeps pressure on: retries a refusal at once (a 429 never suspends,
        # so yield to let everyone else run).
        while not stop:
            if await _call(app, "bulk") == 429:
                await asyncio.sleep(0.001)

    loaders = [asyncio.create_task(loader()) for _ in range(64)]
    await asyncio.sleep(0.1)                      # the load is under way
    lat, refused, served = [], 0, 0
    for _ in range(20):
        t = time.perf_counter()
        for _attempt in range(5):                  # then the client gives up
            if await _call(app, "small") == 200:
                served += 1
                lat.append(time.perf_counter() - t)
                break
            refused += 1
            await asyncio.sleep(retry_after)
    stop = True
    await asyncio.gather(*loaders)
    return sorted(lat), refused, served


def test_another_namespace_is_served_during_a_bulk_load(monkeypatch):
    A, app = _admission(monkeypatch, fair=True)
    lat, refused, served = asyncio.run(_scenario(app, retry_after=0.05))
    assert refused == 0 and served == 20           # admitted at once, every time
    assert lat[-1] < 0.2, lat                      # p100 ~ one handler call
    assert A.Admission.rejected_fair["write"] > 0  # the loader was held to its share
    assert A.Admission.held["write"] == {}


def test_without_fairness_the_bulk_load_takes_every_slot(monkeypatch):
    A, app = _admission(monkeypatch, fair=False)
    lat, refused, served = asyncio.run(_scenario(app, retry_after=0.05))
    # A finished request's slot goes straight back to the loader that held
    # it: the small tenant is refused every time (what fairness fixes).
    assert refused > 0 and served < 20


def test_a_tenant_alone_leaves_a_reserve_and_shares_evenly():
    from app.admission import fair_cap
    assert fair_cap(64, 0) == 56 and fair_cap(64, 1) == 32 and fair_cap(64, 3) == 16
    assert fair_cap(2, 0) == 2                     # a tiny limit has no reserve
    assert fair_cap(4, 10) == 1                    # never starved to zero
