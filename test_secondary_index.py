"""Test feature C: secondary metadata indexes.

Loads the compiled C++ extension (.so) directly, bypassing feather_db/__init__.py
(which imports heavy ML submodules we don't need and that aren't in the test venv).
"""
import os, glob, tempfile, importlib.util, numpy as np

_so = sorted(glob.glob("feather_db/core*.so"))[0]
_spec = importlib.util.spec_from_file_location("core", _so)
feather_db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(feather_db)
Metadata, ContextType = feather_db.Metadata, feather_db.ContextType

DIM = 8
def vec(): return np.random.rand(DIM).astype(np.float32)

def mk(ns, eid, **attrs):
    m = Metadata()
    m.type = ContextType.FACT
    m.namespace_id = ns
    m.entity_id = eid
    for k, v in attrs.items():
        m.set_attribute(k, v)
    return m

fails = []
def check(name, cond):
    print(f"  {'PASS' if cond else 'FAIL'}: {name}")
    if not cond: fails.append(name)

path = tempfile.mktemp(suffix=".feather")
db = feather_db.DB.open(path, dim=DIM)

# --- build via add ---
db.add(id=1, vec=vec(), meta=mk("acme", "u1", channel="instagram"))
db.add(id=2, vec=vec(), meta=mk("acme", "u2", channel="instagram"))
db.add(id=3, vec=vec(), meta=mk("acme", "u3", channel="email"))
db.add(id=4, vec=vec(), meta=mk("globex", "u4", channel="email"))

print("1) basic index lookups")
check("acme has 3", sorted(db.ids_in_namespace("acme")) == [1,2,3])
check("globex has 1", db.ids_in_namespace("globex") == [4])
check("namespace_size acme=3", db.namespace_size("acme") == 3)
check("entity u4", db.ids_for_entity("u4") == [4])
check("attr channel=instagram -> {1,2}", sorted(db.ids_with_attribute("channel","instagram")) == [1,2])
check("attr channel=email -> {3,4}", sorted(db.ids_with_attribute("channel","email")) == [3,4])
check("list_namespaces", sorted(db.list_namespaces()) == ["acme","globex"])
check("missing ns -> []", db.ids_in_namespace("nope") == [])

print("2) update_metadata moves id across indexes")
db.update_metadata(2, mk("globex", "u2", channel="email"))  # move 2 acme->globex, instagram->email
check("acme now {1,3}", sorted(db.ids_in_namespace("acme")) == [1,3])
check("globex now {2,4}", sorted(db.ids_in_namespace("globex")) == [2,4])
check("instagram now {1}", db.ids_with_attribute("channel","instagram") == [1])
check("email now {2,3,4}", sorted(db.ids_with_attribute("channel","email")) == [2,3,4])

print("3) re-add updates index (no stale entries)")
db.add(id=1, vec=vec(), meta=mk("acme", "u1", channel="email"))  # change 1's channel
check("instagram now empty", db.ids_with_attribute("channel","instagram") == [])
check("email now {1,2,3,4}", sorted(db.ids_with_attribute("channel","email")) == [1,2,3,4])

print("4) forget removes from candidate sets")
db.forget(3)
check("acme now {1} (3 forgotten)", db.ids_in_namespace("acme") == [1])
check("email now {1,2,4}", sorted(db.ids_with_attribute("channel","email")) == [1,2,4])

print("5) purge removes whole namespace")
removed = db.purge("globex")
check("purge globex removed 2", removed == 2)
check("globex empty", db.ids_in_namespace("globex") == [])
check("email now {1}", db.ids_with_attribute("channel","email") == [1])

print("6) persistence: save + reload rebuilds indexes (dead excluded)")
db.save()
del db
db2 = feather_db.DB.open(path, dim=DIM)
check("reload acme {1}", db2.ids_in_namespace("acme") == [1])
check("reload globex empty (purged)", db2.ids_in_namespace("globex") == [])
check("reload no forgotten id 3", 3 not in db2.ids_in_namespace("acme"))
check("reload email {1}", db2.ids_with_attribute("channel","email") == [1])
check("reload list_namespaces", db2.list_namespaces() == ["acme"])

os.remove(path)
print()
print("ALL PASS" if not fails else f"{len(fails)} FAILURES: {fails}")
exit(1 if fails else 0)
