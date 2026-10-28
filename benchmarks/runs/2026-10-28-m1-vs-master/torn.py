import os, sys, subprocess, time, tempfile, numpy as np
REPO = sys.argv[1]; secs = float(sys.argv[2])
sys.path.insert(0, REPO)
import feather_db
d = tempfile.mkdtemp(); path = os.path.join(d, "t.feather")
writer = subprocess.Popen([sys.executable, "-c", f"""
import sys, time, numpy as np
sys.path.insert(0, {REPO!r})
import feather_db
db = feather_db.DB.open({path!r}, dim=4)
i, end = 0, time.time() + {secs}
while time.time() < end:
    m = feather_db.Metadata(); m.content = str(i)
    db.add(i, np.full(4, i, np.float32), m); i += 1
    if i % 7 == 0: db.save()
db.close()
"""])
time.sleep(0.3)
opens = torn = 0
end = time.time() + secs - 0.5
while time.time() < end:
    try:
        r = feather_db.DB.open(path, dim=4, read_only=True)
    except Exception as e:
        continue
    ids = sorted(r.get_all_ids("text"))
    r.close()
    opens += 1
    if ids and ids != list(range(len(ids))):
        torn += 1
writer.wait()
print(f"{torn} of {opens} read-only opens saw a torn view")
