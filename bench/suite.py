"""The structure-level evals in one command, LoCoMo optionally.

    python -m bench.suite                       # contradiction + poisoning: free, seconds
    python -m bench.suite --out-dir bench/results
    python -m bench.suite --locomo              # + LoCoMo, the 3 dev conversations (paid)

--model provider:model adds the reader / obedience step to both free evals
(paid, a few cents). LoCoMo's own options are `python -m bench.locomo run -h`.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from . import contradiction, poisoning


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default=None, help="reader for contradiction / poisoning")
    ap.add_argument("--locomo", action="store_true",
                    help="also run LoCoMo on its dev conversations (paid)")
    ap.add_argument("--out-dir", default=None, help="write each JSON report here")
    a = ap.parse_args(argv)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out = Path(a.out_dir) if a.out_dir else None
    if out:
        out.mkdir(parents=True, exist_ok=True)

    for name, mod, kw in (("contradiction", contradiction, {"model": a.model}),
                          ("poisoning", poisoning, {"model": a.model})):
        t0 = time.time()
        report = mod.run(**kw)
        report["wall_seconds"] = round(time.time() - t0, 2)
        print(f"\n## {name}\n\n{mod.table(report)}")
        if out:
            path = out / f"{name}__{stamp}.json"
            path.write_text(json.dumps(report, indent=2))
            print(f"\nwrote {path}", file=sys.stderr)

    if a.locomo:
        from . import locomo
        print("\n## LoCoMo\n")
        locomo.run(out_dir=out)
    return 0


if __name__ == "__main__":
    sys.exit(main())
