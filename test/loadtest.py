#!/usr/bin/env python
"""Empirical load-test harness for brukerapi against resources/testdata.

Walks the testdata tree, finds every supported binary file (fid, 2dseq, rawdata.jobN,
traj, ser), attempts to construct a brukerapi.Dataset for it, and records the outcome:
success (+shape/dtype) or failure (+exception type + short message).
"""

import json
import os
import sys
import traceback
from pathlib import Path

from brukerapi.dataset import Dataset

ROOT = Path("resources/testdata").resolve()

# Binary stems brukerapi claims to support (see DEFAULT_STATES)
SUPPORTED_STEMS = {"fid", "2dseq", "rawdata", "traj", "ser"}


def classify(p: Path):
    stem = p.name.split(".")[0]
    return stem


def find_targets(root: Path):
    targets = []
    for dirpath, dirnames, filenames in os.walk(root):
        # skip git internals and provenance
        parts = set(Path(dirpath).parts)
        if ".git" in parts or "_sources" in parts:
            dirnames[:] = [d for d in dirnames if d != ".git"]
            continue
        for fn in filenames:
            stem = fn.split(".")[0]
            if stem in SUPPORTED_STEMS:
                targets.append(Path(dirpath) / fn)
    return sorted(targets)


def try_load(path: Path):
    rec = {"path": str(path.relative_to(ROOT)), "stem": classify(path), "size": path.stat().st_size}
    try:
        ds = Dataset(str(path))
        rec["ok"] = True
        try:
            rec["shape"] = list(ds.data.shape)
            rec["dtype"] = str(ds.data.dtype)
        except Exception as e:
            rec["ok"] = False
            rec["stage"] = "data-access"
            rec["err"] = f"{type(e).__name__}: {e}"
        return rec
    except Exception as e:
        rec["ok"] = False
        rec["err_type"] = type(e).__name__
        rec["err"] = str(e)[:300]
        rec["tb"] = traceback.format_exc()[-1500:]
        return rec


def main():
    targets = find_targets(ROOT)
    results = [try_load(p) for p in targets]

    ok = [r for r in results if r.get("ok")]
    bad = [r for r in results if not r.get("ok")]

    print(f"TOTAL targets: {len(results)}  OK: {len(ok)}  FAIL: {len(bad)}")
    print("\n=== FAILURE COUNTS BY (stem, err_type) ===")
    from collections import Counter

    c = Counter((r["stem"], r.get("err_type", r.get("stage", "?"))) for r in bad)
    for (stem, et), n in c.most_common():
        print(f"  {n:4d}  {stem:8s} {et}")

    print("\n=== OK COUNTS BY stem ===")
    c2 = Counter(r["stem"] for r in ok)
    for stem, n in c2.most_common():
        print(f"  {n:4d}  {stem}")

    out = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("/tmp/loadtest_results.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"\nFull results -> {out}")


if __name__ == "__main__":
    main()
