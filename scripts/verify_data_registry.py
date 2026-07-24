"""Verify the data registry (fail-closed on corruption/drift).

- Present locally + sha256 MATCHES registry -> ok.
- Present locally + sha256 MISMATCH -> nonzero exit (corruption / drift). The only failure.
- Present locally + no recorded sha256 -> warning (unpublished; run publish_data.py).
- Missing -> fetchable (not a failure): pull with `python3 scripts/fetch_data.py --all`.

Safe to wire into CI as a lightweight gate.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from data_registry_lib import ROOT, load_registry, sha256_file  # noqa: E402


def main() -> int:
    reg = load_registry()
    mismatches, missing, unpublished, ok = [], [], [], []
    for n, d in reg["datasets"].items():
        path = ROOT / d["path"]
        want = d.get("sha256")
        if not path.exists():
            missing.append(n); continue
        if not want:
            unpublished.append(n); continue
        got = sha256_file(path)
        if got == want:
            ok.append(n)
        else:
            mismatches.append(f"{n}: sha256 mismatch (got={got[:12]}… want={want[:12]}…)")

    for n in ok:
        print(f"  [ok] {n}")
    for n in missing:
        print(f"  [fetchable] {n}: not present locally — `fetch_data.py {n}`")
    for n in unpublished:
        print(f"  [WARN] {n}: present but not yet published — `publish_data.py --name {n}`")
    for m in mismatches:
        print(f"  [FAIL] {m}")
    if mismatches:
        print(f"[FATAL] data registry verification failed ({len(mismatches)}).")
        return 1
    print(f"[verify] data registry OK — verified={len(ok)} "
          f"fetchable={len(missing)} unpublished={len(unpublished)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
