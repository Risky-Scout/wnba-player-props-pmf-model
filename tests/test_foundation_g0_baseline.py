"""Foundation Lock tests for the historical G0 baseline proof.

Confirms the pre-Phase-0 baseline is preserved, internally consistent across its JSON /
CSV / Markdown renderings, remains a FAILING (non-promotion-eligible) result, and is
hash-pinned so any change to an output artifact breaks verification.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
D = REPO / "artifacts" / "market_feature_proof" / "G0_baseline"
JSON = D / "market_superiority_proof.json"
CSV = D / "market_superiority_proof.csv"
MD = D / "MARKET_SUPERIORITY_REPORT.md"
MANIFEST = D / "RUN_MANIFEST.json"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_all_props_pass_is_false_and_no_prop_passes():
    d = json.loads(JSON.read_text())
    assert d["all_props_pass"] is False
    for r in d["results"]:
        assert r["market_superiority_gate"] != "PASS", r["prop"]
    # Markdown headline agrees.
    assert "0/" in MD.read_text().split("passing all three gates")[1][:8]


def test_json_csv_markdown_agree():
    d = json.loads(JSON.read_text())
    js = {r["prop"]: r for r in d["results"]}
    df = pd.read_csv(CSV)
    md = MD.read_text()
    assert set(js) == set(df["prop"])
    for _, row in df.iterrows():
        prop = row["prop"]
        assert int(row["n_settled"]) == int(js[prop]["n_settled"])
        assert row["market_superiority_gate"] == js[prop]["market_superiority_gate"]
        # Markdown row carries the same n and gate.
        assert f"| {prop} |" in md
        assert str(int(js[prop]["n_settled"])) in md
        assert js[prop]["market_superiority_gate"] in md


def test_run_manifest_matches_machine_readable_artifact():
    d = json.loads(JSON.read_text())
    man = json.loads(MANIFEST.read_text())
    assert man["promotion_eligible"] is False
    assert man["all_props_pass"] == d["all_props_pass"] == False  # noqa: E712
    js = {r["prop"]: r for r in d["results"]}
    for pp in man["per_prop"]:
        prop = pp["prop"]
        assert int(pp["n_settled"]) == int(js[prop]["n_settled"])
        assert int(pp["n_clusters"]) == int(js[prop]["n_clusters"])
        assert pp["date_min"] == js[prop]["date_min"]
        assert pp["date_max"] == js[prop]["date_max"]
        assert pp["gate"] == js[prop]["market_superiority_gate"]


def test_output_hashes_are_pinned_and_tamper_evident(tmp_path):
    man = json.loads(MANIFEST.read_text())
    outs = man["output_artifacts"]
    # Recorded hashes must match the committed files exactly.
    assert outs["market_superiority_proof.json"] == _sha256(JSON)
    assert outs["market_superiority_proof.csv"] == _sha256(CSV)
    assert outs["MARKET_SUPERIORITY_REPORT.md"] == _sha256(MD)
    # Tampering with any artifact yields a different hash -> verification would fail.
    tampered = tmp_path / "proof.json"
    obj = json.loads(JSON.read_text())
    obj["all_props_pass"] = True  # a forbidden silent "improvement"
    tampered.write_text(json.dumps(obj))
    assert _sha256(tampered) != outs["market_superiority_proof.json"]


def test_label_marks_baseline_not_promotion_eligible():
    man = json.loads(MANIFEST.read_text())
    label = man["label"].lower()
    assert "not promotion-eligible" in label
    assert "pre-phase-0" in label or "baseline" in label
    assert "quote identity" in label
