"""PR 1A #6: scan tracked text files for hidden bidirectional / directional control
characters that GitHub flags. Fails if any are present (outside an explicitly documented
allowlist). Ordinary Unicode arrows/punctuation are NOT flagged - only the control
codepoints below.
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

# Bidirectional / directional control characters.
_BIDI = set(
    list(range(0x202A, 0x202E + 1))    # LRE, RLE, PDF, LRO, RLO
    + list(range(0x2066, 0x2069 + 1))  # LRI, RLI, FSI, PDI
    + [0x200E, 0x200F, 0x061C]         # LRM, RLM, ALM
)
_EXT = {".py", ".json", ".jsonl", ".md", ".yml", ".yaml", ".txt", ".cfg", ".toml"}
_SCAN_DIRS = ["src", "scripts", "config", "docs", "tests", ".github",
              "artifacts/foundation_lock", "artifacts/probability_contract",
              "artifacts/market_feature_proof"]
# Explicitly documented allowlist (path -> reason). Empty: no bidi controls are permitted.
_ALLOWLIST: dict[str, str] = {}


def _iter_text_files():
    for d in _SCAN_DIRS:
        base = REPO / d
        if not base.exists():
            continue
        for p in base.rglob("*"):
            if p.is_file() and p.suffix in _EXT:
                yield p


def test_no_bidirectional_control_characters():
    offenders = []
    for p in _iter_text_files():
        rel = str(p.relative_to(REPO))
        if rel in _ALLOWLIST:
            continue
        try:
            text = p.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        hits = sorted({ord(ch) for ch in text if ord(ch) in _BIDI})
        if hits:
            offenders.append((rel, [f"U+{h:04X}" for h in hits]))
    assert offenders == [], f"bidirectional control characters found: {offenders}"


def test_scanner_detects_a_planted_control(tmp_path):
    # Positive control: the scan logic actually catches a bidi character.
    f = tmp_path / "x.py"
    f.write_text("x = 1  \u202e reversed\n", encoding="utf-8")
    text = f.read_text(encoding="utf-8")
    assert any(ord(ch) in _BIDI for ch in text)
