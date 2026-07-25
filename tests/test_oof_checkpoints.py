"""P7: resumable OOF fold checkpoints - deterministic input hash, immutable write, fail-closed
reuse (only when every hash matches, fit_status==model_oof, all 7 props present)."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("boof", REPO / "scripts" / "build_oof_pmfs.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _pmf_frame():
    rows = []
    for p in ["pts", "reb", "ast", "fg3m", "stl", "blk", "turnover"]:
        rows.append({"game_id": "g1", "player_id": "p1", "stat": p,
                     "pmf_json": '{"0": 0.5, "1": 0.5}', "oof_prediction_type": "model_oof"})
    return pd.DataFrame(rows)


FOLD = {"fold_id": 5, "train_start_date": "2026-05-01", "train_end_date": "2026-06-30",
        "val_start_date": "2026-07-01", "val_end_date": "2026-07-07"}
HASHES = {"features_wide": "a", "features_long": "b", "manifest": "c"}


def test_input_hash_is_deterministic_and_sensitive():
    m = _mod()
    h1 = m._fold_input_hash(FOLD, data_hashes=HASHES, contract_hash="k", config_hash="cfg", code_sha="sha")
    h2 = m._fold_input_hash(FOLD, data_hashes=HASHES, contract_hash="k", config_hash="cfg", code_sha="sha")
    h3 = m._fold_input_hash(FOLD, data_hashes={**HASHES, "features_wide": "X"},
                            contract_hash="k", config_hash="cfg", code_sha="sha")
    assert h1 == h2 and h1 != h3


def test_write_then_resume_roundtrip(tmp_path):
    m = _mod()
    ih = m._fold_input_hash(FOLD, data_hashes=HASHES, contract_hash="k", config_hash="cfg", code_sha="sha")
    m._write_checkpoint(tmp_path, 5, pmf_frame=_pmf_frame(), fold=FOLD, input_hash=ih,
                        data_hashes=HASHES, contract_hash="k", pit_hash="pit", config_hash="cfg",
                        code_sha="sha", encoder_hash="enc", model_hashes={}, fit_status="model_oof")
    loaded = m._load_valid_checkpoint(tmp_path, 5, ih)
    assert loaded is not None and len(loaded) == 7


def test_resume_rejects_on_input_hash_mismatch(tmp_path):
    m = _mod()
    ih = m._fold_input_hash(FOLD, data_hashes=HASHES, contract_hash="k", config_hash="cfg", code_sha="sha")
    m._write_checkpoint(tmp_path, 5, pmf_frame=_pmf_frame(), fold=FOLD, input_hash=ih,
                        data_hashes=HASHES, contract_hash="k", pit_hash="pit", config_hash="cfg",
                        code_sha="sha", encoder_hash="enc", model_hashes={}, fit_status="model_oof")
    assert m._load_valid_checkpoint(tmp_path, 5, "DIFFERENT_HASH") is None


def test_checkpoint_is_immutable(tmp_path):
    m = _mod()
    ih = m._fold_input_hash(FOLD, data_hashes=HASHES, contract_hash="k", config_hash="cfg", code_sha="sha")
    kw = dict(fold=FOLD, input_hash=ih, data_hashes=HASHES, contract_hash="k", pit_hash="pit",
              config_hash="cfg", code_sha="sha", encoder_hash="enc", model_hashes={}, fit_status="model_oof")
    m._write_checkpoint(tmp_path, 5, pmf_frame=_pmf_frame(), **kw)
    with pytest.raises(FileExistsError):
        m._write_checkpoint(tmp_path, 5, pmf_frame=_pmf_frame(), **kw)


def test_resume_rejects_output_hash_tamper(tmp_path):
    m = _mod()
    ih = m._fold_input_hash(FOLD, data_hashes=HASHES, contract_hash="k", config_hash="cfg", code_sha="sha")
    m._write_checkpoint(tmp_path, 5, pmf_frame=_pmf_frame(), fold=FOLD, input_hash=ih,
                        data_hashes=HASHES, contract_hash="k", pit_hash="pit", config_hash="cfg",
                        code_sha="sha", encoder_hash="enc", model_hashes={}, fit_status="model_oof")
    _, data_p = m._checkpoint_paths(tmp_path, 5)
    _pmf_frame().head(3).to_parquet(data_p, index=False)   # tamper: drops props + changes bytes
    assert m._load_valid_checkpoint(tmp_path, 5, ih) is None


def test_resume_rejects_missing_props(tmp_path):
    m = _mod()
    ih = m._fold_input_hash(FOLD, data_hashes=HASHES, contract_hash="k", config_hash="cfg", code_sha="sha")
    partial = _pmf_frame().head(3)  # only 3 of 7 props
    m._write_checkpoint(tmp_path, 5, pmf_frame=partial, fold=FOLD, input_hash=ih,
                        data_hashes=HASHES, contract_hash="k", pit_hash="pit", config_hash="cfg",
                        code_sha="sha", encoder_hash="enc", model_hashes={}, fit_status="model_oof")
    assert m._load_valid_checkpoint(tmp_path, 5, ih) is None
