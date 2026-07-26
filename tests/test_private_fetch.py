"""P4: private authenticated fetching - fail-closed auth priority; no secret in logs/messages."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent


def _mod():
    spec = importlib.util.spec_from_file_location("fd", REPO / "scripts" / "fetch_data.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def test_public_asset_needs_no_token(monkeypatch):
    m = _mod()
    for v in ("PRIVATE_DATA_GH_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    assert m._auth_env(private=False) is None            # public: unchanged, no token needed


def test_private_without_token_fails_closed(monkeypatch):
    m = _mod()
    for v in ("PRIVATE_DATA_GH_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    with pytest.raises(m.PrivateAuthError):
        m._auth_env(private=True)


def test_private_auth_priority(monkeypatch):
    m = _mod()
    monkeypatch.setenv("GH_TOKEN", "fallback")
    monkeypatch.setenv("PRIVATE_DATA_GH_TOKEN", "primary")
    env = m._auth_env(private=True)
    assert env["GH_TOKEN"] == "primary"                  # PRIVATE_DATA_GH_TOKEN wins
    monkeypatch.delenv("PRIVATE_DATA_GH_TOKEN")
    assert m._auth_env(private=True)["GH_TOKEN"] == "fallback"


def test_download_error_message_has_no_token(monkeypatch, tmp_path):
    m = _mod()
    monkeypatch.setenv("PRIVATE_DATA_GH_TOKEN", "supersecrettoken123")
    monkeypatch.setattr(m, "require_gh", lambda: "gh")

    class _Res:
        returncode = 1
        stderr = "HTTP 404: Not Found (release)"  # gh never echoes the token

    captured = {}

    def _fake_run(cmd, env=None):
        captured["cmd"] = cmd
        captured["env_has_token"] = bool(env and env.get("GH_TOKEN"))
        return _Res()

    monkeypatch.setattr(m, "run", _fake_run)
    with pytest.raises(RuntimeError) as exc:
        m._download("Owner/private-repo", "processed-features-v2",
                    "wnba_player_game_features_wide.parquet", tmp_path / "x.parquet", private=True)
    # Token is present in the child ENV (not argv) and never in the raised message.
    assert captured["env_has_token"] is True
    assert "supersecrettoken123" not in " ".join(captured["cmd"])
    assert "supersecrettoken123" not in str(exc.value)


def test_matching_private_asset_succeeds(monkeypatch, tmp_path):
    m = _mod()
    monkeypatch.setenv("PRIVATE_DATA_GH_TOKEN", "tok")
    monkeypatch.setattr(m, "require_gh", lambda: "gh")

    class _OK:
        returncode = 0
        stderr = ""

    monkeypatch.setattr(m, "run", lambda cmd, env=None: _OK())
    m._download("Owner/private-repo", "tag", "asset.parquet", tmp_path / "asset.parquet", private=True)


def test_public_fetch_path_unchanged(monkeypatch, tmp_path):
    # Public assets must not require or inject a token.
    m = _mod()
    for v in ("PRIVATE_DATA_GH_TOKEN", "GH_TOKEN"):
        monkeypatch.delenv(v, raising=False)
    monkeypatch.setattr(m, "require_gh", lambda: "gh")
    seen = {}

    class _OK:
        returncode = 0
        stderr = ""

    def _fake_run(cmd, env=None):
        seen["env"] = env
        return _OK()

    monkeypatch.setattr(m, "run", _fake_run)
    m._download("Owner/public", "tracking-data-v1", "a.parquet", tmp_path / "a.parquet", private=False)
    assert seen["env"] is None                            # public path passes no env/token
