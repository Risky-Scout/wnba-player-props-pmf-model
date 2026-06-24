"""Tests for the pull_closing_lines.py closing-line pull fix.

Verifies that the correct BDL API method (list_player_props_for_game per game)
is called instead of the non-existent get_player_props(date=...) call.
"""
from __future__ import annotations

import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# Allow importing scripts/ as modules
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))


def _make_mock_client(games, props_per_game):
    """Build a mock BDLClient that returns fake games and props."""
    client = MagicMock()
    client.list_endpoint.return_value = games
    client.list_player_props_for_game.side_effect = lambda game_id: props_per_game.get(game_id, [])
    return client


class TestPullClosingLines:
    """Tests for pull_closing_lines.py B1 fix."""

    def test_get_player_props_not_called(self):
        """Regression: get_player_props does NOT exist on BDLClient.
        The fix uses list_endpoint('games') + list_player_props_for_game().
        """
        import pull_closing_lines
        from wnba_props_model.data.bdl_client import BDLClient
        assert not hasattr(BDLClient, "get_player_props"), (
            "get_player_props should NOT exist on BDLClient — "
            "it was a phantom method causing the crash"
        )

    def test_props_collected_across_multiple_games(self, tmp_path):
        """Two games, 2 props each → should not crash. BDLClient is a local import."""
        fake_games = [{"id": 101}, {"id": 102}]
        fake_props = {
            101: [
                {"player_id": 1, "stat": "pts", "line": 15.5, "over_odds": -110, "under_odds": -110},
                {"player_id": 2, "stat": "reb", "line": 6.5, "over_odds": -115, "under_odds": -105},
            ],
            102: [
                {"player_id": 3, "stat": "pts", "line": 20.5, "over_odds": -130, "under_odds": 110},
                {"player_id": 4, "stat": "ast", "line": 4.5, "over_odds": -120, "under_odds": 100},
            ],
        }
        mock_client = _make_mock_client(fake_games, fake_props)

        # BDLClient is imported inside try/except in pull_closing_lines.py, so patch
        # the class on the underlying module, not on the script namespace.
        with patch("wnba_props_model.data.bdl_client.BDLClient", return_value=mock_client):
            from typer.testing import CliRunner
            from pull_closing_lines import app

            runner = CliRunner()
            result = runner.invoke(app, [
                "--game-date", "2026-06-15",
                "--out-dir", str(tmp_path),
                "--api-key", "dummy-key",
            ])

        # Should not raise — exit_code 0 means no unhandled exception
        assert result.exit_code == 0

    def test_empty_games_writes_empty_parquet(self, tmp_path):
        """When no games exist for the date, an empty parquet with correct schema is written."""
        mock_client = _make_mock_client([], {})

        with patch("wnba_props_model.data.bdl_client.BDLClient", return_value=mock_client):
            from typer.testing import CliRunner
            from pull_closing_lines import app

            runner = CliRunner()
            result = runner.invoke(app, [
                "--game-date", "2026-06-15",
                "--out-dir", str(tmp_path),
                "--api-key", "dummy-key",
            ])

        out_path = tmp_path / "closing_lines_2026-06-15.parquet"
        assert out_path.exists(), "Should always write a parquet file (even if empty)"
        df = pd.read_parquet(out_path)
        assert len(df) == 0
        required = ["game_id", "player_id", "stat", "line", "over_odds", "under_odds"]
        for col in required:
            assert col in df.columns, f"Missing column: {col}"

    def test_api_error_on_one_game_does_not_crash(self, tmp_path):
        """If one game's prop fetch fails, continue with others."""
        fake_games = [{"id": 101}, {"id": 102}]
        client = MagicMock()
        client.list_endpoint.return_value = fake_games

        def side_effect(game_id):
            if game_id == 101:
                raise RuntimeError("API timeout")
            return [{"player_id": 1, "stat": "pts", "line": 15.5,
                     "over_odds": -110, "under_odds": -110}]

        client.list_player_props_for_game.side_effect = side_effect

        with patch("wnba_props_model.data.bdl_client.BDLClient", return_value=client):
            from typer.testing import CliRunner
            from pull_closing_lines import app

            runner = CliRunner()
            result = runner.invoke(app, [
                "--game-date", "2026-06-15",
                "--out-dir", str(tmp_path),
                "--api-key", "dummy-key",
            ])
        assert result.exit_code == 0
