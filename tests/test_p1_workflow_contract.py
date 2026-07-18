"""Contract test: the P1 workflow is offline and does not touch production."""
from __future__ import annotations

from pathlib import Path

import yaml

WF = Path(".github/workflows/p1_historical_validation.yml")


def test_p1_workflow_is_offline_and_safe():
    txt = WF.read_text()
    wf = yaml.safe_load(txt)
    on = wf.get("on", wf.get(True))
    # Fired via sentinel push (dispatch unavailable to app tokens) + manual dispatch.
    assert "push" in on and on["push"]["paths"] == [".github/p1_run.trigger"]
    # Must NOT deploy, publish, commit calibration, or run production scripts.
    forbidden = ["ftp_deploy", "generate_web_pages", "generate_distributions_page",
                 "git commit", "git push", "gh pages", "pregame_initial", "fit_calibrators"]
    for f in forbidden:
        assert f not in txt, f"P1 workflow must not reference production step: {f}"
    # Runs the offline backfill + eval only.
    assert "p1_historical_backfill.py" in txt and "p1_build_evaluation.py" in txt
    # Secret is consumed via env, never echoed.
    assert "ODDS_API_KEY: ${{ secrets.ODDS_API_KEY }}" in txt
    assert "echo \"$ODDS_API_KEY\"" not in txt and "echo $ODDS_API_KEY" not in txt


def test_p1_sentinel_exists():
    assert Path(".github/p1_run.trigger").exists()
