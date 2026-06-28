"""Write active_games.json index for live game discovery by the dashboard.

Called by live_inplay.yml after formatting live outputs.
Reads IDS and GAME_DATE from environment variables.
"""
import json
import os
import time

ids_env = os.environ.get("IDS", "")
ids = [int(g) for g in ids_env.split(",") if g.strip()]
payload = {
    "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "game_date": os.environ.get("GAME_DATE", ""),
    "game_ids": ids,
}
out = "tools/odds-scanner/predictions/WNBA/Inplay-Edge/active_games.json"
os.makedirs(os.path.dirname(out), exist_ok=True)
with open(out, "w") as f:
    json.dump(payload, f, indent=2)
print(f"Wrote active_games.json with {len(ids)} game IDs: {ids}")
