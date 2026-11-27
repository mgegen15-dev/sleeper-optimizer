"""Optional, READ-ONLY access to Sleeper's private GraphQL API (the one sleeper.com uses), for one
thing the public API lacks: which game each starter is locked to.

    POST https://sleeper.com/graphql   headers: authorization: <token>, X-Sleeper-GraphQL-Op
    query get_matchup_legs -> matchup_legs(league_id, round) { roster_id starters starters_games ... }

`starters_games` maps player_id -> game_id for every LOCKED starter (unlocked starters are absent).
Verified 2026-11-27 against week 6: each entry's game is exactly the game whose score the public
matchup endpoint shows for that player.

The token comes only from env SLEEPER_TOKEN or a gitignored `.env` file; it is never printed, logged
or written anywhere. No mutations are ever sent. Any failure -> None, and the engine falls back to
lock inference (noted under Assumptions).
"""
from __future__ import annotations

import base64
import json
import os
import time
from pathlib import Path

import requests

URL = "https://sleeper.com/graphql"
QUERY = """query get_matchup_legs($leagueId: String!, $round: Int!) {
  matchup_legs(league_id: $leagueId, round: $round) {
    league_id leg matchup_id roster_id round starters starters_games proj_points
  }
}"""


def load_token(root: Path) -> str | None:
    tok = os.environ.get("SLEEPER_TOKEN")
    if tok:
        return tok.strip()
    env = Path(root) / ".env"
    if env.exists():
        for ln in env.read_text(encoding="utf-8").splitlines():
            if ln.strip().startswith("SLEEPER_TOKEN="):
                v = ln.split("=", 1)[1].strip().strip('"').strip("'")
                return v or None
    return None


def token_expiry(tok: str) -> float | None:
    """JWT 'exp' claim (epoch seconds) without verifying or exposing the token."""
    try:
        payload = tok.split(".")[1]
        return float(json.loads(base64.urlsafe_b64decode(payload + "==="))["exp"])
    except Exception:
        return None


def parse_legs(resp: dict) -> dict:
    """GraphQL response -> {roster_id: {player_id: game_id}} for locked starters."""
    legs = ((resp or {}).get("data") or {}).get("matchup_legs")
    if legs is None:
        raise ValueError("no matchup_legs in response")
    return {int(l["roster_id"]): {str(p): str(g) for p, g in (l.get("starters_games") or {}).items()} for l in legs}


def fetch_locks(league_id: str, week: int, root: Path, cache_dir: Path, failures: list, timeout: float = 20):
    tok = load_token(root)
    if not tok:
        failures.append("Sleeper lock state not read (no SLEEPER_TOKEN); locks inferred from scores")
        return None
    exp = token_expiry(tok)
    if exp and exp - time.time() < 14 * 86400:
        failures.append(f"SLEEPER_TOKEN expires in {(exp - time.time()) / 86400:.0f} days; refresh it (see README)")
    try:
        r = requests.post(URL, timeout=timeout,
                          json={"operationName": "get_matchup_legs", "query": QUERY,
                                "variables": {"leagueId": str(league_id), "round": int(week)}},
                          headers={"authorization": tok, "X-Sleeper-GraphQL-Op": "get_matchup_legs",
                                   "Content-Type": "application/json", "User-Agent": "fantasy-optimizer/0.3"})
        if r.status_code != 200:
            failures.append(f"Sleeper GraphQL lock query HTTP {r.status_code}; locks inferred from scores")
            return None
        data = r.json()
        if data.get("errors"):
            msg = str(data["errors"][0].get("message", "error"))[:120]
            failures.append(f"Sleeper GraphQL lock query error ({msg}); locks inferred from scores")
            return None
        locks = parse_legs(data)
        cache = Path(cache_dir) / f"gql/matchup_legs_week_{week}.json"      # response has no token in it
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(data), encoding="utf-8")
        return locks
    except Exception as e:
        failures.append(f"Sleeper GraphQL lock query failed ({type(e).__name__}); locks inferred from scores")
        return None
