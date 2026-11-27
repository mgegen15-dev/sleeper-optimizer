"""Read-only HTTP client for Sleeper (+ ESPN tip times) with retries, timeouts and a file cache.

Every call degrades gracefully: on failure it falls back to the last cached copy (if any) and
records the problem in `client.failures`, which ends up in the email's Assumptions section.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import requests

APP = "https://api.sleeper.app/v1"
COM = "https://api.sleeper.com"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"


class SleeperClient:
    def __init__(self, cache_dir: str | Path, timeout: float = 30, retries: int = 3, offline: bool = False):
        self.cache = Path(cache_dir)
        self.cache.mkdir(parents=True, exist_ok=True)
        self.timeout, self.retries, self.offline = timeout, retries, offline
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "fantasy-optimizer/0.2"
        self.failures: list[str] = []
        self.fetched: dict[str, str] = {}   # fname -> "live" | "cache" | "stale-cache"

    # -------------------------------------------------------------- core
    def get(self, url: str, fname: str, max_age_h: float | None = 0):
        """max_age_h=0 -> always fetch; None -> any cached copy is fine; N -> refetch if older."""
        path = self.cache / fname
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and (self.offline or max_age_h is None or
                              (max_age_h and time.time() - path.stat().st_mtime < max_age_h * 3600)):
            self.fetched[fname] = "cache"
            return json.loads(path.read_text(encoding="utf-8"))
        err = None
        for attempt in range(self.retries):
            try:
                r = self.s.get(url, timeout=self.timeout)
                if r.status_code == 200:
                    data = r.json()
                    path.write_text(json.dumps(data), encoding="utf-8")
                    self.fetched[fname] = "live"
                    return data
                err = f"HTTP {r.status_code}"
                if r.status_code in (403, 404):
                    break
            except (requests.RequestException, ValueError) as e:
                err = type(e).__name__
            time.sleep(1.5 * 2 ** attempt)
        if path.exists():
            age = (time.time() - path.stat().st_mtime) / 3600
            self.failures.append(f"{url} failed ({err}); used cached copy {age:.1f}h old")
            self.fetched[fname] = "stale-cache"
            return json.loads(path.read_text(encoding="utf-8"))
        self.failures.append(f"{url} failed ({err}); no cache")
        return None

    # -------------------------------------------------------------- Sleeper
    def state(self):
        return self.get(f"{APP}/state/nba", "state_nba.json")

    def league(self, lid):
        return self.get(f"{APP}/league/{lid}", "league.json")

    def rosters(self, lid):
        return self.get(f"{APP}/league/{lid}/rosters", "rosters.json")

    def users(self, lid):
        return self.get(f"{APP}/league/{lid}/users", "users.json", max_age_h=24)

    def matchups(self, lid, week, current=True):
        return self.get(f"{APP}/league/{lid}/matchups/{week}", f"matchups/week_{week}.json", 0 if current else 12)

    def matchups_consensus(self, lid, week, tries: int = 5):
        """Sleeper's matchup endpoint serves DIFFERENT versions of the same week on consecutive requests.
        Fetch `tries` cache-busted copies and use a version only if a strict majority agree. Otherwise
        fall back to the last version that did have a majority (saved as ..._stable.json) and flag it."""
        import collections
        import random
        seen = []
        for i in range(tries):
            d = self.get(f"{APP}/league/{lid}/matchups/{week}?nc={random.randrange(10**9)}", f"matchups/week_{week}_try{i}.json")
            if d is not None:
                seen.append(json.dumps(sorted(d, key=lambda m: m["roster_id"]), sort_keys=True))
        stable = self.cache / f"matchups/week_{week}_stable.json"
        if not seen:
            return self.matchups(lid, week)
        best, cnt = collections.Counter(seen).most_common(1)[0]
        if cnt * 2 > len(seen):
            data = json.loads(best)
            stable.write_text(json.dumps(data), encoding="utf-8")
            if cnt < len(seen):
                self.failures.append(f"matchups week {week}: {len(set(seen))} versions in {len(seen)} requests; used the majority ({cnt}/{len(seen)})")
        elif stable.exists():
            data = json.loads(stable.read_text(encoding="utf-8"))
            self.failures.append(f"matchups week {week}: no majority ({cnt}/{len(seen)} best of {len(set(seen))} versions); "
                                 f"used the last stable version")
        else:
            data = json.loads(best)
            self.failures.append(f"matchups week {week}: no majority ({cnt}/{len(seen)}) and no stable copy; used the most common - numbers may be off")
        (self.cache / f"matchups/week_{week}.json").write_text(json.dumps(data), encoding="utf-8")
        return data

    def transactions(self, lid, week):
        return self.get(f"{APP}/league/{lid}/transactions/{week}", f"transactions_week_{week}.json")

    def players(self):
        return self.get(f"{APP}/players/nba", "players_nba.json", max_age_h=24)   # ~5MB: at most once a day

    def week_stats(self, season, week, current=True):
        """Per-GAME stat lines: one row per player-game (date, game_id, opponent, stats{...})."""
        return self.get(f"{COM}/stats/nba/{season}/{week}?season_type=regular", f"stats/week_{week}.json",
                        0 if current else 12)

    def week_proj(self, season, week, current=True):
        """Per-game projections (rotowire) for the week, including games not yet played."""
        return self.get(f"{COM}/projections/nba/{season}/{week}?season_type=regular", f"projections/week_{week}.json",
                        0 if current else None)

    def schedule(self, season):
        return self.get(f"{COM}/schedule/nba/regular/{season}", "schedule_nba.json", max_age_h=0.25)

    # -------------------------------------------------------------- ESPN (tip times + live status)
    def espn_scoreboard(self, date_iso: str, max_age_h: float | None = 0):
        return self.get(f"{ESPN}?dates={date_iso.replace('-', '')}", f"espn/scoreboard_{date_iso}.json", max_age_h)
