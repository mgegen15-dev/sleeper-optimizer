"""Assemble one consistent snapshot of the league: rosters, matchups, per-game logs, projections,
schedule with tip times (Central) and game status."""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from data.sleeper import SleeperClient
from model.scoring import played, score_line

ESPN2SLP = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}


@dataclass
class Game:
    game_id: str
    date: str          # local NBA date YYYY-MM-DD (as Sleeper lists it)
    week: int
    team: str
    opp: str
    home: bool
    tip: datetime | None = None   # Central time
    started: bool = False
    final: bool = False


@dataclass
class GameLog:
    pid: str
    date: str
    week: int
    team: str | None
    opp: str | None
    fp: float
    game_id: str | None = None


@dataclass
class Snapshot:
    now: datetime
    season: str
    week: int
    league: dict
    rosters: list
    users: list
    matchups: dict
    players: dict
    logs: dict                      # pid -> [GameLog] sorted by date
    proj: dict                      # pid -> {date: projected FP}
    team_games: dict                # team -> [Game] sorted by date
    transactions: list
    failures: list = field(default_factory=list)
    freshness: dict = field(default_factory=dict)
    stat_dates: set = field(default_factory=set)   # (team, date) pairs that have stat lines posted
    reported_locks: dict = field(default_factory=dict)  # rid -> {pid: lock record} you reported
    api_locks: dict | None = None   # rid -> {pid: game_id} from Sleeper's private API (None = unavailable)

    @property
    def scoring(self):
        return self.league["scoring_settings"]

    def roster(self, rid):
        return next(r for r in self.rosters if r["roster_id"] == rid)

    def matchup(self, rid, week=None):
        return next((m for m in self.matchups.get(week or self.week) or [] if m["roster_id"] == rid), None)

    def opponent_rid(self, rid, week=None):
        m = self.matchup(rid, week)
        if not m:
            return None
        return next((x["roster_id"] for x in self.matchups[week or self.week]
                     if x["matchup_id"] == m["matchup_id"] and x["roster_id"] != rid), None)

    def team_name(self, rid):
        r = self.roster(rid)
        u = next((u for u in self.users or [] if u["user_id"] == r.get("owner_id")), {})
        return ((u.get("metadata") or {}).get("team_name") or u.get("display_name") or f"Team {rid}").strip()

    def owner_name(self, rid):
        r = self.roster(rid)
        u = next((u for u in self.users or [] if u["user_id"] == r.get("owner_id")), {})
        return u.get("display_name") or "?"

    def week_games(self, team, week=None):
        return [g for g in self.team_games.get(team, []) if g.week == (week or self.week)]

    def pname(self, pid):
        p = self.players.get(pid, {})
        return f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or pid


def build_logs(stats_by_week: dict, scoring: dict) -> dict:
    logs = defaultdict(list)
    for w, rows in stats_by_week.items():
        for r in rows or []:
            pid = str(r.get("player_id"))
            st = r.get("stats") or {}
            if pid.startswith("TEAM_") or not played(st):
                continue
            logs[pid].append(GameLog(pid, r["date"], int(w), r.get("team"), r.get("opponent"), score_line(st, scoring), r.get("game_id")))
    for v in logs.values():
        v.sort(key=lambda g: g.date)
    return dict(logs)


def build_proj(rows, scoring) -> dict:
    out = defaultdict(dict)
    for r in rows or []:
        if r.get("stats"):
            out[str(r["player_id"])][r["date"]] = score_line(r["stats"], scoring)
    return dict(out)


def load(cfg: dict, now: datetime | None = None, offline: bool = False, extra_proj_weeks=()) -> Snapshot:
    tz = ZoneInfo(cfg["timezone"])
    now = now or datetime.now(tz)
    c = SleeperClient(cfg["cache_dir"], cfg["http"]["timeout_s"], cfg["http"]["retries"], offline=offline)
    lid = cfg["league_id"]
    state = c.state() or {}
    league = c.league(lid)
    if not league:
        raise RuntimeError("league endpoint unavailable and no cache: cannot continue")
    season = state.get("season") or league["season"]
    W = int(state.get("week") or league["settings"].get("leg") or 1)
    last_reg = int(league["settings"].get("playoff_week_start", 19)) - 1
    rosters, users = c.rosters(lid), c.users(lid)
    matchups = {w: (c.matchups_consensus(lid, w) if (W - 1 <= w <= W and not offline) else c.matchups(lid, w, current=False))
                for w in range(1, last_reg + 1)}
    players = c.players() or {}
    stats = {w: c.week_stats(season, w, current=(w >= W - 1)) for w in range(1, W + 1)}
    scoring = league["scoring_settings"]
    logs = build_logs(stats, scoring)
    proj = build_proj(c.week_proj(season, W), scoring)
    for w in extra_proj_weeks:   # historical projections for the backtest
        for pid, d in build_proj(c.week_proj(season, w, current=False), scoring).items():
            proj.setdefault(pid, {}).update(d)
    sched = c.schedule(season) or []
    # ESPN tip times / live status for the current week's remaining dates (+ yesterday for late finals)
    today = now.date().isoformat()
    yday = (now.date() - timedelta(days=1)).isoformat()
    tips = {}
    for d in sorted({g["date"] for g in sched if g.get("week") == W and g["date"] >= yday}):
        sb = c.espn_scoreboard(d)
        for ev in (sb or {}).get("events", []):
            tip = datetime.fromisoformat(ev["date"].replace("Z", "+00:00")).astimezone(tz)
            stt = ev["competitions"][0]["status"]["type"]
            for comp in ev["competitions"][0]["competitors"]:
                ab = comp["team"]["abbreviation"]
                tips[(d, ESPN2SLP.get(ab, ab))] = (tip, stt.get("state"), bool(stt.get("completed")))
    team_games = defaultdict(list)
    for g in sched:
        for side, other in (("home", "away"), ("away", "home")):
            t = g[side]["team"]
            gm = Game(g["game_id"], g["date"], int(g["week"]), t, g[other]["team"], side == "home")
            info = tips.get((g["date"], t))
            if info:
                gm.tip = info[0]
                gm.started = info[1] in ("in", "post") or info[0] <= now
                gm.final = info[2] or info[1] == "post"
            if g.get("status") == "complete":
                gm.started = gm.final = True
            elif g.get("status") not in (None, "pre_game"):
                gm.started = True
            if gm.tip is None and g["date"] < today:
                gm.started = gm.final = gm.final or g.get("status") == "complete" or g["date"] < yday
            team_games[t].append(gm)
    for v in team_games.values():
        v.sort(key=lambda g: (g.date, g.tip or now))
    if not tips and any(g.get("week") == W and g["date"] >= today for g in sched):
        c.failures.append("ESPN scoreboard unavailable: tip times unknown, game status from Sleeper only")
    tx = c.transactions(lid, W) or []
    stat_dates = {(r.get("team"), r.get("date")) for rows in stats.values() for r in rows or [] if r.get("stats")}
    snap = Snapshot(now, season, W, league, rosters, users, matchups, players, logs, proj, dict(team_games), tx,
                    c.failures, c.fetched, stat_dates)
    if cfg.get("private_api", {}).get("enabled", True):
        from pathlib import Path
        from data import sleeper_private as sp
        if offline:
            f = Path(cfg["cache_dir"]) / f"gql/matchup_legs_week_{W}.json"
            snap.api_locks = sp.parse_legs(json.loads(f.read_text(encoding="utf-8"))) if f.exists() else None
        else:
            snap.api_locks = sp.fetch_locks(lid, W, Path(__file__).resolve().parent.parent, Path(cfg["cache_dir"]), c.failures)
    return snap
