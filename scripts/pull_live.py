"""Pull live Sleeper NBA league data + per-game stats, compute player distributions,
and verify lock-in behaviour from matchup data.

Read-only public endpoints, no auth. Raw JSON goes to data_raw/, computed output to
data_raw/summary.json and data_raw/report.md.

Usage:  python scripts/pull_live.py [--no-fetch]   (--no-fetch reuses cached raw files)
"""
from __future__ import annotations

import json
import math
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

import requests

LEAGUE_ID = "1340092651755560960"
MY_ROSTER_ID = 10
TZ = ZoneInfo("America/Chicago")
HALF_LIFE = 7.0
RECENT_N = 15

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data_raw"
RAW.mkdir(exist_ok=True)

S = requests.Session()
S.headers["User-Agent"] = "fantasy-optimizer/0.1"

APP = "https://api.sleeper.app/v1"
COM = "https://api.sleeper.com"
ESPN = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/scoreboard"

NO_FETCH = "--no-fetch" in sys.argv
endpoint_log: list[dict] = []


def get(url: str, fname: str, max_age_h: float | None = None):
    path = RAW / fname
    path.parent.mkdir(parents=True, exist_ok=True)
    fresh = path.exists() and (
        NO_FETCH or (max_age_h is not None and time.time() - path.stat().st_mtime < max_age_h * 3600)
    )
    if fresh:
        endpoint_log.append({"url": url, "status": "cached", "file": fname})
        return json.loads(path.read_text(encoding="utf-8"))
    for attempt in range(4):
        try:
            r = S.get(url, timeout=60)
            break
        except requests.RequestException:
            if attempt == 3:
                raise
            time.sleep(2 ** attempt)
    ok = r.status_code == 200
    data = None
    if ok:
        try:
            data = r.json()
        except ValueError:
            ok = False
    endpoint_log.append({"url": url, "status": r.status_code if ok else f"FAIL {r.status_code}", "file": fname if ok else None})
    if ok:
        path.write_text(json.dumps(data), encoding="utf-8")
    return data


# ---------------------------------------------------------------- fetch
state = get(f"{APP}/state/nba", "state_nba.json")
season, W = state["season"], int(state["week"])
league = get(f"{APP}/league/{LEAGUE_ID}", "league.json")
rosters = get(f"{APP}/league/{LEAGUE_ID}/rosters", "rosters.json")
users = get(f"{APP}/league/{LEAGUE_ID}/users", "users.json")
matchups = {w: get(f"{APP}/league/{LEAGUE_ID}/matchups/{w}", f"matchups/week_{w}.json") for w in range(1, W + 1)}
# Future weeks (pairings for remaining regular season, if Sleeper pre-populates them)
last_reg = int(league["settings"].get("playoff_week_start", W + 1)) - 1
for w in range(W + 1, last_reg + 1):
    matchups[w] = get(f"{APP}/league/{LEAGUE_ID}/matchups/{w}", f"matchups/week_{w}.json")
transactions = get(f"{APP}/league/{LEAGUE_ID}/transactions/{W}", f"transactions_week_{W}.json")
players = get(f"{APP}/players/nba", "players_nba.json", max_age_h=24)

# Per-game stat lines: one row per player-game, with date / game_id / opponent.
stats_by_week = {w: get(f"{COM}/stats/nba/{season}/{w}?season_type=regular", f"stats/week_{w}.json") for w in range(1, W + 1)}
# Per-game projections for the current week (includes games not yet played).
proj_week = get(f"{COM}/projections/nba/{season}/{W}?season_type=regular", f"projections/week_{W}.json")
# Sleeper NBA schedule (no tip times) and ESPN scoreboards (tip times).
schedule = get(f"{COM}/schedule/nba/regular/{season}", "schedule_nba.json")
nba_cdn = get("https://cdn.nba.com/static/json/staticData/scheduleLeagueV2.json", "nba_cdn_schedule.json")

week_dates = sorted({g["date"] for g in schedule if g.get("week") == W})
espn = {}
for d in week_dates:
    espn[d] = get(f"{ESPN}?dates={d.replace('-', '')}", f"espn/scoreboard_{d}.json")

# ---------------------------------------------------------------- scoring
SC = league["scoring_settings"]
BONUS_KEYS = ["dd", "td", "bonus_pt_40p", "bonus_pt_50p", "bonus_reb_20p", "bonus_ast_15p"]


def derived_bonus(st: dict) -> dict:
    pts, reb, ast, stl, blk = (st.get(k, 0) or 0 for k in ("pts", "reb", "ast", "stl", "blk"))
    tens = sum(1 for v in (pts, reb, ast, stl, blk) if v >= 10)
    return {
        "dd": 1.0 if tens >= 2 else 0.0,
        "td": 1.0 if tens >= 3 else 0.0,
        "bonus_pt_40p": 1.0 if pts >= 40 else 0.0,
        "bonus_pt_50p": 1.0 if pts >= 50 else 0.0,
        "bonus_reb_20p": 1.0 if reb >= 20 else 0.0,
        "bonus_ast_15p": 1.0 if ast >= 15 else 0.0,
    }


bonus_audit = {"lines": 0, "mismatch": defaultdict(int), "present_when_earned": defaultdict(int), "earned": defaultdict(int)}


def fp(st: dict, audit: bool = False) -> float:
    st = dict(st or {})
    der = derived_bonus(st)
    if audit:
        bonus_audit["lines"] += 1
        for k in BONUS_KEYS:
            if der[k]:
                bonus_audit["earned"][k] += 1
                if st.get(k):
                    bonus_audit["present_when_earned"][k] += 1
            if bool(st.get(k)) != bool(der[k]):
                bonus_audit["mismatch"][k] += 1
    for k in BONUS_KEYS:  # fill any bonus key Sleeper omitted
        if k not in st:
            st[k] = der[k]
    return round(sum(v * (st.get(k) or 0) for k, v in SC.items()), 2)


def played(st: dict) -> bool:
    return bool(st) and ((st.get("sp") or 0) > 0 or (st.get("min") or 0) > 0)


# game log: pid -> list of {date, week, fp, opp}
glog = defaultdict(list)
for w, rows in stats_by_week.items():
    for r in rows or []:
        st = r.get("stats") or {}
        if not played(st) or str(r["player_id"]).startswith("TEAM_"):
            continue
        glog[r["player_id"]].append({"date": r["date"], "week": w, "game_id": r.get("game_id"), "opp": r.get("opponent"),
                                     "fp": fp(st, audit=True), "min": st.get("min")})
for v in glog.values():
    v.sort(key=lambda g: g["date"])

proj = defaultdict(list)
for r in proj_week or []:
    if r.get("stats"):
        proj[r["player_id"]].append({"date": r["date"], "fp": fp(r["stats"]), "gp": r["stats"].get("gp")})

# ---------------------------------------------------------------- tip times
ESPN2SLP = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS", "PHX": "PHX"}
tips = {}  # (date, team) -> CT datetime str, status
for d, sb in espn.items():
    for ev in (sb or {}).get("events", []):
        utc = datetime.fromisoformat(ev["date"].replace("Z", "+00:00"))
        ct = utc.astimezone(TZ)
        status = ev["competitions"][0]["status"]["type"]["name"]
        for c in ev["competitions"][0]["competitors"]:
            ab = c["team"]["abbreviation"]
            tips[(d, ESPN2SLP.get(ab, ab))] = (ct, status)

now_ct = datetime.now(TZ)
remaining = defaultdict(list)  # team -> list of games still to tip / in progress
for g in schedule:
    if g.get("week") != W or g.get("status") == "complete":
        continue
    for side, other in (("home", "away"), ("away", "home")):
        t = g[side]["team"]
        tip = tips.get((g["date"], t))
        remaining[t].append({
            "date": g["date"],
            "opp": ("vs " if side == "home" else "@") + g[other]["team"],
            "tip_ct": tip[0].strftime("%a %m/%d %I:%M %p CT") if tip else f"{g['date']} (tip n/a)",
            "status": g.get("status"),
        })

# ---------------------------------------------------------------- distributions


def pct(xs, p):
    xs = sorted(xs)
    if not xs:
        return None
    k = (len(xs) - 1) * p
    f, c = math.floor(k), math.ceil(k)
    return round(xs[f] + (xs[c] - xs[f]) * (k - f), 1)


def dist(pid: str) -> dict:
    games = glog.get(pid, [])
    xs = [g["fp"] for g in games]
    n = len(xs)
    out = {"gp": n}
    if n == 0:
        return out
    mean = sum(xs) / n
    sd = math.sqrt(sum((x - mean) ** 2 for x in xs) / (n - 1)) if n > 1 else 0.0
    recent = xs[-RECENT_N:]
    wts = [0.5 ** ((len(recent) - 1 - i) / HALF_LIFE) for i in range(len(recent))]
    rw = sum(w * x for w, x in zip(wts, recent)) / sum(wts)
    skew = (sum((x - mean) ** 3 for x in xs) / n) / (sd ** 3) if n > 2 and sd > 0 else None
    out.update(fpg=round(mean, 1), rw_mean=round(rw, 1), sd=round(sd, 1), p10=pct(xs, .1), p50=pct(xs, .5),
               p90=pct(xs, .9), max=max(xs), skew=round(skew, 2) if skew is not None else None,
               last5=[g["fp"] for g in games[-5:]])
    return out


def player_row(pid: str, slot: str) -> dict:
    p = players.get(pid, {})
    pj = proj.get(pid, [])
    team = p.get("team")
    rem = remaining.get(team, []) if team else []
    return {
        "player_id": pid, "slot": slot,
        "name": f"{p.get('first_name', '')} {p.get('last_name', '')}".strip() or pid,
        "team": team, "positions": p.get("fantasy_positions"),
        "injury": p.get("injury_status"), "injury_note": " ".join(x for x in (p.get("injury_body_part"), p.get("injury_notes")) if x) or None,
        **dist(pid),
        "proj_fpg": round(sum(x["fp"] for x in pj) / len(pj), 1) if pj else None,
        "proj_by_date": {x["date"]: x["fp"] for x in pj},
        "remaining": rem,
        "week_games_played": [g for g in glog.get(pid, []) if g["week"] == W],
    }


def roster_table(rid: int, m: dict) -> list[dict]:
    r = next(x for x in rosters if x["roster_id"] == rid)
    slots = [s for s in league["roster_positions"] if s != "BN"]
    rows, starters = [], (m or {}).get("starters") or r["starters"]
    reserve = set(r.get("reserve") or [])
    for i, pid in enumerate(starters):
        rows.append(player_row(pid, slots[i] if i < len(slots) else f"S{i}"))
    for pid in r["players"]:
        if pid not in starters:
            rows.append(player_row(pid, "IR" if pid in reserve else "BN"))
    pts = (m or {}).get("players_points") or {}
    for row in rows:
        row["matchup_points"] = pts.get(row["player_id"])
    return rows


umap = {u["user_id"]: u for u in users}


def team_name(rid):
    r = next(x for x in rosters if x["roster_id"] == rid)
    u = umap.get(r["owner_id"], {})
    return {"roster_id": rid, "team": (u.get("metadata") or {}).get("team_name") or u.get("display_name"),
            "owner": u.get("display_name"), "record": f"{r['settings']['wins']}-{r['settings']['losses']}"}


cur = matchups[W]
mine = next(m for m in cur if m["roster_id"] == MY_ROSTER_ID)
opp = next(m for m in cur if m["matchup_id"] == mine["matchup_id"] and m["roster_id"] != MY_ROSTER_ID)

# ---------------------------------------------------------------- lock-in verification
all_keys = {}
for w in range(1, W + 1):
    ks = set()
    for m in matchups[w] or []:
        ks |= set(m.keys())
    all_keys[w] = sorted(ks)
lockish = sorted({k for ks in all_keys.values() for k in ks if any(s in k.lower() for s in ("lock", "pick", "meta", "custom"))})

lock_examples = []
match_counts = defaultdict(int)
for w in range(1, W):
    m = next(x for x in matchups[w] if x["roster_id"] == MY_ROSTER_ID)
    for pid, sp in zip(m["starters"], m["starters_points"]):
        gs = [g for g in glog.get(pid, []) if g["week"] == w]
        fps = [g["fp"] for g in gs]
        tags = []
        for i, g in enumerate(gs):
            if abs(g["fp"] - sp) < 0.01:
                tags.append(f"game#{i + 1}/{len(gs)} ({g['date']})" + (" =LAST" if i == len(gs) - 1 else "") + (" =MAX" if g["fp"] == max(fps) else ""))
        if fps and abs(sum(fps) - sp) < 0.01 and len(fps) > 1:
            tags.append("=SUM")
        if not fps and sp == 0:
            tags.append("no games, 0")
        kind = ("last" if any("=LAST" in t for t in tags) else "single-nonlast" if tags and "=SUM" not in tags and "no games" not in tags[0] else
                "sum" if "=SUM" in tags else "none" if not tags else "zero")
        match_counts[kind] += 1
        lock_examples.append({"week": w, "pid": pid, "name": player_row(pid, "")["name"], "starters_points": sp, "game_fps": fps, "match": tags})
    bench_nonzero = {pid: v for pid, v in m["players_points"].items() if pid not in m["starters"] and v}
    lock_examples.append({"week": w, "bench_nonzero_players_points": len(bench_nonzero), "sample": dict(list(bench_nonzero.items())[:4]),
                          "points": m["points"], "sum_starters_points": round(sum(m["starters_points"]), 2)})

# ---------------------------------------------------------------- schedule pairings
pairings = {}
for w in range(1, last_reg + 1):
    ms = matchups.get(w) or []
    if not ms:
        continue
    mid = next((x["matchup_id"] for x in ms if x["roster_id"] == MY_ROSTER_ID), None)
    o = next((x["roster_id"] for x in ms if x["matchup_id"] == mid and x["roster_id"] != MY_ROSTER_ID and mid is not None), None)
    pairings[w] = {"matchup_id": mid, "opponent_roster_id": o, "opponent": team_name(o)["team"] if o else None}

my_r = next(x for x in rosters if x["roster_id"] == MY_ROSTER_ID)
summary = {
    "generated_ct": now_ct.isoformat(), "season": season, "week": W, "state": state,
    "endpoints": endpoint_log,
    "scoring_settings": SC, "roster_positions": league["roster_positions"],
    "league_settings": league["settings"],
    "waiver": {"budget": league["settings"].get("waiver_budget"), "used": my_r["settings"].get("waiver_budget_used"),
               "remaining": (league["settings"].get("waiver_budget") or 0) - (my_r["settings"].get("waiver_budget_used") or 0)},
    "bonus_audit": {k: dict(v) if isinstance(v, defaultdict) else v for k, v in bonus_audit.items()},
    "me": {**team_name(MY_ROSTER_ID), "points": mine["points"], "rows": roster_table(MY_ROSTER_ID, mine)},
    "opp": {**team_name(opp["roster_id"]), "points": opp["points"], "rows": roster_table(opp["roster_id"], opp)},
    "matchup_id": mine["matchup_id"],
    "matchup_keys_by_week": all_keys, "lock_related_keys": lockish,
    "sample_matchup_object": mine,
    "lock_match_counts": dict(match_counts), "lock_examples": lock_examples,
    "pairings": pairings, "transactions_count": len(transactions or []),
}
(RAW / "summary.json").write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")


# ---------------------------------------------------------------- markdown report
def table(rows):
    h = "| slot | name | team | pos | inj | GP | FP/G | rwFP | sd | p10 | p50 | p90 | max | skew | proj | wk pts | remaining (CT) |\n|" + "---|" * 17
    out = [h]
    for r in rows:
        rem = "; ".join(f"{g['tip_ct']} {g['opp']}" for g in r["remaining"]) or "-"
        out.append("| " + " | ".join(str(x if x is not None else "-") for x in [
            r["slot"], r["name"], r["team"], "/".join(r["positions"] or []), r["injury"], r.get("gp"), r.get("fpg"),
            r.get("rw_mean"), r.get("sd"), r.get("p10"), r.get("p50"), r.get("p90"), r.get("max"), r.get("skew"),
            r.get("proj_fpg"), r.get("matchup_points"), rem]) + " |")
    return "\n".join(out)


md = [f"# Week {W} snapshot ({now_ct:%a %b %d %I:%M %p CT})", "",
      f"**{summary['me']['team']}** ({summary['me']['record']}) {mine['points']}  vs  **{summary['opp']['team']}** "
      f"({summary['opp']['owner']}, {summary['opp']['record']}) {opp['points']}", "",
      "## My roster", table(summary["me"]["rows"]), "", "## Opponent", table(summary["opp"]["rows"]), "",
      "## Lock-in match counts (weeks 1-%d, roster %d starters)" % (W - 1, MY_ROSTER_ID), json.dumps(dict(match_counts)), "",
      "## Endpoints", *[f"- {e['status']}  {e['url']}" for e in endpoint_log]]
(RAW / "report.md").write_text("\n".join(md), encoding="utf-8")
print("\n".join(md[:8]))
print("wrote", RAW / "summary.json")
