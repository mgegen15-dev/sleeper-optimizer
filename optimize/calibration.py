"""Forward calibration test of P(win).

Every analysed run logs (timestamp, week, our P(win), Sleeper-style P(win)) in state["calib"] - state
lives in the Actions cache, never in git, and logs print no names. Sleeper-style = Sleeper projections
only, everyone plays, no news. When a week is over, the outcome is read from Sleeper's OFFICIAL roster
record (wins / losses / ties moving), not from the matchup feed, and every checkpoint of that week is
scored with the Brier score and log loss.
"""
from __future__ import annotations

import copy
import math


def sleeper_style_pwin(snap, cfg, n=10000):
    from data.news import game_model  # noqa: F401  (book without news -> plain per-game projections)
    from model.distributions import ModelBook
    from optimize.montecarlo import Draws, p_win, solve_team
    c = copy.deepcopy(cfg)
    c["model"]["blend_weights"] = {"recency": 1e-4, "season": 1e-4, "projection": 1.0}
    c["model"]["injury_play_prob"] = {k: 1.0 for k in c["model"]["injury_play_prob"]}
    wd = {g.date for gs in snap.team_games.values() for g in gs if g.week == snap.week}
    book = ModelBook(snap.logs, snap.players, snap.proj, c, proj_dates=wd)
    dr = Draws(book, n, c["sims"]["seed"] + 101)
    rid = cfg["roster_id"]
    opp = solve_team(snap, snap.opponent_rid(rid), book, dr)
    me = solve_team(snap, rid, book, dr)
    return p_win(me.totals, opp.totals)


def _record(snap, rid):
    st = snap.roster(rid).get("settings") or {}
    return [int(st.get("wins", 0)), int(st.get("losses", 0)), int(st.get("ties", 0))]


def log_run(snap, cfg, state, ours, sleeper):
    cal = state.setdefault("calib", {"weeks": {}})
    wk = cal["weeks"].setdefault(str(snap.week), {"record_at_start": _record(snap, cfg["roster_id"]), "points": []})
    wk["points"].append({"t": snap.now.isoformat(), "ours": round(float(ours), 4), "sleeper": round(float(sleeper), 4)})
    wk["points"] = wk["points"][-200:]
    settle(snap, cfg, state)


def settle(snap, cfg, state):
    """Score any finished week whose outcome shows up in the official record."""
    cal = state.get("calib") or {}
    now_rec = _record(snap, cfg["roster_id"])
    for w, wk in (cal.get("weeks") or {}).items():
        if "outcome" in wk or int(w) >= snap.week:
            continue
        start = wk.get("record_at_start")
        if not start:
            continue
        dw, dl, dt = (now_rec[i] - start[i] for i in range(3))
        if dw + dl + dt != 1:            # record moved by more/less than one game: can't attribute safely
            if dw + dl + dt > 1:
                wk["outcome"] = None
            continue
        wk["outcome"] = 1.0 if dw else (0.0 if dl else 0.5)


def scores(state):
    rows = {"ours": [], "sleeper": []}
    weeks = 0
    for wk in ((state.get("calib") or {}).get("weeks") or {}).values():
        y = wk.get("outcome")
        if y is None:
            continue
        weeks += 1
        for p in wk["points"]:
            for k in rows:
                rows[k].append((p[k], y))
    out = {"weeks": weeks, "n": len(rows["ours"])}
    for k, r in rows.items():
        if r:
            out[k] = dict(brier=sum((p - y) ** 2 for p, y in r) / len(r),
                          logloss=-sum(y * math.log(min(max(p, .01), .99)) + (1 - y) * math.log(1 - min(max(p, .01), .99))
                                       for p, y in r) / len(r))
    return out


def footer_line(state, min_weeks=2):
    s = scores(state)
    if s["weeks"] < min_weeks or "ours" not in s:
        return None
    return (f"Calibration so far ({s['weeks']} weeks, {s['n']} checkpoints): Brier ours {s['ours']['brier']:.3f} vs "
            f"Sleeper-style {s['sleeper']['brier']:.3f} (50/50 guessing = 0.250; lower is better)")
