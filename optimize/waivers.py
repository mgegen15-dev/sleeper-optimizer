"""Waiver / free-agent add-drop evaluation.

For each candidate (top free agents by blended FP/G) paired with my lowest-value droppable player:
  * dP(win this week): rerun the week engine with the swapped roster, counting only the
    candidate's games he can actually be started for (after he is available, not yet tipped).
  * d playoff odds: season sim with the swapped roster (common random numbers).
Availability: league has waiver_type 2 (FAAB), waiver_clear_days, waiver_after_game_start.
  Assumption: a free agent is an instant add unless (a) he was dropped within waiver_clear_days, or
  (b) his team's game today has already tipped; then he is on waivers and available after
  waiver_clear_days (bid required).
FAAB bid = clamp(k * dPlayoff_pp * remaining_budget / weeks_left, min_bid, max_frac * remaining).
  Intuition: spend in proportion to how many playoff percentage points the player buys, relative
  to what the remaining budget has to cover.
"""
from __future__ import annotations

from datetime import timedelta

from optimize.montecarlo import Draws, p_win, solve_team, team_inputs


def free_agents(snap):
    rostered = {str(p) for r in snap.rosters for p in (r.get("players") or [])}
    return [pid for pid, p in snap.players.items()
            if p.get("team") and p.get("active", True) and pid not in rostered and p.get("sport", "nba") == "nba"]


def on_waivers(snap, pid, cfg) -> bool:
    clear = int(snap.league["settings"].get("waiver_clear_days", 2))
    cutoff_ms = (snap.now - timedelta(days=clear)).timestamp() * 1000
    for t in snap.transactions or []:
        if (t.get("drops") or {}).get(pid) is not None and (t.get("created") or 0) >= cutoff_ms:
            return True
    if snap.league["settings"].get("waiver_after_game_start"):
        team = snap.players.get(pid, {}).get("team")
        today = snap.now.date().isoformat()
        if any(g.date == today and g.started for g in snap.week_games(team)):
            return True
    return False


def evaluate(snap, cfg, book, book_ros, me_week, opp_week, season, base_po, today_lineup: set, value_ros: dict):
    wc, sims = cfg["waivers"], cfg["sims"]
    rid = cfg["roster_id"]
    r = snap.roster(rid)
    reserve = set(r.get("reserve") or [])
    frozen_p = {s.pid for s in me_week.states if s.locked}
    droppable = [p for p in r["players"] if p not in reserve and p not in frozen_p
                 and not (wc["protect_starters_today"] and p in today_lineup)]
    if not droppable:
        return []
    drop = min(droppable, key=lambda p: value_ros.get(p, 0.0))
    fas = [p for p in free_agents(snap) if book.get(p).gp >= wc["min_gp"]]
    fas.sort(key=lambda p: -book.get(p).mu)
    fas = fas[: wc["candidates"]]
    clear = int(snap.league["settings"].get("waiver_clear_days", 2))
    budget = snap.league["settings"].get("waiver_budget", 0) - (r["settings"].get("waiver_budget_used") or 0)
    weeks_left = max(1, cfg["season"]["last_regular_week"] - snap.week + 1)
    base_pw = p_win(me_week.totals, opp_week.totals)

    def run(cand, n):
        roster = [p for p in r["players"] if p != drop] + [cand]
        wait = on_waivers(snap, cand, cfg)
        avail = {cand: (snap.now.date() + timedelta(days=clear if wait else 0)).isoformat()}
        inp = team_inputs(snap, rid, roster_override=roster, available_from=avail)
        dr = Draws(book, n, sims["seed"])
        opp_tot = opp_week.totals[:n]
        tw = solve_team(snap, rid, book, dr, vs=opp_tot, inp=inp, passes=1)
        return tw, roster, wait

    screened = []
    base_small = solve_team(snap, rid, book, Draws(book, sims["n_screen"], sims["seed"]),
                            vs=opp_week.totals[: sims["n_screen"]], inp=team_inputs(snap, rid), passes=1)
    base_small_pw = p_win(base_small.totals, opp_week.totals[: sims["n_screen"]])
    for c in fas:
        tw, roster, wait = run(c, sims["n_screen"])
        dpw = p_win(tw.totals, opp_week.totals[: sims["n_screen"]]) - base_small_pw
        d_ros = value_ros_of(book_ros, snap, c) - value_ros.get(drop, 0.0)
        screened.append((dpw * 100 + d_ros, c))
    screened.sort(reverse=True)
    out = []
    for _, c in screened[: wc["finalists"]]:
        tw, roster, wait = run(c, sims["n_week"])
        dpw = p_win(tw.totals, opp_week.totals) - base_pw
        po, _ = season.run(rosters={rid: roster}, week_totals={rid: tw.totals})
        dpo = po[rid] - base_po[rid]
        bid = 0
        if wait:
            raw = wc["faab_k"] * dpo * 100 * budget / weeks_left
            bid = int(max(wc["faab_min_bid"], min(raw, wc["faab_max_frac"] * budget)))
        rec = dpo * 100 >= wc["drop_threshold_playoff_pp"] or dpw * 100 >= wc["drop_threshold_pwin_pp"]
        out.append(dict(add=c, drop=drop, dpwin=dpw, dplayoff=dpo, bid=bid, waivers=wait, recommend=rec,
                        add_mu=book.get(c).mu, drop_mu=book.get(drop).mu))
    out.sort(key=lambda d: -(d["dplayoff"] + d["dpwin"] / 10))
    return out


def value_ros_of(book_ros, snap, pid):
    from optimize.trades import player_week_value
    return player_week_value(book_ros, snap, pid)
