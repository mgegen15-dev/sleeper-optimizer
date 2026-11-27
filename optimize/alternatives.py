"""'Also considered': every candidate for every slot that is open tonight, ranked by P(win).

For each starting slot that can take a new player before the next decision point (tonight's games):
  candidates = every non-IR player on my bench who plays tonight
             + every free agent who can be added AND play tonight (instant add: not on waivers, game not
               started), paired with the drop the waiver logic would make
For each candidate we force him into that slot tonight (the slot's current occupant is benched, which
forfeits his counting score - priced in), let the planner re-optimise everything else, and simulate
with the same random numbers as the main run, so P(win) differences are paired comparisons.
"""
from __future__ import annotations

from optimize.lock import eligible
from optimize.montecarlo import build_plan, p_win, simulate_plan, team_inputs

TIE_PP = 0.5


def _tonight(snap, pid, day):
    team = snap.players.get(pid, {}).get("team")
    return next((g for g in snap.week_games(team) if g.date == day and not g.started), None) if team else None


def evaluate(snap, cfg, book, me, opp_totals, draws, values_ros: dict, max_fa: int = 8):
    from data.news import game_model
    from optimize.waivers import free_agents, on_waivers
    rid = cfg["roster_id"]
    day = snap.now.date().isoformat()
    later = sorted(d for d in me.plan.day_assign if d >= day)
    day = later[0] if later else day
    inp = me.inputs or team_inputs(snap, rid)
    r = snap.roster(rid)
    reserve = set(r.get("reserve") or [])
    starters = {s.pid for s in inp["states"] if s.pid}
    choices = {d.slot_index: d.choice for d in me.decisions}
    frozen = set(inp["frozen"]) | {i for i, c in choices.items() if c == "lock"}
    tonight_assign = me.plan.day_assign.get(day, {})

    def run(inp_x, slot, pid, bench_occupant):
        ch = dict(choices)
        occ = next((s for s in inp_x["states"] if s.index == slot), None)
        if bench_occupant and occ is not None and occ.open_decision:
            ch[slot] = "ride"                      # benching him forfeits his counting score
        inp2 = dict(inp_x)
        inp2["pinned"] = {**inp_x["pinned"], (pid, day): slot}
        pl = build_plan(book, inp2, ch)
        tot = simulate_plan(pl, draws)
        run.plans[(slot, pid)] = pl
        return p_win(tot, opp_totals), pl.ev, tot

    run.plans = {}
    bench = [p for p in map(str, r["players"]) if p not in reserve and p not in starters and _tonight(snap, p, day)]
    bench_idle = [p for p in map(str, r["players"]) if p not in reserve and p not in starters and not _tonight(snap, p, day)]
    droppable = [p for p in map(str, r["players"]) if p not in reserve and p not in starters]
    fas = []
    for p in free_agents(snap):
        g = _tonight(snap, p, day)
        if g is None or book.get(p).gp < cfg["waivers"]["min_gp"] or on_waivers(snap, p, cfg):
            continue
        m, f = game_model(book, p, g)
        fas.append((m.expected(f), p))
    fas = [p for _, p in sorted(fas, reverse=True)[:max_fa]]

    out = []
    for s in inp["states"]:
        if s.index in frozen:
            continue
        moved_here = tonight_assign.get(s.index) not in (None, s.pid)
        bench_occ = not moved_here          # moved slot: keep the engine's choices (it already benches whoever it drops)
        occupant_plays = s.pid and _tonight(snap, s.pid, day) is not None
        cands = []
        for p in bench:
            if eligible(s.slot, inp["positions"].get(p) or snap.players.get(p, {}).get("fantasy_positions")):
                pw, ev, tot = run(inp, s.index, p, bench_occ)
                cands.append(dict(pid=p, name=snap.pname(p), source="bench", drop=None, pwin=pw, ev=ev, tot=tot))
        for p in fas:
            if not eligible(s.slot, snap.players.get(p, {}).get("fantasy_positions")):
                continue
            drop = min((d for d in droppable if d != p), key=lambda d: values_ros.get(d, 0.0), default=None)
            if drop is None:
                continue
            roster = [x for x in map(str, r["players"]) if x != drop] + [p]
            inp_f = team_inputs(snap, rid, roster_override=roster)
            pw, ev, tot = run(inp_f, s.index, p, bench_occ)
            cands.append(dict(pid=p, name=snap.pname(p), source="FA", drop=snap.pname(drop), pwin=pw, ev=ev, tot=tot))
        if not cands:
            continue
        for c in cands:                      # E[pts] tonight (our blend) and Sleeper's own projection, for tie-breaks
            g = _tonight(snap, c["pid"], day)
            m, f = game_model(book, c["pid"], g)
            c["e_pts"] = m.expected(f)
            c["sleeper_proj"] = (snap.proj.get(c["pid"]) or {}).get(g.date)
        cands.sort(key=lambda c: -c["pwin"])
        pick_pid = tonight_assign.get(s.index)
        keep = dict(pid=s.pid, name=snap.pname(s.pid) if s.pid else "—", source="keep", drop=None,
                    pwin=p_win(me.totals, opp_totals), ev=me.ev, tot=me.totals)
        moved = pick_pid is not None and pick_pid != s.pid and pick_pid not in starters
        pick = next((c for c in cands if c["pid"] == pick_pid), None) if moved else keep
        if pick is None:
            pick = keep
        from optimize.selfcheck import paired
        pick_plan = run.plans.get((s.index, pick.get("pid"))) or me.plan
        later_use = {}
        for d_, a_ in pick_plan.day_assign.items():
            for sl, p_ in a_.items():
                if d_ > day:
                    later_use.setdefault(p_, (d_, inp["slot_names"][sl]))
        rest = [c for c in cands if c["pid"] != pick.get("pid")]
        for c in rest:
            c["delta"] = c["pwin"] - pick["pwin"]
            c["se"] = paired(c["tot"], pick["tot"], opp_totals)[1] if pick.get("tot") is not None else 0.0
            c["tie"] = abs(c["delta"]) * 100 < TIE_PP
            m_ = game_model(book, c["pid"], _tonight(snap, c["pid"], day))[0]
            if c["source"] == "FA":
                c["reason"] = f"free agent: costs a roster spot (drop {c['drop']})"
            elif m_.p_play < 0.9:
                c["reason"] = f"{snap.players.get(c['pid'], {}).get('injury_status') or 'questionable'}, {m_.p_play * 100:.0f}% to play"
            elif c["pid"] in later_use:
                d_, sl = later_use[c["pid"]]
                c["reason"] = f"more valuable as the {d_[5:]} fallback for {sl}"
            elif c.get("e_pts", 0) < pick.get("e_pts", 0):
                c["reason"] = f"lower E tonight ({c['e_pts']:.1f} vs {pick.get('e_pts', 0):.1f})"
            else:
                c["reason"] = "lower P(win) once the rest of the week is re-planned"
        idle = [dict(pid=p, name=snap.pname(p), source="bench", drop=None, pwin=None, delta=None, se=None, tie=False,
                     reason="no game tonight" + (f"; used {later_use[p][0][5:]} in {later_use[p][1]}" if p in later_use else ""))
                for p in bench_idle if eligible(s.slot, inp["positions"].get(p) or snap.players.get(p, {}).get("fantasy_positions"))]
        keep_opt = []
        occ_state = next((x for x in inp["states"] if x.index == s.index), None)
        if moved and occ_state is not None and occ_state.pid and occ_state.open_decision:
            pl_k = build_plan(book, inp, {**choices, s.index: "hold"})
            t_k = simulate_plan(pl_k, draws)
            pk = p_win(t_k, opp_totals)
            g_k = _tonight(snap, occ_state.pid, day)
            m_k = game_model(book, occ_state.pid, g_k)[0] if g_k else None
            why = (f"{snap.players.get(occ_state.pid, {}).get('injury_status') or 'healthy'}, {m_k.p_play * 100:.0f}% to play; "
                   f"his own game tonight uses the slot") if m_k else "no game tonight; keeping him blocks the slot"
            keep_opt = [dict(pid=occ_state.pid, name=snap.pname(occ_state.pid), source="keep", drop=None, pwin=pk,
                             delta=pk - pick["pwin"], se=paired(t_k, pick["tot"], opp_totals)[1] if pick.get("tot") is not None else 0.0,
                             tie=abs(pk - pick["pwin"]) * 100 < TIE_PP, reason=why)]
        all_bench = keep_opt + [c for c in rest if c["source"] == "bench"] + idle
        best_fa = [c for c in rest if c["source"] == "FA"][:2]
        runners = rest[:3]
        tiebreak = None
        top = runners[0] if runners else None
        if moved and top and top["tie"] and top.get("e_pts", 0) > pick.get("e_pts", 0):
            tiebreak = (f"Statistical tie with {top['name']} ({top['pwin'] * 100:.1f}% vs {pick['pwin'] * 100:.1f}%); "
                        f"he projects higher tonight ({top['e_pts']:.1f} vs {pick['e_pts']:.1f}), so start {top['name']} instead.")
        sl_note = None
        if moved and top and pick.get("sleeper_proj") is not None and top.get("sleeper_proj") is not None:
            ours = pick.get("e_pts", 0) >= top.get("e_pts", 0)
            theirs = pick["sleeper_proj"] >= top["sleeper_proj"]
            if ours != theirs:
                sl_note = (f"Sleeper's projection disagrees: {top['name']} {top['sleeper_proj']:.1f} vs {pick['name']} {pick['sleeper_proj']:.1f} "
                           f"(ours {top.get('e_pts', 0):.1f} vs {pick.get('e_pts', 0):.1f}).")
        out.append(dict(all_options=all_bench + best_fa, tiebreak=tiebreak, sleeper_note=sl_note, slot=s.slot, slot_index=s.index, occupant=snap.pname(s.pid) if s.pid else None,
                        occupant_plays_tonight=bool(occupant_plays), moved=moved, pick=pick, runners_up=runners,
                        best_alternative=None if moved else (runners[0] if runners else None)))
    return out


def describe(c, pick_name=None):
    src = "bench" if c["source"] == "bench" else f"FA, drop {c['drop']}"
    tie = " (tie)" if c.get("tie") else ""
    return f"{c['name']} ({src}) {c['pwin'] * 100:.1f}% ({c['delta'] * 100:+.1f}pp){tie}"
