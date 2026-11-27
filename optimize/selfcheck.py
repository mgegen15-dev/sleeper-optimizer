"""Self-check: re-test every call against its full option set before the email goes out.

For each open lock decision we evaluate, on the SAME simulated paths (common random numbers):
    lock now · ride (keep him for his next game, lock option after it) · bench now (forfeit, free the slot)
    · ride to the end (never lock) - plus everything the planner does with the other slots.
For each swap we compare the pick with every bench player / addable free agent for that slot
(from optimize/alternatives.py, also on common random numbers).
A call is overturned only if an alternative beats it by more than noise:
    paired difference > 2 standard errors AND > 0.5 percentage points.
Differences within noise are reported as ties. Logs get counts only.
"""
from __future__ import annotations

import copy
from dataclasses import replace

import numpy as np

from optimize.lock import backward_induction
from optimize.montecarlo import build_plan, p_win, simulate_plan

MIN_PP = 0.5
SE_MULT = 2.0
NAMES = {"lock": "lock", "hold": "ride", "ride": "bench", "end": "ride to end"}


def paired(a_tot, b_tot, opp):
    """P(win) difference a - b on the same paths, with its standard error."""
    wa = (a_tot > opp) + 0.5 * (a_tot == opp)
    wb = (b_tot > opp) + 0.5 * (b_tot == opp)
    d = wa.astype(float) - wb.astype(float)
    return float(d.mean()), float(d.std(ddof=1) / np.sqrt(len(d))) if len(d) > 1 else 0.0


def beats(diff, se):
    return diff > SE_MULT * se and diff * 100 > MIN_PP


def _ride_to_end(plan, slot):
    pl = copy.deepcopy(plan)
    for st in pl.slots[slot].seq[:-1]:
        st.threshold = float("inf")
    return pl


def sleeper_bar(snap, book, pid, games):
    """Lock bar if each remaining game's mean were Sleeper's own projection (same shape, same P(play))."""
    from data.news import game_model
    proj = snap.proj.get(pid) or {}
    dists = []
    for g in games:
        m, f = game_model(book, pid, g)
        p = proj.get(g.date)
        if p is None:
            return None
        dists.append((replace(m, mu=float(p) / max(m.p_play, 1e-9) / max(f, 1e-9) / max(float(m.R.mean()), 1e-9)), f))
    V, _ = backward_induction(dists)
    return V[0] if V else None


def check_decisions(snap, book, me, opp_totals, draws):
    """Re-evaluate every lock/ride/bench call. May overturn me's choices (and rebuild its plan).
    Returns per-call results (structured, with names - for the email only)."""
    from data.news import game_model
    choices = {d.slot_index: d.choice for d in me.decisions}
    results = []
    changed = False
    for d in me.decisions:
        opts = (["lock"] if d.lockable else []) + ["hold", "ride"]
        tots = {}
        for c in opts:
            tots[c] = simulate_plan(build_plan(book, me.inputs, {**choices, d.slot_index: c}), draws)
        tots["end"] = simulate_plan(_ride_to_end(build_plan(book, me.inputs, {**choices, d.slot_index: "hold"}), d.slot_index), draws)
        cur = choices[d.slot_index]
        best_alt = max((c for c in tots if c != cur), key=lambda c: p_win(tots[c], opp_totals))
        diff, se = paired(tots[best_alt], tots[cur], opp_totals)
        overturned = best_alt != "end" and beats(diff, se)
        if overturned:
            choices[d.slot_index] = best_alt
            changed = True
        final = choices[d.slot_index]
        # compare with the best option that actually plays out differently (e.g. "bench now" is identical to
        # "ride" when his next game is tonight and he goes straight back in)
        others = {c: t for c, t in tots.items() if c != final and not np.array_equal(t, tots[final])} or                  {c: t for c, t in tots.items() if c != final}
        runner = max(others, key=lambda c: p_win(others[c], opp_totals))
        rdiff, rse = paired(tots[final], tots[runner], opp_totals)
        team = snap.players.get(d.pid, {}).get("team")
        games = [g for g in snap.week_games(team) if not g.started] if team else []
        gl = [(g, game_model(book, d.pid, g)[0].p_play) for g in games]
        sb = sleeper_bar(snap, book, d.pid, games) if games else None
        sleeper = None
        if sb is not None and d.lockable and d.x is not None:
            ours_lock = final == "lock"
            sleeper = "agrees" if (d.x >= sb) == ours_lock else f"disagrees (its bar {sb:.1f})"
        results.append(dict(pid=d.pid, slot_index=d.slot_index, call=final, overturned=overturned, from_call=cur if overturned else None,
                            pwin=p_win(tots[final], opp_totals), runner=runner, runner_pwin=p_win(tots[runner], opp_totals),
                            diff=rdiff, se=rse, tie=not beats(rdiff, rse), games=gl, bar=d.threshold if d.lockable else None,
                            x=d.x, sleeper=sleeper))
    if changed:
        plan = build_plan(book, me.inputs, choices)
        me.plan, me.totals = plan, simulate_plan(plan, draws)
        for d in me.decisions:
            d.choice = choices[d.slot_index]
    return results


def check_swaps(considered, opp_totals):
    """Pick vs best runner-up per slot (alternatives carry their simulated totals)."""
    out = []
    for c in considered:
        pick, runners = c["pick"], c["runners_up"]
        if not runners or pick.get("tot") is None:
            continue
        best = max(runners, key=lambda r: r["pwin"])
        diff, se = paired(best["tot"], pick["tot"], opp_totals)
        out.append(dict(slot=c["slot"], pick=pick["name"], best=best["name"], overturned=c["moved"] and beats(diff, se),
                        diff=-diff, se=se, tie=not beats(abs(diff), se), pick_pwin=pick["pwin"], best_pwin=best["pwin"]))
    return out


def proof_line(r, today):
    def day(g):
        t = g.tip.strftime("%I:%M").lstrip("0") if g.tip else "?"
        return (t if g.date == today else f"{g.tip:%a} {t}") if g.tip else g.date
    parts = ["Games left: " + (", ".join(f"{day(g)} ({p * 100:.0f}%)" for g, p in r["games"]) or "none")]
    if r.get("bar") is not None and r.get("x") is not None:
        parts.append(f"bar {r['bar']:.1f} vs {r['x']:.1f}")
    if r.get("bar") is None and r["call"] in ("hold", "ride"):      # not lockable: riding is the only real choice
        parts.append(f"ride {r['pwin'] * 100:.1f}%")
    else:
        parts.append(f"{NAMES[r['call']]} {r['pwin'] * 100:.1f}% vs {NAMES[r['runner']]} {r['runner_pwin'] * 100:.1f}% "
                     f"({r['diff'] * 100:+.1f} ± {r['se'] * 100:.1f}{', tie' if r['tie'] else ''})")
    if r.get("sleeper"):
        parts.append(f"Sleeper {r['sleeper']}")
    if r.get("overturned"):
        parts.append(f"self-check changed this from {NAMES[r['from_call']]}")
    return " · ".join(parts)


def counts(dec_results, swap_results):
    n = len(dec_results) + len(swap_results)
    over = sum(r["overturned"] for r in dec_results) + sum(r["overturned"] for r in swap_results)
    ties = sum(r["tie"] for r in dec_results) + sum(r["tie"] for r in swap_results)
    return f"{n - over}/{n} pass, {over} overturned, {ties} ties"
