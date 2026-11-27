"""Replay past weeks: what would the lock policy have scored vs what was actually locked?

For each week, distributions are rebuilt using ONLY games before that week (plus that week's
Sleeper projections, which existed beforehand). Each final starter is treated as sitting in his
slot all week (the API keeps only the final lineup), facing his team's games in order; the model
locks the first game whose score beats the continuation value, else takes the last game.
Injury status at the time isn't available historically, so everyone gets the 'healthy' P(play).
"""
from __future__ import annotations

from model.distributions import ModelBook
from optimize.lock import backward_induction


class _AllHealthy(dict):
    def __contains__(self, k):
        return True

    def __getitem__(self, k):
        return None


def run(snap, cfg, weeks, rid):
    rows = []
    for w in weeks:
        m = snap.matchup(rid, w)
        if not m:
            continue
        dates = sorted({g.date for gs in snap.team_games.values() for g in gs if g.week == w})
        book = ModelBook(snap.logs, snap.players, snap.proj, cfg, cutoff=dates[0], proj_dates=set(dates),
                         status_override=_AllHealthy())
        slots = [s for s in snap.league["roster_positions"] if s != "BN"]
        for i, pid in enumerate(m["starters"]):
            if pid in (None, "0"):
                continue
            actual = float(m["starters_points"][i] or 0)
            wlogs = [g for g in snap.logs.get(pid, []) if g.week == w]
            team = wlogs[0].team if wlogs else snap.players.get(pid, {}).get("team")
            games = snap.week_games(team, w) if team else []
            fp_by_date = {g.date: g.fp for g in wlogs}
            fps = [fp_by_date.get(g.date, 0.0) for g in games]
            mdl = book.get(pid)
            V, thr = backward_induction([(mdl, book.opp_factor(pid, g.opp)) for g in games])
            pick = len(games) - 1
            for k, t in enumerate(thr):
                if t is not None and fps[k] >= t:
                    pick = k
                    break
            model_pts = fps[pick] if games else 0.0
            act_idx = next((k for k, x in enumerate(fps) if abs(x - actual) < 0.01), None) if actual > 0 else None
            comparable = act_idx is not None or (actual == 0 and (not fps or fps[-1] == 0))
            rows.append(dict(week=w, slot=slots[i] if i < len(slots) else "", pid=pid, name=snap.pname(pid),
                             games=fps, thresholds=[None if t is None else round(t, 1) for t in thr],
                             model_pick=pick, model_pts=model_pts, actual_pick=act_idx, actual_pts=actual,
                             best=max(fps) if fps else 0.0, comparable=comparable, ev_start=round(V[0], 1) if V else 0))
    return rows


def summarize(rows):
    weeks = sorted({r["week"] for r in rows})
    out = []
    for w in weeks:
        rs = [r for r in rows if r["week"] == w]
        comp = [r for r in rs if r["comparable"]]
        out.append(dict(week=w, model=sum(r["model_pts"] for r in comp), actual=sum(r["actual_pts"] for r in comp),
                        oracle=sum(r["best"] for r in comp), n=len(comp), excluded=[r["name"] for r in rs if not r["comparable"]],
                        diffs=[r for r in comp if r["model_pick"] != r["actual_pick"]]))
    return out
