"""Season simulation -> playoff odds.

Weekly team score for a future week w ~ Normal(mu[team, w], sd), where mu is the planner's expected
lineup value for that team's real NBA schedule that week (number of games matters a lot in a
lock-in league), and sd is the league's empirical week-to-week SD of team scores around each
team's own average. The current week uses the full Monte Carlo totals of each team instead.
Standings: wins (ties = 1/2), then points-for. Playoffs: top `playoff_teams`; if
division_winners_qualify, each division leader is in regardless of rank.
Common random numbers (one fixed matrix of normal shocks) make before/after comparisons of a
trade or waiver move far less noisy than two independent runs.
"""
from __future__ import annotations

import numpy as np

from model.distributions import ModelBook
from optimize.lineup import plan_week


def history(snap):
    """Completed weeks: (wins, ties, pf) per team, plus pooled residual SD of weekly scores."""
    rec = {r["roster_id"]: [0.0, 0.0] for r in snap.rosters}     # wins, pf
    scores = {r["roster_id"]: [] for r in snap.rosters}
    for w in range(1, snap.week):
        ms = snap.matchups.get(w) or []
        by = {}
        for m in ms:
            by.setdefault(m["matchup_id"], []).append(m)
            rec[m["roster_id"]][1] += m["points"] or 0
            scores[m["roster_id"]].append(m["points"] or 0)
        for pair in by.values():
            if len(pair) == 2:
                a, b = pair
                if a["points"] == b["points"]:
                    rec[a["roster_id"]][0] += .5
                    rec[b["roster_id"]][0] += .5
                else:
                    rec[(a if a["points"] > b["points"] else b)["roster_id"]][0] += 1
    res = [x - np.mean(v) for v in scores.values() if len(v) >= 2 for x in v]
    n = len(res)
    k = sum(1 for v in scores.values() if len(v) >= 2)
    sd = float(np.sqrt(np.sum(np.square(res)) / max(n - k, 1))) if n > k else None
    return rec, sd


def team_week_mu(book: ModelBook, snap, roster, week: int) -> float:
    """Expected points for a full future week with this roster (no locks yet)."""
    slot_names = [s for s in snap.league["roster_positions"] if s != "BN"]
    pool = [str(p) for p in roster]
    games = {p: snap.week_games(snap.players.get(p, {}).get("team"), week) if snap.players.get(p, {}).get("team") else []
             for p in pool}
    pos = {p: snap.players.get(p, {}).get("fantasy_positions") for p in pool}
    return plan_week(book, slot_names, pool, games, pos).ev


class SeasonSim:
    def __init__(self, snap, cfg, book_ros: ModelBook, week_totals: dict, n: int, seed: int):
        self.snap, self.cfg, self.book = snap, cfg, book_ros
        self.n = n
        sc = cfg["season"]
        self.weeks = list(range(snap.week + 1, sc["last_regular_week"] + 1))
        self.rids = sorted(r["roster_id"] for r in snap.rosters)
        self.rec, sd = history(snap)
        self.sd = sd or sc["week_sd_fallback"]
        rng = np.random.default_rng(seed)
        self.Z = rng.standard_normal((n, len(self.rids), len(self.weeks)))
        self.idx = rng.integers(0, 10 ** 9, n)
        self.week_totals = week_totals            # rid -> np.array of this week's simulated totals
        self.mu_cache = {}
        self.div = {r["roster_id"]: (r.get("settings") or {}).get("division") for r in snap.rosters}

    def mu(self, rid, roster=None):
        roster = roster if roster is not None else self.snap.roster(rid)["players"]
        key = (rid, tuple(sorted(map(str, roster))))
        if key not in self.mu_cache:
            self.mu_cache[key] = np.array([team_week_mu(self.book, self.snap, roster, w) for w in self.weeks])
        return self.mu_cache[key]

    def run(self, rosters: dict | None = None, week_totals: dict | None = None):
        """rosters: optional {rid: player list} overrides. Returns (playoff_prob, exp_wins) dicts."""
        rosters, wt = rosters or {}, {**self.week_totals, **(week_totals or {})}
        n, T = self.n, len(self.rids)
        col = {rid: i for i, rid in enumerate(self.rids)}
        wins = np.tile([self.rec[r][0] for r in self.rids], (n, 1)).astype(float)
        pf = np.tile([self.rec[r][1] for r in self.rids], (n, 1)).astype(float)
        # current week
        W = self.snap.week
        cur = np.zeros((n, T))
        for rid in self.rids:
            arr = wt[rid]
            cur[:, col[rid]] = arr[self.idx % len(arr)]
        self._apply_week(W, cur, wins, pf, col)
        for j, w in enumerate(self.weeks):
            mus = np.array([self.mu(r, rosters.get(r)) [j] for r in self.rids])
            sc = mus[None, :] + self.sd * self.Z[:, :, j]
            self._apply_week(w, sc, wins, pf, col)
        key = wins * 1e5 + pf
        made = np.zeros((n, T), bool)
        if self.cfg["season"]["division_winners_qualify"]:
            for dv in {d for d in self.div.values() if d}:
                cols = [col[r] for r in self.rids if self.div[r] == dv]
                best = np.array(cols)[np.argmax(key[:, cols], axis=1)]
                made[np.arange(n), best] = True
        order = np.argsort(-key, axis=1)
        need = self.cfg["season"]["playoff_teams"]
        for i in range(n):           # fill remaining spots by rank
            k = int(made[i].sum())
            for c in order[i]:
                if k >= need:
                    break
                if not made[i, c]:
                    made[i, c] = True
                    k += 1
        po = {r: float(made[:, col[r]].mean()) for r in self.rids}
        ew = {r: float(wins[:, col[r]].mean()) for r in self.rids}
        return po, ew

    def _apply_week(self, w, sc, wins, pf, col):
        ms = self.snap.matchups.get(w) or []
        by = {}
        for m in ms:
            by.setdefault(m["matchup_id"], []).append(m["roster_id"])
        pf += sc
        for pair in by.values():
            if len(pair) != 2:
                continue
            a, b = col[pair[0]], col[pair[1]]
            wins[:, a] += (sc[:, a] > sc[:, b]) + 0.5 * (sc[:, a] == sc[:, b])
            wins[:, b] += (sc[:, b] > sc[:, a]) + 0.5 * (sc[:, a] == sc[:, b])
