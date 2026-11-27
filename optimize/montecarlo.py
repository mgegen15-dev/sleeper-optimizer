"""Monte Carlo of the rest of the week and the lock/ride decision engine.

Draws use common random numbers: the outcome of (player, game) is generated from a seed derived
from (seed, player, date), so "lock" and "ride" scenarios are compared on identical game outcomes.
That removes most simulation noise from the *difference* in P(win), like pricing two options on
the same Monte Carlo paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from data.news import game_model
from model.distributions import ModelBook, stable_seed
from optimize.lineup import Plan, plan_week
from optimize.lock import infer_slots


class Draws:
    def __init__(self, book: ModelBook, n: int, seed: int):
        self.book, self.n, self.seed = book, n, seed
        self._c = {}

    def get(self, pid: str, game) -> np.ndarray:
        key = (pid, game.date)
        if key not in self._c:
            m, fac = game_model(self.book, pid, game)
            rng = np.random.default_rng(stable_seed(self.seed, pid, game.date))
            playing = rng.random(self.n) < m.p_play
            idx = rng.integers(0, len(m.R), self.n)
            self._c[key] = np.where(playing, m.mu * fac * m.R[idx], 0.0)
        return self._c[key]


def simulate_plan(plan: Plan, draws: Draws) -> np.ndarray:
    """Final team total per simulation, following the plan's threshold (lock) policy."""
    tot = np.zeros(draws.n)
    for sp in plan.slots:
        if sp.frozen is not None:
            tot += sp.frozen
        elif sp.keep:
            tot += sp.option
        elif sp.seq:
            val = np.zeros(draws.n)
            done = np.zeros(draws.n, bool)
            for st in sp.seq:
                x = draws.get(st.pid, st.game)
                take = ~done if st.threshold is None else (~done & (x >= st.threshold))
                val[take] = x[take]
                done |= take
            tot += val
        elif sp.option is not None:
            tot += sp.option
    return tot


def p_win(me: np.ndarray, opp: np.ndarray) -> float:
    return float(np.mean(me > opp) + 0.5 * np.mean(me == opp))


@dataclass
class Decision:
    slot_index: int
    slot: str
    pid: str
    x: float
    choice: str = "ride"
    ev_lock: float = 0.0
    ev_ride: float = 0.0
    threshold: float = 0.0            # x at which lock and ride have equal expected points
    pwin_lock: float | None = None
    pwin_ride: float | None = None
    next_game: object = None
    lockable: bool = True
    pwin_hold: float | None = None      # keep him in the slot until his next game (no lock)
    pwin_bench: float | None = None     # bench him now and use the slot: forfeits his counting score


@dataclass
class TeamWeek:
    rid: int
    states: list
    plan: Plan
    totals: np.ndarray
    decisions: list = field(default_factory=list)
    inputs: dict = field(default_factory=dict)

    @property
    def ev(self):
        return float(self.totals.mean())


def team_inputs(snap, rid: int, roster_override=None, available_from: dict | None = None, reported: dict | None = None) -> dict:
    if reported is None:
        reported = (getattr(snap, "reported_locks", None) or {}).get(rid)
    states = infer_slots(snap, rid, reported=reported)
    r = snap.roster(rid)
    reserve = set(r.get("reserve") or [])
    roster = [str(p) for p in (roster_override if roster_override is not None else r["players"])]
    frozen = {s.index: s.x for s in states if s.locked}
    frozen_p = {s.pid for s in states if s.locked}
    occupant = {s.pid: s.index for s in states if s.pid}
    pool = [p for p in roster if p not in reserve and p not in frozen_p]
    games, pinned = {}, {}
    for pid in pool:
        team = snap.players.get(pid, {}).get("team")
        gl = []
        for g in snap.week_games(team) if team else []:
            if g.final:
                continue
            if g.started:                    # in progress: only counts for the slot he's sitting in
                if pid in occupant:
                    pinned[(pid, g.date)] = occupant[pid]
                    gl.append(g)
                continue
            if available_from and pid in available_from and g.date < available_from[pid]:
                continue
            gl.append(g)
        games[pid] = gl
    options = {s.index: (s.pid, s.x) for s in states if s.x is not None and not s.locked and s.done and s.pid in roster}
    positions = {p: snap.players.get(p, {}).get("fantasy_positions") for p in roster}
    slot_names = [s for s in snap.league["roster_positions"] if s != "BN"]
    return dict(states=states, frozen=frozen, pool=pool, games=games, pinned=pinned, options=options,
                positions=positions, slot_names=slot_names, roster=roster)


def build_plan(book, inp, choices: dict) -> Plan:
    """choices per open decision: "lock" (freeze his score), "hold" (keep him in the slot until his next
    game; his score is dropped once that game starts), "ride" (bench him now: forfeits the score, frees the slot)."""
    frozen = dict(inp["frozen"])
    pool = list(inp["pool"])
    options = dict(inp["options"])
    blocked = {}
    for s in inp["states"]:
        if s.open_decision and s.pid in inp["roster"]:
            c = choices.get(s.index)
            if c == "lock" and s.lockable:
                frozen[s.index] = s.x
                pool = [p for p in pool if p != s.pid]
            elif c == "hold" and s.next_game is not None:
                blocked[s.index] = s.next_game.date
    return plan_week(book, inp["slot_names"], pool, inp["games"], inp["positions"], frozen, options, inp["pinned"], blocked)


def solve_team(snap, rid, book, draws, vs: np.ndarray | None = None, inp=None, passes: int = 2) -> TeamWeek:
    """Pick lock/ride for every open decision. Without `vs`: maximise expected points (used for the
    opponent: best lineup + threshold policy). With `vs` (opponent totals): maximise P(win)."""
    inp = inp or team_inputs(snap, rid)
    opens = [s for s in inp["states"] if s.open_decision and s.pid in inp["roster"]]
    choices = {s.index: "hold" for s in opens}
    cache = {}

    def evaluate(ch):
        key = tuple(sorted(ch.items()))
        if key not in cache:
            pl = build_plan(book, inp, ch)
            tot = simulate_plan(pl, draws)
            cache[key] = (pl, tot, pl.ev if vs is None else p_win(tot, vs))
        return cache[key]

    for _ in range(passes):          # coordinate ascent: flip one decision at a time if it helps
        changed = False
        for s in opens:
            for c in (["lock"] if s.lockable else []) + ["hold", "ride"]:
                if c == choices[s.index]:
                    continue
                alt = {**choices, s.index: c}
                if evaluate(alt)[2] > evaluate(choices)[2] + 1e-12:
                    choices, changed = alt, True
        if not changed:
            break
    plan, tot, _ = evaluate(choices)
    decs = []
    for s in opens:
        hd = evaluate({**choices, s.index: "hold"})
        bn = evaluate({**choices, s.index: "ride"})
        lk = evaluate({**choices, s.index: "lock"}) if s.lockable else None
        best_ride = hd if hd[0].ev >= bn[0].ev else bn
        d = Decision(s.index, s.slot, s.pid, s.x, choices[s.index], lk[0].ev if lk else float("nan"), best_ride[0].ev,
                     (s.x + (best_ride[0].ev - lk[0].ev)) if lk else float("nan"), next_game=s.next_game, lockable=s.lockable)
        if vs is not None:
            d.pwin_lock = p_win(lk[1], vs) if lk else None
            d.pwin_hold, d.pwin_bench = p_win(hd[1], vs), p_win(bn[1], vs)
            d.pwin_ride = max(d.pwin_hold, d.pwin_bench)
        decs.append(d)
    return TeamWeek(rid, inp["states"], plan, tot, decs, inp)
