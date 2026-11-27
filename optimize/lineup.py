"""Joint lineup + lock planner for the rest of a week.

Each starting slot ends the week with ONE game: a locked one, or the final game its occupant played
while in the slot. A slot can be "streamed": player A on Friday (lock if great), else bench A and
start B on Sunday. So each slot faces a chronological *sequence* of games, possibly from different
players, and its value is the optimal-stopping value of that sequence (see lock.py).

We build the sequences by backward induction over days. Going from the last day to today, for each
day we solve an assignment problem (Hungarian algorithm): which of that day's playing players goes
into which open slot, maximising the total increase in slot values
    gain(p, s) = f_s(E[max(X_p, C_s)]) - f_s(C_s)
where C_s is slot s's continuation value from later days (None if nothing later, then E[X_p]), and
f_s(v) = max(x_s, v) accounts for a value x_s the slot could keep right now. Position eligibility
and "one slot per player per day" are enforced by the assignment. Locked slots are frozen.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import linear_sum_assignment

from data.news import game_model
from model.distributions import ModelBook, expected_max
from optimize.lock import eligible

NEG = -1e9


@dataclass
class Step:
    pid: str
    game: object
    threshold: float | None    # lock this game if score >= threshold (None = last game, forced)


@dataclass
class SlotPlan:
    index: int
    slot: str
    frozen: float | None = None
    option: float | None = None       # value the slot can keep now (current occupant's counting game)
    option_pid: str | None = None
    keep: bool = False                # True: keep the option, ignore the sequence
    seq: list = field(default_factory=list)
    value: float = 0.0


@dataclass
class Plan:
    slots: list
    ev: float
    day_assign: dict                  # date -> {slot_index: pid}


def plan_week(book: ModelBook, slot_names: list, pool: list, games_by_pid: dict, positions: dict,
              frozen: dict | None = None, options: dict | None = None, pinned: dict | None = None) -> Plan:
    """pool: pids available to start. games_by_pid: pid -> remaining Games (chronological).
    frozen: slot -> locked value. options: slot -> (pid, x). pinned: (pid, date) -> slot (game in progress)."""
    frozen, options, pinned = frozen or {}, options or {}, pinned or {}
    S = len(slot_names)
    C = [None] * S
    seqs = [[] for _ in range(S)]
    opt = [options[s][1] if s in options else None for s in range(S)]

    def f(s, v):
        base = opt[s] if opt[s] is not None else 0.0
        return base if v is None else max(base, v)

    days = sorted({g.date for pid in pool for g in games_by_pid.get(pid, [])}, reverse=True)
    day_assign = {}
    slot_of = {}   # a player streams through ONE slot all week (keeps lock bookkeeping consistent)
    for d in days:
        P = [(pid, g) for pid in pool for g in games_by_pid.get(pid, []) if g.date == d]
        if not P:
            continue
        G = np.full((len(P), S), NEG)
        newv = np.zeros((len(P), S))
        for i, (pid, g) in enumerate(P):
            m, fac = game_model(book, pid, g)
            pin = pinned.get((pid, d))
            cvals = np.array([C[s] if C[s] is not None else np.nan for s in range(S)])
            em = expected_max(m, np.nan_to_num(cvals, nan=0.0), fac)
            ex = m.expected(fac)
            for s in range(S):
                if s in frozen or not eligible(slot_names[s], positions.get(pid)):
                    continue
                if pin is not None and pin != s:
                    continue
                if pid in slot_of and slot_of[pid] != s:
                    continue
                nv = ex if C[s] is None else float(em[s])
                newv[i, s] = nv
                G[i, s] = f(s, nv) - f(s, C[s]) + (1e6 if pin == s else 0.0)
        rows, cols = linear_sum_assignment(G, maximize=True)
        day_assign[d] = {}
        for i, s in zip(rows, cols):
            if G[i, s] <= 1e-9:
                continue
            pid, g = P[i]
            seqs[s].insert(0, Step(pid, g, C[s]))
            C[s] = newv[i, s]
            day_assign[d][int(s)] = pid
            slot_of[pid] = int(s)
    out, ev = [], 0.0
    for s in range(S):
        sp = SlotPlan(s, slot_names[s])
        if s in frozen:
            sp.frozen = sp.value = frozen[s]
        else:
            if s in options:
                sp.option_pid, sp.option = options[s]
            sp.seq = seqs[s]
            seq_v = C[s]
            sp.keep = sp.option is not None and (seq_v is None or sp.option >= seq_v)
            sp.value = f(s, seq_v)
            if sp.keep:
                sp.seq = []
                for da in day_assign.values():
                    da.pop(s, None)
        ev += sp.value
        out.append(sp)
    return Plan(out, ev, day_assign)
