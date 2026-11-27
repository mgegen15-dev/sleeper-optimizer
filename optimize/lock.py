"""Lock-in rules, lock inference from matchup data, and the single-slot optimal-stopping solution.

Optimal stopping (backward induction) for one slot facing games X_1..X_n in time order:
    V_n = E[X_n]                    (the last game is forced: you get whatever it is, DNP = 0)
    V_k = E[max(X_k, V_{k+1})]      (after seeing X_k, lock it if X_k >= V_{k+1}, else ride on)
So the lock threshold for game k is V_{k+1}: the value of continuing. It's an American option on
the player's next games: you exercise (lock) when today's payoff beats the continuation value.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from model.distributions import ModelBook, expected_max

ELIG = {"PG": {"PG"}, "SG": {"SG"}, "SF": {"SF"}, "PF": {"PF"}, "C": {"C"},
        "G": {"PG", "SG"}, "F": {"SF", "PF"}, "UTIL": None}


def eligible(slot: str, fantasy_positions) -> bool:
    allowed = ELIG.get(slot)
    return allowed is None or bool(allowed & set(fantasy_positions or []))


def backward_induction(dists):
    """dists: list of (PlayerModel, opp_factor) in time order.
    Returns (V, thresholds): V[k] = value before game k; thresholds[k] = V[k+1] (None for last)."""
    n = len(dists)
    if n == 0:
        return [], []
    V = [0.0] * n
    m, f = dists[-1]
    V[-1] = m.expected(f)
    for k in range(n - 2, -1, -1):
        m, f = dists[k]
        V[k] = float(expected_max(m, V[k + 1], f)[0])
    return V, [V[k + 1] for k in range(n - 1)] + [None]


@dataclass
class SlotState:
    index: int
    slot: str
    pid: str | None
    x: float | None = None        # value currently counting (option to keep), None if nothing counts
    locked: bool = False
    open_decision: bool = False   # has a finished game counting AND a future game: lock-or-ride call
    done: bool = False            # occupant has no games left this week
    games_this_week: list = field(default_factory=list)
    next_game: object = None
    in_progress: object = None
    pending: bool = False         # a later game is final but its stat lines aren't posted yet
    lock_source: str = ""         # "inferred" | "reported" | "sleeper"
    lockable: bool = True         # False: the counting game is ineligible for lock-in (benched since it began)
    conflict: str = ""            # reported lock contradicted by API evidence
    note: str = ""


def infer_slots(snap, rid: int, week: int | None = None, reported: dict | None = None) -> list[SlotState]:
    """Infer each starting slot's lock state from matchup data (the API has no lock field).
    Rule: starters_points equals an earlier game's FP although a later game of his team is final
    -> that earlier game was locked. Equal to the latest game with a future game pending -> open call."""
    W = week or snap.week
    m = snap.matchup(rid, W)
    slots = [s for s in snap.league["roster_positions"] if s != "BN"]
    out = []
    for i, slot in enumerate(slots):
        pid = (m["starters"][i] if m and i < len(m["starters"]) else None)
        pid = None if pid in (None, "0", 0) else str(pid)
        st = SlotState(i, slot, pid)
        if not pid:
            out.append(st)
            continue
        sp = float((m.get("starters_points") or [0] * len(slots))[i] or 0)
        team = snap.players.get(pid, {}).get("team")
        tg = snap.week_games(team, W) if team else []
        logs = [g for g in snap.logs.get(pid, []) if g.week == W]
        st.games_this_week = logs
        future = [g for g in tg if not g.started]
        st.in_progress = next((g for g in tg if g.started and not g.final), None)
        st.next_game = future[0] if future else None
        st.done = not future and st.in_progress is None
        match = next((g for g in reversed(logs) if abs(g.fp - sp) < 0.01), None) if logs else None
        if match is not None and sp > 0:
            later = [g for g in tg if g.final and g.date > match.date]
            posted = [g for g in later if (team, g.date) in snap.stat_dates]
            if later and len(posted) < len(later):
                st.pending, st.x, st.note = True, sp, "stats pending"
            elif later:
                st.locked, st.x, st.note, st.lock_source = True, sp, f"locked {match.date}", "inferred"
            else:
                st.x = sp
                st.open_decision = not st.done and st.in_progress is None
                st.note = f"counting {match.date}"
        elif st.in_progress is not None:
            st.note = "game in progress"
        api = (getattr(snap, "api_locks", None) or {}).get(rid) if getattr(snap, "api_locks", None) is not None and W == snap.week else None
        if api is not None:
            apply_api(snap, st, api.get(pid), tg, logs, sp)
            if reported and pid in reported and not st.locked:
                st.conflict = f"{snap.pname(pid)}: you reported a lock but Sleeper shows none; using Sleeper."
        elif reported and pid in reported:
            apply_reported(snap, st, reported[pid], sp)
        inel = (getattr(snap, "ineligible", None) or {}).get(pid)
        if inel and not st.locked and st.x is not None and match is not None:
            gid = next((str(g.game_id) for g in tg if g.date == match.date), None)
            if "*" in inel or gid in inel:
                st.lockable = False
                st.note = f"not lockable ({match.date} game: benched since it began)"
        out.append(st)
    if reported:
        starters = {s.pid for s in out}
        for pid in reported:
            if pid not in starters:
                out[0].conflict += f" {snap.pname(pid)} reported locked but is not in a starting slot (ignored)."
    return out


def apply_api(snap, st: SlotState, game_id, tg, logs, api_points: float):
    """Sleeper's own lock map is authoritative: present -> locked to that game; absent -> unlocked."""
    if game_id:
        g = next((x for x in tg if str(x.game_id) == str(game_id)), None)
        lg = next((x for x in logs if g is not None and x.date == g.date), None)
        st.locked, st.open_decision, st.lock_source = True, False, "sleeper"
        st.x = lg.fp if lg else api_points
        st.note = f"locked {g.date if g else game_id} (Sleeper)"
        return
    if st.locked:            # inference said locked, Sleeper says not: the counting game was the last one he
        st.locked = False    # played while in the slot; it stays an open call while he has games left
        st.lock_source = ""
        st.open_decision = st.x is not None and not st.done and st.in_progress is None
        st.note = "not locked (Sleeper)"


def apply_reported(snap, st: SlotState, rec: dict, api_points: float):
    """Merge a lock you reported with what the API shows. API evidence wins on conflict."""
    score = float(rec.get("score") or 0)
    name = snap.pname(st.pid)
    if st.locked:
        if abs((st.x or 0) - score) > 0.01:
            st.conflict = f"{name}: you reported a lock at {score:.1f} but the API shows {st.x:.1f} locked; using the API."
        else:
            st.lock_source = "reported"
        return
    if st.x is not None and abs(st.x - score) < 0.01:
        st.locked, st.open_decision, st.lock_source = True, False, "reported"
        st.note = f"locked {rec.get('date', '')} (reported)"
    elif st.x is None or st.pending or api_points == 0:
        st.locked, st.open_decision, st.x, st.lock_source = True, False, score, "reported"
        st.note = f"locked {rec.get('date', '')} (reported; API not showing it yet)"
    else:
        st.conflict = (f"{name}: you reported a lock at {score:.1f} ({rec.get('date', '')}) but the API is counting "
                       f"{st.x:.1f} from a later game; using the API.")


def single_slot_value(book: ModelBook, pid: str, games) -> tuple[float, list]:
    """Value of one player alone in a slot over his remaining games (used for proxies/backtest)."""
    from data.news import game_model
    d = [game_model(book, pid, g) for g in games]
    V, thr = backward_induction(d)
    return (V[0] if V else 0.0), thr
