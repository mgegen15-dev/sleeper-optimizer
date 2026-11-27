"""Lock-in eligibility: a game's score can be locked only if the player was in a starting slot when that
game began AND has not been moved to the bench since (Sleeper: "Player was not in starting lineup OR was
moved to bench since game began"). The API exposes no field for this, so we:
  1. record the starters we observe on every run (state lineup_obs, per week);
  2. mark a started game ineligible if the player was seen OFF the starting lineup at any run after it tipped
     (or the first observation after its tip already shows him benched);
  3. accept manual overrides: config `locks.ineligible` ({week: [names]}), and email replies
     "ineligible kyrie" (stored in state/locks.json under "ineligible").
Result: snap.ineligible = {pid: set(game_ids) or {"*"}} for the current week.
"""
from __future__ import annotations

from datetime import datetime

from data import locks as lk


def observe(snap, cfg, state):
    rid = cfg["roster_id"]
    m = snap.matchup(rid)
    if not m:
        return
    obs = state.setdefault("lineup_obs", {}).setdefault(str(snap.week), [])
    starters = [str(p) for p in m.get("starters") or [] if p not in (None, "0")]
    if not obs or obs[-1]["starters"] != starters:
        obs.append({"t": snap.now.isoformat(), "starters": starters})
    for w in list(state["lineup_obs"]):
        if int(w) < snap.week - 1:
            del state["lineup_obs"][w]


def from_observations(snap, cfg, state) -> dict:
    rid = cfg["roster_id"]
    obs = [(datetime.fromisoformat(o["t"]), set(o["starters"])) for o in state.get("lineup_obs", {}).get(str(snap.week), [])]
    out = {}
    for pid in map(str, snap.roster(rid)["players"]):
        team = snap.players.get(pid, {}).get("team")
        for g in snap.week_games(team) if team else []:
            if not g.started or g.tip is None:
                continue
            after = [s for t, s in obs if t >= g.tip]
            if any(pid not in s for s in after):
                out.setdefault(pid, set()).add(str(g.game_id))
    return out


def manual(snap, cfg, store) -> dict:
    rid = cfg["roster_id"]
    roster = [str(p) for p in snap.roster(rid)["players"]]
    nick, cutoff = cfg.get("locks", {}).get("nicknames", {}), cfg.get("locks", {}).get("fuzzy_cutoff", 0.8)
    out = {}
    for name in (cfg.get("locks", {}).get("ineligible") or {}).get(snap.week, []) or []:
        pid, status, _ = lk.match_name(str(name), roster, snap.players, nick, cutoff)
        if status == "ok":
            out.setdefault(pid, set()).add("*")
    for pid, gids in ((store.get("ineligible") or {}).get(str(snap.week)) or {}).items():
        out.setdefault(pid, set()).update(gids)
    return out


def apply(snap, cfg, state, store):
    """Combine observed + manual ineligibility onto the snapshot. Returns notes for Assumptions."""
    obs = from_observations(snap, cfg, state)
    man = manual(snap, cfg, store)
    merged = {}
    for d in (obs, man):
        for pid, g in d.items():
            merged.setdefault(pid, set()).update(g)
    snap.ineligible = merged
    return [f"{snap.pname(p)}: current game not lockable ({'manual' if p in man else 'seen benched after tip'})"
            for p in merged]
