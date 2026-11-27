"""Trade search: 1-for-1 and 2-for-1 (both directions) with every team.

Stage 1 (screen, cheap): a team-strength proxy
    proxy = best legal 9-man lineup (Hungarian, position eligibility) of each player's average
            weekly *lock-in value* + depth_weight * (sum of the next 4 players)
  where a player's weekly lock-in value is the optimal-stopping value of his own games that week
  (a 4-game week is worth more than a 2-game week), averaged over the remaining regular season.
Stage 2: shortlisted trades get a season sim (fewer sims), finalists a full one. Both teams'
  playoff-odds changes are reported. Trades take effect next week (review period), so the current
  week is unchanged. Assumption: a team receiving 2-for-1 simply benches/drops its worst player.
"""
from __future__ import annotations

import itertools

import numpy as np
from scipy.optimize import linear_sum_assignment

from optimize.lock import eligible, single_slot_value

_pwv_cache = {}


def player_week_value(book_ros, snap, pid, weeks=None) -> float:
    key = (id(book_ros), pid)
    if key in _pwv_cache:
        return _pwv_cache[key]
    weeks = weeks or range(snap.week + 1, 19)
    team = snap.players.get(pid, {}).get("team")
    vals = []
    for w in weeks:
        gs = snap.week_games(team, w) if team else []
        vals.append(single_slot_value(book_ros, pid, gs)[0] if gs else 0.0)
    v = float(np.mean(vals)) if vals else 0.0
    _pwv_cache[key] = v
    return v


def proxy(pids, values, positions, slot_names, depth_weight):
    pids = list(pids)
    if not pids:
        return 0.0
    M = np.full((len(pids), len(slot_names)), -1e6)
    for i, p in enumerate(pids):
        for s, sn in enumerate(slot_names):
            if eligible(sn, positions.get(p)):
                M[i, s] = values.get(p, 0.0)
    r, c = linear_sum_assignment(M, maximize=True)
    used = {pids[i] for i, j in zip(r, c) if M[i, j] > -1e5}
    start = sum(M[i, j] for i, j in zip(r, c) if M[i, j] > -1e5)
    bench = sorted((values.get(p, 0.0) for p in pids if p not in used), reverse=True)[:4]
    return start + depth_weight * sum(bench)


def search(snap, cfg, book_ros, season_screen, base_screen, season_full, base_full, me_rid):
    tc = cfg["trades"]
    slot_names = [s for s in snap.league["roster_positions"] if s != "BN"]
    allp = [str(p) for r in snap.rosters for p in r["players"]]
    values = {p: player_week_value(book_ros, snap, p) for p in allp}
    positions = {p: snap.players.get(p, {}).get("fantasy_positions") for p in allp}
    mine = [str(p) for p in snap.roster(me_rid)["players"]]
    base_me = proxy(mine, values, positions, slot_names, tc["depth_weight"])
    cands = []
    for r in snap.rosters:
        rid = r["roster_id"]
        if rid == me_rid:
            continue
        theirs = [str(p) for p in r["players"]]
        base_th = proxy(theirs, values, positions, slot_names, tc["depth_weight"])
        shapes = [(1, 1), (2, 1), (1, 2)]
        for give_n, get_n in shapes:
            for give in itertools.combinations(mine, give_n):
                for get in itertools.combinations(theirs, get_n):
                    new_me = [p for p in mine if p not in give] + list(get)
                    new_th = [p for p in theirs if p not in get] + list(give)
                    dme = proxy(new_me, values, positions, slot_names, tc["depth_weight"]) - base_me
                    if dme <= 0:
                        continue
                    dth = proxy(new_th, values, positions, slot_names, tc["depth_weight"]) - base_th
                    if dth < -tc.get("screen_their_loss_pts", 6.0):
                        continue
                    cands.append((dme + min(dth, 0) * 0.5, rid, give, get, new_me, new_th))
    cands.sort(key=lambda x: -x[0])
    short = cands[: tc["shortlist"]]
    screened = []
    for _, rid, give, get, new_me, new_th in short:
        po, _ = season_screen.run(rosters={me_rid: new_me, rid: new_th})
        screened.append((po[me_rid] - base_screen[me_rid], rid, give, get, new_me, new_th))
    screened.sort(key=lambda x: -x[0])
    results = []
    for _, rid, give, get, new_me, new_th in screened[: tc["finalists"]]:
        po, _ = season_full.run(rosters={me_rid: new_me, rid: new_th})
        dme, dth = po[me_rid] - base_full[me_rid], po[rid] - base_full[rid]
        if dth * 100 < -tc["max_their_loss_pp"]:
            continue
        results.append(dict(rid=rid, give=give, get=get, dme=dme, dthem=dth,
                            flag=dme * 100 >= tc["threshold_pp"]))
    results.sort(key=lambda d: -d["dme"])
    return results, len(cands)
