from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import yaml

from model.distributions import ModelBook
from optimize import selfcheck as sc
from optimize.montecarlo import Draws, solve_team
from tests.test_locks import make_snap

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
CFG["locks"]["ineligible"] = {}


def solved(jerome=50.0):
    snap = make_snap(jerome=jerome)
    book = ModelBook(snap.logs, snap.players, snap.proj, CFG)
    dr = Draws(book, 4000, 3)
    opp = solve_team(snap, 8, book, dr)
    me = solve_team(snap, 10, book, dr, vs=opp.totals)
    return snap, book, dr, opp, me


def test_overturns_planted_bad_call():
    snap, book, dr, opp, me = solved(jerome=60.0)          # a big counting score: lock is clearly right
    d = next(x for x in me.decisions if x.pid == "2090")
    assert d.choice == "lock"
    d.choice = "hold"                                         # plant a bad call
    res = sc.check_decisions(snap, book, me, opp.totals, dr)
    r = next(x for x in res if x["pid"] == "2090")
    assert r["overturned"] and r["from_call"] == "hold" and r["call"] == "lock"
    assert next(x for x in me.decisions if x.pid == "2090").choice == "lock"


def test_keeps_good_call():
    snap, book, dr, opp, me = solved(jerome=60.0)
    res = sc.check_decisions(snap, book, me, opp.totals, dr)
    r = next(x for x in res if x["pid"] == "2090")
    assert not r["overturned"] and r["call"] == "lock" and r["diff"] > 0


def test_paired_and_noise_rule():
    opp = np.zeros(1000)
    a = np.ones(1000)
    b = np.r_[np.ones(990), -np.ones(10)]
    diff, se = sc.paired(a, b, opp)
    assert abs(diff - 0.01) < 1e-12 and se > 0
    assert sc.beats(0.02, 0.001) and not sc.beats(0.004, 0.0001) and not sc.beats(0.02, 0.015)


def test_proof_line_format():
    CT = ZoneInfo("America/Chicago")
    class G:
        def __init__(self, d, h, m):
            self.date, self.tip = d, datetime(2026, 11, int(d[-2:]), h, m, tzinfo=CT)
    r = dict(games=[(G("2026-11-27", 18, 30), 0.97), (G("2026-11-29", 17, 0), 0.97)], bar=31.2, x=33.0, call="lock",
             runner="hold", pwin=0.596, runner_pwin=0.545, diff=0.051, se=0.001, tie=False, sleeper="agrees", overturned=False)
    assert sc.proof_line(r, "2026-11-27") == ("Games left: 6:30 (97%), Sun 5:00 (97%) · bar 31.2 vs 33.0 · "
                                               "lock 59.6% vs ride 54.5% (+5.1 ± 0.1) · Sleeper agrees")


def test_projection_crosscheck_flags_big_gaps():
    snap, book, dr, opp, me = solved()
    assert sc.proj_crosscheck(snap, book, ["1380"]) == []                 # no Sleeper projection -> nothing to compare
    snap.proj["1380"] = {"2026-11-27": 200.0}                             # absurd projection -> flagged
    book._news_cache = {}
    f = sc.proj_crosscheck(snap, book, ["1380"])
    assert f and f[0]["date"] == "2026-11-27" and f[0]["sleeper"] == 200.0
