import numpy as np
import pytest

from model.distributions import PlayerModel, expected_max
from model.scoring import score_line
from optimize.lock import backward_induction, eligible
from optimize.montecarlo import p_win

SC = {"ast": 1.5, "blk": 3.0, "bonus_ast_15p": 3.0, "bonus_pt_40p": 3.0, "bonus_pt_50p": 5.0, "bonus_reb_20p": 3.0,
      "dd": 2.0, "ff": 0.0, "ftm": 0.0, "pts": 1.0, "reb": 1.5, "stl": 3.0, "td": 3.0, "tf": -0.5, "to": -1.0, "tpm": 0.0}


# ---------------------------------------------------------------- scoring (real lines validated vs Sleeper)
def test_giannis_double_double_line():
    # 2026-11-16: 20 pts, 13 reb, 6 ast, 1 stl, 3 blk, 1 TO, dd=1 -> 20+19.5+9+3+9-1+2
    st = {"pts": 20.0, "reb": 13.0, "ast": 6.0, "stl": 1.0, "blk": 3.0, "to": 1.0, "dd": 1.0, "ftm": 4.0, "tpm": 0}
    assert score_line(st, SC) == pytest.approx(61.5)


def test_bonus_derived_when_missing():
    st = {"pts": 52.0, "reb": 21.0, "ast": 15.0}
    # 52 + 31.5 + 22.5 + dd2 + td3 + 40p3 + 50p5 + reb20 3 + ast15 3
    assert score_line(st, SC) == pytest.approx(52 + 31.5 + 22.5 + 2 + 3 + 3 + 5 + 3 + 3)


def test_triple_double_earns_dd_and_td():
    # real line (player 1739, 2026-10-23): 27/19/10, Sleeper flags dd=1, td=1 -> both bonuses (+5)
    assert score_line({"pts": 27.0, "reb": 19.0, "ast": 10.0, "dd": 1.0, "td": 1.0}, SC) == pytest.approx(27 + 28.5 + 15 + 5)
    # same line with td flag missing -> derived identically
    assert score_line({"pts": 27.0, "reb": 19.0, "ast": 10.0, "dd": 1.0, "td": None}, SC) == pytest.approx(75.5)


def test_technical_and_turnovers():
    assert score_line({"pts": 10, "to": 4, "tf": 1}, SC) == pytest.approx(10 - 4 - 0.5)


# ---------------------------------------------------------------- optimal stopping
def uniform_model(n=200001, p=1.0):
    R = np.linspace(0, 2, n)          # Uniform(0, 2*mu) with mu = 50  -> U(0, 100)
    return PlayerModel("u", 50.0, R, p, "F", 20)


def test_expected_max_uniform():
    m = uniform_model()
    # E[max(U, c)] for U~U(0,100): (c^2 + 100^2) / 200
    for c in (0, 25, 50, 80):
        assert expected_max(m, c)[0] == pytest.approx((c * c + 1e4) / 200, rel=1e-3)


def test_backward_induction_uniform_analytic():
    m = uniform_model()
    V, thr = backward_induction([(m, 1.0)] * 3)
    # V3 = 50; V2 = (50^2+100^2)/200 = 62.5; V1 = (62.5^2+100^2)/200 = 69.53
    assert V[2] == pytest.approx(50, rel=1e-3)
    assert V[1] == pytest.approx(62.5, rel=1e-3)
    assert V[0] == pytest.approx((62.5 ** 2 + 1e4) / 200, rel=1e-3)
    assert thr[0] == pytest.approx(62.5, rel=1e-3) and thr[1] == pytest.approx(50, rel=1e-3) and thr[2] is None


def test_dnp_probability_lowers_value():
    m = uniform_model(p=0.6)
    V, _ = backward_induction([(m, 1.0)])
    assert V[0] == pytest.approx(30, rel=1e-3)


def test_eligibility():
    assert eligible("G", ["PG"]) and eligible("F", ["PF", "C"]) and eligible("UTIL", ["C"])
    assert not eligible("C", ["PF"]) and not eligible("SG", ["PG"])


# ---------------------------------------------------------------- Monte Carlo
class _G:
    def __init__(self, date, opp=None):
        self.date, self.opp = date, opp


class _Book:
    def __init__(self, models):
        self.m = models

    def get(self, pid):
        return self.m[pid]

    def opp_factor(self, pid, opp):
        return 1.0


def _plan_for(pids, dates):
    from optimize.lineup import plan_week
    models = {p: PlayerModel(p, 40.0, np.array([0.5, 0.8, 1.0, 1.2, 1.5]), 0.97, "F", 20) for p in pids}
    book = _Book(models)
    games = {p: [_G(d) for d in dates] for p in pids}
    pos = {p: ["SF", "PF"] for p in pids}
    return book, plan_week(book, ["F", "F", "UTIL"], pids, games, pos)


def test_mc_deterministic_and_symmetric():
    from optimize.montecarlo import Draws, simulate_plan
    dates = ["2026-11-27", "2026-11-28", "2026-11-29"]
    bookA, planA = _plan_for(["a1", "a2", "a3", "a4"], dates)
    bookB, planB = _plan_for(["b1", "b2", "b3", "b4"], dates)
    t1 = simulate_plan(planA, Draws(bookA, 20000, 7))
    t2 = simulate_plan(planA, Draws(bookA, 20000, 7))
    assert np.array_equal(t1, t2)                       # seeded -> identical
    tb = simulate_plan(planB, Draws(bookB, 20000, 7))
    assert abs(p_win(t1, tb) - 0.5) < 0.02              # identical teams -> ~50%
    assert planA.ev == pytest.approx(planB.ev)
    assert t1.mean() == pytest.approx(planA.ev, rel=0.02)   # MC agrees with backward induction


def test_more_games_more_value():
    _, p1 = _plan_for(["a1", "a2", "a3"], ["2026-11-27"])
    _, p3 = _plan_for(["a1", "a2", "a3"], ["2026-11-27", "2026-11-28", "2026-11-29"])
    assert p3.ev > p1.ev
