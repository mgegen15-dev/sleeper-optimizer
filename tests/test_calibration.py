from optimize import calibration as cal
from tests.test_locks import make_snap

CFG = {"roster_id": 10}


def test_log_settle_and_score_from_official_record():
    snap = make_snap()
    snap.rosters[0]["settings"] = {"wins": 4, "losses": 1, "ties": 0}
    state = {}
    cal.log_run(snap, CFG, state, 0.7, 0.6)
    cal.log_run(snap, CFG, state, 0.8, 0.6)
    assert len(state["calib"]["weeks"]["6"]["points"]) == 2
    # week 7: official record shows the week-6 win
    snap.week = 7
    snap.rosters[0]["settings"] = {"wins": 5, "losses": 1, "ties": 0}
    cal.settle(snap, CFG, state)
    assert state["calib"]["weeks"]["6"]["outcome"] == 1.0
    s = cal.scores(state)
    assert s["weeks"] == 1 and s["n"] == 2
    assert abs(s["ours"]["brier"] - ((0.3 ** 2 + 0.2 ** 2) / 2)) < 1e-9 and abs(s["sleeper"]["brier"] - 0.16) < 1e-9
    assert cal.footer_line(state) is None                       # needs 2+ weeks


def test_footer_after_two_weeks_and_loss_tie():
    state = {"calib": {"weeks": {"5": {"points": [{"ours": 0.6, "sleeper": 0.7}], "outcome": 0.0},
                                 "6": {"points": [{"ours": 0.5, "sleeper": 0.9}], "outcome": 0.5}}}}
    line = cal.footer_line(state)
    assert line and "2 weeks, 2 checkpoints" in line and "50/50 guessing = 0.250" in line


def test_ambiguous_record_change_is_not_scored():
    snap = make_snap()
    snap.rosters[0]["settings"] = {"wins": 4, "losses": 1, "ties": 0}
    state = {}
    cal.log_run(snap, CFG, state, 0.7, 0.6)
    snap.week = 8                                                # two games passed before we checked
    snap.rosters[0]["settings"] = {"wins": 5, "losses": 2, "ties": 0}
    cal.settle(snap, CFG, state)
    assert state["calib"]["weeks"]["6"]["outcome"] is None and cal.scores(state)["weeks"] == 0
