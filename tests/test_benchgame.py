from datetime import timedelta

import yaml

from data import eligibility as el
from optimize.lock import infer_slots, played_as_starter
from tests.test_locks import make_snap, t

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
CFG["locks"]["ineligible"] = {}


def test_bench_game_does_not_count_or_lock():
    snap = make_snap()
    # Jerome's 11/25 game: our first lineup check (after that game) shows him on the BENCH
    state = {"lineup_obs": {"6": [{"t": t("2026-11-26", 12).isoformat(), "starters": ["1380"]},
                                  {"t": t("2026-11-27", 12).isoformat(), "starters": ["1380", "2090"]}]}}
    el.apply(snap, CFG, state, {"weeks": {}})
    s = {x.pid: x for x in infer_slots(snap, 10)}
    assert s["2090"].x is None and not s["2090"].open_decision and not s["2090"].lockable
    assert "bench" in s["2090"].note
    assert s["1380"].x == 65.0 and s["1380"].open_decision               # starting all along: still counts


def test_played_as_starter_rules():
    snap = make_snap()
    g = next(x for x in snap.team_games["MEM"] if x.date == "2026-11-25")
    snap.lineup_obs = [(g.tip - timedelta(hours=1), {"2090"}), (g.tip + timedelta(hours=3), {"2090"})]
    assert played_as_starter(snap, "2090", g) is True
    snap.lineup_obs = [(g.tip - timedelta(hours=1), {"2090"}), (g.tip + timedelta(hours=1), set())]
    assert played_as_starter(snap, "2090", g) is False                    # moved to bench after tip
    snap.lineup_obs = [(g.tip + timedelta(hours=5), {"2090"})]
    assert played_as_starter(snap, "2090", g) is None                     # unknown -> assumed, flagged
    snap.lineup_obs = []
    assert played_as_starter(snap, "2090", g) is None


def test_overturned_swap_becomes_the_rendered_action():
    import main as M
    ctx = dict(move_cards=[dict(text="Move B into UTIL, Y to bench", dpwin=0.03)])
    considered = [dict(slot="UTIL", slot_index=1, occupant="Y", occupant_plays_tonight=True, moved=True,
                       pick=dict(pid="b", name="B", source="bench", drop=None, pwin=0.6),
                       runners_up=[dict(pid="c", name="C", source="bench", drop=None, pwin=0.55, delta=-0.05, tie=False, se=0.002,
                                        reason="lower E tonight")],
                       all_options=[], best_alternative=None, tiebreak=None, sleeper_note=None)]
    M.attach_alternatives(ctx, considered, None)
    # the card describes the pick that was finally planned (B), and its runner-ups are relative to B
    assert ctx["move_cards"][0]["text"].startswith("Move B into") and "C (bench) 55.0% (-5.0" in ctx["move_cards"][0]["also_text"]


def test_matchup_consensus_needs_strict_majority(tmp_path):
    from data.sleeper import SleeperClient
    c = SleeperClient(tmp_path)
    versions = iter([[{"roster_id": 1, "points": a}] for a in (1, 2, 3, 1, 4)])
    c.get = lambda url, fname, max_age_h=0: next(versions)
    c.matchups = lambda lid, w: [{"roster_id": 1, "points": -1}]
    (tmp_path / "matchups").mkdir()
    (tmp_path / "matchups/week_6_stable.json").write_text('[{"roster_id": 1, "points": 9}]')
    assert c.matchups_consensus("L", 6)[0]["points"] == 9                 # 2/5 is not a majority -> last stable
    assert "no majority" in c.failures[-1]
    versions2 = iter([[{"roster_id": 1, "points": a}] for a in (5, 5, 5, 1, 2)])
    c.get = lambda url, fname, max_age_h=0: next(versions2)
    assert c.matchups_consensus("L", 6)[0]["points"] == 5                 # 3/5 majority -> used and saved as stable
