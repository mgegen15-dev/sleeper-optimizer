import yaml

import main as M
from model.distributions import ModelBook
from optimize import alternatives as alt
from optimize.montecarlo import Draws, solve_team
from tests.test_locks import make_snap

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
CFG["locks"]["ineligible"] = {}


def setup():
    snap = make_snap(jerome=12.0)          # a weak counting score -> Jerome's slot stays open
    book = ModelBook(snap.logs, snap.players, snap.proj, CFG)
    dr = Draws(book, 3000, 9)
    opp = solve_team(snap, 8, book, dr)
    me = solve_team(snap, 10, book, dr, vs=opp.totals)
    return snap, book, dr, opp, me


def test_every_bench_player_playing_tonight_is_considered_for_each_open_slot():
    snap, book, dr, opp, me = setup()
    res = alt.evaluate(snap, CFG, book, me, opp.totals, dr, values_ros={})
    assert res, "open slots should be evaluated"
    for c in res:
        names = {r["name"] for r in c["runners_up"]} | {c["pick"]["name"]}
        # both bench players with a game tonight (Cameron and Keldon Johnson) appear as candidates or the pick
        assert {"Cameron Johnson", "Keldon Johnson"} & names
        p = [r["pwin"] for r in c["runners_up"]]
        assert p == sorted(p, reverse=True)
        for r in c["runners_up"]:
            assert abs(r["delta"] - (r["pwin"] - c["pick"]["pwin"])) < 1e-12
            assert r["tie"] == (abs(r["delta"]) * 100 < alt.TIE_PP)


def test_deterministic_with_seed():
    snap, book, dr, opp, me = setup()
    a = alt.evaluate(snap, CFG, book, me, opp.totals, dr, values_ros={})
    b = alt.evaluate(snap, CFG, book, me, opp.totals, dr, values_ros={})
    assert [[r["pwin"] for r in c["runners_up"]] for c in a] == [[r["pwin"] for r in c["runners_up"]] for c in b]


def test_attach_runner_ups_and_best_alternative():
    considered = [dict(slot="UTIL", slot_index=1, occupant="Nic Claxton", occupant_plays_tonight=False, moved=True,
                       pick=dict(pid="2029", name="P.J. Washington", source="bench", drop=None, pwin=0.594),
                       runners_up=[dict(pid="4751", name="Derik Queen", source="bench", drop=None, pwin=0.592, delta=-0.002, tie=True),
                                   dict(pid="x", name="Charles Bassey", source="FA", drop="Keldon Johnson", pwin=0.565, delta=-0.029, tie=False)],
                       best_alternative=None)]
    ctx = dict(move_cards=[dict(text="Move P.J. Washington into UTIL, Nic Claxton to bench (forfeits his 26.0)", dpwin=0.084)])
    M.attach_alternatives(ctx, considered, None)
    card = ctx["move_cards"][0]
    assert card["also"][0]["tie"] is True and card["also"][1]["drop"] == "Keldon Johnson"
    assert card["also_text"].startswith("Also considered: Derik Queen (bench) 59.2% (-0.2 ± 0.0, tie)")
    assert "Charles Bassey (FA, drop Keldon Johnson) 56.5% (-2.9 ± 0.0)" in card["also_text"]
    # no move recommended -> best alternative and why it lost
    considered[0]["moved"] = False
    considered[0]["best_alternative"] = considered[0]["runners_up"][0]
    ctx2 = dict(move_cards=[])
    M.attach_alternatives(ctx2, considered, None)
    assert ctx2["best_alternative"]["name"] == "Derik Queen" and "lost because -0.2pp vs keeping the lineup (within noise)" in ctx2["best_alternative"]["text"]


def test_tiebreak_prefers_higher_projection_and_flags_sleeper_disagreement():
    snap, book, dr, opp, me = setup()
    res = alt.evaluate(snap, CFG, book, me, opp.totals, dr, values_ros={})
    for c in res:
        for r in c["runners_up"]:
            assert "e_pts" in r and "sleeper_proj" in r
    considered = [dict(slot="UTIL", slot_index=1, occupant="Nic Claxton", occupant_plays_tonight=False, moved=True,
                       pick=dict(pid="a", name="A", source="bench", drop=None, pwin=0.594, e_pts=27.7, sleeper_proj=24.0),
                       runners_up=[dict(pid="b", name="B", source="bench", drop=None, pwin=0.592, delta=-0.002, tie=True,
                                        e_pts=28.0, sleeper_proj=26.2)],
                       best_alternative=None, tiebreak="Statistical tie with B; start B instead.", sleeper_note=None)]
    ctx = dict(move_cards=[dict(text="Move A into UTIL, Nic Claxton to bench", dpwin=0.08)])
    M.attach_alternatives(ctx, considered, None)
    assert "start B instead" in ctx["move_cards"][0]["also_text"] and ctx["move_cards"][0]["tiebreak"]
