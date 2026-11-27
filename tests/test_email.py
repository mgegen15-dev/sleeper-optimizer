from report import email as em

BASE = dict(date="Fri Nov 27", pwin=0.6, n_actions=0, lineup_date="2026-11-27", lineup=[], moves=[], locks=[], trades=None,
            assumptions=[], locked_list=[], snapshot=dict(week=6, me="A", opp="B", opp_owner="o", my_score=1, opp_score=1,
                                                         my_ev=1, opp_ev=1, margin=0, margin_p10=-1, margin_p90=1))


def w(add, dpwin, dpo, rec):
    return dict(add=add, drop="Nic Claxton", bid=0, waivers=False, dpwin=dpwin, dplayoff=dpo, recommend=rec)


def test_below_threshold_moves_are_not_actions():
    ctx = dict(BASE, waivers=[w("Kenrich Williams", 0.010, 0.0, False), w("Isaiah Joe", -0.006, 0.001, False),
                              w("Quenton Jackson", 0.014, -0.001, False)])
    txt, html = em.text(ctx), em.html(ctx)
    sec = txt.split("WAIVER ADDS")[1].split("ASSUMPTIONS")[0]
    assert "No moves. Best available adds are below the bar." in sec
    assert "Claxton" not in sec and "drop" not in sec.lower() and "Claxton" not in html
    assert sec.count("Watch:") == 2
    assert "+1.4pp win" in sec


def test_cleared_move_listed_with_drop():
    ctx = dict(BASE, waivers=[w("Kenrich Williams", 0.05, 0.02, True), w("Isaiah Joe", 0.01, 0.0, False)])
    sec = em.text(ctx).split("WAIVER ADDS")[1].split("ASSUMPTIONS")[0]
    assert "+Kenrich Williams / -Nic Claxton" in sec and "No moves" not in sec
