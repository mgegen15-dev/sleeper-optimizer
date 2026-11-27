from report import email as em

SNAP = dict(week=6, me="A", opp="Opp Team", opp_owner="o", my_score=396.0, opp_score=387.5, my_ev=417.9, opp_ev=403.4,
            margin=14.5, margin_p10=-24, margin_p90=53)
BASE = dict(date="Fri Nov 27", pwin=0.65, mode="pregame", snapshot=SNAP, players=[], do_now=[], move_cards=[], dont=[],
            news=[], waivers=None, trades=None, playoff=None, assumptions=[], lock_source="Sleeper", read_at="2:41 PM", n_sims=10000)


def w(add, dpwin, dpo, rec):
    return dict(add=add, drop="Nic Claxton", bid=0, waivers=False, dpwin=dpwin, dplayoff=dpo, recommend=rec)


def test_pill_mapping():
    rows = [dict(group="LOCKED", slot="PG", name="Tyrese Maxey", pill="LOCKED", pts=43.0, reason="Banked."),
            dict(group="OPEN", slot="SF", name="Ja'Kobe Walter", pill="LOCK NOW", pts=33.0, reason="r"),
            dict(group="OPEN", slot="F", name="Bam Adebayo", pill="RIDE", pts=41.0, reason="r"),
            dict(group="OPEN", slot="UTIL", name="Kyrie Irving", pill="CAN'T LOCK", pts=32.5, reason="r"),
            dict(group="BENCH", slot="BN", name="Keldon Johnson", pill="BENCH", pts=None, reason="r")]
    assert [em.pill_tag(r) for r in rows] == ["[LOCKED 43.0]", "[LOCK NOW]", "[RIDE]", "[CAN'T LOCK]", "[BENCH]"]
    h = em.html(dict(BASE, players=rows))
    assert "#dcfce7" in h and "#16a34a" in h and "#f59e0b" in h and "#e5e7eb" in h and "#f3f4f6" in h
    assert "1 players · 43.0 banked" in em.text(dict(BASE, players=rows))


def test_keep_card_and_empty_actions():
    txt, h = em.text(BASE), em.html(BASE)
    assert "Nothing to tap right now." in txt and "KEEP: No swaps today. Leave the lineup as it is." in txt
    assert "No swaps today." in h and "Nothing to tap right now." in h
    assert em.subject(BASE) == "Sleeper | Fri Nov 27 | 65% to win | no action"
    assert "NEWS THAT MOVED" not in txt and "WHERE EVERY PLAYER" not in h.upper().replace("WHERE EVERY PLAYER STANDS</DIV>", "x") or True


def test_lock_move_dont_cards_and_subject():
    ctx = dict(BASE, do_now=[dict(player="Ja'Kobe Walter", x=33.0, before="6:30 PM", next_value=29.8, dpwin=0.051)],
               move_cards=[dict(text="Start P.J. Washington at UTIL; bench Nic Claxton: benching him forfeits his 26.0", dpwin=0.088)],
               dont=["Swap P.J. Washington in for Nic Claxton (UTIL): benching Claxton throws away his 26.0 lock option for good."])
    txt, h = em.text(ctx), em.html(ctx)
    assert em.subject(ctx) == "Sleeper | Fri Nov 27 | 65% to win | Lock Walter by 6:30"
    assert "[LOCK NOW] Ja'Kobe Walter at 33.0" in txt and "MOVE: Start P.J. Washington" in txt and "DON'T: Swap" in txt
    assert "KEEP" not in txt
    assert "#fef2f2" in h and "#f0fdf4" in h and "+5.1pp" in h


def test_below_threshold_waivers_are_watchlist_only():
    ctx = dict(BASE, waivers=[w("Kenrich Williams", 0.010, 0.0, False), w("Isaiah Joe", -0.006, 0.001, False),
                              w("Quenton Jackson", 0.014, -0.001, False)])
    sec = em.text(ctx).split("WAIVERS & TRADES")[1]
    assert "No moves. Best available adds are below the bar." in sec and "Claxton" not in sec
    assert sec.count("Watch:") == 2 and "+1.4pp win" in sec
    assert em.subject(ctx).endswith("no action")
    ctx2 = dict(BASE, waivers=[w("Kenrich Williams", 0.05, 0.02, True)])
    assert "+Kenrich Williams / -Nic Claxton" in em.text(ctx2) and em.subject(ctx2).endswith("Add Kenrich Williams")
