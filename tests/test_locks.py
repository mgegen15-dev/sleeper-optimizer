from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pytest
import yaml

from data import locks as lk
from data.loader import Game, GameLog, Snapshot
from model.distributions import ModelBook
from optimize.montecarlo import Draws, p_win, solve_team

TZ = ZoneInfo("America/Chicago")
CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))

PLAYERS = {
    "1380": dict(first_name="Giannis", last_name="Antetokounmpo", team="MIA", fantasy_positions=["C", "PF"], position="PF"),
    "1272": dict(first_name="Anthony", last_name="Davis", team="WAS", fantasy_positions=["C", "PF"], position="C"),
    "2091": dict(first_name="Cameron", last_name="Johnson", team="DEN", fantasy_positions=["SF", "PF"], position="SF"),
    "2065": dict(first_name="Keldon", last_name="Johnson", team="SAS", fantasy_positions=["SF", "PF"], position="SF"),
    "2090": dict(first_name="Ty", last_name="Jerome", team="MEM", fantasy_positions=["PG", "SG"], position="SG"),
    "9001": dict(first_name="Opp", last_name="One", team="BOS", fantasy_positions=["PF"], position="PF"),
    "9002": dict(first_name="Opp", last_name="Two", team="ATL", fantasy_positions=["PG"], position="PG"),
}
MINE = ["1380", "1272", "2091", "2065", "2090"]


def t(d, hh, mm=0):
    return datetime.fromisoformat(f"{d}T{hh:02d}:{mm:02d}").replace(tzinfo=TZ)


def games(team, spec):
    out = []
    for i, (d, hh, final) in enumerate(spec):
        out.append(Game(f"{team}{i}", d, 6, team, "XXX", True, tip=t(d, hh), started=final, final=final))
    return out


def make_snap(now=None, jerome=50.0):
    sp = (65.0, jerome)
    tg = {
        "MIA": games("MIA", [("2026-11-23", 18, True), ("2026-11-25", 18, True), ("2026-11-27", 18, False)]),
        "MEM": games("MEM", [("2026-11-25", 19, True), ("2026-11-27", 19, False)]),
        "WAS": games("WAS", [("2026-11-24", 18, True), ("2026-11-28", 16, False)]),
        "DEN": games("DEN", [("2026-11-27", 20, False)]),
        "SAS": games("SAS", [("2026-11-27", 20, False)]),
        "BOS": games("BOS", [("2026-11-25", 18, True), ("2026-11-27", 18, False)]),
        "ATL": games("ATL", [("2026-11-25", 18, True), ("2026-11-29", 18, False)]),
    }
    rng = np.random.default_rng(1)
    logs = {}
    for pid, base in [("1380", 60), ("1272", 50), ("2091", 24), ("2065", 22), ("2090", 38), ("9001", 55), ("9002", 50)]:
        logs[pid] = [GameLog(pid, f"2026-11-{d:02d}", 5, PLAYERS[pid]["team"], "XXX", float(max(5, rng.normal(base, 10)))) for d in range(10, 22)]
    logs["1380"] += [GameLog("1380", "2026-11-23", 6, "MIA", "XXX", 45.0), GameLog("1380", "2026-11-25", 6, "MIA", "XXX", 65.0)]
    logs["2090"] += [GameLog("2090", "2026-11-25", 6, "MEM", "XXX", jerome)]
    logs["9001"] += [GameLog("9001", "2026-11-25", 6, "BOS", "XXX", 48.0)]
    logs["9002"] += [GameLog("9002", "2026-11-25", 6, "ATL", "XXX", 41.0)]
    league = dict(roster_positions=["UTIL", "UTIL", "BN", "BN", "BN"], scoring_settings={"pts": 1.0},
                  settings={"waiver_clear_days": 2})
    rosters = [dict(roster_id=10, players=MINE, reserve=[], owner_id="u", settings={}),
               dict(roster_id=8, players=["9001", "9002"], reserve=[], owner_id="v", settings={})]
    matchups = {6: [dict(roster_id=10, matchup_id=1, starters=["1380", "2090"], starters_points=list(sp), points=sum(sp)),
                    dict(roster_id=8, matchup_id=1, starters=["9001", "9002"], starters_points=[48.0, 41.0], points=89.0)]}
    snap = Snapshot(now or t("2026-11-27", 12), "2026", 6, league, rosters, [], matchups, PLAYERS, logs, {}, tg, [])
    snap.stat_dates = {(g.team, g.date) for gs in tg.values() for g in gs if g.final}
    return snap


# ---------------------------------------------------------------- parsing
REPLY = """Locked giannis, jerome
and also lock AD

On Fri, Nov 27, 2026 at 5:02 PM Sleeper Bot <bot@example.com> wrote:
> Reply 'locked name, name' to record locks.
> locked Keldon Johnson
"""


def test_quoted_text_stripped_and_names_parsed():
    assert lk.parse_names(REPLY) == ["giannis", "jerome", "AD"]


def test_unlocked_ignored():
    assert lk.parse_names("unlocked giannis\nun-lock jerome") == []


def test_outlook_style_quote_stripped():
    body = "lock bam\n-----Original Message-----\nlocked maxey"
    assert lk.parse_names(body) == ["bam"]


def test_fuzzy_and_nickname_match():
    nick = CFG["locks"]["nicknames"]
    assert lk.match_name("giannis", MINE, PLAYERS, nick)[:2] == ("1380", "ok")
    assert lk.match_name("Antetokoumpo", MINE, PLAYERS, nick)[:2] == ("1380", "ok")   # typo
    assert lk.match_name("AD", MINE, PLAYERS, nick)[:2] == ("1272", "ok")
    assert lk.match_name("ty jerome", MINE, PLAYERS, nick)[:2] == ("2090", "ok")


def test_ambiguous_and_nomatch():
    pid, status, cands = lk.match_name("johnson", MINE, PLAYERS, {})
    assert pid is None and status == "ambiguous" and set(cands) == {"2091", "2065"}
    assert lk.match_name("keldon", MINE, PLAYERS, {})[:2] == ("2065", "ok")
    assert lk.match_name("lebron", MINE, PLAYERS, {})[1] == "nomatch"


# ---------------------------------------------------------------- game resolution
def test_lock_resolves_to_last_completed_game_at_email_time():
    snap = make_snap()
    g, fp = lk.resolve_game(snap, "1380", t("2026-11-25", 21))      # 11/25 game (18:00) over by 21:00
    assert g.date == "2026-11-25" and fp == 65.0
    g, fp = lk.resolve_game(snap, "1380", t("2026-11-25", 19))      # still playing -> previous game
    assert g.date == "2026-11-23" and fp == 45.0
    g, fp = lk.resolve_game(snap, "2091", t("2026-11-27", 12))      # no completed game this week
    assert g is None


class FakeInbox:
    def __init__(self, msgs):
        self.msgs, self.calls = msgs, 0

    def fetch(self, sender, since):
        self.calls += 1
        return self.msgs


def test_ingest_fake_imap_once_per_message(monkeypatch):
    monkeypatch.setenv("EMAIL_TO", "me@example.com")
    snap = make_snap()
    store = {"weeks": {}, "processed": []}
    msgs = [("<a@x>", t("2026-11-26", 10), "Re: Sleeper | Wed Nov 25 | P(win) 60% | 3 actions", REPLY),
            ("<b@x>", t("2026-11-26", 11), "Lunch?", "locked Keldon"),                    # wrong subject
            ("<c@x>", t("2026-11-26", 12), "Re: Sleeper | x", "locked johnson")]          # ambiguous
    notes = lk.ingest(snap, CFG, store, inbox=FakeInbox(msgs))
    wk = store["weeks"]["6"]
    assert set(wk) == {"1380", "2090", "1272"}
    assert wk["1380"]["score"] == 65.0 and wk["1380"]["msg_id"] == "<a@x>" and wk["1380"]["source"] == "email"
    assert wk["1272"]["date"] == "2026-11-24"
    assert any("ambiguous" in n for n in notes)
    assert "<b@x>" not in store["processed"]
    n_before = dict(wk)
    lk.ingest(snap, CFG, store, inbox=FakeInbox(msgs))                                    # re-run: no duplicates
    assert store["weeks"]["6"] == n_before and store["processed"].count("<a@x>") == 1


def test_imap_failure_is_noted(monkeypatch):
    monkeypatch.setenv("EMAIL_TO", "me@example.com")
    class Broken:
        def fetch(self, *a):
            raise OSError("connection refused")
    notes = lk.ingest(make_snap(), CFG, {"weeks": {}, "processed": []}, inbox=Broken())
    assert any("IMAP lock check failed" in n for n in notes)


# ---------------------------------------------------------------- engine respects locks
def _pwin(snap, reported, seed=11):
    snap.reported_locks = {10: reported}
    book = ModelBook(snap.logs, snap.players, snap.proj, CFG)
    dr = Draws(book, 5000, seed)
    opp = solve_team(snap, 8, book, dr)
    me = solve_team(snap, 10, book, dr, vs=opp.totals)
    return me, p_win(me.totals, opp.totals)


def test_engine_freezes_reported_lock_and_is_deterministic():
    snap = make_snap()
    rec = {"1380": dict(game_id="MIA1", date="2026-11-25", score=65.0, source="email")}
    me0, pw0 = _pwin(snap, {})
    me1, pw1 = _pwin(snap, rec)
    me1b, pw1b = _pwin(snap, rec)
    assert pw1 == pw1b                                   # seeded -> identical
    s = me1.states[0]
    assert s.locked and s.lock_source == "reported" and not s.open_decision
    assert all(d.pid != "1380" for d in me1.decisions)   # no open call for a reported lock
    assert any(d.pid == "1380" for d in me0.decisions)
    assert me1.plan.slots[0].frozen == 65.0


def test_reported_lock_changes_pwin_when_engine_would_ride():
    snap = make_snap(jerome=12.0)                         # poor counting score: engine wants to ride
    me0, pw0 = _pwin(snap, {})
    assert next(d for d in me0.decisions if d.pid == "2090").choice in ("hold", "ride")
    rec = {"2090": dict(date="2026-11-25", score=12.0, source="manual")}
    me1, pw1 = _pwin(snap, rec)
    assert me1.plan.slots[1].frozen == 12.0
    assert pw1 < pw0 and me1.plan.ev < me0.plan.ev
    assert _pwin(snap, rec)[1] == pw1


def test_conflicting_report_prefers_api():
    snap = make_snap()
    me, _ = _pwin(snap, {"1380": dict(date="2026-11-23", score=45.0, source="email")})
    s = me.states[0]
    assert not s.locked and "using the API" in s.conflict


# ---------------------------------------------------------------- Sleeper private API lock map (recorded, token-free)
def test_parse_recorded_matchup_legs():
    import json
    from data.sleeper_private import parse_legs
    resp = json.load(open("tests/fixtures/matchup_legs_week6.json", encoding="utf-8"))
    locks = parse_legs(resp)
    assert set(locks[10]) == {"1272", "1380", "1972", "2090", "2126"}       # 5 locked starters in week 6
    assert locks[10]["1380"] == "1393301049854418944"                     # MIA @ PHI, 11/25
    assert locks[10]["1380"] == locks[10]["2126"]                         # Giannis and Maxey locked to the same game
    with pytest.raises(ValueError):
        parse_legs({"errors": [{"message": "unauthorized"}]})


def test_api_locks_override_inference_and_reports():
    snap = make_snap()
    snap.api_locks = {10: {"1380": "MIA1"}, 8: {}}                        # Giannis locked to 11/25 game, Jerome not
    snap.reported_locks = {10: {"2090": dict(date="2026-11-25", score=50.0, source="email")}}
    from optimize.lock import infer_slots
    st = {s.pid: s for s in infer_slots(snap, 10, reported=snap.reported_locks[10])}
    assert st["1380"].locked and st["1380"].lock_source == "sleeper" and st["1380"].x == 65.0
    assert not st["2090"].locked and st["2090"].open_decision and "Sleeper shows none" in st["2090"].conflict


def test_api_absence_unlocks_inferred_lock():
    snap = make_snap()
    # make Giannis look locked by inference: counting the 11/23 game although 11/25 is final
    snap.matchups[6][0]["starters_points"] = [45.0, 50.0]
    from optimize.lock import infer_slots
    assert infer_slots(snap, 10)[0].locked
    snap.api_locks = {10: {}, 8: {}}
    s = infer_slots(snap, 10)[0]
    assert not s.locked and s.open_decision and s.x == 45.0


def test_token_never_needed_offline(monkeypatch, tmp_path):
    from data import sleeper_private as spv
    monkeypatch.delenv("SLEEPER_TOKEN", raising=False)
    fails = []
    assert spv.fetch_locks("1", 6, tmp_path, tmp_path, fails) is None and "no SLEEPER_TOKEN" in fails[0]
