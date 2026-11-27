import json
from datetime import datetime
from zoneinfo import ZoneInfo

import yaml

from data import news
from data.loader import Game, Snapshot

TZ = ZoneInfo("America/Chicago")
CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))
FEED = json.load(open("tests/fixtures/espn_injuries_sample.json", encoding="utf-8"))   # public ESPN data, trimmed

PLAYERS = {
    "1054": dict(first_name="Kyrie", last_name="Irving", team="DAL", injury_status=None, news_updated=None),
    "m1": dict(first_name="LaMelo", last_name="Ball", team="MIN", injury_status="GTD", news_updated=1795700000000),
    "c1": dict(first_name="Jalen", last_name="Smith", team="CHI", injury_status="DTD", news_updated=None),
    "a1": dict(first_name="Jock", last_name="Landale", team="ATL", injury_status="Out", news_updated=None),
    "x1": dict(first_name="Jalen", last_name="Smith Jr.", team="PHX"),   # same-name decoy on another team
}


def g(team, d, started=False):
    return Game(f"{team}-{d}", d, 6, team, "OPP", True, tip=datetime.fromisoformat(f"{d}T19:00").replace(tzinfo=TZ),
                started=started, final=started)


def snap(now="2026-11-27T13:00"):
    tg = {t: [g(t, "2026-11-27"), g(t, "2026-11-28"), g(t, "2026-11-29")] for t in ("DAL", "MIN", "CHI", "ATL", "PHX")}
    league = dict(roster_positions=["UTIL"], scoring_settings={}, settings={})
    return Snapshot(datetime.fromisoformat(now).replace(tzinfo=TZ), "2026", 6, league, [], [], {}, PLAYERS, {}, {}, tg, [])


def adjs(s=None):
    s = s or snap()
    return news.build(s, CFG, news.match(news.parse(FEED), s.players))


def test_parse_and_name_team_matching():
    items = news.parse(FEED)
    assert len(items) == 4 and all(i.date is not None for i in items)
    m = news.match(items, PLAYERS)
    assert set(m) == {"1054", "m1", "c1", "a1"}           # Jalen Smith matched to CHI, not the PHX decoy
    assert news.name_key("Jalen Smith Jr.") == "jalen smith"


def test_status_mapping_and_return_date():
    a = adjs()["a1"]                                       # Landale: Out, returnDate 11/29
    assert a.params("2026-11-27", 0.97)[0] == 0.0
    assert a.params("2026-11-28", 0.97)[0] == 0.0
    assert a.params("2026-11-29", 0.97)[0] == 0.97          # back on the return date


def test_keyword_rules_and_date_targeting():
    A = adjs()
    # LaMelo: "ruled out for Friday's game" -> 0 on Friday; ESPN (18:52Z) newer than Sleeper news_updated
    assert A["m1"].params("2026-11-27", 0.6)[0] == 0.0
    # Kyrie: Day-To-Day, "questionable for Friday's game" -> 0.6 that game
    assert A["1054"].params("2026-11-27", 0.97)[0] == 0.6
    # Jalen Smith: "probable for Saturday's game" -> 0.9 on Saturday only
    assert A["c1"].params("2026-11-28", 0.6)[0] == 0.9
    assert A["c1"].params("2026-11-29", 0.6)[0] == 0.6
    assert news.target_date("out for Friday's game", datetime(2026, 11, 26, 22, 44, tzinfo=ZoneInfo("UTC")), TZ, []) == "2026-11-27"
    assert news.target_date("plays tonight", datetime(2026, 11, 27, 20, 0, tzinfo=ZoneInfo("UTC")), TZ, []) == "2026-11-27"


def _one(short, long="", status="Day-To-Day", date="2026-11-27T18:00Z"):
    feed = {"injuries": [{"injuries": [{"status": status, "date": date, "shortComment": short, "longComment": long,
                                        "details": {}, "athlete": {"displayName": "Kyrie Irving", "team": {"abbreviation": "DAL"}}}]}]}
    s = snap()
    return news.build(s, CFG, news.match(news.parse(feed), s.players)).get("1054")


def test_minutes_rest_starting_and_negation():
    a = _one("Irving will be on a minutes restriction Saturday.")
    assert a.params("2026-11-28", 0.97)[1] == 0.8 and a.params("2026-11-27", 0.97)[1] == 1.0
    a = _one("Irving will rest Sunday (load management).")
    assert a.params("2026-11-29", 0.97)[0] == 0.0 and a.params("2026-11-27", 0.97)[0] == 0.6
    a = _one("Irving has a sore knee and could rest at some point.")      # rest without a date: no effect
    assert all(d.get("p") != 0.0 for d in a.per_date.values())
    a = _one("Irving has not been ruled out.")
    assert a.params("2026-11-27", 0.97)[0] == 0.6                          # negated, status prob only
    a = _one("Irving will move into the starting lineup.", status="Active")
    assert 1.0 < a.params("2026-11-27", 0.97)[1] <= CFG["news"]["mean_mult_bounds"][1]


def test_sleeper_newer_wins():
    p = dict(PLAYERS)
    p["1054"] = dict(p["1054"], news_updated=int(datetime(2026, 11, 27, 20, 0, tzinfo=ZoneInfo("UTC")).timestamp() * 1000))
    s = snap()
    s.players = p
    assert "1054" not in news.build(s, CFG, news.match(news.parse(FEED), p))


def test_feed_failure_falls_back():
    class Dead:
        def get(self, *a, **k):
            return None
    class Book:
        pass
    b = Book()
    notes = news.apply(snap(), CFG, Dead(), b)
    assert notes and "unavailable" in notes[0] and not hasattr(b, "news")
