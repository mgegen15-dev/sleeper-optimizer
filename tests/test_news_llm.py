from types import SimpleNamespace

import anthropic
import httpx2
import yaml

from data import news
from model import news_llm
from model.distributions import ModelBook
from model.news_llm import NewsCalls, PlayerCall
from tests.test_locks import make_snap

CFG = yaml.safe_load(open("config.yaml", encoding="utf-8"))


class FakeClient:
    def __init__(self, calls=None, stop="end_turn", exc=None):
        self.n = 0
        self._calls, self._stop, self._exc = calls, stop, exc
        self.messages = self

    def parse(self, **kw):
        self.n += 1
        assert kw["model"] == "claude-haiku-5-5" and kw["output_format"] is NewsCalls
        assert "temperature" not in kw and "thinking" not in kw
        assert "@" not in kw["messages"][0]["content"]               # no email-like data in the prompt
        if self._exc:
            raise self._exc
        return SimpleNamespace(stop_reason=self._stop, parsed_output=self._calls,
                               usage=SimpleNamespace(input_tokens=1200, output_tokens=300))


def setup():
    snap = make_snap()
    snap.players["1380"]["injury_status"] = None
    book = ModelBook(snap.logs, snap.players, snap.proj, CFG)
    return snap, book


def calls(*items):
    return NewsCalls(players=[PlayerCall(**i) for i in items])


def test_schema_parse_and_caps():
    snap, book = setup()
    c = calls(dict(sleeper_player_id="1380", game_date="2026-11-27", p_play=0.1, minutes_factor=0.5, confidence="high", reason="ankle"),
              dict(sleeper_player_id="2090", game_date=None, p_play=1.0, minutes_factor=1.4, confidence="med", reason="role up"),
              dict(sleeper_player_id="2091", game_date=None, p_play=0.0, minutes_factor=1.0, confidence="low", reason="rumor"))
    st = {}
    status, notes = news_llm.run(snap, CFG, book, st, "pregame", client=FakeClient(c))
    assert status == "ok" and st["news_llm_date"] == "2026-11-27" and len(notes) == 2
    m, _ = news.game_model(book, "1380", snap.team_games["MIA"][2])
    assert abs(m.p_play - (0.97 - 0.4)) < 1e-6                     # moved toward 0.1 by at most 0.4
    assert abs(m.mu - book.get("1380").mu * 0.75) < 1e-6           # minutes factor clamped at 0.75
    m2, _ = news.game_model(book, "2090", snap.team_games["MEM"][1])
    assert abs(m2.mu - book.get("2090").mu * 1.15) < 1e-6          # clamped at 1.15
    assert "2091" not in book.news                                  # low confidence ignored
    assert book.news_llm_usage["usd"] > 0


def test_cannot_raise_official_out():
    snap, book = setup()
    snap.players["1380"]["injury_status"] = "Out"
    book = ModelBook(snap.logs, snap.players, snap.proj, CFG)
    c = calls(dict(sleeper_player_id="1380", p_play=0.9, minutes_factor=1.0, confidence="high", reason="says he's fine"))
    news_llm.run(snap, CFG, book, {}, "pregame", client=FakeClient(c))
    m, _ = news.game_model(book, "1380", snap.team_games["MIA"][2])
    assert m.p_play == 0.0


def test_refusal_and_errors_fall_back():
    snap, book = setup()
    st, notes = news_llm.run(snap, CFG, book, {}, "pregame", client=FakeClient(calls(), stop="refusal"))
    assert st == "failed" and "declined" in notes[0]
    req = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    err = anthropic.RateLimitError("slow down", response=httpx2.Response(429, request=req), body=None)
    st, notes = news_llm.run(snap, CFG, book, {}, "pregame", client=FakeClient(exc=err))
    assert st == "failed" and "rate-limited" in notes[0]
    st, notes = news_llm.run(snap, CFG, book, {}, "pregame", client=FakeClient(exc=anthropic.APIConnectionError(request=req)))
    assert st == "failed" and not getattr(book, "news", {})


def test_once_per_day_and_mode_gating(monkeypatch):
    snap, book = setup()
    fc = FakeClient(calls())
    assert news_llm.run(snap, CFG, book, {"news_llm_date": "2026-11-27"}, "pregame", client=fc)[0] == "skipped"
    assert news_llm.run(snap, CFG, book, {}, "postgame", client=fc)[0] == "skipped"
    assert fc.n == 0
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    st, notes = news_llm.run(snap, CFG, book, {}, "pregame")
    assert st == "skipped" and "ANTHROPIC_API_KEY" in notes[0]
