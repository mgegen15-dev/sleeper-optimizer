"""Optional AI news step (Claude Haiku 5.5) layered on top of the rule-based ESPN adjustments.

Pregame run only, at most once per day (date kept in state). For players on my roster and this
week's opponent who still have games, we send ONLY public data - Sleeper player id, name, team,
remaining game dates, ESPN injury status + comments, and ESPN news headlines from the last 48h
that mention them - and ask for a structured judgement per player.

Applied with guard rails (config `news_llm`):
  * low-confidence items are ignored;
  * P(play) moves from the rules value toward the model's value by at most max_dp_play;
  * the model can never raise P(play) for a player officially Out (ESPN/Sleeper);
  * minutes_factor is clamped to [min, max] and multiplies the mean for that game.
Any failure (no key, refusal, API error, bad output) -> rules-only, noted under Assumptions.
"""
from __future__ import annotations

import os
from datetime import datetime, timedelta
from typing import Literal, Optional

from pydantic import BaseModel

from data.news import NewsAdj, game_model, name_key

NEWS_URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/news?limit=50"

SYSTEM = (
    "You assess NBA player availability for fantasy basketball from public injury reports and news. "
    "For each player listed, judge their next relevant game: probability they play (0-1) and a minutes "
    "factor versus their normal role (1.0 = normal; below 1 for a minutes limit or reduced role, above 1 "
    "for an expanded role). Use only the text provided. If the reports say nothing beyond a routine status, "
    "say so with confidence 'low'. Keep each reason under 15 words."
)


class PlayerCall(BaseModel):
    sleeper_player_id: str
    game_date: Optional[str] = None       # YYYY-MM-DD; null = next game
    p_play: float
    minutes_factor: float
    confidence: Literal["low", "med", "high"]
    reason: str


class NewsCalls(BaseModel):
    players: list[PlayerCall]


def _headlines(feed, since: datetime):
    out = []
    for a in (feed or {}).get("articles", []):
        try:
            pub = datetime.fromisoformat(a.get("published", "").replace("Z", "+00:00"))
        except ValueError:
            continue
        if pub >= since:
            out.append((pub, f"{a.get('headline', '')} - {a.get('description', '')}"))
    return out


def candidates(snap, cfg, book):
    rid = cfg["roster_id"]
    opp = snap.opponent_rid(rid)
    pids = [str(p) for r in (rid, opp) if r for p in snap.roster(r)["players"]]
    out = []
    for pid in dict.fromkeys(pids):
        team = snap.players.get(pid, {}).get("team")
        games = [g for g in snap.week_games(team) if not g.started] if team else []
        if games:
            out.append((pid, games))
    return out


def build_prompt(snap, cfg, book, cands, news_feed) -> str:
    items = getattr(book, "news_items", {}) or {}
    heads = _headlines(news_feed, snap.now - timedelta(hours=48))
    lines = []
    for pid, games in cands:
        p = snap.players.get(pid, {})
        name = f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
        ln = [f"id={pid} | {name} ({p.get('team')}) | games: {', '.join(g.date for g in games)}",
              f"  sleeper status: {p.get('injury_status') or 'none'}"]
        it = items.get(pid)
        if it:
            ln.append(f"  ESPN {it.status} ({it.date:%Y-%m-%d %H:%MZ}): {it.short} {it.long}".strip())
        last = name_key(p.get("last_name", ""))
        hits = [h for _, h in heads if last and (name_key(name) in name_key(h) or (len(last) > 4 and last in name_key(h)))][:3]
        ln += [f"  news: {h}" for h in hits]
        lines.append("\n".join(ln))
    return (f"Today is {snap.now:%A %Y-%m-%d} (US Central). Players:\n\n" + "\n\n".join(lines) +
            "\n\nReturn one entry per player id above.")


def run(snap, cfg, book, state: dict, mode: str, client=None, news_feed=None) -> tuple[str, list[str]]:
    """Returns (status, notes). status in {"ok", "skipped", "failed"}."""
    lc = cfg.get("news_llm", {})
    today = snap.now.date().isoformat()
    if not lc.get("enabled", False) or mode != "pregame":
        return "skipped", []
    if state.get("news_llm_date") == today:
        return "skipped", ["AI news step already ran today (once-per-day limit); rule-based news only this run."]
    if client is None and not os.environ.get("ANTHROPIC_API_KEY"):
        return "skipped", ["AI news step skipped (ANTHROPIC_API_KEY not set); rule-based news only."]
    cands = candidates(snap, cfg, book)
    if not cands:
        return "skipped", []
    import anthropic
    try:
        client = client or anthropic.Anthropic(max_retries=2, timeout=90.0)
        resp = client.messages.parse(
            model=lc.get("model", "claude-haiku-5-5"),
            max_tokens=16000,
            output_config={"effort": "low"},
            system=SYSTEM,
            messages=[{"role": "user", "content": build_prompt(snap, cfg, book, cands, news_feed)}],
            output_format=NewsCalls,
        )
    except anthropic.RateLimitError:
        return "failed", ["AI news step rate-limited; rule-based news only."]
    except anthropic.APIStatusError as e:
        return "failed", [f"AI news step API error {e.status_code}; rule-based news only."]
    except anthropic.APIConnectionError:
        return "failed", ["AI news step could not connect; rule-based news only."]
    if getattr(resp, "stop_reason", None) == "refusal":
        return "failed", ["AI news step declined the request; rule-based news only."]
    calls = getattr(resp, "parsed_output", None)
    if calls is None:
        return "failed", ["AI news step returned no usable output; rule-based news only."]
    notes = apply_calls(snap, cfg, book, calls, dict(cands))
    u = getattr(resp, "usage", None)
    if u is not None:
        cost = (u.input_tokens * lc.get("usd_per_mtok_in", 0.10) + u.output_tokens * lc.get("usd_per_mtok_out", 0.50)) / 1e6
        book.news_llm_usage = dict(input_tokens=u.input_tokens, output_tokens=u.output_tokens, usd=round(cost, 5))
    state["news_llm_date"] = today
    return "ok", notes


def apply_calls(snap, cfg, book, calls: NewsCalls, cand_games: dict) -> list[str]:
    lc = cfg.get("news_llm", {})
    max_dp = lc.get("max_dp_play", 0.4)
    lo, hi = lc.get("minutes_factor_bounds", [0.75, 1.15])
    if not hasattr(book, "news"):
        book.news = {}
    notes = []
    for c in calls.players:
        pid = str(c.sleeper_player_id)
        if c.confidence == "low" or pid not in cand_games:
            continue
        games = cand_games[pid]
        g = next((x for x in games if x.date == c.game_date), games[0])
        m, _ = game_model(book, pid, g)
        p_rules = m.p_play
        base_m = book.get(pid)
        mult_rules = m.mu / base_m.mu if base_m.mu else 1.0
        official_out = (snap.players.get(pid, {}).get("injury_status") in ("Out", "Sus", "IR")
                        or (pid in book.news and book.news[pid].params(g.date, 1.0)[0] == 0.0))
        target = min(1.0, max(0.0, float(c.p_play)))
        p_new = p_rules + max(-max_dp, min(max_dp, target - p_rules))
        if official_out:
            p_new = min(p_new, p_rules)
        mf = min(hi, max(lo, float(c.minutes_factor)))
        mult_new = min(hi, max(lo, mult_rules * mf))
        if abs(p_new - p_rules) < 0.01 and abs(mult_new - mult_rules) < 0.01:
            continue
        adj = book.news.get(pid) or NewsAdj(pid, "llm", snap.players.get(pid, {}).get("injury_status") or "")
        slot = adj.per_date.setdefault(g.date, {})
        slot["p"], slot["mult"] = round(p_new, 3), round(mult_new, 3)
        reason = c.reason.strip()[:120]
        note = f"AI ({c.confidence}): P(play) {p_rules:.2f}->{p_new:.2f}, mean x{mult_new:.2f} on {g.date} ({reason})"
        adj.notes.append(note)
        book.news[pid] = adj
        book.__dict__.get("_news_cache", {}).pop((pid, g.date), None)
        notes.append(f"{snap.pname(pid)}: {note}")
    return notes
