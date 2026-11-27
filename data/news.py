"""Free injury/news adjustments from ESPN's public NBA injuries feed (no AI, rule-based).

Feed: GET https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries
  {"timestamp", "status", "season", "injuries": [ {team "id", "displayName",
     "injuries": [ {"status": "Out"|"Day-To-Day"|..., "date": "2026-11-26T22:44Z",
                    "shortComment", "longComment",
                    "details": {"fantasyStatus": {...}, "returnDate": "YYYY-MM-DD", ...},
                    "athlete": {"firstName", "lastName", "displayName", "team": {"abbreviation"}}} ]} ]}

For each listed player (matched to Sleeper by name + team):
  * Source choice: ESPN status is used only if its item is newer than Sleeper's news_updated; if
    Sleeper has no timestamp, ESPN is used only when it is more pessimistic and < 24h old.
  * Status -> P(play) (config). An "Out" with a returnDate only covers games before that date.
  * Keyword rules on the comments (config, case-insensitive, word boundaries): override P(play)
    or scale the mean (capped) for the game the note refers to: a weekday/"tonight"/"tomorrow"
    in the note picks that date, otherwise the team's next game. Rules flagged needs_date only
    fire when a date is mentioned (e.g. "resting Saturday").
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta

from data.locks import norm

URL = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
ESPN2SLP = {"NY": "NYK", "SA": "SAS", "GS": "GSW", "NO": "NOP", "UTAH": "UTA", "WSH": "WAS"}
SUFFIX = re.compile(r"\b(jr|sr|ii|iii|iv|v)\b")
DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
DATE_RE = re.compile(r"\b(monday|tuesday|wednesday|thursday|friday|saturday|sunday|tonight|today|tomorrow)\b", re.I)


def name_key(s: str) -> str:
    return re.sub(r"\s+", " ", SUFFIX.sub("", norm(s))).strip()


@dataclass
class NewsItem:
    name: str
    team: str | None
    status: str
    fantasy_status: str | None
    date: datetime | None
    short: str
    long: str
    return_date: str | None


@dataclass
class NewsAdj:
    pid: str
    source: str                     # "espn" | "sleeper"
    status: str
    base_p: float | None = None     # applies to all games this week (None = keep model's)
    until: str | None = None        # base_p applies only to games before this date (returnDate)
    per_date: dict = field(default_factory=dict)   # date -> {"p": .., "mult": ..}
    notes: list = field(default_factory=list)

    def params(self, date: str, p0: float):
        p = p0
        if self.base_p is not None and (self.until is None or date < self.until):
            p = self.base_p
        d = self.per_date.get(date, {})
        return d.get("p", p), d.get("mult", 1.0)


def parse(feed: dict) -> list[NewsItem]:
    out = []
    for t in (feed or {}).get("injuries", []):
        for it in t.get("injuries", []):
            a = it.get("athlete") or {}
            ab = ((a.get("team") or {}).get("abbreviation"))
            when = None
            if it.get("date"):
                try:
                    when = datetime.fromisoformat(it["date"].replace("Z", "+00:00"))
                except ValueError:
                    pass
            fs = ((it.get("details") or {}).get("fantasyStatus") or {}).get("description")
            out.append(NewsItem(a.get("displayName") or f"{a.get('firstName', '')} {a.get('lastName', '')}",
                                ESPN2SLP.get(ab, ab), it.get("status") or "", fs, when,
                                it.get("shortComment") or "", it.get("longComment") or "",
                                (it.get("details") or {}).get("returnDate")))
    return out


def match(items, players: dict) -> dict:
    """-> {pid: NewsItem}. Name + team first; unique name across the league as fallback."""
    by_nt, by_n = {}, {}
    for pid, p in players.items():
        if not p.get("team"):
            continue
        k = name_key(f"{p.get('first_name', '')} {p.get('last_name', '')}")
        by_nt[(k, p["team"])] = pid
        by_n.setdefault(k, []).append(pid)
    out = {}
    for it in items:
        k = name_key(it.name)
        pid = by_nt.get((k, it.team))
        if pid is None and len(by_n.get(k, [])) == 1:
            pid = by_n[k][0]
        if pid:
            out[pid] = it
    return out


def status_prob(item: NewsItem, table: dict):
    for s in (item.status, item.fantasy_status):
        if s and s in table:
            return table[s]
        if s and s.title() in table:
            return table[s.title()]
    return None


def target_date(text: str, at: datetime, tz, future_dates: list[str]) -> str | None:
    """Date the note refers to (weekday / tonight / tomorrow), resolved relative to the note time."""
    m = DATE_RE.search(text or "")
    if not m or at is None:
        return None
    w = m.group(1).lower()
    base = at.astimezone(tz).date()
    if w in ("tonight", "today"):
        d = base
    elif w == "tomorrow":
        d = base + timedelta(days=1)
    else:
        d = base + timedelta(days=(DAYS.index(w) - base.weekday()) % 7)
    return d.isoformat()


def build(snap, cfg, items_by_pid: dict) -> dict:
    nc = cfg["news"]
    table = nc["status_play_prob"]
    lo, hi = nc["mean_mult_bounds"]
    tz = snap.now.tzinfo
    rules = [(r, re.compile(r["pattern"], re.I), re.compile(r["unless"], re.I) if r.get("unless") else None) for r in nc["rules"]]
    out = {}
    for pid, it in items_by_pid.items():
        p = snap.players.get(pid, {})
        sl_ms = p.get("news_updated")
        sl_time = datetime.fromtimestamp(sl_ms / 1000, tz) if sl_ms else None
        sp = status_prob(it, table)
        sleeper_p = table.get(p.get("injury_status") or "healthy", table.get("healthy"))
        fresh = it.date is not None and (snap.now - it.date) <= timedelta(hours=nc["recent_hours"])
        if it.date is not None and sl_time is not None:
            use = it.date > sl_time
        else:
            use = fresh and sp is not None and sp <= sleeper_p   # never relax Sleeper's status on an untimed note
        if not use:
            continue
        adj = NewsAdj(pid, "espn", it.status, base_p=sp,
                      until=it.return_date if (it.status or "").lower() == "out" and it.return_date else None)
        adj.notes.append(f"status {it.status or '?'} (ESPN {it.date:%m/%d %H:%M}Z) -> P(play) {sp if sp is not None else 'unchanged'}"
                         + (f" until {it.return_date}" if adj.until else ""))
        team = p.get("team")
        future = [g.date for g in snap.team_games.get(team, []) if not g.started]
        text = f"{it.short} {it.long}"
        tgt = target_date(it.short, it.date, tz, future) or target_date(it.long, it.date, tz, future)
        nxt = future[0] if future else None
        for r, rx, unless in rules:
            if not rx.search(text) or (unless is not None and unless.search(text)):
                continue
            d = tgt if tgt else (None if r.get("needs_date") else nxt)
            if d is None or d not in future:
                continue
            slot = adj.per_date.setdefault(d, {})
            if "play_prob" in r and "p" not in slot:          # first matching rule wins for P(play)
                slot["p"] = r["play_prob"]
                adj.notes.append(f"rule '{r['name']}' -> P(play) {r['play_prob']} on {d}")
            if "mean_mult" in r:
                slot["mult"] = min(hi, max(lo, slot.get("mult", 1.0) * r["mean_mult"]))
                adj.notes.append(f"rule '{r['name']}' -> mean x{slot['mult']:.2f} on {d}")
        out[pid] = adj
    return out


def apply(snap, cfg, client, book) -> list[str]:
    """Fetch, match and attach adjustments to the ModelBook. Returns failure notes."""
    if not cfg.get("news", {}).get("enabled", True):
        return []
    feed = client.get(URL, "espn/injuries.json", max_age_h=0.25)
    if not feed or "injuries" not in feed:
        return ["ESPN injuries feed unavailable; using Sleeper injury status only"]
    try:
        items = match(parse(feed), snap.players)
        book.news_items = items
        book.news = build(snap, cfg, items)
        return []
    except Exception as e:
        return [f"ESPN injuries feed could not be applied ({type(e).__name__}); using Sleeper status only"]


def game_model(book, pid, game):
    """(PlayerModel for that specific game, opponent factor).
    Mean = history blend re-weighted with THAT game's Sleeper projection (not the week's average),
    then news adjustments (P(play), minutes multiplier) for that date."""
    m = book.get(pid)
    fac = book.opp_factor(pid, game.opp)
    mu = m.mu
    parts = getattr(m, "parts", {}) or {}
    pj = None if getattr(book, "ros", False) else ((getattr(book, "proj", {}) or {}).get(pid) or {}).get(game.date)
    if pj is not None and "w_proj" in parts:
        h = parts.get("hist")
        mu = pj if h is None else (parts["w_hist"] * h + parts["w_proj"] * pj) / (parts["w_hist"] + parts["w_proj"])
    adj = getattr(book, "news", {}).get(pid)
    p, mult = (adj.params(game.date, m.p_play) if adj is not None else (m.p_play, 1.0))
    if p == m.p_play and mult == 1.0 and abs(mu - m.mu) < 1e-9:
        return m, fac
    key = (pid, game.date)
    cache = book.__dict__.setdefault("_news_cache", {})
    if key not in cache:
        cache[key] = replace(m, p_play=float(p), mu=max(0.0, mu) * mult)
    return cache[key], fac
