"""Locks reported by you (email reply, workflow input, or hand-edited state/locks.json).

Email grammar (case-insensitive, only the new text above the quoted reply is read):
    "locked giannis, jerome"   "lock maxey and AD"   "Locked: Bam"
"unlocked ..." is ignored (locks are irreversible). Names are matched against MY roster only:
full name, last name, first name, initials ("AD") and config nicknames; then difflib fuzzy match.
Two or more candidates -> reported back as ambiguous, never guessed.
A reported lock applies to the player's most recent completed game at the time of the report.

state/locks.json format:
    {"weeks": {"6": {"1380": {"game_id": "...", "date": "2026-11-25", "score": 65.0,
                              "source": "email|manual|file", "msg_id": "<...>", "reported_at": "ISO"}}},
     "processed": ["<message-id>", ...]}
"""
from __future__ import annotations

import difflib
import email
import email.utils
import imaplib
import json
import re
import unicodedata
from datetime import datetime, timedelta
from email.header import decode_header, make_header
from pathlib import Path

QUOTE_MARKERS = [
    re.compile(r"^\s*On .+wrote:\s*$", re.I),
    re.compile(r"^\s*-{2,}\s*Original Message\s*-{2,}", re.I),
    re.compile(r"^\s*From:\s", re.I),
    re.compile(r"^\s*Sent from my ", re.I),
    re.compile(r"^\s*_{5,}"),
]
LOCK_RE = re.compile(r"(?<![a-z])(?:locked|lock)\b\s*:?\s*(.+)", re.I)
UNLOCK_RE = re.compile(r"\bun-?lock", re.I)


def norm(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z0-9 ]", "", s.lower().replace("-", " ")).strip()


def new_text(body: str) -> str:
    """Text above the quoted reply."""
    out = []
    lines = (body or "").replace("\r\n", "\n").split("\n")
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith(">"):
            break
        # Gmail sometimes wraps "On ... wrote:" over two lines
        joined = ln + (" " + lines[i + 1] if i + 1 < len(lines) else "")
        if any(m.search(ln) for m in QUOTE_MARKERS) or QUOTE_MARKERS[0].search(joined):
            break
        out.append(ln)
    return "\n".join(out).strip()


INELIG_RE = re.compile(r"(?<![a-z])ineligible\b\s*:?\s*(.+)", re.I)


def parse_ineligible(body: str) -> list[str]:
    names = []
    for ln in new_text(body).split("\n"):
        m = INELIG_RE.search(ln)
        if m:
            for tok in re.split(r",|;|/|&|\band\b|\+", m.group(1), flags=re.I):
                tok = tok.strip(" .!?:\t")
                if tok:
                    names.append(tok)
    return names


def parse_names(body: str) -> list[str]:
    names = []
    for ln in new_text(body).split("\n"):
        if UNLOCK_RE.search(ln) or INELIG_RE.search(ln):
            continue
        m = LOCK_RE.search(ln)
        if not m:
            continue
        for tok in re.split(r",|;|/|&|\band\b|\+", m.group(1), flags=re.I):
            tok = tok.strip(" .!?:\t")
            if tok:
                names.append(tok)
    return names


def name_keys(p: dict) -> set[str]:
    first, last = norm(p.get("first_name", "")), norm(p.get("last_name", ""))
    keys = {f"{first} {last}".strip(), last, first, (first[:1] + last[:1]) if first and last else ""}
    keys |= {k.replace(" ", "") for k in keys}
    return {k for k in keys if k}


def match_name(token: str, roster: list, players: dict, nicknames: dict | None = None, cutoff: float = 0.8):
    """-> (pid, 'ok') | (None, 'ambiguous', [pids]) | (None, 'nomatch', [])"""
    t = norm(token)
    nick = {norm(k): norm(v) for k, v in (nicknames or {}).items()}
    t = nick.get(t, t)
    index = {}
    for pid in roster:
        for k in name_keys(players.get(pid, {})):
            index.setdefault(k, set()).add(pid)
    hits = index.get(t) or index.get(t.replace(" ", ""))
    if not hits:
        close = difflib.get_close_matches(t, list(index), n=5, cutoff=cutoff)
        hits = set().union(*(index[c] for c in close)) if close else set()
        if close and len(hits) > 1:      # keep only the best-scoring key's players if clearly better
            best = difflib.SequenceMatcher(None, t, close[0]).ratio()
            second = [c for c in close if difflib.SequenceMatcher(None, t, c).ratio() >= best - 0.05]
            hits = set().union(*(index[c] for c in second))
    if len(hits) == 1:
        return next(iter(hits)), "ok", []
    return None, ("ambiguous" if hits else "nomatch"), sorted(hits)


def resolve_game(snap, pid: str, at: datetime):
    """Most recent completed game of the player's team (this week) before time `at`."""
    team = snap.players.get(pid, {}).get("team")
    done = []
    for g in snap.week_games(team) if team else []:
        if g.tip is not None:      # a game takes ~2h; it must have ended before the report was sent
            ended = g.final and g.tip + timedelta(hours=2) <= at
        else:
            ended = g.final and g.date < at.date().isoformat()
        if ended:
            done.append(g)
    if not done:
        return None, None
    g = done[-1]
    log = next((x for x in snap.logs.get(pid, []) if x.date == g.date), None)
    return g, (log.fp if log else 0.0)


# ------------------------------------------------------------------ state file
def load_locks(path) -> dict:
    p = Path(path)
    try:
        d = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        d = {}
    d.setdefault("weeks", {})
    d.setdefault("processed", [])
    return d


def save_locks(path, d):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    d["processed"] = d.get("processed", [])[-500:]
    Path(path).write_text(json.dumps(d, indent=1), encoding="utf-8")


def record(snap, locks: dict, names: list, at: datetime, source: str, roster: list, nicknames, cutoff, msg_id=None):
    """Resolve names -> games and store. Returns human-readable notes."""
    notes = []
    wk = locks["weeks"].setdefault(str(snap.week), {})
    for nm in names:
        pid, status, cands = match_name(nm, roster, snap.players, nicknames, cutoff)
        if status != "ok":
            what = ("ambiguous: " + " / ".join(snap.pname(c) for c in cands)) if cands else "no match on your roster"
            notes.append(f"lock '{nm}' ({source}) not recorded - {what}")
            continue
        if pid in wk:
            notes.append(f"{snap.pname(pid)} already recorded as locked ({wk[pid]['date']}, {wk[pid]['score']})")
            continue
        g, fp = resolve_game(snap, pid, at)
        if g is None:
            notes.append(f"lock '{nm}' -> {snap.pname(pid)}: no completed game this week before {at:%a %H:%M}; not recorded")
            continue
        wk[pid] = dict(game_id=g.game_id, date=g.date, score=fp, source=source, msg_id=msg_id, reported_at=at.isoformat())
        notes.append(f"recorded lock: {snap.pname(pid)} = {fp:.1f} ({g.date} {'vs ' if g.home else '@'}{g.opp}, {source})")
    return notes


# ------------------------------------------------------------------ Gmail IMAP
class GmailInbox:
    """Minimal read-only IMAP reader. fetch() -> [(msg_id, datetime, subject, plain_text_body)]."""

    def __init__(self, user, password, host="imap.gmail.com", port=993):
        self.user, self.password, self.host, self.port = user, password, host, port

    def fetch(self, sender: str, since: datetime):
        out = []
        M = imaplib.IMAP4_SSL(self.host, self.port, timeout=30)
        try:
            M.login(self.user, self.password)
            M.select("INBOX", readonly=True)
            typ, data = M.search(None, "FROM", f'"{sender}"', "SINCE", since.strftime("%d-%b-%Y"))
            for num in (data[0].split() if data and data[0] else []):
                typ, msg_data = M.fetch(num, "(BODY.PEEK[])")
                raw = next((part[1] for part in msg_data if isinstance(part, tuple)), None)
                if raw is None:
                    continue
                msg = email.message_from_bytes(raw)
                subj = str(make_header(decode_header(msg.get("Subject", ""))))
                when = email.utils.parsedate_to_datetime(msg.get("Date")) if msg.get("Date") else None
                out.append((msg.get("Message-ID", f"num-{num.decode()}"), when, subj, plain_body(msg)))
        finally:
            try:
                M.logout()
            except Exception:
                pass
        return out


def plain_body(msg) -> str:
    if msg.is_multipart():
        for part in msg.walk():
            if part.get_content_type() == "text/plain" and "attachment" not in str(part.get("Content-Disposition", "")):
                return part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
        for part in msg.walk():
            if part.get_content_type() == "text/html":
                html = part.get_payload(decode=True).decode(part.get_content_charset() or "utf-8", "replace")
                html = re.sub(r"(?is)<blockquote.*", "", html)          # drop quoted reply
                return re.sub(r"<[^>]+>", "\n", html)
        return ""
    return msg.get_payload(decode=True).decode(msg.get_content_charset() or "utf-8", "replace")


def is_lock_subject(subj: str) -> bool:
    s = (subj or "").strip()
    return s.lower().startswith("re: sleeper |") or "locked" in s.lower()


def ingest(snap, cfg, locks: dict, inbox=None, manual: str | None = None):
    """Pull email replies (and a manual comma list) into `locks`. Returns notes for Assumptions."""
    notes = []
    lc = cfg.get("locks", {})
    r = snap.roster(cfg["roster_id"])
    roster = [str(p) for p in r["players"]]
    nick, cutoff = lc.get("nicknames", {}), lc.get("fuzzy_cutoff", 0.8)
    monday = (snap.now - timedelta(days=snap.now.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
    if inbox is not None:
        import os
        sender = os.environ.get("EMAIL_TO") or cfg["email"].get("to") or os.environ.get(cfg["email"]["user_env"])
        if not sender:
            raise RuntimeError("no recipient address configured")
        try:
            msgs = inbox.fetch(sender, monday)
            for mid, when, subj, body in sorted(msgs, key=lambda m: m[1] or snap.now):
                if mid in locks["processed"] or not is_lock_subject(subj):
                    continue
                when = (when or snap.now).astimezone(snap.now.tzinfo)
                if when < monday:
                    continue
                names = parse_names(body)
                if names:
                    notes += record(snap, locks, names, when, "email", roster, nick, cutoff, mid)
                for nm in parse_ineligible(body):
                    pid, status, cands = match_name(nm, roster, snap.players, nick, cutoff)
                    if status != "ok":
                        notes.append(f"ineligible '{nm}' not recorded ({status})")
                        continue
                    team = snap.players.get(pid, {}).get("team")
                    gids = [str(g.game_id) for g in snap.week_games(team) if g.started and (g.tip is None or g.tip <= when)]
                    locks.setdefault("ineligible", {}).setdefault(str(snap.week), {}).setdefault(pid, [])
                    locks["ineligible"][str(snap.week)][pid] = sorted(set(locks["ineligible"][str(snap.week)][pid]) | set(gids))
                    notes.append(f"recorded: {snap.pname(pid)} games so far not lockable (email)")
                locks["processed"].append(mid)
        except Exception as e:
            notes.append(f"IMAP lock check failed ({type(e).__name__}: {e}); using stored/inferred locks only")
    if manual:
        names = [t.strip() for t in re.split(r",|;", manual) if t.strip()]
        notes += record(snap, locks, names, snap.now, "manual", roster, nick, cutoff)
    return notes
