"""When to send. Pure functions on timezone-aware datetimes (America/Chicago), so DST is handled
by zoneinfo and the logic is unit-testable with simulated clocks.

Pregame: computed per day from that day's FIRST tip across the whole NBA slate.
    send_at = first_tip - lead (config pregame_lead_min, default 90)
    * now >= send_at and before first tip, not yet sent today  -> send ("on-time", or "catch-up" if
      more than one cron interval late - GitHub cron can run 10-30+ min late)
    * first tip already passed, not sent today, but some of MY or my OPPONENT's players have not
      tipped yet -> send anyway, labelled "late"
    * if tip times are unknown (ESPN down) but there are games today, assume first tip at
      fallback_first_tip (conservatively early).
Lock window: any open lock/ride call whose next game tips within lock_window_min and that hasn't
    had a reminder -> send now (back-to-backs, early games).
"""
from __future__ import annotations

from datetime import datetime, time, timedelta


def first_tip(tips, day, tz, fallback: str | None):
    tips = [t for t in tips if t is not None]
    if tips:
        return min(tips)
    if fallback:
        hh, mm = (int(x) for x in fallback.split(":"))
        return datetime.combine(day, time(hh, mm), tzinfo=tz)
    return None


def pregame_decision(now: datetime, all_tips_today, my_tips_today, sent_today: bool, lead_min: int,
                     cron_interval_min: int = 15, fallback_first_tip: str | None = "12:00", have_games: bool = True):
    """-> (send: bool, label: str | None, send_at: datetime | None)"""
    if sent_today or not have_games:
        return False, None, None
    ft = first_tip(all_tips_today, now.date(), now.tzinfo, fallback_first_tip)
    if ft is None:
        return False, None, None
    send_at = ft - timedelta(minutes=lead_min)
    if now < send_at:
        return False, None, send_at
    if now < ft:
        late_by = (now - send_at).total_seconds() / 60
        return True, ("on-time" if late_by <= cron_interval_min + 5 else "catch-up"), send_at
    if any(t is not None and t > now for t in my_tips_today):
        return True, "late", send_at
    return False, None, send_at


def lock_window_due(now: datetime, open_calls, alerted: set, window_min: int):
    """open_calls: [(key, next_tip)]. Returns keys whose next tip is within the window and unalerted."""
    due = []
    for key, tip in open_calls:
        if tip is None or key in alerted:
            continue
        mins = (tip - now).total_seconds() / 60
        if 0 < mins <= window_min:
            due.append(key)
    return due


def keys_to_mark(now: datetime, open_calls, horizon_min: int):
    """After sending, mark calls whose tip falls within the horizon (one email per tip cluster)."""
    return [k for k, tip in open_calls if tip is not None and 0 < (tip - now).total_seconds() / 60 <= horizon_min]
