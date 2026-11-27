from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from optimize import schedule as sch

CT = ZoneInfo("America/Chicago")


def t(s):
    return datetime.fromisoformat(s).replace(tzinfo=CT)


SUN = [t("2026-11-29T14:30"), t("2026-11-29T15:00"), t("2026-11-29T19:30")]   # MIA @ CLE 2:30 PM CT first


def test_send_time_follows_first_tip_of_the_day():
    assert sch.pregame_decision(t("2026-11-29T12:59"), SUN, SUN, False, 90)[:2] == (False, None)
    send, label, at = sch.pregame_decision(t("2026-11-29T13:00"), SUN, SUN, False, 90)
    assert send and label == "on-time" and at == t("2026-11-29T13:00")
    noon = [t("2026-12-25T11:00"), t("2026-12-25T13:30")]                       # synthetic early slate
    assert sch.pregame_decision(t("2026-12-25T09:30"), noon, noon, False, 90)[:2] == (True, "on-time")
    assert sch.pregame_decision(t("2026-12-25T09:15"), noon, noon, False, 90)[0] is False


def test_catch_up_when_window_missed():
    # cron delayed: first run after the window is 25 min before tip -> still sends, labelled catch-up
    send, label, _ = sch.pregame_decision(t("2026-11-29T14:05"), SUN, SUN, False, 90, cron_interval_min=15)
    assert send and label == "catch-up"
    assert sch.pregame_decision(t("2026-11-29T14:05"), SUN, SUN, True, 90)[0] is False   # already sent today


def test_late_send_only_if_my_players_still_to_tip():
    mine = [t("2026-11-29T19:30")]
    send, label, _ = sch.pregame_decision(t("2026-11-29T16:00"), SUN, mine, False, 90)
    assert send and label == "late"
    assert sch.pregame_decision(t("2026-11-29T20:00"), SUN, mine, False, 90)[0] is False  # everyone tipped


def test_no_games_and_fallback_tip():
    assert sch.pregame_decision(t("2026-11-26T15:00"), [], [], False, 90, have_games=False)[0] is False
    # ESPN down: tips unknown but games exist -> assume 12:00, send from 10:30
    assert sch.pregame_decision(t("2026-11-28T10:29"), [None], [None], False, 90)[0] is False
    send, label, at = sch.pregame_decision(t("2026-11-28T10:30"), [None], [None], False, 90)
    assert send and at == t("2026-11-28T10:30")


def test_dst_transitions():
    # spring forward 2027-03-14 02:00 CST -> CDT; fall back 2027-11-07
    spring = [t("2027-03-14T14:00")]
    send, _, at = sch.pregame_decision(t("2027-03-14T12:30"), spring, spring, False, 90)
    assert send and at.utcoffset() == timedelta(hours=-5) and at.hour == 12 and at.minute == 30
    fall = [t("2027-11-07T14:00")]
    send, _, at = sch.pregame_decision(t("2027-11-07T12:30"), fall, fall, False, 90)
    assert send and at.utcoffset() == timedelta(hours=-6)
    # UTC-scheduled cron at 18:30Z is 12:30 CST in November and 13:30 CDT in March: the logic keys off CT
    utc_run = datetime(2027, 11, 7, 18, 30, tzinfo=ZoneInfo("UTC")).astimezone(CT)
    assert sch.pregame_decision(utc_run, fall, fall, False, 90)[0] is True


def test_lock_window_reminders():
    calls = [("1054:g1", t("2026-11-27T20:00")), ("1846:g2", t("2026-11-27T18:00")), ("2032:g3", t("2026-11-28T19:00"))]
    now = t("2026-11-27T17:20")
    assert sch.lock_window_due(now, calls, set(), 45) == ["1846:g2"]
    assert sch.lock_window_due(now, calls, {"1846:g2"}, 45) == []
    # after sending, one reminder covers every call tipping within 2 h (batching), not tomorrow's
    assert sch.keys_to_mark(now, calls, 120) == ["1846:g2"]
    assert sorted(sch.keys_to_mark(now, calls, 180)) == ["1054:g1", "1846:g2"]
