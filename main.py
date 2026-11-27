"""Sleeper NBA decision engine.

  python main.py --mode pregame|postgame|auto [--dry-run]
  python main.py --backtest --weeks 1-5
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import re
import traceback

QUIET = False   # CI / --quiet: never print league data (public Actions logs)


def scrub(msg: str) -> str:
    msg = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<email>", str(msg))
    return re.sub(r"eyJ[A-Za-z0-9_.-]+", "<token>", msg)[:200]
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from data.loader import load  # noqa: E402
from model.distributions import ModelBook  # noqa: E402
from optimize import backtest as bt  # noqa: E402
from optimize import trades as tr  # noqa: E402
from optimize import waivers as wv  # noqa: E402
from optimize.montecarlo import Draws, p_win, solve_team  # noqa: E402
from optimize.season import SeasonSim  # noqa: E402
from report import email as em  # noqa: E402
from data import locks as lk  # noqa: E402


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_state(cfg):
    p = ROOT / cfg["state_file"]
    try:
        return json.loads(p.read_text(encoding="utf-8")) or {}
    except Exception:
        return {}


def save_state(cfg, st):
    p = ROOT / cfg["state_file"]
    p.parent.mkdir(parents=True, exist_ok=True)
    st["pregame_sent"] = st.get("pregame_sent", [])[-30:]
    st["finals_alerted"] = st.get("finals_alerted", [])[-300:]
    st["lock_alerts"] = st.get("lock_alerts", [])[-300:]
    p.write_text(json.dumps(st, indent=1), encoding="utf-8")


def tip_str(g):
    return g.tip.strftime("%a %-I:%M%p") if g and g.tip and sys.platform != "win32" else (g.tip.strftime("%a %#I:%M%p") if g and g.tip else "?")


def game_str(g):
    return ("vs " if g.home else "@") + g.opp if g else ""


# ---------------------------------------------------------------------------------------- engine
def analyse(snap, cfg, mode, timings, state=None):
    sims = cfg["sims"]
    rid = cfg["roster_id"]
    W = snap.week
    wdates = {g.date for gs in snap.team_games.values() for g in gs if g.week == W}
    book = ModelBook(snap.logs, snap.players, snap.proj, cfg, proj_dates=wdates)
    from data import news as nw
    from data.sleeper import SleeperClient
    news_client = SleeperClient(cfg["cache_dir"], cfg["http"]["timeout_s"], cfg["http"]["retries"], offline=getattr(snap, "offline", False))
    snap.failures += nw.apply(snap, cfg, news_client, book)
    from model import news_llm
    llm_status, llm_notes = "skipped", []
    if mode == "pregame" and cfg.get("news_llm", {}).get("enabled"):
        try:
            feed = news_client.get(news_llm.NEWS_URL, "espn/news.json", max_age_h=0.25)
            llm_status, llm_notes = news_llm.run(snap, cfg, book, state if state is not None else {}, mode, news_feed=feed)
        except Exception as e:
            llm_status, llm_notes = "failed", [f"AI news step failed ({type(e).__name__}); rule-based news only."]
    timings["news_llm"] = llm_status
    if llm_status == "ok":
        timings["news_llm_adjusted"] = len(llm_notes)
        u = getattr(book, "news_llm_usage", None)
        if u and not QUIET:
            print(f"news_llm usage: {u['input_tokens']} in / {u['output_tokens']} out tokens, ~${u['usd']}")
    else:
        snap.lock_notes = list(getattr(snap, "lock_notes", [])) + llm_notes   # shown under Assumptions
    draws = Draws(book, sims["n_week"], sims["seed"])
    t = time.time()
    opp_rid = snap.opponent_rid(rid)
    teams = {}
    for r in snap.rosters:
        if r["roster_id"] != rid:
            teams[r["roster_id"]] = solve_team(snap, r["roster_id"], book, draws)
    opp = teams[opp_rid]
    me = solve_team(snap, rid, book, draws, vs=opp.totals)
    teams[rid] = me
    timings["week_engine_s"] = round(time.time() - t, 1)
    pw = p_win(me.totals, opp.totals)
    margin = me.totals - opp.totals
    m_me, m_opp = snap.matchup(rid), snap.matchup(opp_rid)

    # ---------------- tonight's lineup
    today = snap.now.date().isoformat()
    later_today = sorted(d for d in wdates if d >= today)
    lineup_date = later_today[0] if later_today else today
    assign = me.plan.day_assign.get(lineup_date, {})
    plays_on = lambda pid, d: any(g.date == d and not g.started for g in snap.week_games(snap.players.get(pid, {}).get("team")))
    lineup, moves, tonight, changed = [], [], set(), []
    used = set()
    for sp, stt in zip(me.plan.slots, me.states):
        row = dict(slot=sp.slot, player="—", game="", tip="", lock="", note="")
        if sp.frozen is not None:
            row.update(player=snap.pname(stt.pid), lock="—",
                       note=(f"LOCKED {sp.frozen:.1f} ({stt.lock_source or 'inferred'})" if stt.locked else f"LOCK NOW at {sp.frozen:.1f}"))
            used.add(stt.pid)
        elif sp.keep:
            row.update(player=snap.pname(sp.option_pid), note=f"keep {sp.option:.1f}", lock="—")
            used.add(sp.option_pid)
            if plays_on(sp.option_pid, lineup_date):
                row["note"] += " (has game: LOCK it)"
        elif sp.index in assign:
            pid = assign[sp.index]
            step = next(s for s in sp.seq if s.pid == pid and s.game.date == lineup_date)
            row.update(player=snap.pname(pid), game=game_str(step.game), tip=tip_str(step.game),
                       lock="last game" if step.threshold is None else f"{step.threshold:.1f}")
            used.add(pid)
            tonight.add(pid)
        else:
            # nobody should play in this slot tonight: park a player without a game tonight
            nxt = next((s.pid for s in sp.seq if not plays_on(s.pid, lineup_date)), None)
            if nxt is None and stt.pid and not plays_on(stt.pid, lineup_date):
                nxt = stt.pid
            row.update(player=(snap.pname(nxt) if nxt else "any player w/o game tonight"), note="no game tonight", lock="—")
            if nxt:
                used.add(nxt)
        new_pid = assign.get(sp.index)
        if new_pid and new_pid != stt.pid:
            changed.append((sp.slot, new_pid, stt))
        lineup.append(row)
    current = {s.pid for s in me.states if s.pid}
    in_lineup = used | tonight
    for slot, pid, stt in changed:       # net changes only: a swap between two UTIL slots isn't a move
        if pid not in current:
            moves.append(f"start {snap.pname(pid)} at {slot}")
        elif not (slot == "UTIL" and next(s.slot for s in me.states if s.pid == pid) == "UTIL"):
            moves.append(f"move {snap.pname(pid)} to {slot}")
    frozen_idx = {sp.index for sp in me.plan.slots if sp.frozen is not None}
    for s in me.states:                  # anyone leaving the lineup: say what it costs
        if not s.pid or s.locked or s.index in frozen_idx or s.pid in in_lineup:
            continue
        cost = f": benching him forfeits his {s.x:.1f}" if s.x is not None else ""
        if s.x is None and not plays_on(s.pid, lineup_date):
            continue
        moves.append(f"bench {snap.pname(s.pid)}" + (cost or " tonight (his game shouldn't count)"))

    locked_list = [f"{snap.pname(s.pid)} {s.x:.1f} ({s.lock_source or 'inferred'})" for s in me.states if s.locked]

    # ---------------- lock / ride calls
    locks = []
    LABEL = {"lock": "LOCK", "hold": "RIDE", "ride": "BENCH"}
    for d in me.decisions:
        pwmap = {"lock": d.pwin_lock, "hold": d.pwin_hold, "ride": d.pwin_bench}
        chosen = pwmap[d.choice]
        others = [v for k, v in pwmap.items() if k != d.choice and v is not None]
        dp = chosen - max(others) if others else 0.0
        note = "" if d.lockable else "not lockable (benched since that game began)"
        if d.choice == "ride":
            note = (note + "; " if note else "") + f"benching {snap.pname(d.pid)} forfeits his {d.x:.1f}"
        locks.append(dict(decision=LABEL[d.choice], player=snap.pname(d.pid), slot=d.slot, x=d.x,
                          threshold=d.threshold if d.lockable else None, pw_lock=d.pwin_lock, pw_hold=d.pwin_hold,
                          pw_bench=d.pwin_bench, pw_ride=d.pwin_ride, dpwin=dp, note=note, deadline=tip_str(d.next_game)))
    locks.sort(key=lambda d: -abs(d["dpwin"]))

    assumptions = []
    waivers = trades = None
    playoff = None
    if mode == "pregame":
        try:
            t = time.time()
            book_ros = ModelBook(snap.logs, snap.players, snap.proj, cfg, proj_dates=wdates, ros=True)
            wt = {k: v.totals for k, v in teams.items()}
            full = SeasonSim(snap, cfg, book_ros, wt, sims["n_season"], sims["seed"])
            base = full.run()[0]
            playoff = base[rid]
            timings["season_base_s"] = round(time.time() - t, 1)
            values = {str(p): tr.player_week_value(book_ros, snap, str(p)) for p in snap.roster(rid)["players"]}
            t = time.time()
            try:
                res = wv.evaluate(snap, cfg, book, book_ros, me, opp, full, base, tonight | used, values)
                waivers = [dict(w, add=snap.pname(w["add"]), drop=snap.pname(w["drop"])) for w in res]
            except Exception as e:
                assumptions.append(f"Waiver analysis failed: {type(e).__name__}: {e}")
            timings["waivers_s"] = round(time.time() - t, 1)
            t = time.time()
            try:
                screen = SeasonSim(snap, cfg, book_ros, wt, sims["n_season_screen"], sims["seed"] + 1)
                screen.mu_cache = full.mu_cache
                base_s = screen.run()[0]
                res, ncand = tr.search(snap, cfg, book_ros, screen, base_s, full, base, rid)
                trades = [dict(t_, team=snap.team_name(t_["rid"]), give=" + ".join(map(snap.pname, t_["give"])),
                               get=" + ".join(map(snap.pname, t_["get"]))) for t_ in res[: cfg["trades"]["show"]]]
                timings["trade_candidates_screened"] = ncand
            except Exception as e:
                assumptions.append(f"Trade search failed: {type(e).__name__}: {e}")
            timings["trades_s"] = round(time.time() - t, 1)
            sd = full.sd
            assumptions.append(f"Playoff odds now {playoff * 100:.1f}% (season sim {sims['n_season']:,} runs; team weekly SD {sd:.1f} from league history; "
                               f"top {cfg['season']['playoff_teams']}, division winners in, tiebreak record then PF).")
        except Exception as e:
            assumptions.append(f"Season sim failed: {type(e).__name__}: {e}")
            if not QUIET:
                traceback.print_exc()

    # ---------------- assumptions / freshness
    ip = cfg["model"]["injury_play_prob"]
    inj = [f"{snap.pname(s)} {snap.players[s].get('injury_status')}" for s in snap.roster(rid)["players"]
           if snap.players.get(s, {}).get("injury_status") and s not in (snap.roster(rid).get("reserve") or [])]
    assumptions += [
        f"P(play): healthy {ip['healthy']}, Questionable/GTD/DTD {ip['GTD']}, Doubtful {ip['Doubtful']}, Out/Sus 0. DNP = 0 pts.",
        "My active players with an injury tag: " + (", ".join(inj) if inj else "none"),
        ("Locks read from Sleeper (matchup_legs.starters_games, read-only)." if snap.api_locks is not None else
         "Locks are inferred (public API has no lock field). A counting score whose next game hasn't tipped is shown as an open call; if you already locked it, ignore the call."),
        f"Mean = blend recency (half-life {cfg['model']['half_life_games']} g) / season / Sleeper projection "
        f"({cfg['model']['blend_weights']['recency']}/{cfg['model']['blend_weights']['season']}/{cfg['model']['blend_weights']['projection']}); "
        f"opponent adj capped ±{cfg['model']['opp_adjust_cap'] * 100:.0f}%. {sims['n_week']:,} week sims, seed {sims['seed']}.",
        "Opponent modelled as best lineup + threshold lock policy (expected-points optimal).",
        f"Data as of {snap.now:%a %b %d %I:%M %p} CT. {sum(v == 'live' for v in snap.freshness.values())} feeds live; from cache by design: "
        + (", ".join(sorted({k.split('/')[0].replace('.json', '') for k, v in snap.freshness.items() if v == 'cache'})) or "none")
        + " (players <=24h, schedule <=15 min, completed weeks <=12h)"
        + ("; STALE fallback: " + ", ".join(k for k, v in snap.freshness.items() if v == 'stale-cache') if 'stale-cache' in snap.freshness.values() else "") + ".",
    ]
    assumptions += [s.conflict.strip() for s in me.states if s.conflict]
    mine_opp = [str(p) for p in snap.roster(rid)["players"] + snap.roster(opp_rid)["players"]]
    for pid in mine_opp:
        adj = getattr(book, "news", {}).get(pid)
        if adj:
            assumptions.append(f"News: {snap.pname(pid)} - " + "; ".join(adj.notes))
    assumptions += getattr(snap, "lock_notes", [])
    stale = [s for s in me.states if s.pending]
    if stale:
        assumptions.append("Stats not yet posted for: " + ", ".join(snap.pname(s.pid) for s in stale) + " (treated as unknown).")
    assumptions += [f"SOURCE FAILURE: {f}" for f in snap.failures]

    n_actions = len(moves) + sum(1 for d in locks if d["decision"] == "LOCK") + sum(1 for w in waivers or [] if w["recommend"])
    ctx = dict(date=snap.now.strftime("%a %b %d"), pwin=pw, n_actions=n_actions, lineup_date=lineup_date,
               snapshot=dict(week=W, me=snap.team_name(rid), opp=snap.team_name(opp_rid), opp_owner=snap.owner_name(opp_rid),
                             my_score=m_me["points"], opp_score=m_opp["points"], my_ev=me.ev, opp_ev=opp.ev,
                             margin=float(margin.mean()), margin_p10=float(np.percentile(margin, 10)),
                             margin_p90=float(np.percentile(margin, 90))),
               lineup=lineup, moves=moves, locks=locks, locked_list=locked_list, waivers=waivers, trades=trades, assumptions=assumptions,
               playoff=playoff)
    return ctx, me


# ---------------------------------------------------------------------------------------- auto mode
def open_calls(snap, cfg):
    """[(key, next_tip)] for my open lock/ride decisions (after reported/Sleeper locks are applied)."""
    from optimize.lock import infer_slots
    rid = cfg["roster_id"]
    out = []
    for s in infer_slots(snap, rid, reported=(snap.reported_locks or {}).get(rid)):
        if s.open_decision and s.next_game is not None:
            out.append((f"{s.pid}:{s.next_game.game_id}", s.next_game.tip))
    return out


def decide_auto(snap, cfg, state):
    """-> (mode, finals_to_mark, label). See optimize/schedule.py for the rules."""
    from optimize import schedule as sch
    sc = cfg["schedule"]
    today = snap.now.date().isoformat()
    games_today = [g for gs in snap.team_games.values() for g in gs if g.date == today]
    tips = [g.tip for g in games_today]
    rid = cfg["roster_id"]
    opp = snap.opponent_rid(rid)
    mine_opp_teams = {snap.players.get(str(p), {}).get("team") for r in (rid, opp) if r for p in snap.roster(r)["players"]} - {None}
    my_tips = [g.tip for g in games_today if g.team in mine_opp_teams and not g.started]
    send, label, _ = sch.pregame_decision(snap.now, tips, my_tips, today in state.get("pregame_sent", []),
                                          sc["pregame_lead_min"], sc["cron_interval_min"], sc["fallback_first_tip"],
                                          have_games=bool(games_today))
    if send:
        return "pregame", [], label
    # lock window: an open call whose next game tips soon and hasn't been flagged yet
    calls = open_calls(snap, cfg)
    if sch.lock_window_due(snap.now, calls, set(state.get("lock_alerts", [])), sc["lock_window_min"]):
        return "postgame", [], "lock-window"
    r = snap.roster(rid)
    active = [p for p in r["players"] if p not in (r.get("reserve") or [])]
    teams = {snap.players.get(p, {}).get("team") for p in active} - {None}
    yday = (snap.now.date() - timedelta(days=1)).isoformat()
    new_finals, running = [], []
    for t in teams:
        for g in snap.week_games(t):
            if g.date in (today, yday) and g.final and g.game_id not in state.get("finals_alerted", []):
                if (t, g.date) in snap.stat_dates:     # wait until Sleeper has posted the stat lines
                    new_finals.append(g.game_id)
            elif g.started and not g.final and g.tip:
                running.append(g)
    if new_finals:
        # batch: if another of my games should finish within ~25 min, wait for it
        soon = [g for g in running if 0 < (g.tip + timedelta(minutes=150) - snap.now).total_seconds() / 60 < 25]
        if soon:
            return None, [], None
        return "postgame", sorted(set(new_finals)), None
    return None, [], None


def ingest_locks(snap, cfg, a, persist=True):
    """Read lock reports (Gmail replies + --locked) into state/locks.json and onto the snapshot."""
    import os
    path = ROOT / cfg.get("locks", {}).get("file", "state/locks.json")
    store = lk.load_locks(path)
    inbox = None
    user, pwd = os.environ.get(cfg["email"]["user_env"]), os.environ.get(cfg["email"]["password_env"])
    notes = []
    if a.no_imap:
        pass
    elif user and pwd:
        inbox = lk.GmailInbox(user, pwd, cfg.get("locks", {}).get("imap_host", "imap.gmail.com"))
    else:
        notes.append("Lock replies not checked (GMAIL_USER / GMAIL_APP_PASSWORD not set); using stored + inferred locks.")
    notes += lk.ingest(snap, cfg, store, inbox=inbox, manual=a.locked or os.environ.get("LOCKED_INPUT"))
    if persist:
        lk.save_locks(path, store)
    snap.reported_locks = {cfg["roster_id"]: store["weeks"].get(str(snap.week), {})}
    snap.lock_store = store
    snap.lock_notes = notes
    return notes


# ---------------------------------------------------------------------------------------- main
def main(argv=None):
    global QUIET
    QUIET = "--quiet" in (argv if argv is not None else sys.argv[1:]) or os.environ.get("CI") == "true"
    try:
        return _main(argv)
    except SystemExit:
        raise
    except Exception as e:                     # never dump data into public logs
        if not QUIET:
            raise
        tb = traceback.extract_tb(e.__traceback__)
        where = f"{Path(tb[-1].filename).stem}.{tb[-1].name}" if tb else "?"
        print(f"failed: {type(e).__name__} in {where}: {scrub(e)}")
        return 1


def _main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["pregame", "postgame", "auto"], default="auto")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--backtest", action="store_true")
    ap.add_argument("--weeks", default="1-5")
    ap.add_argument("--config", default=str(ROOT / "config.yaml"))
    ap.add_argument("--offline", action="store_true", help="use cached API data only")
    ap.add_argument("--now", help="override current time (ISO, Central) for testing")
    ap.add_argument("--locked", default="", help="comma-separated names you have locked (same as an email reply)")
    ap.add_argument("--no-imap", action="store_true", help="skip reading lock replies from Gmail")
    ap.add_argument("--quiet", action="store_true", help="CI mode: log only mode/result/timings, never league data")
    ap.add_argument("--test", action="store_true", help="test send: subject prefixed [TEST], nothing recorded in state")
    a = ap.parse_args(argv)
    cfg = load_cfg(a.config)
    import os
    os.chdir(ROOT)
    tz = ZoneInfo(cfg["timezone"])
    now = datetime.fromisoformat(a.now).replace(tzinfo=tz) if a.now else datetime.now(tz)
    t0 = time.time()
    timings = {}

    if a.backtest:
        lo, hi = (int(x) for x in a.weeks.split("-"))
        snap = load(cfg, now=now, offline=a.offline, extra_proj_weeks=range(lo, hi + 1))
        rows = bt.run(snap, cfg, range(lo, hi + 1), cfg["roster_id"])
        summ = bt.summarize(rows)
        print("week | model | actual | best-possible | starters compared | excluded (0 pts, not in slot)")
        for s in summ:
            print(f"{s['week']:>4} | {s['model']:6.1f} | {s['actual']:6.1f} | {s['oracle']:6.1f} | {s['n']} | {', '.join(s['excluded']) or '-'}")
        tm = sum(s["model"] for s in summ)
        ta = sum(s["actual"] for s in summ)
        print(f"TOTAL | {tm:6.1f} | {ta:6.1f} | {sum(s['oracle'] for s in summ):6.1f} | model - actual = {tm - ta:+.1f}")
        print("\nDecisions that differed (games in order; model pick vs your pick):")
        for s in summ:
            for r in s["diffs"]:
                print(f"  wk{r['week']} {r['name']:<24} games {r['games']} thresholds {r['thresholds']} -> "
                      f"model g{r['model_pick'] + 1}={r['model_pts']:.1f}, you {'g' + str(r['actual_pick'] + 1) if r['actual_pick'] is not None else 'final(DNP)'}={r['actual_pts']:.1f} "
                      f"({r['model_pts'] - r['actual_pts']:+.1f})")
        Path(cfg["cache_dir"], "backtest.json").write_text(json.dumps(rows, indent=1, default=str), encoding="utf-8")
        print(f"\nruntime {time.time() - t0:.1f}s")
        return 0

    state = load_state(cfg)
    snap = load(cfg, now=now, offline=a.offline)
    timings["load_s"] = round(time.time() - t0, 1)
    ingest_locks(snap, cfg, a, persist=not a.dry_run)
    from data import eligibility as elig
    elig.observe(snap, cfg, state)
    snap.lock_notes = list(getattr(snap, "lock_notes", [])) + elig.apply(snap, cfg, state, snap.lock_store)
    mode, finals, label = a.mode, [], None
    if mode == "auto":
        mode, finals, label = decide_auto(snap, cfg, state)
        if not mode:
            print(f"[auto] nothing to send ({time.time() - t0:.1f}s)" if QUIET else f"[{now:%Y-%m-%d %H:%M} CT] auto: nothing to send")
            if not a.dry_run:
                save_state(cfg, state)    # keeps lineup observations (lock eligibility); cached, not committed
            return 0
    ctx, me = analyse(snap, cfg, mode, timings, state)
    if mode == "postgame" and label != "lock-window" and not ctx["locks"] and not ctx["moves"]:
        print("[postgame] nothing to send (no open calls)")
        state.setdefault("finals_alerted", []).extend(finals)
        if not a.dry_run:
            save_state(cfg, state)
        return 0
    timings["total_s"] = round(time.time() - t0, 1)
    if QUIET and snap.failures:
        timings["source_issues"] = len(snap.failures)
    ctx["assumptions"].append("Runtime: " + ", ".join(f"{k} {v}" for k, v in timings.items()))
    subj, txt, html = em.subject(ctx), em.text(ctx), em.html(ctx)
    if label in ("late", "catch-up", "lock-window"):
        subj = subj.replace("Sleeper |", f"Sleeper ({label}) |", 1)
    if a.test:
        subj = "[TEST] " + subj
    Path(cfg["cache_dir"], "last_email.html").write_text(html, encoding="utf-8")
    Path(cfg["cache_dir"], "last_email.txt").write_text(subj + "\n\n" + txt, encoding="utf-8")
    if a.dry_run and QUIET:
        print(f"[{mode}] dry run: output suppressed in CI")
        return 0
    if a.dry_run:
        print("Subject:", subj)
        print()
        print(txt)
        return 0
    import os
    if not (os.environ.get(cfg["email"]["user_env"]) and os.environ.get(cfg["email"]["password_env"])):
        print(f"[{mode}] not sent: email secrets not set" if QUIET else f"would send: {subj} (not sent: {cfg['email']['user_env']} / {cfg['email']['password_env']} not set)")
        return 0          # quiet no-op until the secrets exist; nothing marked as sent
    from report.smtp import send
    send(cfg, subj, txt, html)
    if a.test:                      # a test send must not use up today's pregame slot or mark finals
        print(f"[{mode}] test email sent; " + ", ".join(f"{k} {v}" for k, v in timings.items()) if QUIET else f"test sent: {subj}")
        return 0
    if mode == "pregame":
        state.setdefault("pregame_sent", []).append(now.date().isoformat())
    state.setdefault("finals_alerted", []).extend(finals)
    from optimize import schedule as sch   # one reminder per tip cluster: mark calls this email covered
    horizon = cfg["schedule"]["lock_batch_min"] + (cfg["schedule"]["pregame_lead_min"] if mode == "pregame" else 0)
    state.setdefault("lock_alerts", []).extend(k for k in sch.keys_to_mark(now, open_calls(snap, cfg), horizon)
                                               if k not in state["lock_alerts"])
    state["last_run"] = now.isoformat()
    state["last_sent"] = {"at": now.isoformat(), "mode": mode, "subject": subj}
    save_state(cfg, state)
    print(f"[{mode}] sent; " + ", ".join(f"{k} {v}" for k, v in timings.items()) if QUIET else f"sent: {subj}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
