"""Alert email: email-safe HTML (tables + inline CSS, 560px, dark-mode block) and a plain-text mirror.
Data-driven: every section is skipped when it has nothing to show."""
from __future__ import annotations

from html import escape

FOOTER = "Reply 'locked name, name' to record locks."

PILL = {  # tag -> (background, text colour, label prefix)
    "LOCKED": ("#dcfce7", "#166534", "\U0001F512 "),
    "LOCK NOW": ("#16a34a", "#ffffff", ""),
    "RIDE": ("#f59e0b", "#ffffff", ""),
    "CAN'T LOCK": ("#e5e7eb", "#374151", ""),
    "BENCH": ("#f3f4f6", "#4b5563", ""),
}
GROUPS = [("LOCKED", "Locked"), ("OPEN", "Still open"), ("BENCH", "Bench"), ("IR", "IR")]


def _pp(x):
    return "—" if x is None else f"{x * 100:+.1f}pp"


def _pts(x):
    return "—" if x is None else f"{x:.1f}"


def split_waivers(ws):
    """Only moves that clear the configured thresholds are actions; at most 2 others as a watchlist."""
    ws = ws or []
    rec = [w for w in ws if w["recommend"]]
    score = lambda w: w["dplayoff"] * 100 + 0.5 * w["dpwin"] * 100      # playoff pp + half a this-week pp
    watch = sorted((w for w in ws if not w["recommend"] and score(w) > 0), key=lambda w: -score(w))[:2]
    return rec, watch


def watch_line(w):
    return f"{w['add']} ({w['dpwin'] * 100:+.1f}pp win, {w['dplayoff'] * 100:+.1f}pp playoffs; not worth a roster spot yet)"


def pill_tag(row):
    """Plain-text tag for a player row."""
    if row["pill"] == "LOCKED":
        return f"[LOCKED {_pts(row['pts'])}]"
    return f"[{row['pill']}]"


def top_action(ctx):
    for d in ctx.get("do_now") or []:
        last = d["player"].split()[-1]
        return f"Lock {last} by {d['before'].replace(' PM', '').replace(' AM', '')}"
    for m in ctx.get("move_cards") or []:
        return m["text"].split(":")[0].split(" (")[0].split(";")[0][:40]
    rec, _ = split_waivers(ctx.get("waivers"))
    if rec:
        return f"Add {rec[0]['add']}"
    return "no action"


def subject(ctx) -> str:
    return f"Sleeper | {ctx['date']} | {ctx['pwin'] * 100:.0f}% to win | {top_action(ctx)}"


def _mode_label(ctx):
    return {"pregame": "Pregame", "postgame": "Post-game"}.get(ctx.get("mode"), "Update")


# ------------------------------------------------------------------------------------------- text
def text(ctx) -> str:
    s = ctx["snapshot"]
    L = [f"WEEK {s['week']} · {ctx['date']} · {_mode_label(ctx).upper()}   vs {s['opp']}",
         f"{ctx['pwin'] * 100:.0f}% chance you win this week",
         f"Live {s['my_score']:.1f} - {s['opp_score']:.1f}   Projected final {s['my_ev']:.1f} – {s['opp_ev']:.1f} ({s['margin']:+.1f})", ""]
    L += ["DO THIS NOW"]
    if ctx.get("do_now"):
        for d in ctx["do_now"]:
            L.append(f"  [LOCK NOW] {d['player']} at {d['x']:.1f}  before {d['before']} tip  (riding {d.get('ride_games', 'his remaining games')} is worth ~{d['next_value']:.1f}, {_pp(d['dpwin'])})")
            if d.get("proof"):
                L.append(f"        {d['proof']}")
    else:
        L.append("  Nothing to tap right now.")
    L += ["", "LINEUP MOVES"]
    if ctx.get("move_cards"):
        for m in ctx["move_cards"]:
            L.append(f"  MOVE: {m['text']}" + (f"  ({_pp(m['dpwin'])})" if m.get("dpwin") is not None else ""))
            if m.get("proof"):
                L.append(f"        {m['proof']}")
            if m.get("also_text"):
                L.append(f"        {m['also_text']}")
    else:
        L.append("  KEEP: No swaps today. Leave the lineup as it is.")
        if ctx.get("best_alternative"):
            L.append(f"        {ctx['best_alternative']['text']}")
    for d in ctx.get("dont") or []:
        L.append(f"  DON'T: {d}")
    rows = ctx.get("players") or []
    if rows:
        L += ["", "WHERE EVERY PLAYER STANDS"]
        for g, title in GROUPS:
            grp = [r for r in rows if r["group"] == g]
            if not grp:
                continue
            head = title.upper()
            if g == "LOCKED":
                head += f" ({len(grp)} players · {sum(r['pts'] or 0 for r in grp):.1f} banked)"
            L.append(f"  {head}")
            for r in grp:
                L.append(f"    {r['slot']:<5} {r['name']:<24} {pill_tag(r):<14} {r['reason']}")
                if r.get("proof"):
                    L.append(f"          {r['proof']}")
    if ctx.get("news"):
        L += ["", "NEWS THAT MOVED THE ODDS"] + [f"  - {n}" for n in ctx["news"]]
    if ctx.get("waivers") is not None or ctx.get("trades") is not None or ctx.get("playoff") is not None:
        L += ["", "WAIVERS & TRADES"]
        rec, watch = split_waivers(ctx.get("waivers"))
        for w in rec:
            L.append(f"  +{w['add']} / -{w['drop']}  bid ${w['bid']}{' (waivers)' if w['waivers'] else ' (instant FA)'}  "
                     f"P(win) {_pp(w['dpwin'])}  playoffs {_pp(w['dplayoff'])}")
        if ctx.get("waivers") is not None and not rec:
            L.append("  No moves. Best available adds are below the bar.")
        for w in watch:
            L.append(f"  Watch: {watch_line(w)}")
        for t in ctx.get("trades") or []:
            L.append(f"  Trade{' *' if t['flag'] else ''} with {t['team']}: give {t['give']} | get {t['get']}  "
                     f"my playoffs {_pp(t['dme'])}, theirs {_pp(t['dthem'])}")
        if ctx.get("playoff") is not None:
            L.append(f"  Playoff odds: {ctx['playoff'] * 100:.1f}%")
    if ctx.get("assumptions"):
        L += ["", "ASSUMPTIONS & DATA"] + [f"  - {a}" for a in ctx["assumptions"]]
    L += ["", f"{FOOTER}  Locks read from {ctx.get('lock_source', 'inferred')} at {ctx.get('read_at', '?')} · {ctx.get('n_sims', 0):,} sims."]
    return "\n".join(L)


# ------------------------------------------------------------------------------------------- html
CARD = 'style="background:#ffffff;border:1px solid #e3e6ec;border-radius:12px;padding:14px 16px"'
H2 = 'style="font-size:13px;letter-spacing:.06em;text-transform:uppercase;color:#6b7280;margin:0 0 8px;font-weight:700"'


def _pill_html(row):
    bg, fg, pre = PILL.get(row["pill"], PILL["BENCH"])
    label = f"{pre}{_pts(row['pts'])}" if row["pill"] == "LOCKED" else (
        f"{row['pill']} {_pts(row['pts'])}" if row["pts"] is not None else row["pill"])
    return (f'<span style="display:inline-block;background:{bg};color:{fg};border-radius:999px;padding:3px 9px;'
            f'font-size:12px;font-weight:700;white-space:nowrap">{escape(label)}</span>')


def _action_card(kind, title, sub, pp=None):
    colours = {"LOCK": ("#f0fdf4", "#16a34a"), "MOVE": ("#eff6ff", "#2563eb"), "KEEP": ("#f9fafb", "#6b7280"),
               "DON'T": ("#fef2f2", "#dc2626")}
    bg, bd = colours[kind]
    right = f'<td align="right" style="font-weight:700;color:{bd};font-size:14px;white-space:nowrap;padding-right:12px">{escape(_pp(pp))}</td>' if pp is not None else ""
    return (f'<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="background:{bg};border-left:4px solid {bd};'
            f'border-radius:8px;margin:0 0 8px"><tr><td style="padding:10px 12px">'
            f'<div style="font-size:11px;font-weight:800;color:{bd};letter-spacing:.06em">{escape(kind)}</div>'
            f'<div style="font-size:15px;font-weight:600;color:#111827;margin-top:2px">{escape(title)}</div>'
            + (f'<div style="font-size:13px;color:#4b5563;margin-top:2px">{escape(sub)}</div>' if sub else "") +
            f'</td>{right}</tr></table>')


def html(ctx) -> str:
    s = ctx["snapshot"]
    pw = max(0.0, min(1.0, ctx["pwin"]))
    out = ['<!doctype html><html><head><meta name="viewport" content="width=device-width,initial-scale=1">'
           '<meta name="color-scheme" content="light dark"><style>'
           '@media (max-width:480px){.wrap{padding:8px!important}.card{padding:12px!important}.big{font-size:40px!important}}'
           '@media (prefers-color-scheme:dark){body,.bg{background:#0f1115!important}.card{background:#171a21!important;border-color:#2a2f3a!important}'
           '.t{color:#e5e7eb!important}.m{color:#9ca3af!important}}'
           '</style></head><body style="margin:0;background:#f2f3f6" class="bg">'
           '<table role="presentation" width="100%" cellspacing="0" cellpadding="0" class="bg" style="background:#f2f3f6"><tr><td align="center" class="wrap" style="padding:16px">'
           '<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="max-width:560px;font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif">']
    # 1) header
    out.append(
        '<tr><td style="background:#166534;background-image:linear-gradient(135deg,#166534,#0f3d24);border-radius:14px;padding:18px 18px 16px;color:#ffffff">'
        '<table role="presentation" width="100%"><tr>'
        f'<td style="font-size:12px;opacity:.85;color:#ffffff">Week {s["week"]} · {escape(ctx["date"])} · {_mode_label(ctx)}</td>'
        f'<td align="right" style="font-size:12px;opacity:.85;color:#ffffff">vs {escape(s["opp"])}</td></tr></table>'
        f'<div class="big" style="font-size:48px;font-weight:800;line-height:1.05;margin-top:8px">{pw * 100:.0f}%</div>'
        '<div style="font-size:13px;opacity:.9">chance you win this week</div>'
        f'<div style="font-size:14px;margin-top:10px">Live {s["my_score"]:.1f} – {s["opp_score"]:.1f} · Projected final {s["my_ev"]:.1f} – {s["opp_ev"]:.1f} ({s["margin"]:+.1f})</div>'
        '<table role="presentation" width="100%" cellspacing="0" cellpadding="0" style="margin-top:10px;border-radius:6px;overflow:hidden"><tr>'
        f'<td width="{max(1, round(pw * 100))}%" style="background:#4ade80;height:8px;font-size:0;line-height:0">&nbsp;</td>'
        f'<td style="background:#14532d;height:8px;font-size:0;line-height:0">&nbsp;</td></tr></table>'
        '</td></tr>')

    def card(title, inner):
        out.append(f'<tr><td style="height:12px"></td></tr><tr><td class="card" {CARD}><div {H2} class="m">{escape(title)}</div>{inner}</td></tr>')

    # 2) do this now
    if ctx.get("do_now"):
        inner = "".join(_action_card("LOCK", f"{d['player']} at {d['x']:.1f}",
                                     f"Before {d['before']} tip · riding {d.get('ride_games', 'his remaining games')} is worth ~{d['next_value']:.1f}"
                                     + (f" — {d['proof']}" if d.get("proof") else ""), d["dpwin"])
                        for d in ctx["do_now"])
    else:
        inner = '<div class="t" style="font-size:14px;color:#374151">Nothing to tap right now.</div>'
    card("Do this now", inner)
    # 3) lineup moves
    if ctx.get("move_cards"):
        inner = "".join(_action_card("MOVE", m["text"], " — ".join(x for x in (m.get("proof"), m.get("also_text")) if x), m.get("dpwin"))
                        for m in ctx["move_cards"])
    else:
        inner = _action_card("KEEP", "No swaps today. Leave the lineup as it is.",
                             (ctx.get("best_alternative") or {}).get("text", ""))
    inner += "".join(_action_card("DON'T", d, "") for d in ctx.get("dont") or [])
    card("Lineup moves", inner)
    # 4) where every player stands
    rows = ctx.get("players") or []
    if rows:
        legend = " ".join(_pill_html(dict(pill=k, pts=(43.0 if k == "LOCKED" else None))) for k in PILL)
        inner = f'<div style="margin:0 0 10px;line-height:2">{legend}</div>'
        for g, title in GROUPS:
            grp = [r for r in rows if r["group"] == g]
            if not grp:
                continue
            head = escape(title)
            if g == "LOCKED":
                head += f' <span style="font-weight:400;color:#6b7280">({len(grp)} players · {sum(r["pts"] or 0 for r in grp):.1f} banked)</span>'
            inner += f'<div class="t" style="font-size:13px;font-weight:700;color:#111827;margin:10px 0 4px">{head}</div>'
            inner += '<table role="presentation" width="100%" cellspacing="0" cellpadding="0">'
            for r in grp:
                inner += ('<tr>'
                          f'<td class="m" style="width:44px;font-size:12px;color:#6b7280;padding:7px 0;border-top:1px solid #eef0f3;vertical-align:top">{escape(r["slot"])}</td>'
                          f'<td style="padding:7px 6px;border-top:1px solid #eef0f3"><div class="t" style="font-size:14px;font-weight:600;color:#111827">{escape(r["name"])}</div>'
                          f'<div class="m" style="font-size:12px;color:#6b7280">{escape(r["reason"])}</div>'
                          + (f'<div class="m" style="font-size:11px;color:#9ca3af">{escape(r["proof"])}</div>' if r.get("proof") else "") + '</td>'
                          f'<td align="right" style="padding:7px 0;border-top:1px solid #eef0f3;vertical-align:top">{_pill_html(r)}</td></tr>')
            inner += "</table>"
        card("Where every player stands", inner)
    # 5) news, waivers & trades
    if ctx.get("news"):
        card("News that moved the odds", "".join(f'<div class="t" style="font-size:13px;color:#374151;margin:0 0 6px">{escape(n)}</div>' for n in ctx["news"]))
    if ctx.get("waivers") is not None or ctx.get("trades") is not None or ctx.get("playoff") is not None:
        rec, watch = split_waivers(ctx.get("waivers"))
        inner = "".join(_action_card("MOVE", f"Add {w['add']}, drop {w['drop']}",
                                     f"Bid ${w['bid']}{' (waivers)' if w['waivers'] else ' (instant free agent)'} · playoffs {_pp(w['dplayoff'])}", w["dpwin"]) for w in rec)
        if ctx.get("waivers") is not None and not rec:
            inner += '<div class="t" style="font-size:14px;color:#374151">No moves. Best available adds are below the bar.</div>'
        inner += "".join(f'<div class="m" style="font-size:12px;color:#9ca3af;margin-top:4px">Watch: {escape(watch_line(w))}</div>' for w in watch)
        for t in ctx.get("trades") or []:
            inner += (f'<div class="t" style="font-size:13px;color:#374151;margin-top:8px">{"★ " if t["flag"] else ""}Trade with {escape(t["team"])}: '
                      f'give {escape(t["give"])}, get {escape(t["get"])} · mine {_pp(t["dme"])}, theirs {_pp(t["dthem"])}</div>')
        if ctx.get("playoff") is not None:
            inner += f'<div class="m" style="font-size:12px;color:#6b7280;margin-top:8px">Playoff odds: {ctx["playoff"] * 100:.1f}%</div>'
        card("Waivers & trades", inner)
    if ctx.get("assumptions"):
        card("Assumptions & data", '<ul style="margin:0;padding-left:18px">' + "".join(
            f'<li class="m" style="font-size:11px;color:#6b7280;margin:0 0 3px">{escape(a)}</li>' for a in ctx["assumptions"]) + "</ul>")
    # 6) footer
    out.append(f'<tr><td class="m" style="font-size:12px;color:#6b7280;text-align:center;padding:14px 8px 18px">{escape(FOOTER)}<br>'
               f'Locks read from {escape(ctx.get("lock_source", "inferred"))} at {escape(ctx.get("read_at", "?"))} · {ctx.get("n_sims", 0):,} sims</td></tr>')
    out.append("</table></td></tr></table></body></html>")
    return "".join(out)
