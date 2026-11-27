"""Build the alert email: plain text + mobile-friendly HTML (single column, inline styles)."""
from __future__ import annotations

from html import escape


FOOTER = "Reply 'locked name, name' to record locks."


def split_waivers(ws):
    """Only moves that clear the configured thresholds are actions; at most 2 others as a watchlist."""
    ws = ws or []
    rec = [w for w in ws if w["recommend"]]
    score = lambda w: w["dplayoff"] * 100 + 0.5 * w["dpwin"] * 100      # playoff pp + half a this-week pp
    watch = sorted((w for w in ws if not w["recommend"] and score(w) > 0), key=lambda w: -score(w))[:2]
    return rec, watch


def watch_line(w):
    return f"{w['add']} ({w['dpwin'] * 100:+.1f}pp win, {w['dplayoff'] * 100:+.1f}pp playoffs; not worth a roster spot yet)"


def subject(ctx) -> str:
    return f"Sleeper | {ctx['date']} | P(win) {ctx['pwin'] * 100:.0f}% | {ctx['n_actions']} actions"


def _pp(x):
    return "—" if x is None else f"{x * 100:+.1f}pp"


def text(ctx) -> str:
    L = []
    s = ctx["snapshot"]
    L += ["MATCHUP SNAPSHOT",
          f"Week {s['week']}: {s['me']} {s['my_score']:.1f}  vs  {s['opp']} ({s['opp_owner']}) {s['opp_score']:.1f}",
          f"P(win) {ctx['pwin'] * 100:.1f}%  |  projected final {s['my_ev']:.1f} - {s['opp_ev']:.1f}  "
          f"(margin {s['margin']:+.1f}, 80% range {s['margin_p10']:+.0f} to {s['margin_p90']:+.0f})", ""]
    L += [f"TONIGHT'S LINEUP ({ctx['lineup_date']})"]
    for r in ctx["lineup"]:
        L.append(f"  {r['slot']:<5} {r['player']:<24} {r['game']:<10} {r['tip']:<9} {r['lock']}" + (f"  [{r['note']}]" if r['note'] else ""))
    if ctx.get("moves"):
        L += ["  Moves: " + "; ".join(ctx["moves"])]
    L += ["  Locked (reported/inferred): " + ("; ".join(ctx.get("locked_list") or []) or "none")]
    L += [""]
    L += ["LOCK/RIDE CALLS"]
    if ctx["locks"]:
        for d in ctx["locks"]:
            thr = f"vs threshold {d['threshold']:.1f}" if d.get("threshold") is not None else "(not lockable)"
            pws = " / ".join(f"{k} {v * 100:.1f}%" for k, v in (("lock", d.get("pw_lock")), ("ride", d.get("pw_hold")), ("bench", d.get("pw_bench"))) if v is not None)
            L.append(f"  {d['decision']:<5} {d['player']:<24} {d['slot']:<5} score {d['x']:.1f} {thr}"
                     f"  P(win) {pws} ({_pp(d['dpwin'])})  decide before {d['deadline']}" + (f"  [{d['note']}]" if d.get("note") else ""))
    else:
        L.append("  None open.")
    L += [""]
    if ctx.get("waivers") is not None:
        L += ["WAIVER ADDS"]
        rec, watch = split_waivers(ctx["waivers"])
        for w in rec:
            L.append(f"  +{w['add']} / -{w['drop']}  bid ${w['bid']}{' (waivers)' if w['waivers'] else ' (instant FA)'}  "
                     f"P(win) {_pp(w['dpwin'])}  playoffs {_pp(w['dplayoff'])}")
        if not rec:
            L.append("  No moves. Best available adds are below the bar.")
        for w in watch:
            L.append(f"  Watch: {watch_line(w)}")
        L += [""]
    if ctx.get("trades") is not None:
        L += ["TRADE IDEAS"]
        if ctx["trades"]:
            for t in ctx["trades"]:
                L.append(f"  {'*' if t['flag'] else ' '} with {t['team']}: give {t['give']} | get {t['get']}  "
                         f"my playoffs {_pp(t['dme'])}, theirs {_pp(t['dthem'])}")
        else:
            L.append("  Nothing clears the bar.")
        L += [""]
    L += ["ASSUMPTIONS"] + [f"  - {a}" for a in ctx["assumptions"]]
    return "\n".join(L)


def html(ctx) -> str:
    s = ctx["snapshot"]
    td = 'style="padding:4px 6px;border-bottom:1px solid #eee;font-size:13px;vertical-align:top"'
    th = 'style="padding:4px 6px;text-align:left;font-size:12px;color:#666;border-bottom:1px solid #ccc"'

    def table(head, rows):
        h = "".join(f"<th {th}>{escape(x)}</th>" for x in head)
        b = "".join("<tr>" + "".join(f"<td {td}>{escape(str(c))}</td>" for c in r) + "</tr>" for r in rows)
        return f'<table style="border-collapse:collapse;width:100%">{"<tr>" + h + "</tr>"}{b}</table>'

    def sec(t):
        return f'<h3 style="margin:18px 0 6px;font-size:15px;color:#222">{escape(t)}</h3>'

    parts = [f'<div style="font-family:-apple-system,Segoe UI,Roboto,Arial,sans-serif;max-width:640px;margin:auto;color:#222">',
             sec("Matchup snapshot"),
             f'<div style="font-size:22px;font-weight:600">P(win) {ctx["pwin"] * 100:.1f}%</div>',
             f'<div style="font-size:14px">{escape(s["me"])} <b>{s["my_score"]:.1f}</b> vs {escape(s["opp"])} <b>{s["opp_score"]:.1f}</b><br>'
             f'Projected final {s["my_ev"]:.1f} – {s["opp_ev"]:.1f} (margin {s["margin"]:+.1f}; 80% range {s["margin_p10"]:+.0f} to {s["margin_p90"]:+.0f})</div>',
             sec(f"Tonight's lineup ({ctx['lineup_date']})"),
             table(["Slot", "Player", "Game", "Tip CT", "Lock if ≥"], [[r["slot"], r["player"] + (f" ({r['note']})" if r["note"] else ""), r["game"], r["tip"], r["lock"]] for r in ctx["lineup"]])]
    if ctx.get("moves"):
        parts.append(f'<p style="font-size:13px"><b>Moves:</b> {escape("; ".join(ctx["moves"]))}</p>')
    parts.append(f'<p style="font-size:13px"><b>Locked (reported/inferred):</b> {escape("; ".join(ctx.get("locked_list") or []) or "none")}</p>')
    parts.append(sec("Lock / ride calls"))
    if ctx["locks"]:
        parts.append(table(["Call", "Player", "Score", "Threshold", "ΔP(win)", "Before"],
                           [[d["decision"], d["player"] + (f" ({d['note']})" if d.get("note") else ""), f"{d['x']:.1f}",
                             f"{d['threshold']:.1f}" if d.get("threshold") is not None else "not lockable", _pp(d["dpwin"]), d["deadline"]] for d in ctx["locks"]]))
    else:
        parts.append('<p style="font-size:13px">None open.</p>')
    if ctx.get("waivers") is not None:
        parts.append(sec("Waiver adds"))
        rec, watch = split_waivers(ctx["waivers"])
        parts.append(table(["Add", "Drop", "Bid", "ΔP(win)", "ΔPlayoffs"],
                           [[w["add"], w["drop"], f"${w['bid']}" + (" W" if w["waivers"] else " FA"), _pp(w["dpwin"]), _pp(w["dplayoff"])]
                            for w in rec]) if rec else '<p style="font-size:13px">No moves. Best available adds are below the bar.</p>')
        for w in watch:
            parts.append(f'<p style="font-size:12px;color:#888;margin:4px 0">Watch: {escape(watch_line(w))}</p>')
    if ctx.get("trades") is not None:
        parts.append(sec("Trade ideas"))
        parts.append(table(["Team", "Give", "Get", "Mine", "Theirs"],
                           [[("★ " if t["flag"] else "") + t["team"], t["give"], t["get"], _pp(t["dme"]), _pp(t["dthem"])] for t in ctx["trades"]])
                     if ctx["trades"] else '<p style="font-size:13px">Nothing clears the bar.</p>')
    parts.append(sec("Assumptions"))
    parts.append('<ul style="font-size:12px;color:#555;padding-left:18px">' + "".join(f"<li>{escape(a)}</li>" for a in ctx["assumptions"]) + "</ul>"
                 f'<p style="font-size:13px;color:#333;border-top:1px solid #ddd;padding-top:8px">{escape(FOOTER)}</p></div>')
    return "".join(parts)
