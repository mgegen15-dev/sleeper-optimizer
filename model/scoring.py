"""Fantasy points for one stat line, using the league's scoring_settings (read from the API at runtime).

Sleeper's per-game lines already carry the bonus flags (dd, td, bonus_pt_40p, ...) when earned
(verified on 6,033 player-games, 0 mismatches). If a flag is missing we derive it from the raw
box score so the function also works on lines from other sources.
FP = sum_k  weight_k * stat_k     (a triple-double line has dd=1 AND td=1, so it earns both)
"""
from __future__ import annotations

BONUS_KEYS = ("dd", "td", "bonus_pt_40p", "bonus_pt_50p", "bonus_reb_20p", "bonus_ast_15p")


def derive_bonuses(st: dict) -> dict:
    g = lambda k: st.get(k) or 0
    tens = sum(1 for k in ("pts", "reb", "ast", "stl", "blk") if g(k) >= 10)
    return {
        "dd": float(tens >= 2), "td": float(tens >= 3),
        "bonus_pt_40p": float(g("pts") >= 40), "bonus_pt_50p": float(g("pts") >= 50),
        "bonus_reb_20p": float(g("reb") >= 20), "bonus_ast_15p": float(g("ast") >= 15),
    }


def score_line(stats: dict | None, scoring: dict) -> float:
    st = dict(stats or {})
    der = derive_bonuses(st)
    for k in BONUS_KEYS:
        if k not in st or st[k] is None:
            st[k] = der[k]
    return round(sum(w * (st.get(k) or 0) for k, w in scoring.items()), 2)


def played(stats: dict | None) -> bool:
    """A real appearance: some seconds/minutes played."""
    return bool(stats) and ((stats.get("sp") or 0) > 0 or (stats.get("min") or 0) > 0)
