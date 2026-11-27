"""Per-player, per-game fantasy-point distributions.

Level (the mean, "mu"):
    mu = w_r * recency_mean + w_s * season_mean + w_p * sleeper_projection   (weights renormalised
    over the pieces that exist). recency_mean is an exponentially-weighted average of the last N
    games: weight_i = 0.5 ** (age_i / half_life), age 0 = most recent game.
    With fewer than `small_sample_gp` games the history part is shrunk toward the positional prior:
    hist = a * own + (1 - a) * prior,  a = GP / small_sample_gp   (credibility weighting).

Shape: the player's own game-to-game *ratios* r_i = x_i / season_mean (mean 1, keeps right skew),
    rescaled to mu. Small samples are mixed with the positional pool of ratios in the same a : 1-a
    proportion. No history at all -> gamma distribution with CV = fallback_cv.

Outcome of one game:  X = 0 with prob 1 - p_play (DNP), else mu * opp_factor * R, R ~ ratio sample.
"""
from __future__ import annotations

import zlib
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

GROUP = {"PG": "G", "SG": "G", "G": "G", "SF": "F", "PF": "F", "F": "F", "C": "C"}


def pos_group(p: dict) -> str:
    pos = p.get("position") or (p.get("fantasy_positions") or ["F"])[0]
    return GROUP.get(pos, "F")


def stable_seed(*parts) -> int:
    return zlib.crc32("|".join(map(str, parts)).encode())


@dataclass
class PlayerModel:
    pid: str
    mu: float
    R: np.ndarray          # outcome ratios, mean ~1
    p_play: float
    group: str
    gp: int
    parts: dict = field(default_factory=dict)

    def sample_values(self, factor: float = 1.0) -> np.ndarray:
        return self.mu * factor * self.R

    def expected(self, factor: float = 1.0) -> float:
        return self.p_play * self.mu * factor * float(self.R.mean())


def expected_max(m: PlayerModel, c, factor: float = 1.0):
    """E[max(X, c)] for X = player's game outcome (DNP mass included). c may be an array."""
    c = np.atleast_1d(np.asarray(c, dtype=float))
    s = m.sample_values(factor)
    played = np.maximum(s[:, None], c[None, :]).mean(axis=0)
    return (1 - m.p_play) * np.maximum(0.0, c) + m.p_play * played


def recency_mean(xs, half_life: float, n: int) -> float:
    xs = list(xs)[-n:]
    w = np.array([0.5 ** ((len(xs) - 1 - i) / half_life) for i in range(len(xs))])
    return float(np.dot(w, xs) / w.sum())


def play_prob(status: str | None, table: dict) -> float:
    if not status:
        return table["healthy"]
    return table.get(status, table.get(status.capitalize(), table["healthy"]))


class ModelBook:
    """Builds and caches PlayerModels for a given information cutoff."""

    def __init__(self, logs: dict, players: dict, proj: dict, cfg: dict, cutoff: str | None = None,
                 proj_dates: set | None = None, ros: bool = False, status_override: dict | None = None,
                 prior_pids: set | None = None):
        self.cfg, self.mc = cfg, cfg["model"]
        self.players, self.proj, self.proj_dates = players, proj, proj_dates
        self.ros = ros
        self.status_override = {} if status_override is None else status_override
        self.logs = {pid: [g for g in gl if cutoff is None or g.date < cutoff] for pid, gl in logs.items()}
        self._cache: dict[str, PlayerModel] = {}
        self._build_priors(prior_pids)
        self._build_opp_factors()

    # ------------------------------------------------------------------ priors
    def _build_priors(self, prior_pids):
        means, pools = defaultdict(list), defaultdict(list)
        for pid, gl in self.logs.items():
            if prior_pids is not None and pid not in prior_pids:
                continue
            xs = np.array([g.fp for g in gl])
            if len(xs) >= self.mc["small_sample_gp"] and xs.mean() > 0:
                grp = pos_group(self.players.get(pid, {}))
                means[grp].append(xs.mean())
                pools[grp].append(xs / xs.mean())
        self.prior_mu = {g: float(np.mean(v)) for g, v in means.items()}
        self.prior_pool = {g: np.concatenate(v) for g, v in pools.items()}

    def _build_opp_factors(self):
        k, cap = self.mc["opp_adjust_shrink_games"], self.mc["opp_adjust_cap"]
        acc = defaultdict(list)
        for pid, gl in self.logs.items():
            if len(gl) < 5:
                continue
            m = np.mean([g.fp for g in gl])
            if m <= 0:
                continue
            grp = pos_group(self.players.get(pid, {}))
            for g in gl:
                if g.opp:
                    acc[(g.opp, grp)].append(g.fp / m)
        self.opp = {key: float(np.clip((sum(v) + k) / (len(v) + k), 1 - cap, 1 + cap)) for key, v in acc.items()}

    def opp_factor(self, pid: str, opp: str | None) -> float:
        if not opp:
            return 1.0
        return self.opp.get((opp, pos_group(self.players.get(pid, {}))), 1.0)

    # ------------------------------------------------------------------ player model
    def status(self, pid):
        if pid in self.status_override:
            return self.status_override[pid]
        return self.players.get(pid, {}).get("injury_status")

    def get(self, pid: str) -> PlayerModel:
        if pid in self._cache:
            return self._cache[pid]
        mc, p = self.mc, self.players.get(pid, {})
        grp = pos_group(p)
        xs = np.array([g.fp for g in self.logs.get(pid, [])])
        n = len(xs)
        pj = self.proj.get(pid, {})
        if self.proj_dates is not None:
            pj = {d: v for d, v in pj.items() if d in self.proj_dates}
        proj_mean = float(np.mean(list(pj.values()))) if pj else None
        prior = self.prior_mu.get(grp)
        w = mc["blend_weights"]
        parts = {"gp": n, "proj": proj_mean}
        hist = None
        if n:
            season, rec = float(xs.mean()), recency_mean(xs, mc["half_life_games"], mc["recent_games"])
            parts.update(season=round(season, 1), recency=round(rec, 1))
            hist = (w["recency"] * rec + w["season"] * season) / (w["recency"] + w["season"])
        a = min(1.0, n / mc["small_sample_gp"])
        if prior is not None and a < 1:
            hist = a * (hist or 0.0) + (1 - a) * prior
        wh = w["recency"] + w["season"]
        if hist is not None and proj_mean is not None:
            mu = (wh * hist + w["projection"] * proj_mean) / (wh + w["projection"])
        else:
            mu = hist if hist is not None else (proj_mean if proj_mean is not None else 15.0)
        # shape
        K = mc["residual_samples"]
        rng = np.random.default_rng(stable_seed("shape", pid))
        own = xs / xs.mean() if n and xs.mean() > 0 else np.array([])
        pool = self.prior_pool.get(grp)
        if n >= mc["small_sample_gp"]:
            R = own
        elif pool is not None and len(pool):
            k_own = int(round(K * a)) if len(own) else 0
            R = np.concatenate([rng.choice(own, k_own) if k_own else np.array([]), rng.choice(pool, K - k_own)])
        elif len(own):
            R = rng.choice(own, K)
        else:
            cv = mc["fallback_cv"]
            R = rng.gamma(1 / cv ** 2, cv ** 2, K)
        R = R / R.mean()
        table = mc["ros_play_prob"] if self.ros else mc["injury_play_prob"]
        m = PlayerModel(pid, float(max(mu, 0.0)), np.asarray(R, float), play_prob(self.status(pid), table), grp, n, parts)
        self._cache[pid] = m
        return m
