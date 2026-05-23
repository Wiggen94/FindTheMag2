"""V2 project-selection algorithm for FindTheMag2.

Replaces the legacy "winner-takes-all within 10%" picker with a probabilistic
selector based on Thompson sampling and softmax allocation.

Goals over the legacy algorithm:
    * Uncertainty-aware: a project's allocation reflects how confident we
      are in its observed credit-per-hour, not just the point estimate.
      Projects with few completed tasks get wider posterior samples, which
      both forces exploration when data is thin and naturally damps once
      data is plentiful.
    * Smooth weight changes: softmax over scores replaces the bimodal
      10%-threshold cliff. Projects don't flap in/out of the winner set as
      ratios drift across an arbitrary line.
    * Diversification: an optional uniform mixin caps single-project
      exposure (mitigates project-shutdown risk).
    * Smoothed mag-per-credit: the blockchain-reported magnitude ratios
      are EWMA-smoothed across runs so that single-snapshot spikes don't
      whipsaw allocations.

Out-of-scope for this initial cut (tracked as follow-ups):
    * CPU vs GPU separation (requires per-WU resource type, not just
      aggregated stats).
    * Peer-hardware priors (à la QuickMag) for cold-start estimates.
    * Power / thermal awareness in the selector itself.

This module has no dependencies outside the standard library.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from math import exp
from random import Random
from typing import Dict, Iterable, Mapping, Optional


# --------------------------------------------------------------------- config


@dataclass(frozen=True)
class V2Config:
    """Tunable parameters for the v2 selector.

    Attributes:
        temperature: Softmax temperature. ``"auto"`` (default) scales
            temperature to ``(max_score - min_score) / 5`` so the softmax
            spread is calibrated to the actual score range. A float pins
            a fixed temperature (in mag/hr units): lower = greedier, higher
            = closer to uniform.
        diversification_lambda: 0..1. Fraction of weight pushed to a uniform
            distribution over eligible projects. ``0`` disables; ``1``
            forces equal weights. Default 0.05 gives a 5% safety net
            against single-project shutdowns.
        n_samples: Number of Thompson draws per allocation pass. Higher
            reduces run-to-run variance; 128 is plenty for ~20 projects.
        min_tasks_for_local_data: Below this task count, a project is
            sampled from the prior only — its own observed credit-per-hour
            is ignored. Prevents one fluke task from crowning a project.
        prior_mean_credit_per_hour: Prior expectation of credit-per-hour
            for any unobserved project. Used by the Gamma-Poisson prior.
            Default reflects a typical mid-range BOINC project.
        prior_strength_hours: Pseudo-hours of prior observation. Higher =
            slower to be pulled away from prior by local data. 0.5 is
            weakly informative (local data dominates after ~5 hours).
        mag_ratio_half_life_days: Half-life of the EWMA over magnitude
            ratios reported by the blockchain. 14 days means a spike from
            today contributes ~50% weight after two weeks. Set to 0 to
            disable smoothing (use raw current ratios).
        rng_seed: Optional integer for deterministic runs. ``None`` for
            nondeterministic. Set this in tests, leave it unset in prod.
    """

    temperature: float | str = "auto"
    diversification_lambda: float = 0.05
    n_samples: int = 128
    min_tasks_for_local_data: int = 3
    prior_mean_credit_per_hour: float = 50.0
    prior_strength_hours: float = 0.5
    mag_ratio_half_life_days: float = 14.0
    rng_seed: Optional[int] = None


# ----------------------------------------------------- mag-ratio EWMA storage


@dataclass
class _MagRatioEMAEntry:
    value: float
    last_updated_ts: float


class MagRatioEMA:
    """Persisted EWMA over per-project ``mag_per_credit`` snapshots.

    Lives in a tiny JSON file (``selector_v2_state.json``) next to
    ``stats.json``. Each ``update_many`` call reads today's ratios from
    the blockchain, blends them with the stored running average using a
    half-life decay in real (wall-clock) time, and persists.
    """

    def __init__(self, state_path: str, half_life_days: float):
        self.state_path = state_path
        self.half_life_days = half_life_days
        self._entries: Dict[str, _MagRatioEMAEntry] = {}
        self._load()

    def _load(self) -> None:
        if not os.path.exists(self.state_path):
            return
        try:
            with open(self.state_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
        except (OSError, json.JSONDecodeError):
            return
        for url, payload in (raw.get("mag_ratio_ema") or {}).items():
            self._entries[url] = _MagRatioEMAEntry(
                value=float(payload["value"]),
                last_updated_ts=float(payload["last_updated_ts"]),
            )

    def _save(self) -> None:
        payload = {
            "mag_ratio_ema": {
                url: {"value": e.value, "last_updated_ts": e.last_updated_ts}
                for url, e in self._entries.items()
            }
        }
        tmp_path = self.state_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, sort_keys=True)
        os.replace(tmp_path, self.state_path)

    def update_many(self, new_ratios: Mapping[str, float], now_ts: Optional[float] = None) -> Dict[str, float]:
        """Blend new snapshot into the EWMA and return the smoothed map.

        If ``half_life_days`` is 0, the EWMA is bypassed and the new ratios
        are returned (and persisted) verbatim.
        """
        if now_ts is None:
            now_ts = time.time()
        smoothed: Dict[str, float] = {}
        for url, new_val in new_ratios.items():
            entry = self._entries.get(url)
            if entry is None or self.half_life_days <= 0:
                self._entries[url] = _MagRatioEMAEntry(value=new_val, last_updated_ts=now_ts)
                smoothed[url] = new_val
                continue
            days = max(0.0, (now_ts - entry.last_updated_ts) / 86400.0)
            alpha = 1.0 - 0.5 ** (days / self.half_life_days)
            new_smoothed = entry.value * (1.0 - alpha) + new_val * alpha
            self._entries[url] = _MagRatioEMAEntry(value=new_smoothed, last_updated_ts=now_ts)
            smoothed[url] = new_smoothed
        self._save()
        return smoothed


# ----------------------------------------------------------------- math core


def posterior_sample(
    total_credit: float,
    total_hours: float,
    rng: Random,
    prior_mean: float,
    prior_strength_hours: float,
) -> float:
    """Thompson-sample credit-per-hour from a Gamma posterior.

    Model: rate has a Gamma(α, β) prior with mean ``prior_mean`` and
    "pseudo-observation" of ``prior_strength_hours`` hours. Each real
    observation of (credit, hours) updates α += credit, β += hours.
    A draw from the posterior is a Gamma sample with shape α and scale 1/β.

    For low-data projects the posterior is wide → samples have high
    variance → occasional optimistic draws push the project up the
    rankings, giving it exploratory weight. For well-observed projects
    the posterior concentrates tightly around the empirical mean.
    """
    alpha = prior_mean * prior_strength_hours + total_credit
    beta = prior_strength_hours + total_hours
    if alpha <= 0 or beta <= 0:
        return prior_mean
    return rng.gammavariate(alpha, 1.0 / beta)


def softmax(scores: Mapping[str, float], temperature: float) -> Dict[str, float]:
    """Numerically stable softmax with explicit temperature.

    Temperature 0 collapses to argmax. Very large temperature → uniform.
    Returns an empty dict on empty input.
    """
    if not scores:
        return {}
    if temperature <= 0:
        best = max(scores, key=lambda k: scores[k])
        return {k: (1.0 if k == best else 0.0) for k in scores}
    m = max(scores.values())
    exps = {k: exp((v - m) / temperature) for k, v in scores.items()}
    s = sum(exps.values()) or 1.0
    return {k: v / s for k, v in exps.items()}


# ---------------------------------------------------------- public entry point


def select_and_weight(
    combined_stats: Mapping[str, Mapping],
    mag_ratios: Mapping[str, float],
    approved_project_urls: Iterable[str],
    preferred_projects: Mapping[str, float],
    ignored_projects: Iterable[str],
    *,
    total_weight: float = 1000.0,
    preferred_pct: float = 10.0,
    config: Optional[V2Config] = None,
) -> Dict[str, float]:
    """Compute BOINC resource-share weights.

    Args:
        combined_stats: As produced by ``config_files_to_stats`` — a dict
            keyed by canonical project URL, each value containing a
            ``COMPILED_STATS`` sub-dict with at minimum ``TOTALCREDIT``,
            ``TOTALWALLTIME``, and ``TOTALTASKS``.
        mag_ratios: Magnitude per credit per project (typically EWMA-
            smoothed via ``MagRatioEMA.update_many`` before being passed
            in).
        approved_project_urls: Whitelist of project URLs eligible for
            mining allocation.
        preferred_projects: Map of URL → percentage. These bypass the
            score-driven allocation and split ``preferred_pct`` of total
            weight pro-rata across them.
        ignored_projects: URLs explicitly excluded; weight is forced to 0.
        total_weight: Sum of all weights returned (BOINC convention: 1000).
        preferred_pct: Fraction (0..100) of ``total_weight`` reserved for
            preferred projects.
        config: ``V2Config`` instance; defaults are reasonable.

    Returns:
        Dict mapping project URL to weight. Sums approximately to
        ``total_weight`` minus any ignored allocations.
    """
    cfg = config or V2Config()
    rng = Random(cfg.rng_seed)
    ignored_set = set(ignored_projects)
    approved_set = set(approved_project_urls)
    preferred_set = set(preferred_projects)
    eligible = [
        url for url in approved_set
        if url not in ignored_set and url not in preferred_set
    ]

    preferred_slice = total_weight * (preferred_pct / 100.0)
    mining_slice = total_weight - preferred_slice

    def _stats(url: str) -> tuple[float, float, int, float]:
        cs = combined_stats.get(url, {}).get("COMPILED_STATS", {})
        return (
            float(cs.get("TOTALCREDIT", 0.0)),
            float(cs.get("TOTALWALLTIME", 0.0)),
            int(cs.get("TOTALTASKS", 0)),
            float(mag_ratios.get(url, 0.0)),
        )

    # If no eligible project can earn magnitude, fall back to uniform across
    # eligible (keeps the host attached / responsive without committing) and
    # we let the legacy weak-stats path handle minimum-weight semantics.
    earning = [u for u in eligible if _stats(u)[3] > 0]
    if not earning:
        per = mining_slice / max(len(eligible), 1) if eligible else 0.0
        weights: Dict[str, float] = {u: per for u in eligible}
    else:
        # Thompson sampling: average softmax allocations across n_samples draws.
        allocations: Dict[str, float] = {u: 0.0 for u in eligible}
        for _ in range(cfg.n_samples):
            sample_scores: Dict[str, float] = {}
            for url in earning:
                tc, twh, nt, mr = _stats(url)
                if nt < cfg.min_tasks_for_local_data:
                    rate = posterior_sample(
                        0.0, 0.0, rng,
                        cfg.prior_mean_credit_per_hour,
                        cfg.prior_strength_hours,
                    )
                else:
                    rate = posterior_sample(
                        tc, twh, rng,
                        cfg.prior_mean_credit_per_hour,
                        cfg.prior_strength_hours,
                    )
                sample_scores[url] = rate * mr

            if cfg.temperature == "auto":
                spread = max(sample_scores.values()) - min(sample_scores.values())
                temp = max(spread / 5.0, 1e-9)
            else:
                temp = float(cfg.temperature)

            draw_alloc = softmax(sample_scores, temp)
            for u, w in draw_alloc.items():
                allocations[u] += w / cfg.n_samples

        # Diversification mixin: blend with uniform over eligible.
        lam = cfg.diversification_lambda
        if lam > 0 and eligible:
            uniform = 1.0 / len(eligible)
            for u in eligible:
                allocations[u] = (1.0 - lam) * allocations[u] + lam * uniform

        total = sum(allocations.values()) or 1.0
        weights = {u: (allocations[u] / total) * mining_slice for u in eligible}

    # Preferred slice: pro-rata over the user-supplied percentages.
    if preferred_projects:
        preferred_total = sum(preferred_projects.values()) or 1.0
        for url, pct in preferred_projects.items():
            weights[url] = (pct / preferred_total) * preferred_slice

    # Ignored projects always at 0.
    for url in ignored_set:
        weights[url] = 0.0

    return weights
