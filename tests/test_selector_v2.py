"""Unit tests for the v2 project selector.

Run from repo root: ``python -m pytest tests/test_selector_v2.py -v``
or just ``python tests/test_selector_v2.py``.
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import tempfile

# Allow running both via pytest and as a script from any cwd.
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

import selector_v2 as sv2  # noqa: E402


def _stats(total_credit: float, total_hours: float, total_tasks: int) -> dict:
    """Tiny helper to build the COMPILED_STATS shape main.py uses."""
    return {"COMPILED_STATS": {
        "TOTALCREDIT": total_credit,
        "TOTALWALLTIME": total_hours,
        "TOTALTASKS": total_tasks,
    }}


# ----------------------------------------------------- posterior_sample tests


def test_posterior_concentrates_with_data():
    import random
    rng = random.Random(42)
    # Project with 1000 hours of solid data at 100 credit/hr should sample tightly
    samples = [
        sv2.posterior_sample(100_000.0, 1000.0, rng, prior_mean=50.0, prior_strength_hours=0.5)
        for _ in range(2000)
    ]
    mean = statistics.fmean(samples)
    stdev = statistics.stdev(samples)
    assert abs(mean - 100.0) < 1.0, f"posterior mean drifted: {mean}"
    # CoV should be tight (<5%) with that much data
    assert stdev / mean < 0.05, f"posterior too wide: stdev={stdev}, mean={mean}"


def test_posterior_wide_with_low_data():
    import random
    rng = random.Random(42)
    # With NO data, samples come from the prior — should be very noisy
    no_data = [
        sv2.posterior_sample(0.0, 0.0, rng, prior_mean=50.0, prior_strength_hours=0.5)
        for _ in range(2000)
    ]
    # With lots of data — should be narrow
    lots = [
        sv2.posterior_sample(50_000.0, 500.0, rng, prior_mean=50.0, prior_strength_hours=0.5)
        for _ in range(2000)
    ]
    cov_no_data = statistics.stdev(no_data) / statistics.fmean(no_data)
    cov_lots = statistics.stdev(lots) / statistics.fmean(lots)
    assert cov_no_data > cov_lots * 5, (
        f"prior should be much wider than posterior-with-data: "
        f"no_data CoV={cov_no_data}, lots CoV={cov_lots}"
    )


# -------------------------------------------------------------- softmax tests


def test_softmax_argmax_at_zero_temp():
    out = sv2.softmax({"a": 1.0, "b": 2.0, "c": 0.5}, temperature=0.0)
    assert out == {"a": 0.0, "b": 1.0, "c": 0.0}


def test_softmax_uniform_at_high_temp():
    out = sv2.softmax({"a": 1.0, "b": 2.0, "c": 0.5}, temperature=1e6)
    # All should be ~1/3
    for v in out.values():
        assert abs(v - 1.0 / 3) < 1e-3


def test_softmax_preserves_ranking():
    out = sv2.softmax({"a": 1.0, "b": 2.0, "c": 0.5}, temperature=0.5)
    assert out["b"] > out["a"] > out["c"]
    assert abs(sum(out.values()) - 1.0) < 1e-9


def test_softmax_empty():
    assert sv2.softmax({}, temperature=1.0) == {}


# --------------------------------------- posterior closed-form helpers tests


def test_posterior_mean_matches_empirical_with_lots_of_data():
    # 10_000 credit over 100 hours = 100 c/hr; with a weak prior, posterior
    # mean should be very close to 100.
    m = sv2.posterior_mean(10_000.0, 100.0, prior_mean=50.0, prior_strength_hours=0.5)
    assert abs(m - 100.0) < 0.5


def test_posterior_mean_falls_back_to_prior_with_no_data():
    m = sv2.posterior_mean(0.0, 0.0, prior_mean=42.0, prior_strength_hours=0.5)
    assert m == 42.0


def test_posterior_std_shrinks_with_more_data():
    s_thin = sv2.posterior_std(50.0, 0.5, prior_mean=50.0, prior_strength_hours=0.5)
    s_thick = sv2.posterior_std(50_000.0, 500.0, prior_mean=50.0, prior_strength_hours=0.5)
    assert s_thick < s_thin / 10


def test_v2_annotates_combined_stats():
    cs = {"http://a/": _stats(1000.0, 10.0, 50)}  # 100 c/hr
    sv2.select_and_weight(
        combined_stats=cs,
        mag_ratios={"http://a/": 0.005},
        approved_project_urls=["http://a/"],
        preferred_projects={},
        ignored_projects=[],
        config=sv2.V2Config(rng_seed=1, n_samples=8),
    )
    stats = cs["http://a/"]["COMPILED_STATS"]
    assert "V2_POSTERIOR_MEAN_CR" in stats
    assert "V2_POSTERIOR_STD_CR" in stats
    assert "V2_EXP_MAG" in stats
    # Posterior mean should land near 100 c/hr (50 tasks of solid data)
    assert abs(stats["V2_POSTERIOR_MEAN_CR"] - 100.0) < 5.0
    # V2_EXP_MAG = mean * smoothed_ratio
    assert abs(stats["V2_EXP_MAG"] - stats["V2_POSTERIOR_MEAN_CR"] * 0.005) < 1e-9
    # AVGMAGPERHOUR should now match V2_EXP_MAG (v2 overwrites the legacy value)
    assert stats["AVGMAGPERHOUR"] == stats["V2_EXP_MAG"]


# -------------------------------------------------- select_and_weight integration


def test_select_no_eligible_returns_zero_for_ignored():
    out = sv2.select_and_weight(
        combined_stats={},
        mag_ratios={},
        approved_project_urls=["http://a/"],
        preferred_projects={},
        ignored_projects=["http://b/"],
    )
    assert out["http://b/"] == 0.0


def test_select_single_eligible_gets_all_mining_weight():
    out = sv2.select_and_weight(
        combined_stats={"http://a/": _stats(1000.0, 10.0, 50)},
        mag_ratios={"http://a/": 0.001},
        approved_project_urls=["http://a/"],
        preferred_projects={},
        ignored_projects=[],
        total_weight=1000.0,
        preferred_pct=10.0,
        config=sv2.V2Config(rng_seed=1, diversification_lambda=0.0),
    )
    # With no preferred projects, the whole mining slice (900) plus the
    # unused preferred slice (the latter only allocated if preferred set
    # is non-empty) stays with the one eligible project. By design the
    # preferred-slice "vanishes" when no preferred projects exist — that's
    # not the v2 selector's job to redistribute. So we expect exactly
    # mining_slice = 900.
    assert abs(out["http://a/"] - 900.0) < 1e-6


def test_select_preferred_projects_get_exact_percentage():
    out = sv2.select_and_weight(
        combined_stats={"http://a/": _stats(1000.0, 10.0, 50)},
        mag_ratios={"http://a/": 0.001, "http://pref/": 0.0},
        approved_project_urls=["http://a/", "http://pref/"],
        preferred_projects={"http://pref/": 100.0},  # 100% of preferred slice
        ignored_projects=[],
        total_weight=1000.0,
        preferred_pct=10.0,
        config=sv2.V2Config(rng_seed=1, diversification_lambda=0.0),
    )
    assert abs(out["http://pref/"] - 100.0) < 1e-6  # 10% of 1000
    assert abs(out["http://a/"] - 900.0) < 1e-6


def test_select_ignored_always_zero():
    out = sv2.select_and_weight(
        combined_stats={
            "http://a/": _stats(1000.0, 10.0, 50),
            "http://ign/": _stats(99999.0, 1.0, 1000),  # would otherwise dominate
        },
        mag_ratios={"http://a/": 0.001, "http://ign/": 0.005},
        approved_project_urls=["http://a/", "http://ign/"],
        preferred_projects={},
        ignored_projects=["http://ign/"],
        config=sv2.V2Config(rng_seed=1, diversification_lambda=0.0),
    )
    assert out["http://ign/"] == 0.0
    assert out["http://a/"] > 0.0


def test_select_better_project_gets_more_weight():
    # Two projects, one earns 10x mag/hr. Expect winner to dominate.
    out = sv2.select_and_weight(
        combined_stats={
            "http://hi/": _stats(10_000.0, 100.0, 200),  # 100 c/hr
            "http://lo/": _stats(1000.0, 100.0, 200),   # 10 c/hr
        },
        mag_ratios={"http://hi/": 0.001, "http://lo/": 0.001},
        approved_project_urls=["http://hi/", "http://lo/"],
        preferred_projects={},
        ignored_projects=[],
        config=sv2.V2Config(rng_seed=1, diversification_lambda=0.0, n_samples=256),
    )
    assert out["http://hi/"] > out["http://lo/"] * 5, (
        f"high-earner should dominate, got hi={out['http://hi/']:.2f} "
        f"lo={out['http://lo/']:.2f}"
    )


def test_diversification_lambda_one_gives_uniform():
    out = sv2.select_and_weight(
        combined_stats={
            "http://hi/": _stats(10_000.0, 100.0, 200),
            "http://lo/": _stats(1000.0, 100.0, 200),
        },
        mag_ratios={"http://hi/": 0.001, "http://lo/": 0.001},
        approved_project_urls=["http://hi/", "http://lo/"],
        preferred_projects={},
        ignored_projects=[],
        config=sv2.V2Config(rng_seed=1, diversification_lambda=1.0),
    )
    # λ=1 → pure uniform → exactly equal weights
    assert abs(out["http://hi/"] - out["http://lo/"]) < 1e-6


# --------------------------------------------------- UCB cold-start tests


def test_ucb_bonus_is_max_emag_for_zero_tasks():
    bonus = sv2.ucb_exploration_bonus(n_tasks=0, total_tasks=1000, max_emag=0.78, c=1.0)
    assert bonus == 0.78


def test_ucb_bonus_decays_with_tasks():
    high = sv2.ucb_exploration_bonus(n_tasks=1, total_tasks=1000, max_emag=1.0, c=1.0)
    mid = sv2.ucb_exploration_bonus(n_tasks=100, total_tasks=1000, max_emag=1.0, c=1.0)
    low = sv2.ucb_exploration_bonus(n_tasks=10_000, total_tasks=10_000, max_emag=1.0, c=1.0)
    assert high > mid > low
    assert low < 0.05


def test_ucb_bonus_zero_when_c_is_zero():
    assert sv2.ucb_exploration_bonus(n_tasks=0, total_tasks=1000, max_emag=10.0, c=0.0) == 0.0


def test_ucb_cold_start_can_beat_established_leader():
    """The original motivating case: cold-start Asteroids vs confident NumberFields."""
    # Two projects: one has tons of data and a confident lower EMag, the other
    # has zero local data but a higher mag/credit ratio.
    out = sv2.select_and_weight(
        combined_stats={
            "http://leader/": _stats(110_000.0, 100.0, 500),  # 1100 c/hr, lots of data
            "http://cold/": _stats(0.0, 0.0, 0),              # zero data
        },
        mag_ratios={
            "http://leader/": 0.0007,   # leader EMag ≈ 0.77
            "http://cold/": 0.003,      # cold project — best mag/cr if any data
        },
        approved_project_urls=["http://leader/", "http://cold/"],
        preferred_projects={},
        ignored_projects=[],
        config=sv2.V2Config(
            rng_seed=1,
            diversification_lambda=0.0,
            ucb_exploration_c=1.0,   # default
            n_samples=512,
        ),
    )
    # Cold project should get a meaningful share (>15%) of mining weight
    # — without UCB it would get effectively 0 against the confident leader.
    cold_share = out["http://cold/"] / (out["http://cold/"] + out["http://leader/"])
    assert cold_share > 0.15, (
        f"cold-start project should get meaningful weight via UCB; "
        f"got cold={out['http://cold/']:.1f}, leader={out['http://leader/']:.1f}, "
        f"share={cold_share:.3f}"
    )


def test_ucb_disabled_keeps_old_behavior():
    """With c=0, the cold-start project gets minimal weight (proves UCB is what unlocks it)."""
    out = sv2.select_and_weight(
        combined_stats={
            "http://leader/": _stats(110_000.0, 100.0, 500),
            "http://cold/": _stats(0.0, 0.0, 0),
        },
        mag_ratios={
            "http://leader/": 0.0007,
            "http://cold/": 0.003,
        },
        approved_project_urls=["http://leader/", "http://cold/"],
        preferred_projects={},
        ignored_projects=[],
        config=sv2.V2Config(
            rng_seed=1,
            diversification_lambda=0.0,
            ucb_exploration_c=0.0,   # disabled
            n_samples=512,
        ),
    )
    cold_share = out["http://cold/"] / (out["http://cold/"] + out["http://leader/"])
    assert cold_share < 0.05, (
        f"without UCB, cold-start project should be starved; got share={cold_share:.3f}"
    )


def test_low_data_projects_get_exploration_weight():
    # New project with zero history should get *some* weight via prior sampling.
    out = sv2.select_and_weight(
        combined_stats={
            "http://known/": _stats(10_000.0, 100.0, 200),
            "http://new/": _stats(0.0, 0.0, 0),  # cold start
        },
        mag_ratios={"http://known/": 0.001, "http://new/": 0.001},
        approved_project_urls=["http://known/", "http://new/"],
        preferred_projects={},
        ignored_projects=[],
        config=sv2.V2Config(rng_seed=1, diversification_lambda=0.0, n_samples=512),
    )
    # Cold-start project should get nonzero exploration weight (more than the
    # diversification floor, which is 0 here)
    assert out["http://new/"] > 0.0, "v2 should explore unknown projects"


# -------------------------------------------------------- EWMA persistence tests


def test_ema_first_observation_just_stored():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "state.json")
        ema = sv2.MagRatioEMA(path, half_life_days=14.0)
        out = ema.update_many({"http://a/": 0.005})
        assert out == {"http://a/": 0.005}
        assert os.path.exists(path)


def test_ema_decays_toward_new_value_over_time():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "state.json")
        ema = sv2.MagRatioEMA(path, half_life_days=14.0)
        ema.update_many({"http://a/": 1.0}, now_ts=0.0)
        # 14 days later, new value 3.0 → smoothed should be ~midpoint (2.0)
        out = ema.update_many({"http://a/": 3.0}, now_ts=14 * 86400)
        assert abs(out["http://a/"] - 2.0) < 0.01


def test_ema_half_life_zero_no_smoothing():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "state.json")
        ema = sv2.MagRatioEMA(path, half_life_days=0.0)
        ema.update_many({"http://a/": 1.0}, now_ts=0.0)
        out = ema.update_many({"http://a/": 99.0}, now_ts=1 * 86400)
        assert out["http://a/"] == 99.0


def test_ema_persists_across_instances():
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "state.json")
        ema1 = sv2.MagRatioEMA(path, half_life_days=14.0)
        ema1.update_many({"http://a/": 0.005}, now_ts=1_000_000.0)
        ema2 = sv2.MagRatioEMA(path, half_life_days=14.0)
        # No new data → same value (decays toward itself, no change)
        assert ema2._entries["http://a/"].value == 0.005


def test_ema_atomic_write_with_tmp_file():
    # Persistence uses os.replace through a .tmp file → no partial-write
    # corruption on crash.
    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "state.json")
        ema = sv2.MagRatioEMA(path, half_life_days=14.0)
        ema.update_many({"http://a/": 0.005})
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        assert "mag_ratio_ema" in data
        assert data["mag_ratio_ema"]["http://a/"]["value"] == 0.005


if __name__ == "__main__":
    # Allow ``python tests/test_selector_v2.py`` without pytest installed.
    import traceback
    funcs = [g for n, g in sorted(globals().items()) if n.startswith("test_")]
    failed = 0
    for fn in funcs:
        try:
            fn()
            print(f"  PASS  {fn.__name__}")
        except Exception:
            failed += 1
            print(f"  FAIL  {fn.__name__}")
            traceback.print_exc()
    print(f"\n{len(funcs) - failed}/{len(funcs)} passed")
    sys.exit(1 if failed else 0)
