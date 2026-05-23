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
