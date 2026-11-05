import math

import pytest

from mood_steering.eval.stats import (
    Interval,
    bootstrap_ci,
    excludes_zero,
    interval_from_draws,
    paired_bootstrap_diff,
    paired_bootstrap_stat,
    paired_shift,
    percentile,
)


def test_percentile_interpolates_linearly():
    v = [0.0, 10.0, 20.0, 30.0, 40.0]
    assert percentile(v, 0.0) == 0.0
    assert percentile(v, 1.0) == 40.0
    assert percentile(v, 0.5) == 20.0
    assert percentile(v, 0.125) == pytest.approx(5.0)
    assert percentile([7.0], 0.9) == 7.0


def test_interval_from_draws_drops_nan_and_handles_all_nan():
    iv = interval_from_draws([math.nan, 1.0, 2.0, 3.0], 2.0)
    assert (iv.mean, iv.n) == (2.0, 3)
    assert iv.low == pytest.approx(1.05) and iv.high == pytest.approx(2.95)
    empty = interval_from_draws([math.nan], 1.0)
    assert empty.n == 0 and math.isnan(empty.low) and math.isnan(empty.high)


def test_bootstrap_ci_deterministic_and_errors():
    v = [1.0, 2.0, 3.0, 4.0]
    a = bootstrap_ci(v, n_boot=200, seed=3)
    assert a == bootstrap_ci(v, n_boot=200, seed=3)
    assert a.mean == 2.5 and a.low < 2.5 < a.high and a.n == 4
    with pytest.raises(ValueError, match="aucune valeur"):
        bootstrap_ci([])
    with pytest.raises(ValueError, match="n_boot"):
        bootstrap_ci(v, n_boot=0)


def test_paired_bootstrap_diff_matches_ci_of_differences_and_errors():
    a, b = [3.0, 4.0, 5.0, 6.0], [1.0, 1.0, 2.0, 2.0]
    got = paired_bootstrap_diff(a, b, n_boot=100, seed=1)
    assert got == bootstrap_ci([2.0, 3.0, 3.0, 4.0], n_boot=100, seed=1)
    with pytest.raises(ValueError, match="appariées"):
        paired_bootstrap_diff([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="appariées"):
        paired_bootstrap_diff([], [])


def test_paired_bootstrap_stat_resamples_items_together():
    cols = {"x": [0.0, 1.0, 2.0, 3.0], "y": [0.0, 1.0, 2.0, 3.0]}
    # x - y vaut 0 pour tout rééchantillon : les colonnes partagent les indices tirés
    iv = paired_bootstrap_stat(cols, lambda c: sum(c["x"]) - sum(c["y"]), n_boot=50, seed=2)
    assert (iv.mean, iv.low, iv.high) == (0.0, 0.0, 0.0)
    with pytest.raises(ValueError, match="longueurs"):
        paired_bootstrap_stat({"a": [1.0], "b": [1.0, 2.0]}, lambda c: 0.0)
    with pytest.raises(ValueError, match="longueurs"):
        paired_bootstrap_stat({"a": []}, lambda c: 0.0)


def test_excludes_zero_branches():
    pos, neg, mid = Interval(1, 0.1, 2, 5), Interval(-1, -2, -0.1, 5), Interval(0, -1, 1, 5)
    assert excludes_zero(pos) and excludes_zero(neg) and not excludes_zero(mid)
    assert excludes_zero(pos, 1) and not excludes_zero(pos, -1)
    assert excludes_zero(neg, -1) and not excludes_zero(neg, 1)
    assert not excludes_zero(Interval(0, 0.0, 1.0, 5), 1)  # 0 sur la borne : non exclu
    assert not excludes_zero(None) and not excludes_zero(Interval(0, math.nan, math.nan, 0))


def test_paired_shift_direction_and_pairing():
    base = [0.0, 0.1, -0.1, 0.05] * 10
    up = [x + 0.5 for x in base]
    assert paired_shift(up, base, direction=1, n_boot=200)
    assert not paired_shift(up, base, direction=-1, n_boot=200)
    assert paired_shift(base, up, direction=-1, n_boot=200)
    assert not paired_shift(base, base, direction=1, n_boot=200)


def test_no_duplicate_helpers_left_in_repo():
    import inspect
    from pathlib import Path

    from mood_steering.eval.runner import run_bench

    root = Path(__file__).resolve().parents[1]
    banned = ("_read_part", "_percentile", "sys.modules", "spec_from_file_location")
    for path in [*(root / "scripts").glob("*.py"), *(root / "src").rglob("*.py")]:
        text = path.read_text(encoding="utf-8")
        for word in banned:
            assert word not in text, f"{word} dans {path.name}"
    assert "energy_tracker" not in inspect.signature(run_bench).parameters
