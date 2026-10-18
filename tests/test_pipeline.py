import numpy as np

from src.ou import adf_pvalue, build_features
from src.strategy import run_backtest
from src.synthetic import generate_synthetic


def test_synthetic_spread_is_stationary():
    df = generate_synthetic(n_days=20, seed=3)
    features = build_features(
        df, window=60, x_multiplier=10_000.0, min_pairs=15
    )
    spread = features["spread"].dropna()
    assert len(spread) > 1000
    _, pvalue = adf_pvalue(spread)
    assert pvalue < 0.05


def test_backtest_smoke():
    df = generate_synthetic(n_days=10, seed=4)
    log, trades, final_value, _ = run_backtest(
        df,
        window=60,
        a=1.5,
        b=0.5,
        c=3.0,
        x_multiplier=10_000.0,
        max_x_units=8,
        min_y_units=100,
        min_ou_pairs=15,
        initial_cash=10_000_000.0,
    )

    assert len(log) == len(df)
    assert np.isfinite(final_value)
    assert final_value > 0
    assert log["value"].iloc[0] > 0
    # The synthetic pair is constructed to mean-revert, so the strategy
    # should trade at least once over ten days.
    assert len(trades) > 0
