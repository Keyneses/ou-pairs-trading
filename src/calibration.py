"""Fast vectorised engine used for threshold calibration.

This engine is deliberately simpler than backtrader: it sizes positions at
each bar and earns the next bar's price change. It is used only to rank
``(a, b, c)`` combinations on the training half of the sample. Final numbers
reported in the README are always recomputed with the backtrader engine.
"""

from __future__ import annotations

import numpy as np

from .metrics import performance


def _position_size(
    abs_score: float,
    beta: float,
    y_price: float,
    *,
    a: float,
    c: float,
    max_x_units: int,
    min_y_units: int,
    initial_cash: float,
) -> tuple[int, int]:
    frac = (abs_score - a) / (c - a)
    frac = min(1.0, max(0.0, frac))
    if frac < 0.02 or not np.isfinite(beta) or beta <= 0:
        return 0, 0

    n_x = int(round(max_x_units * frac))
    n_x = max(1, min(n_x, max_x_units))

    n_y = int(round(n_x / beta))
    n_y_cash = int((initial_cash * 0.90) // y_price) if y_price > 0 else 0
    if n_y_cash < min_y_units:
        return 0, 0
    n_y = min(max(n_y, min_y_units), n_y_cash)

    n_x = int(round(n_y * beta))
    n_x = max(1, min(n_x, max_x_units))
    return n_y, n_x


def fast_backtest(
    features,
    *,
    a: float,
    b: float,
    c: float,
    x_multiplier: float,
    max_x_units: int = 8,
    min_y_units: int = 100,
    initial_cash: float = 10_000_000.0,
):
    """Run the simplified engine on a features DataFrame.

    Returns ``(equity, n_y, n_x)`` where the position arrays are signed.
    """
    score = features["zscore"].to_numpy(dtype=np.float64)
    beta = features["beta_roll"].to_numpy(dtype=np.float64)
    y_close = features["y_close"].to_numpy(dtype=np.float64)
    x_close = features["x_close"].to_numpy(dtype=np.float64) * x_multiplier

    n = score.size
    state = 0
    pos = np.zeros(n, dtype=np.int8)
    n_y = np.zeros(n, dtype=np.float64)
    n_x = np.zeros(n, dtype=np.float64)

    for i in range(n):
        s = score[i]
        if np.isfinite(s):
            if state == 0:
                if s < -a:
                    state = 1
                elif s > a:
                    state = -1
            elif abs(s) > c or abs(s) < b:
                state = 0

        pos[i] = state
        if state != 0:
            sy, sx = _position_size(
                abs(s) if np.isfinite(s) else 0.0,
                beta[i],
                y_close[i],
                a=a,
                c=c,
                max_x_units=max_x_units,
                min_y_units=min_y_units,
                initial_cash=initial_cash,
            )
            if sy and sx:
                n_y[i] = sy * state
                n_x[i] = sx * state

    pnl = np.zeros(n)
    for i in range(1, n):
        d_y = y_close[i] - y_close[i - 1]
        d_x = x_close[i] - x_close[i - 1]
        pnl[i] = n_y[i - 1] * d_y - n_x[i - 1] * d_x

    equity = initial_cash + np.cumsum(pnl)
    return equity, n_y, n_x


def score_params(
    features,
    *,
    a: float,
    b: float,
    c: float,
    x_multiplier: float,
    bars_per_year: int,
    max_x_units: int = 8,
    min_y_units: int = 100,
    initial_cash: float = 10_000_000.0,
) -> dict:
    equity, n_y, n_x = fast_backtest(
        features,
        a=a,
        b=b,
        c=c,
        x_multiplier=x_multiplier,
        max_x_units=max_x_units,
        min_y_units=min_y_units,
        initial_cash=initial_cash,
    )
    stats = performance(
        equity, initial_cash=initial_cash, bars_per_year=bars_per_year
    )
    stats.update(
        {
            "a": a,
            "b": b,
            "c": c,
            "open_bars": int((n_x != 0).sum()),
        }
    )
    return stats


def grid_search(
    features,
    grid: list[tuple[float, float, float]],
    *,
    x_multiplier: float,
    bars_per_year: int,
    max_x_units: int = 8,
    min_y_units: int = 100,
    initial_cash: float = 10_000_000.0,
):
    """Score every ``(a, b, c)`` combination and return a sorted DataFrame."""
    import pandas as pd

    rows = []
    for a, b, c in grid:
        if not (0 < b < a < c):
            continue
        rows.append(
            score_params(
                features,
                a=a,
                b=b,
                c=c,
                x_multiplier=x_multiplier,
                bars_per_year=bars_per_year,
                max_x_units=max_x_units,
                min_y_units=min_y_units,
                initial_cash=initial_cash,
            )
        )
    return pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
