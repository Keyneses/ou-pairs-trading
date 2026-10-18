"""Performance metrics for bar-level equity curves."""

from __future__ import annotations

import numpy as np


def performance(
    equity,
    *,
    initial_cash: float,
    bars_per_year: int,
) -> dict:
    """Compute total return, annualised return, Sharpe and max drawdown.

    ``equity`` is the bar-level mark-to-market account value. Sharpe uses
    per-bar returns annualised by ``sqrt(bars_per_year)``.
    """
    eq = np.asarray(equity, dtype=np.float64)
    if eq.size < 2:
        raise ValueError("equity curve needs at least two points")
    if not np.all(np.isfinite(eq)):
        raise ValueError("equity curve contains non-finite values")
    if eq[0] <= 0:
        raise ValueError("equity curve must stay positive")

    final = float(eq[-1])
    total_return = final / initial_cash - 1.0
    years = eq.size / float(bars_per_year)
    annual_return = (
        (final / initial_cash) ** (1.0 / years) - 1.0
        if years > 0 and final > 0
        else np.nan
    )

    returns = np.diff(eq) / eq[:-1]
    vol = returns.std(ddof=1)
    sharpe = (
        float(returns.mean() / vol * np.sqrt(bars_per_year))
        if vol > 0
        else 0.0
    )

    peak = np.maximum.accumulate(eq)
    max_drawdown = float((eq / peak - 1.0).min())

    return {
        "final_value": final,
        "total_return": float(total_return),
        "annual_return": float(annual_return),
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "bars": int(eq.size),
        "years": float(years),
    }
