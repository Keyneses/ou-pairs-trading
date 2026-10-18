"""Synthetic cointegrated 1-minute price data.

The generator is intentionally simple and fully reproducible:

* ``x_quote`` follows a random walk (the "futures" leg, quoted per 100 face
  value).
* ``y`` is linked to the per-contract price of ``x`` through a constant hedge
  ratio plus an OU spread.
* Both observed closes carry a small amount of i.i.d. microstructure noise.

The bundled data is a sanity-check fixture, not evidence that the strategy is
profitable on real markets.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _intraday_index(start: str, n_days: int) -> pd.DatetimeIndex:
    """240 one-minute bars per business day: 09:30-11:29 and 13:00-14:59."""
    days = pd.bdate_range(start=start, periods=n_days)
    stamps: list[pd.DatetimeIndex] = []
    for day in days:
        morning = pd.date_range(
            day + pd.Timedelta(hours=9, minutes=30), periods=120, freq="min"
        )
        afternoon = pd.date_range(
            day + pd.Timedelta(hours=13), periods=120, freq="min"
        )
        stamps.append(morning.append(afternoon))
    return stamps[0].append(stamps[1:]) if len(stamps) > 1 else stamps[0]


def _ohlc_from_close(
    close: np.ndarray,
    rng: np.random.Generator,
    tick: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = close.size
    noise = rng.normal(0.0, tick, size=n)
    open_ = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] + noise[1:]

    body_hi = np.maximum(open_, close)
    body_lo = np.minimum(open_, close)
    wick = np.abs(rng.normal(0.0, tick, size=n))
    high = body_hi + wick
    low = body_lo - np.abs(rng.normal(0.0, tick, size=n))

    return open_, high, low, close


def generate_synthetic(
    *,
    n_days: int = 120,
    bars_per_day: int = 240,
    start: str = "2024-01-02",
    seed: int = 7,
    x_start: float = 112.0,
    x_drift: float = 0.0,
    x_sigma: float = 0.0025,
    alpha: float = -0.50,
    beta: float = 0.00011,
    ou_theta: float = np.log(2.0) / 60.0,
    ou_sigma_eq: float = 0.05,
    y_noise: float = 0.004,
    x_noise: float = 0.0015,
) -> pd.DataFrame:
    """Generate an aligned 1-minute synthetic pair.

    ``bars_per_day`` must be 240 to match the default intraday calendar.
    """
    if bars_per_day != 240:
        raise ValueError("the bundled calendar is fixed at 240 bars per day")

    rng = np.random.default_rng(seed)
    idx = _intraday_index(start, n_days)
    n = idx.size

    # Futures leg: random walk in quote space (per 100 face value).
    x_steps = rng.normal(x_drift, x_sigma, size=n)
    x_quote = x_start + np.cumsum(x_steps)
    x_contract = x_quote * 10_000.0

    # OU spread, simulated at its exact AR(1) discretisation.
    phi = np.exp(-ou_theta)
    z = np.empty(n)
    z[0] = rng.normal(0.0, ou_sigma_eq)
    shocks = rng.normal(0.0, ou_sigma_eq * np.sqrt(1.0 - phi**2), size=n)
    for i in range(1, n):
        z[i] = phi * z[i - 1] + shocks[i]

    y_true = alpha + beta * x_contract + z

    # Observed closes: true value plus independent microstructure noise.
    y_close = y_true + rng.normal(0.0, y_noise, size=n)
    x_quote_obs = x_quote + rng.normal(0.0, x_noise, size=n)

    y_open, y_high, y_low, _ = _ohlc_from_close(y_close, rng, tick=y_noise)
    x_open, x_high, x_low, _ = _ohlc_from_close(x_quote_obs, rng, tick=x_noise)

    volume_y = rng.lognormal(mean=8.0, sigma=0.6, size=n).astype(int)
    volume_x = rng.lognormal(mean=9.0, sigma=0.5, size=n).astype(int)

    df = pd.DataFrame(
        {
            "datetime": idx,
            "y_open": np.round(y_open, 4),
            "y_high": np.round(y_high, 4),
            "y_low": np.round(y_low, 4),
            "y_close": np.round(y_close, 4),
            "y_volume": volume_y,
            "x_open": np.round(x_open, 4),
            "x_high": np.round(x_high, 4),
            "x_low": np.round(x_low, 4),
            "x_close": np.round(x_quote_obs, 4),
            "x_volume": volume_x,
        }
    )

    # OHLC consistency after rounding.
    for leg in ("y", "x"):
        df[f"{leg}_high"] = df[[f"{leg}_open", f"{leg}_high", f"{leg}_close"]].max(axis=1)
        df[f"{leg}_low"] = df[[f"{leg}_open", f"{leg}_low", f"{leg}_close"]].min(axis=1)

    return df
