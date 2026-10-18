"""OLS and Ornstein-Uhlenbeck estimation.

Conventions
-----------
* Prices are observed at equally spaced bars. ``theta`` and ``sigma`` are
  therefore expressed per bar, not per year.
* OU parameters are estimated from the exact AR(1) discretisation of the
  continuous process ``dZ = theta * (mu - Z) dt + sigma dW``:

      Z_t = c + phi * Z_{t-1} + eps,   phi = exp(-theta * dt)
      mu  = c / (1 - phi)
      sigma_eq = s_eps / sqrt(1 - phi^2)

* All rolling estimates use data up to bar ``t-1`` and are applied to bar
  ``t``. Nothing in this module looks ahead.
"""

from __future__ import annotations

import warnings

import numpy as np

MIN_OU_PAIRS = 30


def ols_beta_alpha(x, y) -> tuple[float, float]:
    """Centred OLS of ``y = alpha + beta * x + eps``.

    Returns ``(beta, alpha)``. The centred form avoids the catastrophic
    cancellation that can occur when x is large (e.g. futures notional).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.size != y.size or x.size < 2:
        return np.nan, np.nan

    xm = x.mean()
    ym = y.mean()
    xc = x - xm
    yc = y - ym
    denom = float(xc @ xc)
    if denom < 1e-12:
        return np.nan, np.nan

    beta = float(xc @ yc) / denom
    alpha = float(ym - beta * xm)
    return beta, alpha


def estimate_ou_ar1(z, min_pairs: int = MIN_OU_PAIRS) -> dict | None:
    """Estimate OU parameters from an AR(1) regression on a spread series.

    Returns ``None`` when the series is too short, non-finite, or does not
    imply a stationary AR(1) with ``0 < phi < 1``.
    """
    z = np.asarray(z, dtype=np.float64)
    z = z[np.isfinite(z)]
    if z.size < min_pairs + 1:
        return None

    z_lag = z[:-1]
    z_next = z[1:]
    phi, c = ols_beta_alpha(z_lag, z_next)
    if not np.isfinite(phi) or not (0.0 < phi < 1.0):
        return None

    resid = z_next - (c + phi * z_lag)
    s_eps = float(resid.std(ddof=2))
    if not np.isfinite(s_eps) or s_eps < 1e-12:
        return None

    theta = -np.log(phi)
    mu = c / (1.0 - phi)
    sigma = s_eps * np.sqrt(2.0 * theta / (1.0 - phi**2))
    sigma_eq = sigma / np.sqrt(2.0 * theta)
    half_life = np.log(2.0) / theta

    return {
        "mu": float(mu),
        "theta": float(theta),
        "sigma": float(sigma),
        "sigma_eq": float(sigma_eq),
        "half_life": float(half_life),
        "phi": float(phi),
        "s_eps": s_eps,
        "n": int(z.size),
    }


def rolling_ols(y, x, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling OLS with parameters estimated on the window ending at t-1.

    ``alpha[i]`` and ``beta[i]`` are fitted on ``[i-window, i)`` and are
    intended to be applied to bar ``i``.
    """
    y = np.asarray(y, dtype=np.float64)
    x = np.asarray(x, dtype=np.float64)
    if y.size != x.size:
        raise ValueError("y and x must have the same length")

    n = y.size
    alpha = np.full(n, np.nan)
    beta = np.full(n, np.nan)
    if n <= window:
        return alpha, beta

    cum_y = np.concatenate(([0.0], np.cumsum(y)))
    cum_x = np.concatenate(([0.0], np.cumsum(x)))
    cum_xy = np.concatenate(([0.0], np.cumsum(y * x)))
    cum_xx = np.concatenate(([0.0], np.cumsum(x * x)))

    for i in range(window, n):
        sy = cum_y[i] - cum_y[i - window]
        sx = cum_x[i] - cum_x[i - window]
        sxy = cum_xy[i] - cum_xy[i - window]
        sxx = cum_xx[i] - cum_xx[i - window]
        denom = window * sxx - sx * sx
        if denom < 1e-12:
            continue
        b = (window * sxy - sx * sy) / denom
        a = (sy - b * sx) / window
        beta[i] = b
        alpha[i] = a

    return alpha, beta


def rolling_ou(
    z,
    window: int,
    min_pairs: int = MIN_OU_PAIRS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Rolling OU estimates on the window ending at t-1.

    Returns ``(mu, theta, sigma_eq, half_life)``. Each entry at ``i`` uses
    at most ``window`` historical z values and is applied to ``z[i]``.
    """
    z = np.asarray(z, dtype=np.float64)
    n = z.size
    mu = np.full(n, np.nan)
    theta = np.full(n, np.nan)
    sigma_eq = np.full(n, np.nan)
    half_life = np.full(n, np.nan)

    for i in range(n):
        seg = z[max(0, i - window) : i]
        if seg.size < min_pairs + 1 or not np.all(np.isfinite(seg)):
            continue
        est = estimate_ou_ar1(seg, min_pairs=min_pairs)
        if est is None:
            continue
        mu[i] = est["mu"]
        theta[i] = est["theta"]
        sigma_eq[i] = est["sigma_eq"]
        half_life[i] = est["half_life"]

    return mu, theta, sigma_eq, half_life


def build_features(
    df,
    *,
    window: int = 180,
    y_col: str = "y_close",
    x_col: str = "x_close",
    x_multiplier: float = 1.0,
    min_pairs: int = MIN_OU_PAIRS,
):
    """Attach rolling hedge ratio, spread and z-score columns to a DataFrame."""
    import pandas as pd  # local import keeps the numerical core dependency-free

    out = df.copy()
    y = out[y_col].to_numpy(dtype=np.float64)
    x = out[x_col].to_numpy(dtype=np.float64) * float(x_multiplier)

    alpha, beta = rolling_ols(y, x, window)
    spread = y - alpha - beta * x
    mu, theta, sigma_eq, half_life = rolling_ou(spread, window, min_pairs)

    out["x_price"] = x
    out["alpha_roll"] = alpha
    out["beta_roll"] = beta
    out["spread"] = spread
    out["mu_roll"] = mu
    out["theta_roll"] = theta
    out["sigma_eq_roll"] = sigma_eq
    out["half_life_roll"] = half_life
    out["zscore"] = (spread - mu) / sigma_eq

    return out


def adf_pvalue(series) -> tuple[float, float]:
    """ADF statistic and p-value; returns NaNs when the series is too short."""
    from statsmodels.tsa.stattools import adfuller

    s = np.asarray(series, dtype=np.float64)
    s = s[np.isfinite(s)]
    if s.size < 20:
        return np.nan, np.nan
    with warnings.catch_warnings():
        # statsmodels 0.15 warns about an upcoming return-type change.
        warnings.simplefilter("ignore", FutureWarning)
        stat, pvalue, *_ = adfuller(s, autolag="AIC")
    return float(stat), float(pvalue)
