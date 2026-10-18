import numpy as np
import pytest

from src.ou import estimate_ou_ar1, ols_beta_alpha, rolling_ols


def test_ols_recovers_exact_line():
    x = np.linspace(-5.0, 5.0, 100)
    y = 2.0 + 3.0 * x
    beta, alpha = ols_beta_alpha(x, y)
    assert np.isclose(beta, 3.0)
    assert np.isclose(alpha, 2.0)


def test_ols_handles_degenerate_x():
    beta, alpha = ols_beta_alpha(np.ones(10), np.arange(10.0))
    assert np.isnan(beta)
    assert np.isnan(alpha)


def test_ou_recovers_simulated_parameters():
    rng = np.random.default_rng(0)
    phi_true = 0.95
    c_true = 0.5
    mu_true = c_true / (1.0 - phi_true)
    s_eps = 0.1

    n = 50_000
    z = np.empty(n)
    z[0] = mu_true
    eps = rng.normal(0.0, s_eps, size=n)
    for i in range(1, n):
        z[i] = c_true + phi_true * z[i - 1] + eps[i]

    est = estimate_ou_ar1(z)
    assert est is not None
    assert est["phi"] == pytest.approx(phi_true, abs=0.01)
    assert est["mu"] == pytest.approx(mu_true, abs=0.1)
    assert est["theta"] == pytest.approx(-np.log(phi_true), abs=0.005)
    assert est["sigma_eq"] == pytest.approx(
        s_eps / np.sqrt(1.0 - phi_true**2), rel=0.05
    )


def test_ou_rejects_explosive_series():
    z = 1.05 ** np.arange(1000)
    assert estimate_ou_ar1(z) is None


def test_rolling_ols_matches_direct_regression():
    rng = np.random.default_rng(1)
    x = rng.normal(size=300)
    y = 1.5 + 0.8 * x + rng.normal(0.0, 0.1, size=300)

    window = 40
    alpha, beta = rolling_ols(y, x, window)

    for i in (window, 120, 299):
        b_direct, a_direct = ols_beta_alpha(x[i - window : i], y[i - window : i])
        assert beta[i] == pytest.approx(b_direct, rel=1e-10)
        assert alpha[i] == pytest.approx(a_direct, rel=1e-10)


def test_rolling_ols_has_no_lookahead():
    """A change at bar i must not affect beta[i], but must affect beta[i+1]."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=100)
    y = rng.normal(size=100)
    window = 20

    _, beta_before = rolling_ols(y, x, window)
    y2 = y.copy()
    y2[60] += 1000.0
    _, beta_after = rolling_ols(y2, x, window)

    # beta[60] is fitted on [40:60], which does not contain y[60].
    assert np.isclose(beta_before[60], beta_after[60])
    # beta[61] is fitted on [41:61], which does contain y[60].
    assert not np.isclose(beta_before[61], beta_after[61])
