"""Backtrader implementation of the three-threshold OU pairs strategy.

Signal convention
-----------------
At each bar ``t`` the strategy has just observed the close. It estimates the
hedge ratio and OU parameters on the trailing window ending at ``t-1``, forms
the spread and z-score at ``t``, and submits market orders. Backtrader fills
those orders at the next bar's open, so the signal never trades on its own
bar's prices.
"""

from __future__ import annotations

from pathlib import Path

import backtrader as bt
import numpy as np
import pandas as pd

from .data import to_backtrader_frames
from .ou import estimate_ou_ar1, ols_beta_alpha


class PairTradingStrategy(bt.Strategy):
    params = (
        ("window", 180),
        ("a", 2.0),
        ("b", 0.5),
        ("c", 3.5),
        ("x_multiplier", 1.0),
        ("max_x_units", 8),
        ("min_y_units", 100),
        ("min_ou_pairs", 30),
    )

    def __init__(self):
        self.win_y: list[float] = []
        self.win_x: list[float] = []
        self.win_z: list[float] = []
        self.log: list[dict] = []
        self.trades: list[dict] = []

    def next(self):
        dt = self.datas[0].datetime.datetime(0)
        y_price = float(self.datas[0].close[0])
        x_price = float(self.datas[1].close[0]) * float(self.p.x_multiplier)

        self.win_y.append(y_price)
        self.win_x.append(x_price)

        if len(self.win_y) < self.p.window + 1:
            self._record(dt, np.nan, np.nan, "warmup", 0, 0)
            return

        hist_x = self.win_x[-self.p.window - 1 : -1]
        hist_y = self.win_y[-self.p.window - 1 : -1]
        beta, alpha = ols_beta_alpha(hist_x, hist_y)
        if not np.isfinite(beta) or not np.isfinite(alpha) or beta <= 0:
            self._record(dt, np.nan, np.nan, "noparam", 0, 0)
            return

        z_now = y_price - alpha - beta * x_price
        self.win_z.append(z_now)
        if len(self.win_z) > self.p.window + 1:
            self.win_z.pop(0)

        hist_z = self.win_z[-self.p.window - 1 : -1]
        ou = estimate_ou_ar1(hist_z, min_pairs=self.p.min_ou_pairs)
        if ou is None:
            self._record(dt, np.nan, np.nan, "nonstat", 0, 0)
            return

        score = (z_now - ou["mu"]) / ou["sigma_eq"]
        y_pos = self.getposition(self.datas[0]).size
        x_pos = self.getposition(self.datas[1]).size
        flat = y_pos == 0 and x_pos == 0
        action = "hold"

        if flat:
            if score < -self.p.a:
                self._open_position(dt, score, z_now, "long_spread", beta)
                action = "open_long"
            elif score > self.p.a:
                self._open_position(dt, score, z_now, "short_spread", beta)
                action = "open_short"
        else:
            if abs(score) > self.p.c:
                self._close_position()
                action = "stop_loss"
            elif abs(score) < self.p.b:
                self._close_position()
                action = "close"

        self._record(dt, score, z_now, action, y_pos, x_pos)

    def _open_position(self, dt, score, z_now, side, beta):
        y_price = float(self.datas[0].close[0])
        x_price = float(self.datas[1].close[0]) * float(self.p.x_multiplier)
        cash = float(self.broker.getcash())

        frac = (abs(score) - self.p.a) / (self.p.c - self.p.a)
        frac = min(1.0, max(0.0, frac))
        if frac < 0.02:
            return

        n_x = int(round(self.p.max_x_units * frac))
        n_x = max(1, min(n_x, self.p.max_x_units))

        if beta <= 0 or y_price <= 0 or x_price <= 0:
            return

        # Units of y that hedge n_x units of x: n_y * dY = n_x * dX.
        n_y = int(round(n_x / beta))
        n_y_cash = int((cash * 0.90) // y_price)
        if n_y_cash < self.p.min_y_units:
            return
        n_y = min(max(n_y, self.p.min_y_units), n_y_cash)

        # Reconcile the x leg with the cash-constrained y leg.
        n_x = int(round(n_y * beta))
        n_x = max(1, min(n_x, self.p.max_x_units))

        if side == "long_spread":
            self.buy(data=self.datas[0], size=n_y)
            self.sell(data=self.datas[1], size=n_x)
        else:
            self.sell(data=self.datas[0], size=n_y)
            self.buy(data=self.datas[1], size=n_x)

        self.trades.append(
            {
                "datetime": dt,
                "side": side,
                "zscore": float(score),
                "spread": float(z_now),
                "n_y": n_y,
                "n_x": n_x,
                "y_price": y_price,
                "x_price": x_price,
                "beta": float(beta),
                "size_frac": float(frac),
            }
        )

    def _close_position(self):
        for data in (self.datas[0], self.datas[1]):
            if self.getposition(data).size != 0:
                self.close(data=data)

    def _record(self, dt, score, spread, action, y_pos, x_pos):
        self.log.append(
            {
                "datetime": dt,
                "zscore": score,
                "spread": spread,
                "action": action,
                "y_pos": y_pos,
                "x_pos": x_pos,
                "value": float(self.broker.getvalue()),
            }
        )


def run_backtest(
    df: pd.DataFrame,
    *,
    window: int = 180,
    a: float = 2.0,
    b: float = 0.5,
    c: float = 3.5,
    x_multiplier: float = 1.0,
    max_x_units: int = 8,
    min_y_units: int = 100,
    min_ou_pairs: int = 30,
    initial_cash: float = 10_000_000.0,
    save_dir: str | Path | None = None,
    tag: str = "run",
):
    """Run the strategy on an aligned two-leg DataFrame.

    Returns ``(log, trades, final_value, strategy)``.
    """
    if not (0 < b < a < c):
        raise ValueError("thresholds must satisfy 0 < b < a < c")
    if window < min_ou_pairs + 1:
        raise ValueError("window must exceed min_ou_pairs")

    y_frame, x_frame = to_backtrader_frames(df)

    cerebro = bt.Cerebro()
    cerebro.adddata(bt.feeds.PandasData(dataname=y_frame, datetime="datetime"), name="Y")
    cerebro.adddata(bt.feeds.PandasData(dataname=x_frame, datetime="datetime"), name="X")
    cerebro.addstrategy(
        PairTradingStrategy,
        window=window,
        a=a,
        b=b,
        c=c,
        x_multiplier=x_multiplier,
        max_x_units=max_x_units,
        min_y_units=min_y_units,
        min_ou_pairs=min_ou_pairs,
    )
    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.0)

    strat = cerebro.run()[0]
    log = pd.DataFrame(strat.log)
    trades = pd.DataFrame(strat.trades)
    final_value = float(strat.broker.getvalue())

    if save_dir is not None:
        out = Path(save_dir)
        out.mkdir(parents=True, exist_ok=True)
        log.to_csv(out / f"backtest_log_{tag}.csv", index=False, encoding="utf-8-sig")
        if len(trades):
            trades.to_csv(
                out / f"backtest_trades_{tag}.csv", index=False, encoding="utf-8-sig"
            )

    return log, trades, final_value, strat
