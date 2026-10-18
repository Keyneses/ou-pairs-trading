"""End-to-end demo: features -> backtest -> metrics -> figures."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import load_aligned_csv, split_train_test
from src.metrics import performance
from src.ou import adf_pvalue, build_features, estimate_ou_ar1
from src.plots import (
    plot_equity_curves,
    plot_spread_position,
    plot_zscore_distribution,
    plot_zscore_timeseries,
)
from src.strategy import run_backtest
from src.synthetic import generate_synthetic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/synthetic_1min.csv")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--figure-dir", default="docs/figures")
    p.add_argument("--window", type=int, default=180)
    p.add_argument("--a", type=float, default=2.0)
    p.add_argument("--b", type=float, default=0.5)
    p.add_argument("--c", type=float, default=3.5)
    p.add_argument("--x-multiplier", type=float, default=10_000.0)
    p.add_argument("--max-x-units", type=int, default=8)
    p.add_argument("--min-y-units", type=int, default=100)
    p.add_argument("--min-ou-pairs", type=int, default=30)
    p.add_argument("--initial-cash", type=float, default=10_000_000.0)
    p.add_argument("--train-frac", type=float, default=0.5)
    p.add_argument("--bars-per-year", type=int, default=242 * 240)
    return p.parse_args()


def main() -> None:
    args = parse_args()

    data_path = Path(args.data)
    if not data_path.exists():
        print(f"{data_path} not found; generating the synthetic demo dataset")
        data_path.parent.mkdir(parents=True, exist_ok=True)
        generate_synthetic(n_days=120, seed=7).to_csv(data_path, index=False)

    df = load_aligned_csv(data_path)
    print(f"loaded {len(df):,} bars: {df['datetime'].min()} -> {df['datetime'].max()}")

    features = build_features(
        df,
        window=args.window,
        x_multiplier=args.x_multiplier,
        min_pairs=args.min_ou_pairs,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    features.to_csv(out_dir / "features.csv", index=False, encoding="utf-8-sig")

    train, _ = split_train_test(features, args.train_frac)
    train_spread = train["spread"].dropna()

    adf_stat, adf_p = adf_pvalue(train_spread)
    static_ou = estimate_ou_ar1(train_spread, min_pairs=args.min_ou_pairs)

    print()
    print("=== spread diagnostics (training half) ===")
    print(f"bars: {len(train):,}")
    print(f"ADF statistic: {adf_stat:.4f}   p-value: {adf_p:.4g}")
    if static_ou:
        print(
            f"static OU: mu={static_ou['mu']:.4f}  theta={static_ou['theta']:.6f}/bar"
            f"  half-life={static_ou['half_life']:.1f} bars"
        )
    valid_hl = features["half_life_roll"].dropna()
    if len(valid_hl):
        print(f"rolling half-life median: {valid_hl.median():.1f} bars")

    print()
    print("=== backtrader backtest ===")
    log, trades, final_value, _ = run_backtest(
        df,
        window=args.window,
        a=args.a,
        b=args.b,
        c=args.c,
        x_multiplier=args.x_multiplier,
        max_x_units=args.max_x_units,
        min_y_units=args.min_y_units,
        min_ou_pairs=args.min_ou_pairs,
        initial_cash=args.initial_cash,
        save_dir=out_dir,
        tag="baseline",
    )

    stats = performance(
        log["value"].to_numpy(),
        initial_cash=args.initial_cash,
        bars_per_year=args.bars_per_year,
    )
    stats.update(
        {
            "tag": "baseline",
            "window": args.window,
            "a": args.a,
            "b": args.b,
            "c": args.c,
            "trades": int(len(trades)),
        }
    )

    print(f"final value: {final_value:,.0f}")
    print(f"total return: {stats['total_return']:.2%}")
    print(f"annual return: {stats['annual_return']:.2%}")
    print(f"Sharpe: {stats['sharpe']:.2f}")
    print(f"max drawdown: {stats['max_drawdown']:.2%}")
    print(f"open trades: {len(trades):,}")

    pd.DataFrame([stats]).to_csv(
        out_dir / "summary.csv", index=False, encoding="utf-8-sig"
    )

    fig_dir = Path(args.figure_dir)
    plot_zscore_timeseries(
        log,
        a=args.a,
        b=args.b,
        c=args.c,
        window=args.window,
        path=fig_dir / "fig1_zscore_timeseries.png",
    )
    plot_zscore_distribution(
        log,
        train_frac=args.train_frac,
        path=fig_dir / "fig2_zscore_distribution.png",
    )
    plot_spread_position(log, path=fig_dir / "fig3_spread_position.png")
    plot_equity_curves(
        {"baseline": log},
        initial_cash=args.initial_cash,
        path=fig_dir / "fig4_equity_curve.png",
    )

    print()
    print(f"wrote outputs to {out_dir}/ and figures to {fig_dir}/")


if __name__ == "__main__":
    main()
