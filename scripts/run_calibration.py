"""Grid-search thresholds on the training half, validate on the second half."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.calibration import grid_search, score_params
from src.data import load_aligned_csv, split_train_test
from src.metrics import performance
from src.ou import build_features
from src.plots import plot_equity_curves
from src.strategy import run_backtest
from src.synthetic import generate_synthetic


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", default="data/synthetic_1min.csv")
    p.add_argument("--output-dir", default="output")
    p.add_argument("--figure-dir", default="docs/figures")
    p.add_argument("--window", type=int, default=180)
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
    features = build_features(
        df,
        window=args.window,
        x_multiplier=args.x_multiplier,
        min_pairs=args.min_ou_pairs,
    )
    train, test = split_train_test(features, args.train_frac)

    grid = [
        (a, b, c)
        for a in (1.5, 2.0, 2.5)
        for b in (0.3, 0.5, 0.8)
        for c in (3.0, 3.5, 4.0)
        if 0 < b < a < c
    ]

    print(f"scanning {len(grid)} combinations on {len(train):,} training bars")
    results = grid_search(
        train,
        grid,
        x_multiplier=args.x_multiplier,
        bars_per_year=args.bars_per_year,
        max_x_units=args.max_x_units,
        min_y_units=args.min_y_units,
        initial_cash=args.initial_cash,
    )

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_dir / "calibration_grid.csv", index=False, encoding="utf-8-sig")

    display = results[
        ["a", "b", "c", "total_return", "sharpe", "max_drawdown", "open_bars"]
    ].head(10)
    print()
    print("=== training grid, top 10 by Sharpe ===")
    print(display.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    print()
    print("=== validation half, top 3 training params ===")
    validation = []
    for _, row in results.head(3).iterrows():
        stats = score_params(
            test,
            a=float(row["a"]),
            b=float(row["b"]),
            c=float(row["c"]),
            x_multiplier=args.x_multiplier,
            bars_per_year=args.bars_per_year,
            max_x_units=args.max_x_units,
            min_y_units=args.min_y_units,
            initial_cash=args.initial_cash,
        )
        stats.update({"a": float(row["a"]), "b": float(row["b"]), "c": float(row["c"])})
        validation.append(stats)
        print(
            f"a={row['a']:.1f} b={row['b']:.1f} c={row['c']:.1f}"
            f"  return={stats['total_return']:.2%}"
            f"  Sharpe={stats['sharpe']:.2f}"
            f"  maxDD={stats['max_drawdown']:.2%}"
        )

    pd.DataFrame(validation).to_csv(
        out_dir / "calibration_validation.csv", index=False, encoding="utf-8-sig"
    )

    best = results.iloc[0]
    configs = {
        "baseline (2.0, 0.5, 3.5)": (2.0, 0.5, 3.5),
        f"best ({best['a']:g}, {best['b']:g}, {best['c']:g})": (
            float(best["a"]),
            float(best["b"]),
            float(best["c"]),
        ),
    }

    print()
    print("=== backtrader, full sample ===")
    summaries = []
    curves = {}
    for label, (a, b, c) in configs.items():
        tag = label.split(" ")[0].replace("(", "").replace(")", "").replace(",", "")
        log, trades, final_value, _ = run_backtest(
            df,
            window=args.window,
            a=a,
            b=b,
            c=c,
            x_multiplier=args.x_multiplier,
            max_x_units=args.max_x_units,
            min_y_units=args.min_y_units,
            min_ou_pairs=args.min_ou_pairs,
            initial_cash=args.initial_cash,
            save_dir=out_dir,
            tag=tag,
        )
        stats = performance(
            log["value"].to_numpy(),
            initial_cash=args.initial_cash,
            bars_per_year=args.bars_per_year,
        )
        stats.update(
            {
                "config": label,
                "a": a,
                "b": b,
                "c": c,
                "trades": int(len(trades)),
            }
        )
        summaries.append(stats)
        curves[label] = log
        print(
            f"{label}: final={final_value:,.0f}"
            f"  return={stats['total_return']:.2%}"
            f"  Sharpe={stats['sharpe']:.2f}"
            f"  maxDD={stats['max_drawdown']:.2%}"
            f"  trades={len(trades):,}"
        )

    pd.DataFrame(summaries).to_csv(
        out_dir / "backtest_summary.csv", index=False, encoding="utf-8-sig"
    )

    fig_dir = Path(args.figure_dir)
    plot_equity_curves(
        curves,
        initial_cash=args.initial_cash,
        path=fig_dir / "fig5_equity_comparison.png",
    )
    print()
    print(f"wrote {out_dir}/calibration_grid.csv and comparison figure")


if __name__ == "__main__":
    main()
