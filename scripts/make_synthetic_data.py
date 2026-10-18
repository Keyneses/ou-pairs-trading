"""Generate the bundled synthetic cointegrated 1-minute dataset."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.synthetic import generate_synthetic


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", default="data/synthetic_1min.csv")
    parser.add_argument("--days", type=int, default=120)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--start", default="2024-01-02")
    args = parser.parse_args()

    df = generate_synthetic(n_days=args.days, seed=args.seed, start=args.start)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)

    print(f"wrote {len(df):,} bars to {out}")
    print(f"date range: {df['datetime'].min()} -> {df['datetime'].max()}")
    print(f"y_close range: {df['y_close'].min():.4f} .. {df['y_close'].max():.4f}")
    print(f"x_close range: {df['x_close'].min():.4f} .. {df['x_close'].max():.4f}")


if __name__ == "__main__":
    main()
