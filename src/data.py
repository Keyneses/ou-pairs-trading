"""Loading, validation and splitting helpers for aligned 1-minute data."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REQUIRED_COLUMNS = (
    "datetime",
    "y_open",
    "y_high",
    "y_low",
    "y_close",
    "x_open",
    "x_high",
    "x_low",
    "x_close",
)


def load_aligned_csv(path: str | Path, time_col: str = "datetime") -> pd.DataFrame:
    """Load an aligned two-leg 1-minute CSV and validate the schema."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python scripts/make_synthetic_data.py` "
            "or point --data at your own aligned CSV."
        )

    df = pd.read_csv(path)
    if time_col not in df.columns:
        raise ValueError(f"missing time column: {time_col}")

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            "missing required columns: "
            + ", ".join(missing)
            + ". See README section 'Data schema'."
        )

    df[time_col] = pd.to_datetime(df[time_col])
    df = (
        df.sort_values(time_col)
        .drop_duplicates(subset=[time_col], keep="last")
        .reset_index(drop=True)
    )

    if not df[["y_close", "x_close"]].notna().all().all():
        raise ValueError("y_close and x_close must be fully populated")

    return df


def split_train_test(df: pd.DataFrame, train_frac: float = 0.5):
    """Chronological split; never shuffle time-series data."""
    if not 0.0 < train_frac < 1.0:
        raise ValueError("train_frac must be in (0, 1)")
    cut = int(len(df) * train_frac)
    if cut < 2 or cut >= len(df) - 1:
        raise ValueError("not enough rows for the requested split")
    return df.iloc[:cut].copy(), df.iloc[cut:].copy()


def to_backtrader_frames(
    df: pd.DataFrame,
    *,
    time_col: str = "datetime",
    y_col: str = "y_close",
    x_col: str = "x_close",
):
    """Build the two OHLC frames consumed by backtrader."""
    y = df[
        [
            time_col,
            "y_open",
            "y_high",
            "y_low",
            "y_close",
        ]
    ].copy()
    y.columns = ["datetime", "open", "high", "low", "close"]

    x = df[
        [
            time_col,
            "x_open",
            "x_high",
            "x_low",
            "x_close",
        ]
    ].copy()
    x.columns = ["datetime", "open", "high", "low", "close"]

    if y_col != "y_close" or x_col != "x_close":
        raise ValueError("this repo expects y_close/x_close as the close columns")

    return y, x
