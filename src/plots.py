"""Figures for the README and the output directory."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE = "#4C72B0"
RED = "#C44E52"
GREEN = "#55A868"
PURPLE = "#8172B2"
GREY = "#8C8C8C"


def _prepare(log: pd.DataFrame) -> pd.DataFrame:
    out = log.copy()
    out["datetime"] = pd.to_datetime(out["datetime"])
    return out.sort_values("datetime").reset_index(drop=True)


def plot_zscore_timeseries(
    log: pd.DataFrame,
    *,
    a: float,
    b: float,
    c: float,
    window: int,
    path: str | Path,
) -> Path:
    df = _prepare(log).dropna(subset=["zscore"])
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 4.8))
    ax.plot(df["datetime"], df["zscore"], lw=0.35, color=BLUE, label="z-score")
    for level, color, style, label in [
        (a, RED, "--", f"open |z|={a:g}"),
        (b, GREEN, "-.", f"close |z|={b:g}"),
        (c, PURPLE, ":", f"stop |z|={c:g}"),
    ]:
        ax.axhline(level, color=color, lw=0.8, ls=style, label=label)
        ax.axhline(-level, color=color, lw=0.8, ls=style)
    ax.axhline(0, color="black", lw=0.5)

    signals = {
        "open_long": ("^", RED, "open long spread"),
        "open_short": ("^", BLUE, "open short spread"),
        "close": ("v", GREEN, "close"),
        "stop_loss": ("X", PURPLE, "stop"),
    }
    for action, (marker, color, label) in signals.items():
        sub = df[df["action"] == action]
        if len(sub):
            ax.scatter(
                sub["datetime"], sub["zscore"], marker=marker, s=14, color=color,
                label=label, zorder=3,
            )

    ax.set_title(
        f"z-score and three-threshold signals (W={window}, a={a:g}, b={b:g}, c={c:g})"
    )
    ax.set_ylabel("z-score")
    ax.legend(loc="upper left", fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_zscore_distribution(
    log: pd.DataFrame,
    *,
    train_frac: float,
    path: str | Path,
) -> Path:
    df = _prepare(log).dropna(subset=["zscore"])
    cut = int(len(df) * train_frac)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for ax, sample, name in [
        (axes[0], df["zscore"].iloc[:cut], "training half"),
        (axes[1], df["zscore"].iloc[cut:], "validation half"),
    ]:
        ax.hist(
            sample, bins=100, density=True, alpha=0.6, color=BLUE,
            label=f"sample (mean={sample.mean():.2f}, std={sample.std():.2f})",
        )
        xx = np.linspace(-6, 6, 300)
        ax.plot(xx, np.exp(-xx**2 / 2) / np.sqrt(2 * np.pi), color=RED, lw=1.2,
                label="N(0, 1)")
        ax.set_title(name)
        ax.set_xlabel("z-score")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    fig.suptitle("z-score distribution vs standard normal")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_spread_position(log: pd.DataFrame, *, path: str | Path) -> Path:
    df = _prepare(log)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(2, 1, figsize=(15, 6.5), sharex=True)
    axes[0].plot(df["datetime"], df["spread"], lw=0.35, color=BLUE, label="spread")
    axes[0].axhline(0, color="black", lw=0.5)
    axes[0].set_ylabel("spread")
    axes[0].set_title("Spread and positions")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=8)

    axes[1].fill_between(
        df["datetime"], df["y_pos"].fillna(0), step="pre", alpha=0.5,
        color=BLUE, label="y position",
    )
    axes[1].plot(df["datetime"], df["x_pos"].fillna(0), lw=0.6, color=RED,
                 label="x position")
    axes[1].set_ylabel("units")
    axes[1].legend(loc="upper left", fontsize=8)
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def plot_equity_curves(
    curves: dict[str, pd.DataFrame],
    *,
    initial_cash: float,
    path: str | Path,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(15, 4.4))
    colors = [RED, GREEN, GREY, BLUE]
    for (label, log), color in zip(curves.items(), colors):
        df = _prepare(log)
        ax.plot(df["datetime"], df["value"], lw=0.9, label=label, color=color)

    ax.axhline(initial_cash, color="black", lw=0.5, ls="--")
    ax.set_title("Equity curves")
    ax.set_ylabel("account value")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
