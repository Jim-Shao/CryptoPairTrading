#
# File Name: plot.py
# Create Time: 2025-09-29 23:33:51
# Modified Time: 2025-10-01 16:01:41
#


import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_equity_with_trades(
    daily: pd.DataFrame,
    title: str = "Event-Driven Coint Backtest (OOP)",
    save_path: str | None = None,
):
    """Plot equity curve and annotate entries/exits."""
    plt.figure(figsize=(14, 7))
    plt.plot(daily["time"], daily["pnl"], color="black", label="PnL")

    # Shade cointegration-active periods
    if "in_coint_period" in daily.columns:
        on = daily["in_coint_period"].astype(bool).values
        times = daily["time"].values
        i = 0
        while i < len(on):
            if on[i]:
                j = i
                while j + 1 < len(on) and on[j + 1]:
                    j += 1
                plt.axvspan(times[i], times[j], alpha=0.08)
                i = j + 1
            else:
                i += 1

    ent_long = daily[(daily["entry_flag"]) & (daily["entry_side"] == 1)]
    ent_short = daily[(daily["entry_flag"]) & (daily["entry_side"] == -1)]
    exits = daily[daily["exit_flag"]]

    if len(ent_long):
        plt.scatter(
            ent_long["time"], ent_long["pnl"], marker="^", s=60, label="Entry long"
        )
    if len(ent_short):
        plt.scatter(
            ent_short["time"], ent_short["pnl"], marker="v", s=60, label="Entry short"
        )
    if len(exits):
        plt.scatter(
            exits["time"],
            exits["pnl"],
            marker="o",
            s=60,
            facecolors="none",
            edgecolors="gray",
            label="Exit",
        )

    plt.title(title)
    plt.xlabel("Time")
    plt.ylabel("Portfolio value")
    plt.legend()
    plt.grid(True)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()  # disable for headless


def _trade_segments_from_daily(daily: pd.DataFrame) -> list[dict]:
    """Extract trade segments [entry_idx .. exit_idx] from the daily log."""
    entries = daily.index[daily["entry_flag"]].tolist()
    exits = daily.index[daily["exit_flag"]].tolist()
    segs, e_ptr, x_ptr = [], 0, 0
    while e_ptr < len(entries):
        e_idx = entries[e_ptr]
        while x_ptr < len(exits) and exits[x_ptr] <= e_idx:
            x_ptr += 1
        x_idx = exits[x_ptr] if x_ptr < len(exits) else (len(daily) - 1)
        segs.append(
            {
                "entry_idx": e_idx,
                "exit_idx": x_idx,
                "side": int(daily.loc[e_idx, "entry_side"]),
                "entry_time": daily.loc[e_idx, "time"],
                "exit_time": daily.loc[x_idx, "time"],
                "entry_sigma": (
                    float(daily.loc[e_idx, "sigma"])
                    if pd.notna(daily.loc[e_idx, "sigma"])
                    else None
                ),
                "entry_h_units": (
                    float(daily.loc[e_idx, "h_units"])
                    if pd.notna(daily.loc[e_idx, "h_units"])
                    else None
                ),
            }
        )
        e_ptr += 1
        if x_ptr < len(exits) and exits[x_ptr] == x_idx:
            x_ptr += 1
    return segs


def plot_trade_residual_paths(
    daily: pd.DataFrame,
    *,
    entry_k: float = 2.0,
    exit_k: float = 0.5,
    max_trades: int | None = None,
    title_prefix: str = "[Residual] Trade paths",
    save_path: str | None = None,
    dpi: int = 150,
):
    """Per-trade residual path with entry/exit bands (log-scale)."""
    segs = _trade_segments_from_daily(daily)
    if not segs:
        print("No trades to plot.")
        return
    if max_trades is not None:
        segs = segs[:max_trades]

    n = len(segs)
    cols = 3
    rows = math.ceil(n / cols)
    plt.figure(figsize=(6 * cols, 3.8 * rows))

    for i, seg in enumerate(segs, 1):
        e, x = seg["entry_idx"], seg["exit_idx"]
        window = daily.loc[e:x].reset_index(drop=True)
        es = seg["entry_sigma"]
        if es is None or np.isnan(es) or es <= 0:
            continue

        ax = plt.subplot(rows, cols, i)
        ax.plot(window["time"], window["residual"], label="Residual", lw=1.5)

        up = entry_k * es
        dn = -entry_k * es
        up_exit = exit_k * es
        dn_exit = -exit_k * es
        ax.hlines(
            [up, dn],
            window["time"].iloc[0],
            window["time"].iloc[-1],
            linestyles="--",
            colors="tab:orange",
            label="+/- entry band" if i == 1 else None,
        )
        ax.hlines(
            [up_exit, dn_exit],
            window["time"].iloc[0],
            window["time"].iloc[-1],
            linestyles="--",
            colors="tab:green",
            label="+/- exit band" if i == 1 else None,
        )
        ax.axhline(0.0, ls="--", c="red", lw=1.0, label="Zero" if i == 1 else None)

        ax.scatter(
            window["time"].iloc[0],
            window["residual"].iloc[0],
            marker="^" if daily.loc[e, "entry_side"] == 1 else "v",
            s=60,
            label="Entry" if i == 1 else None,
        )
        ax.scatter(
            window["time"].iloc[-1],
            window["residual"].iloc[-1],
            marker="o",
            s=60,
            facecolors="none",
            edgecolors="gray",
            label="Exit" if i == 1 else None,
        )

        ax.set_title(
            f"Trade {i}: {pd.to_datetime(daily.loc[e, 'time']).date()} → {pd.to_datetime(daily.loc[x, 'time']).date()} ({'LONG' if daily.loc[e, 'entry_side']==1 else 'SHORT'})"
        )
        ax.set_ylabel("Residual (log)")
        ax.grid(True)

    plt.suptitle(
        f"{title_prefix} (k_entry={entry_k}, k_exit={exit_k})", y=1.02, fontsize=14
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_trade_linear_spread_paths(
    daily: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    max_trades: int | None = None,
    title_prefix: str = "[Linear spread] Trade paths",
    save_path: str | None = None,
    dpi: int = 150,
):
    """
    For each trade, plot S_t = close_1 - h_units_entry * close_2,
    rebased to zero at entry (directly comparable across trades).
    """
    px = prices_df[["open_time", "close_1", "close_2"]].rename(
        columns={"open_time": "time"}
    )
    d = pd.merge(daily, px, on="time", how="left").dropna(subset=["close_1", "close_2"])

    segs = _trade_segments_from_daily(d)
    if not segs:
        print("No trades to plot.")
        return
    if max_trades is not None:
        segs = segs[:max_trades]

    n = len(segs)
    cols = 3
    rows = math.ceil(n / cols)
    plt.figure(figsize=(6 * cols, 3.8 * rows))

    for i, seg in enumerate(segs, 1):
        e, x = seg["entry_idx"], seg["exit_idx"]
        h_units_e = seg["entry_h_units"]
        if h_units_e is None or np.isnan(h_units_e):
            continue

        w = d.loc[e:x].reset_index(drop=True)
        s = w["close_1"] - h_units_e * w["close_2"]
        s_rel = s - s.iloc[0]

        ax = plt.subplot(rows, cols, i)
        ax.plot(w["time"], s_rel, label="S_t - S_entry", lw=1.5, color="black")
        ax.axhline(0.0, ls="--", c="gray", lw=1.0)
        ax.scatter(
            w["time"].iloc[0],
            s_rel.iloc[0],
            marker="^" if d.loc[e, "entry_side"] == 1 else "v",
            s=60,
            label="Entry" if i == 1 else None,
        )
        ax.scatter(
            w["time"].iloc[-1],
            s_rel.iloc[-1],
            marker="o",
            s=60,
            facecolors="none",
            edgecolors="tab:orange",
            label="Exit" if i == 1 else None,
        )

        ax.set_title(
            f"Trade {i}: {pd.to_datetime(d.loc[e, 'time']).date()} → {pd.to_datetime(d.loc[x, 'time']).date()} ({'LONG' if d.loc[e, 'entry_side']==1 else 'SHORT'})"
        )
        ax.set_ylabel("Spread (rebased)")
        ax.grid(True)

    plt.suptitle(
        f"{title_prefix} (S_t = close_1 - h_units*close_2)", y=1.02, fontsize=14
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def heatmap_from_pivot(
    piv: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    xlabel: str = "entry_k",
    ylabel: str = "train_len_days",
    figsize: tuple = (10, 6),
    rotate_xticks: int = 45,
):
    """
    Generic heatmap for a DataFrame pivot (index x columns).
    """
    fig, ax = plt.subplots(figsize=figsize)
    data = piv.values
    im = ax.imshow(data, origin="lower", aspect="auto")
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(
        [str(c) for c in piv.columns], rotation=rotate_xticks, ha="right"
    )
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(i) for i in piv.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.show()
    plt.close(fig)


def heatmap_from_pivot_mask_nan(
    piv: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    xlabel: str = "entry_k",
    ylabel: str = "train_len_days",
    figsize: tuple = (10, 6),
    rotate_xticks: int = 45,
):
    """
    Heatmap that treats NaNs as a separate color (useful for final_equity grids).
    """
    fig, ax = plt.subplots(figsize=figsize)

    data = piv.values.astype(float)
    mask = ~np.isfinite(data)
    masked = np.ma.masked_array(data, mask=mask)

    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="lightgray")

    finite_vals = data[np.isfinite(data)]
    vmin = np.min(finite_vals) if finite_vals.size else None
    vmax = np.max(finite_vals) if finite_vals.size else None

    im = ax.imshow(
        masked, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax
    )
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(
        [str(c) for c in piv.columns], rotation=rotate_xticks, ha="right"
    )
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(i) for i in piv.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    plt.tight_layout()
    plt.show()
    plt.close(fig)
