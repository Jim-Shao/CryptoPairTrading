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

from utils import drop_warmup_rows


_EXIT_MARKER_STYLES = {
    "normal": {
        "facecolors": "none",
        "edgecolors": "gray",
        "linewidths": 1.2,
        "label": "Exit (normal)",
    },
    "stop_loss": {
        "facecolors": "tab:red",
        "edgecolors": "tab:red",
        "label": "Exit (stop loss)",
    },
    "no_longer_coint": {
        "facecolors": "none",
        "edgecolors": "tab:purple",
        "linewidths": 1.6,
        "label": "Exit (no longer coint)",
    },
}
_EXIT_FALLBACK_STYLE = {
    "facecolors": "none",
    "edgecolors": "black",
    "label": "Exit (other)",
}


def _scatter_exit(ax, x, y, reason: str, used_labels: set[str], *, size: int = 60):
    style = _EXIT_MARKER_STYLES.get(reason, _EXIT_FALLBACK_STYLE).copy()
    label = style.pop("label", None)
    if label and label in used_labels:
        label = None
    ax.scatter(x, y, marker="o", s=size, label=label, **style)
    if label:
        used_labels.add(label)


def plot_equity_with_trades(
    daily: pd.DataFrame,
    title: str = "Event-Driven Coint Backtest (OOP)",
    save_path: str | None = None,
):
    """Plot equity curve and annotate entries/exits."""
    daily = drop_warmup_rows(daily)
    if daily.empty:
        return

    fig, ax = plt.subplots(figsize=(14, 7))
    times = pd.to_datetime(daily["time"])
    ax.plot(times, daily["pnl"], color="black", label="PnL")

    start_eq = float(daily["pnl"].iloc[0]) if "pnl" in daily.columns else np.nan
    end_eq = float(daily["pnl"].iloc[-1]) if "pnl" in daily.columns else np.nan
    total_return = np.nan
    ann_return = np.nan
    if np.isfinite(start_eq) and start_eq > 0 and np.isfinite(end_eq) and end_eq > 0:
        ratio = end_eq / start_eq
        total_return = ratio - 1.0
        years = max((times.iloc[-1] - times.iloc[0]).days / 365.25, 1e-9)
        ann_return = ratio ** (1.0 / years) - 1.0 if years > 0 else np.nan

    segs = _trade_segments_from_daily(daily)
    total_trades = len(segs)
    reason_returns: dict[str, list[float]] = {}
    for seg in segs:
        entry_idx = seg["entry_idx"]
        exit_idx = seg["exit_idx"]
        entry_equity = float(daily.loc[entry_idx, "pnl"])
        exit_equity = float(daily.loc[exit_idx, "pnl"])
        trade_ret = np.nan
        if entry_equity > 0:
            trade_ret = exit_equity / entry_equity - 1.0
        reason = seg.get("exit_reason", "normal") or "normal"
        reason_returns.setdefault(reason, []).append(trade_ret)

    entry_flags = daily.get("entry_flag", pd.Series(False, index=daily.index))
    entry_sides = daily.get("entry_side", pd.Series(0, index=daily.index))
    ent_long = daily[entry_flags & (entry_sides == 1)]
    ent_short = daily[entry_flags & (entry_sides == -1)]

    if len(ent_long):
        ax.scatter(
            pd.to_datetime(ent_long["time"]),
            ent_long["pnl"],
            marker="^",
            s=60,
            label="Entry long",
        )
    if len(ent_short):
        ax.scatter(
            pd.to_datetime(ent_short["time"]),
            ent_short["pnl"],
            marker="v",
            s=60,
            label="Entry short",
        )

    exit_flags = daily.get("exit_flag", pd.Series(False, index=daily.index))
    exits = daily[exit_flags]
    if len(exits):
        reason_series = exits.get(
            "force_close_reason", pd.Series(index=exits.index, data="normal")
        )
        reason_series = (
            reason_series.fillna("normal")
            .replace("", "normal")
            .astype(str)
            .replace({"nan": "normal"})
        )
        used_labels: set[str] = set()
        for reason in reason_series.unique():
            mask = reason_series == reason
            _scatter_exit(
                ax,
                pd.to_datetime(exits.loc[mask, "time"]),
                exits.loc[mask, "pnl"],
                reason,
                used_labels,
                size=70,
            )

    first_line_parts = [
        f"{title}: {times.iloc[0].date()} → {times.iloc[-1].date()}"
    ]
    if np.isfinite(total_return):
        first_line_parts.append(f"return={total_return:+.2%}")
    if np.isfinite(ann_return):
        first_line_parts.append(f"ann={ann_return:+.2%}")
    first_line = " | ".join(first_line_parts)

    reason_order = list(_EXIT_MARKER_STYLES.keys())
    extra_reasons = sorted(set(reason_returns) - set(reason_order))
    ordered_reasons = reason_order + extra_reasons

    def _avg_label(values: list[float]) -> str:
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return "avg=—"
        return f"avg={np.mean(finite):+.2%}"

    second_parts = [f"trades={total_trades}"]
    for reason in ordered_reasons:
        vals = reason_returns.get(reason, [])
        count = len(vals)
        second_parts.append(f"{reason}={count} ({_avg_label(vals)})")
    second_line = " | ".join(second_parts)

    ax.set_title(f"{first_line}\n{second_line}")
    ax.set_xlabel("Time")
    ax.set_ylabel("Portfolio value")
    ax.legend()
    ax.grid(True)

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    else:
        plt.show()
    plt.close(fig)


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
                "entry_residual": (
                    float(daily.loc[e_idx, "residual"])
                    if pd.notna(daily.loc[e_idx, "residual"])
                    else None
                ),
                "entry_h_units": (
                    float(daily.loc[e_idx, "h_units"])
                    if pd.notna(daily.loc[e_idx, "h_units"])
                    else None
                ),
                "entry_qty_y": (
                    float(daily.loc[e_idx, "qty_y"])
                    if "qty_y" in daily.columns and pd.notna(daily.loc[e_idx, "qty_y"])
                    else None
                ),
                "entry_qty_x": (
                    float(daily.loc[e_idx, "qty_x"])
                    if "qty_x" in daily.columns and pd.notna(daily.loc[e_idx, "qty_x"])
                    else None
                ),
                "stop_k": (
                    float(daily.loc[e_idx, "stop_k"])
                    if (
                        "stop_k" in daily.columns
                        and pd.notna(daily.loc[e_idx, "stop_k"])
                    )
                    else (
                        float(daily.loc[e_idx, "exit_plus_k"])
                        if (
                            "exit_plus_k" in daily.columns
                            and pd.notna(daily.loc[e_idx, "exit_plus_k"])
                        )
                        else None
                    )
                ),
                "exit_reason": (
                    str(
                        daily.loc[x_idx, "force_close_reason"]
                        if "force_close_reason" in daily.columns
                        else "normal"
                    )
                    if (
                        "force_close_reason" in daily.columns
                        and pd.notna(daily.loc[x_idx, "force_close_reason"])
                        and str(daily.loc[x_idx, "force_close_reason"]).strip() != ""
                    )
                    else "normal"
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
    exit_cross_k: float | None = None,
    stop_k: float | None = None,
    max_trades: int | None = None,
    title_prefix: str = "[Residual] Trade paths",
    save_path: str | None = None,
    dpi: int = 150,
    **legacy_kwargs,
):
    """Per-trade residual path with entry/exit bands (log-scale)."""
    if exit_cross_k is None:
        exit_cross_k = float(legacy_kwargs.pop("exit_k", 0.5))
    else:
        legacy_kwargs.pop("exit_k", None)
        exit_cross_k = float(exit_cross_k)
    if stop_k is None:
        fallback = legacy_kwargs.pop("exit_plus_k", None)
        stop_k = float(fallback) if fallback is not None else None
    else:
        legacy_kwargs.pop("exit_plus_k", None)
        stop_k = float(stop_k)
    if legacy_kwargs:
        raise TypeError(f"Unexpected keyword arguments: {tuple(legacy_kwargs.keys())}")

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

    exit_label_used: set[str] = set()

    for i, seg in enumerate(segs, 1):
        e, x = seg["entry_idx"], seg["exit_idx"]
        window = daily.loc[e:x].reset_index(drop=True)
        es = seg["entry_sigma"]
        if es is None or np.isnan(es) or es <= 0:
            continue

        ax = plt.subplot(rows, cols, i)
        ax.plot(window["time"], window["residual"], label="Residual", lw=1.5)

        side = seg.get("side", 0)
        entry_line = None
        exit_line = None

        if side == -1:  # short
            entry_line = entry_k * es
            exit_line = exit_cross_k * es
        elif side == 1:  # long
            entry_line = -entry_k * es
            exit_line = -exit_cross_k * es

        if entry_line is not None and np.isfinite(entry_line):
            ax.axhline(
                entry_line,
                linestyle="--",
                color="tab:orange",
                label="Entry band" if i == 1 else None,
            )

        if exit_line is not None and np.isfinite(exit_line):
            ax.axhline(
                exit_line,
                linestyle="--",
                color="tab:green",
                label="Exit band" if i == 1 else None,
            )
        stop_val = seg.get("stop_k")
        if stop_val is None and stop_k is not None:
            stop_val = stop_k
        entry_res = seg.get("entry_residual")
        if entry_res is None or not np.isfinite(entry_res):
            entry_res = window["residual"].iloc[0]
        if stop_val is not None:
            try:
                stop_val = float(stop_val)
            except (TypeError, ValueError):
                stop_val = np.nan
        if (
            stop_val is not None
            and np.isfinite(stop_val)
            and stop_val > 0
            and np.isfinite(entry_res)
        ):
            if side == -1:  # short
                stop_line = entry_res + stop_val * es
            elif side == 1:  # long
                stop_line = entry_res - stop_val * es
            else:
                stop_line = np.nan

            if np.isfinite(stop_line):
                ax.axhline(
                    stop_line,
                    linestyle="-.",
                    color="tab:purple",
                    label="Stop band" if i == 1 else None,
                )
        ax.axhline(0.0, ls="--", c="red", lw=1.0, label="Zero" if i == 1 else None)

        entry_marker = "^" if side == 1 else "v"
        ax.scatter(
            window["time"].iloc[0],
            window["residual"].iloc[0],
            marker=entry_marker,
            s=60,
            label="Entry" if i == 1 else None,
        )
        _scatter_exit(
            ax,
            window["time"].iloc[-1],
            window["residual"].iloc[-1],
            seg.get("exit_reason", "normal") or "normal",
            exit_label_used,
        )

        ax.set_title(
            f"Trade {i}: {pd.to_datetime(daily.loc[e, 'time']).date()} → {pd.to_datetime(daily.loc[x, 'time']).date()} ({'LONG' if daily.loc[e, 'entry_side']==1 else 'SHORT'})"
        )
        ax.set_ylabel("Residual (log)")
        ax.grid(True)

    stop_label = f"{stop_k}" if stop_k is not None else "trade-specific"
    plt.suptitle(
        f"{title_prefix} (k_entry={entry_k}, k_cross={exit_cross_k}, k_stop={stop_label})",
        y=1.02,
        fontsize=14,
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


def plot_trade_residual_spread_paths(
    daily: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    entry_k: float = 2.0,
    exit_cross_k: float | None = None,
    stop_k: float | None = None,
    max_trades: int | None = None,
    title_prefix: str = "[Residual & Spread] Trade paths",
    residual_label: str = "Residual",
    spread_label: str = "PnL",
    residual_color: str = "tab:blue",
    spread_color: str = "black",
    save_path: str | None = None,
    dpi: int = 150,
    **legacy_kwargs,
):
    """Per-trade plot with residual (left axis) and rebased spread (right axis)."""

    if exit_cross_k is None:
        exit_cross_k = float(legacy_kwargs.pop("exit_k", 0.5))
    else:
        legacy_kwargs.pop("exit_k", None)
        exit_cross_k = float(exit_cross_k)

    if stop_k is None:
        fallback = legacy_kwargs.pop("exit_plus_k", None)
        stop_k = float(fallback) if fallback is not None else None
    else:
        legacy_kwargs.pop("exit_plus_k", None)
        stop_k = float(stop_k)

    if legacy_kwargs:
        raise TypeError(f"Unexpected keyword arguments: {tuple(legacy_kwargs.keys())}")

    px = prices_df[["open_time", "close_1", "close_2"]].rename(
        columns={"open_time": "time"}
    )
    merged = (
        pd.merge(daily, px, on="time", how="left")
        .dropna(subset=["close_1", "close_2", "residual"])
        .reset_index(drop=True)
    )

    segs = _trade_segments_from_daily(merged)
    if not segs:
        print("No trades to plot.")
        return
    if max_trades is not None:
        segs = segs[:max_trades]

    n = len(segs)
    cols = 3
    rows = math.ceil(n / cols)
    plt.figure(figsize=(6 * cols, 3.8 * rows))

    exit_label_used: set[str] = set()

    for i, seg in enumerate(segs, 1):
        e, x = seg["entry_idx"], seg["exit_idx"]
        es = seg.get("entry_sigma")
        h_units = seg.get("entry_h_units")
        side = seg.get("side", 0)

        if (
            es is None
            or np.isnan(es)
            or es <= 0
            or h_units is None
            or np.isnan(h_units)
        ):
            continue

        window = merged.loc[e:x].reset_index(drop=True)
        if window.empty:
            continue

        qty_y = seg.get("entry_qty_y")
        qty_x = seg.get("entry_qty_x")
        if (
            qty_y is None
            or qty_x is None
            or not np.isfinite(qty_y)
            or not np.isfinite(qty_x)
        ):
            continue

        entry_close1 = window["close_1"].iloc[0]
        entry_close2 = window["close_2"].iloc[0]
        delta_y = window["close_1"] - entry_close1
        delta_x = window["close_2"] - entry_close2

        if (
            not np.isfinite(delta_y.to_numpy()).all()
            or not np.isfinite(delta_x.to_numpy()).all()
            or not np.isfinite(window["residual"].to_numpy()).any()
        ):
            continue

        ax = plt.subplot(rows, cols, i)
        ax2 = ax.twinx()

        ax.plot(
            window["time"],
            window["residual"],
            color=residual_color,
            lw=1.5,
        )
        pnl_series = qty_y * delta_y - qty_x * delta_x

        ax2.plot(
            window["time"],
            pnl_series,
            color=spread_color,
            lw=1.4,
        )

        entry_line = exit_line = None
        if side == -1:  # short
            entry_line = entry_k * es
            exit_line = exit_cross_k * es
        elif side == 1:  # long
            entry_line = -entry_k * es
            exit_line = -exit_cross_k * es

        entry_marker = "^" if side == 1 else "v"

        if entry_line is not None and np.isfinite(entry_line):
            ax.axhline(
                entry_line,
                linestyle="--",
                color="tab:orange",
            )

        if exit_line is not None and np.isfinite(exit_line):
            ax.axhline(
                exit_line,
                linestyle="--",
                color="tab:green",
            )

        entry_res = seg.get("entry_residual")
        if entry_res is None or not np.isfinite(entry_res):
            entry_res = window["residual"].iloc[0]

        stop_val = seg.get("stop_k")
        if stop_val is None and stop_k is not None:
            stop_val = stop_k
        if stop_val is not None:
            try:
                stop_val = float(stop_val)
            except (TypeError, ValueError):
                stop_val = np.nan

        if (
            stop_val is not None
            and np.isfinite(stop_val)
            and stop_val > 0
            and np.isfinite(entry_res)
        ):
            if side == -1:
                stop_line = entry_res + stop_val * es
            elif side == 1:
                stop_line = entry_res - stop_val * es
            else:
                stop_line = np.nan

            if np.isfinite(stop_line):
                ax.axhline(
                    stop_line,
                    linestyle="-.",
                    color="tab:purple",
                )

        ax.axhline(0.0, ls="--", c="red", lw=1.0)
        ax2.axhline(0.0, ls=":", c="gray", lw=1.0)

        ax.scatter(
            window["time"].iloc[0],
            window["residual"].iloc[0],
            marker=entry_marker,
            s=60,
        )
        exit_reason = seg.get("exit_reason", "normal") or "normal"
        _scatter_exit(
            ax,
            window["time"].iloc[-1],
            window["residual"].iloc[-1],
            exit_reason,
            exit_label_used,
        )

        entry_time = pd.to_datetime(window["time"].iloc[0]).date()
        exit_time = pd.to_datetime(window["time"].iloc[-1]).date()
        side_label = "LONG" if side == 1 else "SHORT" if side == -1 else "FLAT"
        return_suffix = ""
        if "pnl" in window.columns:
            pnl_entry = float(window["pnl"].iloc[0])
            pnl_exit = float(window["pnl"].iloc[-1])
            if (
                np.isfinite(pnl_entry)
                and np.isfinite(pnl_exit)
                and not np.isclose(pnl_entry, 0.0)
            ):
                trade_return = pnl_exit / pnl_entry - 1.0
                return_suffix = f", return={trade_return:+.2%}"
        main_line = f"Trade {i}: {entry_time} → {exit_time}"
        meta_line = f"{side_label}, exit={exit_reason}{return_suffix}"
        ax.set_title(f"{main_line}\n{meta_line}")
        ax.set_ylabel(residual_label, color=residual_color)
        ax2.set_ylabel(spread_label, color=spread_color)

        ax.grid(True)
        ax2.grid(False)
        ax.tick_params(axis="x", labelrotation=60)
        ax.tick_params(axis="y", colors=residual_color)
        ax2.tick_params(axis="y", colors=spread_color)
        ax.spines["left"].set_color(residual_color)
        ax2.spines["right"].set_color(spread_color)

    stop_label = f"{stop_k}" if stop_k is not None else "trade-specific"
    plt.suptitle(
        f"{title_prefix} (k_entry={entry_k}, k_cross={exit_cross_k}, k_stop={stop_label})",
        y=1.02,
        fontsize=14,
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close()


def plot_beta_and_pvalue(
    daily: pd.DataFrame,
    *,
    title: str = "[Beta & p-value] Regime diagnostics",
    save_path: str | None = None,
    dpi: int = 150,
    alpha_active: float = 0.08,
):
    """Plot beta1 and p-value through time (shared time axis)."""

    d = drop_warmup_rows(daily)
    if d is None or d.empty:
        print("No data to plot.")
        return

    times = pd.to_datetime(d["time"])
    beta_series = pd.to_numeric(d.get("beta1"), errors="coerce")
    p_series = pd.to_numeric(d.get("p_value"), errors="coerce")

    fig, (ax_beta, ax_p) = plt.subplots(
        2, 1, sharex=True, figsize=(14, 7), dpi=dpi
    )

    ax_beta.plot(times, beta_series, color="tab:blue", label="beta1")
    entry_flags = d.get("entry_flag", pd.Series(False, index=d.index))
    exit_flags = d.get("exit_flag", pd.Series(False, index=d.index))

    if entry_flags.any():
        entries = d[entry_flags & np.isfinite(beta_series)]
        ax_beta.scatter(
            pd.to_datetime(entries["time"]),
            entries["beta1"],
            marker="^",
            color="tab:green",
            s=50,
            label="Entry",
        )
    if exit_flags.any():
        exits = d[exit_flags & np.isfinite(beta_series)]
        ax_beta.scatter(
            pd.to_datetime(exits["time"]),
            exits["beta1"],
            marker="v",
            color="tab:red",
            s=50,
            label="Exit",
        )

    if "in_coint_period" in d.columns:
        mask = d["in_coint_period"].astype(bool).values
        t_values = times.values
        i = 0
        while i < len(mask):
            if mask[i]:
                j = i
                while j + 1 < len(mask) and mask[j + 1]:
                    j += 1
                ax_beta.axvspan(
                    t_values[i], t_values[j], alpha=alpha_active, color="gray"
                )
                i = j + 1
            else:
                i += 1

    ax_beta.set_ylabel("beta1")
    ax_beta.legend(loc="best")
    ax_beta.grid(True)

    ax_p.plot(times, p_series, color="tab:orange", label="p-value")

    if "pval_alpha" in d.columns:
        ax_p.plot(
            times,
            pd.to_numeric(d["pval_alpha"], errors="coerce"),
            linestyle="--",
            color="tab:blue",
            label="pval_alpha",
        )
    if "relaxed_pval_alpha" in d.columns:
        ax_p.plot(
            times,
            pd.to_numeric(d["relaxed_pval_alpha"], errors="coerce"),
            linestyle=":",
            color="tab:red",
            label="relaxed_alpha",
        )

    ax_p.set_ylabel("p-value")
    ax_p.set_xlabel("Time")
    ax_p.set_ylim(0.0, 1.05)
    ax_p.grid(True)
    ax_p.legend(loc="best")

    plt.suptitle(title)
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        fig.savefig(save_path, dpi=dpi, bbox_inches="tight")
        print(f"Saved: {save_path}")
    else:
        plt.show()
    plt.close(fig)


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
