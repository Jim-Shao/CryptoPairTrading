import os
import math
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from utils import drop_warmup_rows
from metrics import summarize_performance
import matplotlib.dates as mdates


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

    # Overlay cumulative fees (right axis) if provided.
    fee_entry_col = daily.get("fee_entry")
    fee_exit_col = daily.get("fee_exit")
    per_bar_fee = None
    if fee_entry_col is not None or fee_exit_col is not None:
        zeros = pd.Series(0.0, index=daily.index)
        fee_entry = fee_entry_col.fillna(0.0) if fee_entry_col is not None else zeros
        fee_exit = fee_exit_col.fillna(0.0) if fee_exit_col is not None else zeros
        per_bar_fee = (fee_entry + fee_exit).astype(float)

    fees_cumulative_col = daily.get("fees_cumulative")
    fees_cumulative = None
    total_fee_cost = np.nan
    if fees_cumulative_col is not None and len(fees_cumulative_col):
        fees_cumulative = pd.to_numeric(fees_cumulative_col, errors="coerce")
        fees_cumulative = (
            fees_cumulative.replace([np.inf, -np.inf], np.nan).ffill().fillna(0.0)
        )
        if len(fees_cumulative):
            try:
                total_fee_cost = float(fees_cumulative.iloc[-1])
            except (TypeError, ValueError):
                total_fee_cost = np.nan

    if np.isnan(total_fee_cost) and per_bar_fee is not None and len(per_bar_fee):
        try:
            total_fee_cost = float(per_bar_fee.sum())
        except (TypeError, ValueError):
            total_fee_cost = np.nan

    fee_ax = None
    if fees_cumulative is not None and fees_cumulative.notna().any():
        fee_ax = ax.twinx()
        fee_ax.set_ylabel("Fees", color="tab:red")
        fee_ax.tick_params(axis="y", colors="tab:red")
        fee_ax.spines["right"].set_color("tab:red")
        fee_ax.step(
            times,
            fees_cumulative,
            where="post",
            color="tab:red",
            linewidth=1.8,
            label="Fees (cumulative)",
        )

    perf_metrics = None
    try:
        perf_metrics, _ = summarize_performance(daily, pnl_col="pnl", time_col="time")
    except Exception:
        perf_metrics = None

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

    first_line = f"{title}: {times.iloc[0].date()} → {times.iloc[-1].date()}"

    second_parts: list[str] = []
    if perf_metrics:
        ret = perf_metrics.get("Ret")
        if ret is not None and np.isfinite(ret):
            second_parts.append(f"Ret={ret:+.2%}")
        ann_ret = perf_metrics.get("AnnRet")
        if ann_ret is not None and np.isfinite(ann_ret):
            second_parts.append(f"AnnRet={ann_ret:+.2%}")
        ann_vol = perf_metrics.get("AnnVol")
        if ann_vol is not None and np.isfinite(ann_vol):
            second_parts.append(f"AnnVol={ann_vol:.2%}")
        sharpe = perf_metrics.get("Sharpe")
        if sharpe is not None and np.isfinite(sharpe):
            second_parts.append(f"Sharpe={sharpe:.2f}")
        max_dd = perf_metrics.get("MaxDrawdown")
        if max_dd is not None and np.isfinite(max_dd):
            second_parts.append(f"MaxDD={max_dd:.2%}")
        maxdd_start = perf_metrics.get("MaxDD_Start")
        if isinstance(maxdd_start, pd.Timestamp):
            if not pd.isna(maxdd_start):
                second_parts.append(f"DD_start={maxdd_start.date()}")
        maxdd_end = perf_metrics.get("MaxDD_End")
        if isinstance(maxdd_end, pd.Timestamp):
            if not pd.isna(maxdd_end):
                second_parts.append(f"DD_end={maxdd_end.date()}")
    second_line = " | ".join(second_parts) if second_parts else ""

    reason_order = list(_EXIT_MARKER_STYLES.keys())
    extra_reasons = sorted(set(reason_returns) - set(reason_order))
    ordered_reasons = reason_order + extra_reasons

    def _avg_label(values: list[float]) -> str:
        finite = [v for v in values if np.isfinite(v)]
        if not finite:
            return "avg=—"
        return f"avg={np.mean(finite):+.2%}"

    third_parts = [f"trades={total_trades}"]
    if np.isfinite(total_fee_cost) and not math.isclose(
        total_fee_cost, 0.0, rel_tol=0.0, abs_tol=1e-12
    ):
        third_parts.append(f"fees={total_fee_cost:,.2f}")
    for reason in ordered_reasons:
        vals = reason_returns.get(reason, [])
        count = len(vals)
        third_parts.append(f"{reason}={count} ({_avg_label(vals)})")
    third_line = " | ".join(third_parts)

    title_lines = [first_line]
    if second_line:
        title_lines.append(second_line)
    title_lines.append(third_line)
    ax.set_title("\n".join(title_lines))
    ax.set_xlabel("Time")
    ax.set_ylabel("Portfolio value")
    handles, labels = ax.get_legend_handles_labels()
    if fee_ax is not None:
        fee_handles, fee_labels = fee_ax.get_legend_handles_labels()
        handles += fee_handles
        labels += fee_labels
    if handles:
        ax.legend(handles, labels)
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
                "stop_loss_pct": (
                    float(daily.loc[e_idx, "stop_loss_pct"])
                    if (
                        "stop_loss_pct" in daily.columns
                        and pd.notna(daily.loc[e_idx, "stop_loss_pct"])
                    )
                    else (
                        float(daily.loc[e_idx, "stop_k"])
                        if (
                            "stop_k" in daily.columns
                            and pd.notna(daily.loc[e_idx, "stop_k"])
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


def plot_trade_residual_spread_paths(
    daily: pd.DataFrame,
    prices_df: pd.DataFrame,
    *,
    entry_k: float = 2.0,
    exit_cross_k: float | None = None,
    stop_loss_pct: float | None = None,
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

    if stop_loss_pct is None:
        fallback = legacy_kwargs.pop("stop_k", None)
        if fallback is None:
            fallback = legacy_kwargs.pop("exit_plus_k", None)
        stop_loss_pct = float(fallback) if fallback is not None else None
    else:
        legacy_kwargs.pop("stop_k", None)
        legacy_kwargs.pop("exit_plus_k", None)
        stop_loss_pct = float(stop_loss_pct)

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
        stop_pct_seg = seg.get("stop_loss_pct")
        if stop_pct_seg is None:
            stop_pct_seg = stop_loss_pct
        stop_suffix = ""
        if stop_pct_seg is not None and np.isfinite(stop_pct_seg):
            stop_suffix = f", stop={float(stop_pct_seg):.0%}"
        meta_line = f"{side_label}, exit={exit_reason}{stop_suffix}{return_suffix}"
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

    if stop_loss_pct is not None:
        try:
            stop_label = f"{float(stop_loss_pct):.0%}"
        except (TypeError, ValueError):
            stop_label = str(stop_loss_pct)
    else:
        stop_label = "trade-specific"
    plt.suptitle(
        f"{title_prefix} (k_entry={entry_k}, k_cross={exit_cross_k}, stop_loss={stop_label})",
        y=1.02,
        fontsize=14,
    )
    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight")
    else:
        plt.show()
    plt.close()


def plot_portfolio_equity(curve: pd.DataFrame, save_path: str | None = None) -> None:
    df = curve.copy()
    if df.empty:
        return
    if "time" not in df.columns or "portfolio_equity" not in df.columns:
        return

    times = pd.to_datetime(df["time"])
    equity = pd.to_numeric(df["portfolio_equity"], errors="coerce")

    if equity.isna().all():
        return

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(times, equity, color="tab:blue", linewidth=1.5, label="Portfolio Equity")

    if "portfolio_cash" in df.columns:
        cash = pd.to_numeric(df["portfolio_cash"], errors="coerce")
        if not cash.isna().all():
            ax.plot(times, cash, color="tab:green", linewidth=1.2, label="Cash")

    if "portfolio_margin_used" in df.columns:
        margin = pd.to_numeric(df["portfolio_margin_used"], errors="coerce")
        if not margin.isna().all():
            ax.plot(times, margin, color="tab:orange", linewidth=1.2, label="Margin Used")

    if "portfolio_fees" in df.columns:
        fees = pd.to_numeric(df["portfolio_fees"], errors="coerce")
        if not fees.isna().all():
            ax.plot(times, fees, color="tab:red", linewidth=1.2, label="Fees (cum)")

    ax.set_xlabel("Time")
    ax.set_ylabel("Value")
    ax.set_title("Portfolio Metrics Over Time")
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.legend(loc="best")

    locator = mdates.AutoDateLocator()
    formatter = mdates.ConciseDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    fig.autofmt_xdate()

    plt.tight_layout()
    if save_path:
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()
    plt.close(fig)


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

    fig, (ax_beta, ax_p) = plt.subplots(2, 1, sharex=True, figsize=(14, 7), dpi=dpi)

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
    cmap: str = "viridis",
    vmin: float | None = None,
    vmax: float | None = None,
    fmt: str = "{:.2f}",
    annot: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Generic heatmap for a DataFrame pivot (index x columns)."""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data = piv.values
    if vmin is None or vmax is None:
        finite = data[np.isfinite(data)]
        if finite.size:
            if vmin is None:
                vmin = float(np.min(finite))
            if vmax is None:
                vmax = float(np.max(finite))

    im = ax.imshow(data, origin="lower", aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks(np.arange(len(piv.columns)))
    ax.set_xticklabels(
        [str(c) for c in piv.columns], rotation=rotate_xticks, ha="right"
    )
    ax.set_yticks(np.arange(len(piv.index)))
    ax.set_yticklabels([str(i) for i in piv.index])
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    if annot:
        for (i, j), val in np.ndenumerate(data):
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center", color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return ax


def heatmap_from_pivot_mask_nan(
    piv: pd.DataFrame,
    *,
    title: str,
    cbar_label: str,
    xlabel: str = "entry_k",
    ylabel: str = "train_len_days",
    figsize: tuple = (10, 6),
    rotate_xticks: int = 45,
    cmap: str = "viridis",
    fmt: str = "{:.2f}",
    annot: bool = True,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Heatmap variant that renders NaNs with a dedicated color."""

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure

    data = piv.values.astype(float)
    mask = ~np.isfinite(data)
    masked = np.ma.masked_array(data, mask=mask)

    cmap_obj = plt.get_cmap(cmap).copy()
    cmap_obj.set_bad(color="lightgray")

    finite = data[np.isfinite(data)]
    vmin = float(np.min(finite)) if finite.size else None
    vmax = float(np.max(finite)) if finite.size else None

    im = ax.imshow(
        masked, origin="lower", aspect="auto", cmap=cmap_obj, vmin=vmin, vmax=vmax
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

    if annot:
        for (i, j), val in np.ndenumerate(data):
            if np.isfinite(val):
                ax.text(j, i, fmt.format(val), ha="center", va="center", color="white")

    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(cbar_label)
    fig.tight_layout()
    return ax


def heatmap_from_results(
    results: pd.DataFrame,
    *,
    metric_col: str = "sharpe",
    title: str | None = None,
    cbar_label: str | None = None,
    annot: bool = True,
    fmt: str = "{:+.2f}",
    mask_nan: bool = True,
    xlabel: str = "entry_k",
    ylabel: str = "train_len_days",
    cmap: str = "viridis",
    ax: plt.Axes | None = None,
) -> plt.Axes:
    """Plot a heatmap for a metric contained in results_df (train_len_days x entry_k)."""

    required = {"train_len_days", "entry_k", metric_col}
    missing = required - set(results.columns)
    if missing:
        raise ValueError(f"results is missing columns: {sorted(missing)}")

    pivot = results.pivot(index="train_len_days", columns="entry_k", values=metric_col)

    title = title or f"{metric_col} heatmap"
    cbar = cbar_label or metric_col

    if mask_nan:
        return heatmap_from_pivot_mask_nan(
            pivot,
            title=title,
            cbar_label=cbar,
            xlabel=xlabel,
            ylabel=ylabel,
            annot=annot,
            fmt=fmt,
            cmap=cmap,
            ax=ax,
        )

    return heatmap_from_pivot(
        pivot,
        title=title,
        cbar_label=cbar,
        xlabel=xlabel,
        ylabel=ylabel,
        annot=annot,
        fmt=fmt,
        cmap=cmap,
        ax=ax,
    )


def heatmap_success_rate(
    results: pd.DataFrame,
    *,
    metric_col: str = "normal_exit_pct",
    title: str = "Success rate heatmap",
    cbar_label: str = "Success rate",
    fmt: str = "{:.0%}",
    **kwargs,
) -> plt.Axes:
    """Plot success-rate (normal exits / total trades) heatmap for results_df."""

    return heatmap_from_results(
        results,
        metric_col=metric_col,
        title=title,
        cbar_label=cbar_label,
        fmt=fmt,
        **kwargs,
    )


def heatmap_entry_count(
    results: pd.DataFrame,
    *,
    metric_col: str = "entry_count",
    title: str = "Entry count heatmap",
    cbar_label: str = "Entries",
    fmt: str = "{:.0f}",
    mask_nan: bool = False,
    **kwargs,
) -> plt.Axes:
    """Plot number-of-entries heatmap for results_df."""

    return heatmap_from_results(
        results,
        metric_col=metric_col,
        title=title,
        cbar_label=cbar_label,
        fmt=fmt,
        mask_nan=mask_nan,
        **kwargs,
    )


def heatmap_final_equity(
    results: pd.DataFrame,
    *,
    metric_col: str = "final_equity",
    title: str = "Final equity heatmap",
    cbar_label: str = "Final equity",
    fmt: str = "{:.0f}",
    mask_nan: bool = True,
    **kwargs,
) -> plt.Axes:
    """Plot final equity heatmap for results_df."""

    return heatmap_from_results(
        results,
        metric_col=metric_col,
        title=title,
        cbar_label=cbar_label,
        fmt=fmt,
        mask_nan=mask_nan,
        **kwargs,
    )
