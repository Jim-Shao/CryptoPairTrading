#
# File Name: metrics.py
# Create Time: 2025-10-01 15:00:29
# Modified Time: 2025-10-01 15:35:46
#


import numpy as np
import pandas as pd


# ---------- helpers ----------


def _ensure_sorted(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    if not df[time_col].is_monotonic_increasing:
        df = df.sort_values(time_col).reset_index(drop=True)
    return df


def _infer_annualization_factor(times: pd.Series) -> float:
    """Infer per-year factor from median spacing of timestamps."""
    t = pd.to_datetime(times)
    if len(t) < 2:
        return 252.0  # fallback
    dt = (t.diff().dropna().median()).total_seconds()
    sec_year = 365.25 * 24 * 3600
    return float(sec_year / max(dt, 1e-9))


def _cagr(first_value: float, last_value: float, times: pd.Series) -> float:
    t = pd.to_datetime(times)
    years = max((t.iloc[-1] - t.iloc[0]).days / 365.25, 1e-9)
    if first_value <= 0:
        return np.nan
    return (last_value / first_value) ** (1.0 / years) - 1.0


def _drawdown_stats(equity: pd.Series):
    """Return (dd series in pct), max dd, start idx, end idx, length (bars)."""
    peak = equity.cummax()
    dd = equity / peak - 1.0
    mdd = dd.min()
    end_idx = dd.idxmin()
    start_idx = equity.loc[:end_idx].idxmax()
    length = (
        int((dd.loc[start_idx:end_idx]).shape[0]) if start_idx is not None else np.nan
    )
    return dd, float(mdd), start_idx, end_idx, length


# ---------- public API ----------


def equity_to_returns(
    daily: pd.DataFrame,
    *,
    pnl_col: str = "pnl",
    time_col: str = "time",
    fill_start: float = 0.0,
) -> pd.DataFrame:
    """Add 'ret' (pct change), 'peak', 'dd' (drawdown %) to a copy of daily."""
    d = _ensure_sorted(daily.copy(), time_col)
    eq = d[pnl_col].astype(float)
    d["ret"] = eq.pct_change()
    if fill_start is not None and len(d):
        d.loc[d.index[0], "ret"] = fill_start
    d["peak"] = eq.cummax()
    d["dd"] = eq / d["peak"] - 1.0
    return d


def summarize_performance(
    daily: pd.DataFrame,
    *,
    pnl_col: str = "pnl",
    time_col: str = "time",
    rf_annual: float = 0.0,
) -> tuple[dict, pd.DataFrame]:
    """
    Compute a minimal set of core performance metrics:
      - CAGR, Annualized Volatility, Sharpe
      - Max Drawdown (value, start/end, length in bars)
    Returns (metrics_dict, augmented_daily_with_ret_peak_dd).
    """
    d = equity_to_returns(daily, pnl_col=pnl_col, time_col=time_col)
    d = _ensure_sorted(d, time_col)

    ann_factor = _infer_annualization_factor(d[time_col])

    r = d["ret"].astype(float)
    r_ex = r - rf_annual / ann_factor

    vol_ann = float(r.std(ddof=1) * np.sqrt(ann_factor)) if len(r) > 1 else np.nan
    sharpe = (
        float(r_ex.mean() / r.std(ddof=1) * np.sqrt(ann_factor))
        if r.std(ddof=1) > 0
        else np.nan
    )

    eq = d[pnl_col].astype(float)
    dd_series, mdd, dd_start, dd_end, dd_len = _drawdown_stats(eq)

    cagr = _cagr(eq.iloc[0], eq.iloc[-1], d[time_col])

    metrics = {
        "CAGR": cagr,
        "AnnVol": vol_ann,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,  # negative number, e.g., -0.23
        "MaxDD_Start": pd.to_datetime(dd_start) if dd_start is not None else None,
        "MaxDD_End": pd.to_datetime(dd_end) if dd_end is not None else None,
        "MaxDD_Length_Bars": dd_len,
    }
    return metrics, d


def summarize_close_reasons(
    daily: pd.DataFrame,
    *,
    pnl_col: str = "pnl",
    time_col: str = "time",
) -> tuple[dict, pd.DataFrame]:
    """
    Aggregate exit counts and PnL by reason.
    Reasons:
      - 'normal'            : exit_flag == True and force_close_reason is NaN/None
      - 'stop_loss'         : force_close_reason == 'stop_loss'
      - 'no_longer_coint'   : force_close_reason == 'no_longer_coint'

    Returns:
      - summary: {
           'normal': {'count': int, 'pnl': float},
           'stop_loss': {'count': int, 'pnl': float},
           'no_longer_coint': {'count': int, 'pnl': float},
           'total_trades': int,
           'total_pnl': float,
        }
      - trades_df: one row per trade with columns:
           ['entry_time','exit_time','entry_idx','exit_idx','reason','pnl']
    """
    d = _ensure_sorted(daily.copy(), time_col)
    eq = d[pnl_col].astype(float)

    # Pair entries and exits into trades
    entries = d.index[d.get("entry_flag", pd.Series(False, index=d.index))].tolist()
    exits = d.index[d.get("exit_flag", pd.Series(False, index=d.index))].tolist()

    trades = []
    i = j = 0
    while i < len(entries):
        e = entries[i]
        # find the first exit strictly after entry
        while j < len(exits) and exits[j] <= e:
            j += 1
        x = exits[j] if j < len(exits) else (len(d) - 1)
        # reason at exit
        reason_raw = d.get(
            "force_close_reason", pd.Series(index=d.index, data=None)
        ).iloc[x]
        if pd.isna(reason_raw) or reason_raw is None:
            reason = "normal"
        elif reason_raw in ("stop_loss", "no_longer_coint"):
            reason = str(reason_raw)
        else:
            # fallback in case of unforeseen label
            reason = str(reason_raw)

        pnl = float(eq.iloc[x] - eq.iloc[e])

        trades.append(
            {
                "entry_idx": int(e),
                "exit_idx": int(x),
                "entry_time": pd.to_datetime(d.loc[e, time_col]),
                "exit_time": pd.to_datetime(d.loc[x, time_col]),
                "reason": reason,
                "pnl": pnl,
            }
        )
        i += 1
        if j < len(exits) and exits[j] == x:
            j += 1

    trades_df = pd.DataFrame(trades)

    # Aggregate by reason
    reasons = ["normal", "stop_loss", "no_longer_coint"]
    summary = {r: {"count": 0, "pnl": 0.0} for r in reasons}
    if not trades_df.empty:
        grp = trades_df.groupby("reason")["pnl"].agg(["count", "sum"]).reset_index()
        for _, row in grp.iterrows():
            r = str(row["reason"])
            if r not in summary:
                summary[r] = {"count": 0, "pnl": 0.0}
            summary[r]["count"] = int(row["count"])
            summary[r]["pnl"] = float(row["sum"])

    total_trades = int(trades_df.shape[0]) if not trades_df.empty else 0
    total_pnl = float(trades_df["pnl"].sum()) if not trades_df.empty else 0.0
    summary["total_trades"] = total_trades
    summary["total_pnl"] = total_pnl

    return summary, trades_df
