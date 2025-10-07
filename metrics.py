#
# File Name: metrics.py
# Modified Time: 2025-10-07 04:34:34
#


import numpy as np
import pandas as pd

from utils import drop_warmup_rows


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


def _total_return(first_value: float, last_value: float) -> float:
    if first_value <= 0:
        return np.nan
    return last_value / first_value - 1.0


def _ann_return(first_value: float, last_value: float, times: pd.Series) -> float:
    t = pd.to_datetime(times)
    years = max((t.iloc[-1] - t.iloc[0]).days / 365.25, 1e-9)
    if first_value <= 0:
        return np.nan
    return (last_value / first_value) ** (1.0 / years) - 1.0


def _drawdown_stats(equity: pd.Series):
    """Return (dd series in pct), max dd, start idx, end idx, length (bars)."""
    if equity is None or equity.empty:
        return pd.Series(dtype=float), np.nan, None, None, np.nan

    series = equity.astype(float)
    peak = series.cummax()
    dd = series / peak - 1.0
    if dd.empty:
        return dd, np.nan, None, None, np.nan

    mdd = float(dd.min()) if len(dd) else np.nan
    if not np.isfinite(mdd):
        return dd, mdd, None, None, np.nan

    mask = np.isclose(dd.to_numpy(), mdd, atol=1e-12, rtol=1e-9)
    hit_indices = np.flatnonzero(mask)
    if hit_indices.size == 0:
        return dd, mdd, None, None, np.nan
    first_hit_pos = int(hit_indices[0])
    end_idx = dd.index[first_hit_pos]

    start_idx = None
    equity_up_to = series.loc[:end_idx]
    if not equity_up_to.empty:
        start_idx = equity_up_to.idxmax()

    length = np.nan
    if start_idx is not None and end_idx is not None:
        if start_idx <= end_idx:
            length = int(dd.loc[start_idx:end_idx].shape[0])
        else:
            length = int(dd.loc[end_idx:start_idx].shape[0])

    return dd, mdd, start_idx, end_idx, length


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
    d = drop_warmup_rows(d)
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

    total_ret = _total_return(eq.iloc[0], eq.iloc[-1])
    ann_ret = _ann_return(eq.iloc[0], eq.iloc[-1], d[time_col])

    maxdd_start_time = None
    if dd_start is not None and dd_start in d.index:
        maxdd_start_time = pd.to_datetime(d.loc[dd_start, time_col])

    maxdd_end_time = None
    if dd_end is not None and dd_end in d.index:
        maxdd_end_time = pd.to_datetime(d.loc[dd_end, time_col])

    metrics = {
        "Ret": total_ret,
        "AnnRet": ann_ret,
        "AnnVol": vol_ann,
        "Sharpe": sharpe,
        "MaxDrawdown": mdd,  # negative number, e.g., -0.23
        "MaxDD_Start": maxdd_start_time,
        "MaxDD_End": maxdd_end_time,
        "MaxDD_Length_Bars": dd_len,
    }
    return metrics, d
