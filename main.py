# %%
from utils import load_crypto_dir, preprocess_ohlc, align_two
from trade import Exchange, CointBacktester
from plot import (
    plot_equity_with_trades,
    plot_trade_residual_paths,
    plot_trade_linear_spread_paths,
    plot_trade_residual_spread_paths,
    plot_beta_and_pvalue,
    heatmap_from_pivot,
    heatmap_from_pivot_mask_nan,
    _trade_segments_from_daily,
)
from metrics import summarize_performance, summarize_close_reasons, equity_to_returns
from tqdm import tqdm

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from pathlib import Path

try:  # pragma: no cover - IPython may not be available in CLI runs
    from IPython.display import display  # type: ignore
except ImportError:  # fallback for pure CLI usage
    def display(obj):
        print(obj)

# %%
data_dir = "/home/jim/CryptoPairTrading/data/futures"
start_date = start_date = (datetime(2023, 1, 1) - timedelta(days=119)).strftime(
    "%Y-%m-%d"
)
end_date = "2024-12-31"

df_dict = load_crypto_dir(data_dir, start_date=start_date, end_date=end_date)
sorted(df_dict.keys())

# %%
key1 = "KSMUSDT_1h"
key2 = "CELOUSDT_1h"

df1 = preprocess_ohlc(df_dict[key1])
df2 = preprocess_ohlc(df_dict[key2])

coint_var = "log_close"
merged_df = align_two(df1, df2, coint_var=coint_var)
merged_df.head()

# %%
ex = Exchange(merged_df, coint_var=coint_var)

start = merged_df["open_time"].min()
end = merged_df["open_time"].max()

bt = CointBacktester(
    exchange=ex,
    start_time=start,
    end_time=end,
    coint_var=coint_var,
    train_len=120 * 24,  # window length in bars
    gap=24,  # decision cadence in bars
    pval_alpha=0.05,
    entry_k=1.5,
    exit_k=0.5,
    relaxed_pval_alpha=0.20,  # optional
    stop_k=1.5,  # optional
    stop_loss_cooling_days=5.0,  # optional
    initial_balance=1_000_000.0,
    trade_frac=0.05,
    margin_rate=0.10,
    fee_rate=5e-4,
)

daily, fits = bt.run()
fits.head()

# %%
plot_beta_and_pvalue(daily, title=f"Beta & P-Value {key1} vs {key2}")

plot_equity_with_trades(daily, title=f"Backtest {key1} vs {key2}")

plot_trade_residual_spread_paths(
    daily,
    prices_df=merged_df[["open_time", "close_1", "close_2"]],
    entry_k=bt.entry_k,
    exit_cross_k=bt.exit_k,
    stop_k=bt.stop_k,
    max_trades=None,
    title_prefix=f"[Residual & Spread] {key1} vs {key2}",
)

# %%
bars_per_day = 24
INITIAL_BAL = 1_000_000.0
FIG_ROOT = Path("figures")
FIG_ROOT.mkdir(exist_ok=True)
prices_for_plots = merged_df[["open_time", "close_1", "close_2"]]


def run_once_return_daily(
    train_len_days: int, entry_k: float, gap_bars: int = 24
) -> tuple[pd.DataFrame, pd.DataFrame, CointBacktester]:
    bt = CointBacktester(
        exchange=ex,
        start_time=start,
        end_time=end,
        coint_var=coint_var,
        train_len=int(train_len_days * bars_per_day),
        gap=gap_bars,
        pval_alpha=0.05,
        entry_k=entry_k,
        exit_k=0.5,
        relaxed_pval_alpha=0.20,
        stop_k=1.5,
        stop_loss_cooling_days=5.0,
        initial_balance=INITIAL_BAL,
        trade_frac=0.05,
        margin_rate=0.10,
        fee_rate=5e-4,
    )
    daily_i, fits_i = bt.run()
    return daily_i, fits_i, bt


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        val = float(value)
        return val if np.isfinite(val) else float("nan")
    except (TypeError, ValueError):
        return float("nan")


def _trade_duration_stats(daily_df: pd.DataFrame) -> tuple[int, float, float]:
    segs = _trade_segments_from_daily(daily_df)
    if not segs:
        return 0, float("nan"), float("nan")

    durations = []
    for seg in segs:
        entry_time = pd.to_datetime(seg.get("entry_time"))
        exit_time = pd.to_datetime(seg.get("exit_time"))
        if pd.isna(entry_time) or pd.isna(exit_time):
            continue
        delta_days = (exit_time - entry_time).total_seconds() / (24 * 3600.0)
        durations.append(delta_days)

    avg_days = float(np.mean(durations)) if durations else float("nan")
    median_days = float(np.median(durations)) if durations else float("nan")
    return len(segs), avg_days, median_days


def _combo_slug(train_len_days: int, entry_k: float) -> str:
    return f"train_{train_len_days:03d}_entry_{entry_k:.1f}".replace(".", "p")


# Grid search across parameters
train_lens_days = list(range(30, 121, 30))
entry_ks = [round(x, 1) for x in np.arange(1.0, 2.5 + 1e-9, 0.5)]

grid_rows: list[dict] = []
for tl in tqdm(train_lens_days, desc="train_len_days"):
    for ek in tqdm(entry_ks, desc=f"entry_k (train={tl}d)", leave=False):
        daily_i, fits_i, bt_i = run_once_return_daily(train_len_days=tl, entry_k=ek, gap_bars=24)

        # Performance metrics (net of fees)
        try:
            perf_i, _ = summarize_performance(daily_i, pnl_col="pnl", time_col="time")
        except Exception:
            perf_i = {}

        # Trade duration stats
        n_trades, avg_revert_days, median_revert_days = _trade_duration_stats(daily_i)

        final_equity = float(daily_i["pnl"].iloc[-1]) if len(daily_i) else float("nan")
        final_pnl = final_equity - INITIAL_BAL
        ret_pct = _safe_float(perf_i.get("Ret")) if perf_i else (
            final_pnl / INITIAL_BAL if np.isfinite(final_pnl) else float("nan")
        )

        combo_dir = FIG_ROOT / _combo_slug(tl, ek)
        combo_dir.mkdir(parents=True, exist_ok=True)
        title_suffix = f"train={tl}d | entry_k={ek:.1f}"

        plot_equity_with_trades(
            daily_i,
            title=f"Backtest {key1} vs {key2} ({title_suffix})",
            save_path=str(combo_dir / "equity.png"),
        )
        plot_beta_and_pvalue(
            daily_i,
            title=f"Beta & P-Value {key1} vs {key2} ({title_suffix})",
            save_path=str(combo_dir / "beta_pvalue.png"),
        )
        plot_trade_residual_spread_paths(
            daily_i,
            prices_df=prices_for_plots,
            entry_k=bt_i.entry_k,
            exit_cross_k=bt_i.exit_k,
            stop_k=bt_i.stop_k,
            max_trades=None,
            title_prefix=f"[Residual & Spread] {key1} vs {key2} ({title_suffix})",
            save_path=str(combo_dir / "residual_spread.png"),
        )

        grid_rows.append(
            {
                "train_len_days": tl,
                "entry_k": ek,
                "n_trades": n_trades,
                "avg_revert_days": avg_revert_days,
                "median_revert_days": median_revert_days,
                "final_equity": final_equity,
                "final_pnl": final_pnl,
                "return_pct": ret_pct,
                "ann_ret": _safe_float(perf_i.get("AnnRet")) if perf_i else float("nan"),
                "ann_vol": _safe_float(perf_i.get("AnnVol")) if perf_i else float("nan"),
                "sharpe": _safe_float(perf_i.get("Sharpe")) if perf_i else float("nan"),
                "max_drawdown": _safe_float(perf_i.get("MaxDrawdown")) if perf_i else float("nan"),
            }
        )

results_df = (
    pd.DataFrame(grid_rows)
    .sort_values("sharpe", ascending=False, na_position="last")
    .reset_index(drop=True)
)
print("Top parameter sets by Sharpe ratio:\n", results_df.head())


# %%
# %%
pivot_cnt = results_df.pivot(
    index="train_len_days", columns="entry_k", values="n_trades"
)
pivot_avg = results_df.pivot(
    index="train_len_days", columns="entry_k", values="avg_revert_days"
)
pivot_eq = results_df.pivot(
    index="train_len_days", columns="entry_k", values="final_equity"
)

display(pivot_cnt)
display(pivot_avg)
display(pivot_eq)

# Uses plot.py helpers
heatmap_from_pivot(
    pivot_cnt,
    title="Trade count",
    cbar_label="count",
    xlabel="entry_k",
    ylabel="train_len_days",
)

heatmap_from_pivot(
    pivot_avg,
    title="Avg reversion time (days)",
    cbar_label="days",
    xlabel="entry_k",
    ylabel="train_len_days",
)

heatmap_from_pivot_mask_nan(
    pivot_eq,
    title="Final equity",
    cbar_label="USD",
    xlabel="entry_k",
    ylabel="train_len_days",
)

# %%
