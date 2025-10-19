from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd
from tqdm import tqdm

from fetch_data import run as fetch_run
from plot import (
    plot_beta_and_pvalue,
    plot_equity_with_trades,
    plot_trade_residual_spread_paths,
    plot_portfolio_equity,
)
from portfolio import PairSpec, Portfolio
from trade import Exchange
from utils import align_two, load_crypto_dir, preprocess_ohlc


logger = logging.getLogger(__name__)

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")


DATA_DIR = Path("/home/jim/CryptoPairTrading/data/futures")
DATA_ROOT = DATA_DIR.parent
PAIR_CSV = Path("/home/jim/CryptoPairTrading/export_after_signal_test.csv")
OUTPUT_DIR = Path("runs/portfolio")

TRAIN_START = pd.Timestamp("2023-03-01")
TRAIN_END = pd.Timestamp("2025-02-28")
TEST_START = pd.Timestamp("2025-03-01")
TEST_END = pd.Timestamp("2025-09-30")

GRID_TRAIN_LEN = [30 * 24, 60 * 24, 120 * 24, 180 * 24]
GRID_ENTRY_K = [0.8, 1.0, 1.5, 2.0, 2.5, 3.0]
GRID_EXIT_K = [0, 0.5]

INITIAL_BALANCE = 1_000_000.0
RESET_LEN = 24
DEACTIVATE_EQUITY_RATIO = 0.20
DEFAULT_BACKTEST_KWARGS = dict(
    require_cointegration=True,
    reset_len=RESET_LEN,
    pval_alpha=0.1,
    exit_k=0.5,
    relaxed_pval_alpha=0.50,
    stop_loss_pct=0.20,
    stop_loss_cooling_days=5.0,
    margin_rate=0.10,
    trade_frac=0.50,
    fee_rate=5e-4,
    deactivate_equity_ratio=DEACTIVATE_EQUITY_RATIO,
)


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[str]:
    months = []
    cur = pd.Timestamp(start.year, start.month, 1)
    end_month = pd.Timestamp(end.year, end.month, 1)
    while cur <= end_month:
        months.append(cur.strftime("%Y-%m"))
        cur += pd.DateOffset(months=1)
    return months


def _missing_months_for_symbol(symbol: str, months: list[str]) -> list[str]:
    sym_dir = DATA_DIR / symbol / "1h"
    missing = []
    for ym in months:
        csv_path = sym_dir / f"{symbol}-1h-{ym}.csv"
        if not csv_path.exists():
            missing.append(ym)
    return missing


def ensure_data_for_pairs(
    pairs: Iterable[tuple[str, str]],
    months: list[str],
) -> None:
    symbols = {sym for pair in pairs for sym in pair}
    missing_symbols: dict[str, list[str]] = {}
    for symbol in symbols:
        missing = _missing_months_for_symbol(symbol, months)
        if missing:
            missing_symbols[symbol] = missing

    if not missing_symbols:
        logger.info("All required data present for %d symbols.", len(symbols))
        return

    for sym, miss_list in missing_symbols.items():
        logger.info(
            "Missing %d months for %s: %s", len(miss_list), sym, ", ".join(miss_list)
        )

    bases = sorted(
        {sym[:-4] if sym.endswith("USDT") else sym for sym in missing_symbols}
    )
    if not bases:
        return

    logger.info(
        "Fetching data for symbols: %s (months %s -> %s)",
        ", ".join(bases),
        months[0],
        months[-1],
    )
    try:
        fetch_run(
            cryptocurrencies=bases,
            quote_currencies=["USDT"],
            data_types=["futures"],
            begin=months[0],
            end=months[-1],
            interval="1h",
            save_root=str(DATA_ROOT),
            threads=8,
            proxy=None,
            timeout=30,
            retries=3,
        )
    except Exception as exc:
        logger.exception("Failed to download missing data: %s", exc)
        raise

    remaining = [sym for sym in symbols if _missing_months_for_symbol(sym, months)]
    if remaining:
        raise RuntimeError(f"Data still missing after download: {remaining}")


def _load_pair_data(
    symbol1: str,
    symbol2: str,
    *,
    data_cache: dict[str, pd.DataFrame],
) -> Optional[PairSpec]:
    key1 = f"{symbol1}_1h"
    key2 = f"{symbol2}_1h"
    if key1 not in data_cache or key2 not in data_cache:
        logger.warning("Missing data for pair %s-%s; skipping.", symbol1, symbol2)
        return None

    df1 = preprocess_ohlc(data_cache[key1])
    df2 = preprocess_ohlc(data_cache[key2])
    merged = align_two(df1, df2, coint_var="log_close")
    if merged.empty:
        logger.warning(
            "Aligned dataframe empty for pair %s-%s; skipping.", symbol1, symbol2
        )
        return None
    ex = Exchange(merged, coint_var="log_close")
    return PairSpec(
        name=f"{symbol1}-{symbol2}",
        exchange=ex,
        train_window=(TRAIN_START, TRAIN_END),
        test_window=(TEST_START, TEST_END),
        grid={
            "train_len": GRID_TRAIN_LEN,
            "entry_k": GRID_ENTRY_K,
            "exit_k": GRID_EXIT_K,
        },
        backtester_kwargs=dict(DEFAULT_BACKTEST_KWARGS),
        metrics_kwargs={"pnl_col": "pnl", "time_col": "time"},
    )


def _build_pair_specs(
    top_pairs: Iterable[tuple[str, str]],
    *,
    data_cache: dict[str, pd.DataFrame],
) -> list[PairSpec]:
    top_list = list(top_pairs)
    total = len(top_list)
    specs: list[PairSpec] = []
    skipped = 0
    for idx, (sym1, sym2) in enumerate(top_list, start=1):
        logger.info("Preparing pair %s-%s (%d/%d)...", sym1, sym2, idx, total)
        spec = _load_pair_data(sym1, sym2, data_cache=data_cache)
        if spec is None:
            skipped += 1
            continue
        specs.append(spec)
    if skipped:
        logger.info("Skipped %d pairs due to missing data.", skipped)
    return specs


def _load_top_pairs(n: int) -> list[tuple[str, str]]:
    df = pd.read_csv(PAIR_CSV, encoding="utf-8-sig")
    df_sorted = df.sort_values("Best IC").head(n)
    pairs: list[tuple[str, str]] = []
    for _, row in df_sorted.iterrows():
        c1 = str(row["Coin1"]).strip()
        c2 = str(row["Coin2"]).strip()
        pairs.append((c1, c2))
    return pairs


def _progress_worker(total: int, read_fd: int) -> None:
    with os.fdopen(read_fd, "rb", closefd=True) as pipe:
        with tqdm(total=total, desc="Backtests", dynamic_ncols=True) as bar:
            completed = 0
            while completed < total:
                chunk = pipe.read(1)
                if not chunk:
                    break
                bar.update(len(chunk))
                completed += len(chunk)

            if completed < total:
                bar.update(total - completed)


def main(
    top_n: int = 100,
    max_workers: Optional[int] = None,
    mp_context: Optional[mp.context.BaseContext] = None,
    run_tag: str | None = None,
) -> None:
    output_dir = OUTPUT_DIR
    if run_tag:
        safe_tag = str(run_tag).strip().replace(" ", "_")
        if safe_tag:
            output_dir = OUTPUT_DIR.parent / f"{OUTPUT_DIR.name}_{safe_tag}"
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / "backtest_params.json"
    with params_path.open("w", encoding="utf-8") as params_file:
        json.dump(DEFAULT_BACKTEST_KWARGS, params_file, indent=2, sort_keys=True)
    logger.info("Saved backtest parameters to %s", params_path)

    mp_ctx = mp_context or mp.get_context("fork")

    logger.info("Loading top %d pairs from %s...", top_n, PAIR_CSV)
    pairs = _load_top_pairs(top_n)
    logger.info("Loaded %d candidate pairs.", len(pairs))
    config_msg = (
        "Backtest configuration:\n"
        f"  Train window : {TRAIN_START.date()} -> {TRAIN_END.date()}\n"
        f"  Test window  : {TEST_START.date()} -> {TEST_END.date()}\n"
        f"  Train lengths: {GRID_TRAIN_LEN}\n"
        f"  Entry k grid : {GRID_ENTRY_K}\n"
        f"  Exit k grid  : {GRID_EXIT_K}\n"
        f"  Deactivate at: {DEACTIVATE_EQUITY_RATIO:.0%} of initial equity\n"
        f"  CPU workers : {max_workers if max_workers is not None else os.cpu_count()}\n"
        f"  Output dir  : {output_dir}"
    )
    print(config_msg)
    logger.info(config_msg)

    max_train_len = max(GRID_TRAIN_LEN)
    earliest_needed = TRAIN_START - pd.Timedelta(hours=max_train_len)
    months_needed = _month_range(earliest_needed, TEST_END)

    ensure_data_for_pairs(pairs, months_needed)
    symbols_needed = {sym for pair in pairs for sym in pair}

    cache_start = earliest_needed.strftime("%Y-%m-%d")
    cache_end = TEST_END.strftime("%Y-%m-%d")

    logger.info("Loading market data into cache from %s...", DATA_DIR)
    data_cache = load_crypto_dir(
        str(DATA_DIR),
        start_date=cache_start,
        end_date=cache_end,
        symbols=symbols_needed,
    )
    logger.info("Loaded %d symbol-frequency datasets.", len(data_cache))

    specs = _build_pair_specs(pairs, data_cache=data_cache)
    logger.info("Constructed %d PairSpec objects.", len(specs))
    if not specs:
        raise RuntimeError("No valid pairs found for backtesting.")

    total_runs = sum(
        len(spec.grid["train_len"])
        * len(spec.grid["entry_k"])
        * len(spec.grid["exit_k"])
        for spec in specs
    )
    logger.info("Total backtests to run: %d", total_runs)

    read_fd, write_fd = os.pipe()
    progress_process = mp_ctx.Process(
        target=_progress_worker, args=(total_runs, read_fd)
    )
    progress_process.start()
    os.close(read_fd)

    portfolio = Portfolio(
        specs,
        initial_balance=INITIAL_BALANCE,
        sort_by="Sharpe",
        ascending=False,
        max_workers=max_workers,
        mp_context=mp_ctx,
    )

    try:
        train_results = portfolio.train(progress_fd=write_fd)
    finally:
        os.close(write_fd)
        progress_process.join()
    logger.info("Training stage completed.")
    test_results = portfolio.run_test()
    logger.info("Testing stage completed.")
    equity_curve = portfolio.aggregate_equity()

    spec_map = {spec.name: spec for spec in specs}

    train_tables = {name: summary.table for name, summary in train_results.items()}
    for name, table in train_tables.items():
        path = output_dir / f"{name}_train_summary.csv"
        table.to_csv(path, index=False)
        logger.info("Saved training summary for %s to %s", name, path)

    # Plot only best training combo for each pair, with progress feedback
    best_plot_bar = tqdm(
        total=len(train_results), desc="Training plots", dynamic_ncols=True
    )
    train_best_rows: list[dict[str, object]] = []
    for name, summary in train_results.items():
        row: dict[str, object] = {"pair": name}
        best_params = summary.best_params or {}
        best_metrics = summary.best_metrics or {}
        top_row = (
            summary.table.iloc[0]
            if summary.table is not None and not summary.table.empty
            else None
        )
        if "train_len" in best_params:
            row["train_len"] = best_params["train_len"]
        elif top_row is not None and pd.notna(top_row.get("train_len")):
            row["train_len"] = int(top_row["train_len"])
        if "entry_k" in best_params:
            row["entry_k"] = best_params["entry_k"]
        elif top_row is not None and pd.notna(top_row.get("entry_k")):
            row["entry_k"] = float(top_row["entry_k"])
        if "exit_k" in best_params:
            row["exit_k"] = best_params["exit_k"]
        elif top_row is not None and pd.notna(top_row.get("exit_k")):
            row["exit_k"] = float(top_row["exit_k"])
        if top_row is not None:
            for col in ("Trades", "Exits", "FinalEquity"):
                if col in top_row:
                    row[col] = top_row[col]
        for key, value in best_metrics.items():
            row[key] = value
        train_best_rows.append(row)

        spec = spec_map.get(name)
        if spec is None:
            best_plot_bar.update(1)
            continue
        best_daily = summary.best_daily
        best_fits = summary.best_fits
        if best_daily is None or best_daily.empty:
            best_plot_bar.update(1)
            continue

        combo_dir = output_dir / name / "train"
        combo_dir.mkdir(parents=True, exist_ok=True)

        title_base = (
            f"{name} [Train] train_len={int(best_params.get('train_len', 0))}"
            f", entry_k={best_params.get('entry_k', 0):.3f}, exit_k={best_params.get('exit_k', 0):.3f}"
        )
        plot_equity_with_trades(
            best_daily,
            title=f"{title_base} Equity Curve",
            save_path=str(combo_dir / "equity.png"),
        )
        plot_beta_and_pvalue(
            best_daily,
            title=f"{title_base} Beta & P-Value",
            save_path=str(combo_dir / "beta_pvalue.png"),
        )
        prices_df = spec.exchange.df[["open_time", "close_1", "close_2"]]
        entry_k_val = best_params.get(
            "entry_k", spec.backtester_kwargs.get("entry_k", 0.0)
        )
        exit_k_val = best_params.get(
            "exit_k", spec.backtester_kwargs.get("exit_k", 0.5)
        )
        plot_trade_residual_spread_paths(
            best_daily,
            prices_df=prices_df,
            entry_k=entry_k_val,
            exit_cross_k=exit_k_val,
            stop_loss_pct=DEFAULT_BACKTEST_KWARGS.get("stop_loss_pct"),
            title_prefix=f"{title_base} Residual & Spread",
            save_path=str(combo_dir / "residual_spread.png"),
        )
        logger.info("Saved training plots for %s", name)
        best_plot_bar.update(1)

        summary.best_daily = None
        summary.best_fits = None
    best_plot_bar.close()

    if train_best_rows:
        train_best_df = pd.DataFrame(train_best_rows)
        train_best_df = train_best_df.sort_values("pair").reset_index(drop=True)
        train_best_path = output_dir / "train_pair_summary.csv"
        train_best_df.to_csv(train_best_path, index=False)
        logger.info("Saved best-train summary table to %s", train_best_path)

    test_best_rows: list[dict[str, object]] = []
    for name, result in test_results.items():
        pair_dir = output_dir / name
        pair_dir.mkdir(parents=True, exist_ok=True)

        row: dict[str, object] = {"pair": name}
        best_params = result.train_summary.best_params or {}
        row["train_len"] = best_params.get("train_len")
        row["entry_k"] = best_params.get("entry_k")
        row["exit_k"] = best_params.get("exit_k")

        daily = result.test_daily
        train_len_val = best_params.get("train_len")
        entry_k_val = best_params.get("entry_k")
        exit_k_val = best_params.get("exit_k")

        if daily is not None and not daily.empty:
            entry_flags = daily.get("entry_flag")
            exit_flags = daily.get("exit_flag")
            if entry_flags is not None:
                entry_series = pd.to_numeric(pd.Series(entry_flags), errors="coerce")
                row["Trades"] = int(entry_series.fillna(0).sum())
            if exit_flags is not None:
                exit_series = pd.to_numeric(pd.Series(exit_flags), errors="coerce")
                row["Exits"] = int(exit_series.fillna(0).sum())
            pnl_series = pd.to_numeric(daily["pnl"], errors="coerce")
            if not pnl_series.empty and pd.notna(pnl_series.iloc[-1]):
                row["FinalEquity"] = float(pnl_series.iloc[-1])
            if "fees_cumulative" in daily.columns:
                fees_series = pd.to_numeric(daily["fees_cumulative"], errors="coerce")
                if not fees_series.empty and pd.notna(fees_series.iloc[-1]):
                    row["FinalFees"] = float(fees_series.iloc[-1])

            daily_path = pair_dir / "test_daily.csv"
            daily.to_csv(daily_path, index=False)
            logger.info("Saved test daily log for %s to %s", name, daily_path)

            entry_k = entry_k_val if entry_k_val is not None else GRID_ENTRY_K[0]
            exit_k = (
                exit_k_val
                if exit_k_val is not None
                else DEFAULT_BACKTEST_KWARGS.get("exit_k", 0.5)
            )
            stop_loss_pct = DEFAULT_BACKTEST_KWARGS.get("stop_loss_pct")

            train_len_fmt = (
                f"{int(train_len_val)}" if train_len_val is not None else "?"
            )
            title_prefix = (
                f"{name} [Test] train_len={train_len_fmt}, "
                f"entry_k={float(entry_k):.3f}, exit_k={float(exit_k):.3f}"
            )
            plot_equity_with_trades(
                daily,
                title=f"{title_prefix} Equity Curve",
                save_path=str(pair_dir / "equity.png"),
            )
            logger.info("Saved equity plot for %s", name)
            plot_beta_and_pvalue(
                daily,
                title=f"{title_prefix} Beta & P-Value",
                save_path=str(pair_dir / "beta_pvalue.png"),
            )
            logger.info("Saved beta/p-value plot for %s", name)
            prices_df = result.pair.exchange.df[["open_time", "close_1", "close_2"]]
            plot_trade_residual_spread_paths(
                daily,
                prices_df=prices_df,
                entry_k=entry_k,
                exit_cross_k=exit_k,
                stop_loss_pct=stop_loss_pct,
                title_prefix=f"{title_prefix} Residual & Spread",
                save_path=str(pair_dir / "residual_spread.png"),
            )
            logger.info("Saved residual/spread plot for %s", name)

        if result.test_metrics is not None:
            metrics_path = pair_dir / "test_metrics.csv"
            pd.Series(result.test_metrics).to_csv(metrics_path)
            logger.info("Saved test metrics for %s to %s", name, metrics_path)

        metrics = result.test_metrics or {}
        for key, value in metrics.items():
            row[key] = value
        test_best_rows.append(row)

    if not equity_curve.empty:
        port_path = output_dir / "portfolio_equity.csv"
        equity_curve.to_csv(port_path, index=False)
        logger.info("Saved portfolio equity curve to %s", port_path)

        equity_plot_path = output_dir / "portfolio_equity.png"
        plot_portfolio_equity(equity_curve, str(equity_plot_path))
        logger.info("Saved portfolio equity plot to %s", equity_plot_path)
    else:
        logger.warning("Portfolio equity curve is empty; nothing saved.")

    if test_best_rows:
        test_best_df = pd.DataFrame(test_best_rows)
        test_best_df = test_best_df.sort_values("pair").reset_index(drop=True)
        test_best_path = output_dir / "test_pair_summary.csv"
        test_best_df.to_csv(test_best_path, index=False)
        logger.info("Saved best-test summary table to %s", test_best_path)

    logger.info("Backtest workflow complete.")


if __name__ == "__main__":
    RUN_TAG = "coint_1018_20stop"  # e.g. "no_coint_run"
    TOP_N = 100
    MAX_WORKERS = os.cpu_count() - 2

    logging.basicConfig(
        filename="backtest.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        force=True,
    )
    mp_ctx = mp.get_context("fork")
    main(
        top_n=TOP_N,
        max_workers=MAX_WORKERS,
        mp_context=mp_ctx,
        run_tag=RUN_TAG,
    )
