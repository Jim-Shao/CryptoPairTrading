from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any, Callable, Iterable, Sequence
import logging

import numpy as np
import pandas as pd

from metrics import summarize_performance
from trade import CointBacktester, Exchange

logger = logging.getLogger(__name__)


@dataclass
class GridSearchSummary:
    """
    Container for grid-search results.

    Attributes
    ----------
    table : pd.DataFrame
        Metrics table with one row per parameter combination (sorted).
    best_params : dict[str, Any] | None
        Parameter dictionary for the top-ranked row (None if no successful runs).
    best_metrics : dict[str, Any] | None
        Metrics dictionary for the best row (None if no successful runs).
    best_daily : pd.DataFrame | None
        Daily equity log for the best run (None if unavailable).
    best_fits : pd.DataFrame | None
        Fits log for the best run (None if unavailable).
    """

    table: pd.DataFrame
    best_params: dict[str, Any] | None
    best_metrics: dict[str, Any] | None
    best_daily: pd.DataFrame | None
    best_fits: pd.DataFrame | None
    runs: list[dict[str, Any]]


def _as_list(seq: Sequence[Any] | Iterable[Any] | Any) -> list[Any]:
    if isinstance(seq, (str, bytes)):
        return [seq]
    if isinstance(seq, Iterable):
        return list(seq)
    return [seq]


def grid_search_parameters(
    exchange: Exchange,
    *,
    start_time: pd.Timestamp,
    end_time: pd.Timestamp,
    train_len_values: Sequence[int] | Iterable[int],
    entry_k_values: Sequence[float] | Iterable[float],
    exit_k_values: Sequence[float] | Iterable[float],
    sort_by: str = "Sharpe",
    ascending: bool | None = None,
    backtester_kwargs: dict[str, Any] | None = None,
    metrics_kwargs: dict[str, Any] | None = None,
    raise_on_error: bool = False,
    progress_cb: Callable[[dict[str, Any]], None] | None = None,
) -> GridSearchSummary:
    """
    Run a brute-force grid search over (train_len, entry_k, exit_k).

    Parameters
    ----------
    exchange : Exchange
        Pre-aligned data source.
    start_time, end_time : pd.Timestamp
        Inclusive window for each backtest.
    train_len_values, entry_k_values, exit_k_values :
        Sequences defining the grid for each dimension.
    sort_by : str
        Column used to rank the resulting table (default: 'Sharpe').
    ascending : bool | None
        Sort direction. If None, defaults to descending (largest first) except
        for MaxDrawdown where ascending is True.
    backtester_kwargs : dict | None
        Additional keyword arguments forwarded to CointBacktester.
    metrics_kwargs : dict | None
        Keyword arguments for summarize_performance (e.g., rf_annual).
    raise_on_error : bool
        If True, abort on the first failed run. Otherwise, log the error row.
    """

    train_lens = _as_list(train_len_values)
    entry_ks = _as_list(entry_k_values)
    exit_ks = _as_list(exit_k_values)

    if not train_lens or not entry_ks or not exit_ks:
        raise ValueError("All parameter grids must contain at least one value.")

    combos = list(product(train_lens, entry_ks, exit_ks))
    bt_kwargs = dict(backtester_kwargs or {})
    metric_kwargs = dict(metrics_kwargs or {})
    log_prefix = bt_kwargs.pop("log_prefix", None)
    results: list[dict[str, Any]] = []
    run_records: list[dict[str, Any]] = []
    successful_runs: list[
        tuple[dict[str, Any], dict[str, Any], pd.DataFrame, pd.DataFrame]
    ] = []
    total_combos = len(combos)

    for idx, (train_len, entry_k, exit_k) in enumerate(combos, start=1):
        params = {
            "train_len": int(train_len),
            "entry_k": float(entry_k),
            "exit_k": float(exit_k),
        }
        base_start = pd.to_datetime(start_time)
        combo_start = base_start - pd.to_timedelta(params["train_len"], unit="h")
        earliest_available = pd.to_datetime(exchange.time_index().min())
        if combo_start < earliest_available:
            combo_start = earliest_available
        run_kwargs = {
            "exchange": exchange,
            "start_time": combo_start,
            "end_time": end_time,
            **bt_kwargs,
            **params,
        }

        try:
            if log_prefix:
                logger.info(
                    "[%s] Grid combo %d/%d: train_len=%s entry_k=%.3f exit_k=%.3f",
                    log_prefix,
                    idx,
                    total_combos,
                    params["train_len"],
                    params["entry_k"],
                    params["exit_k"],
                )
            bt = CointBacktester(**run_kwargs)
            daily, fits = bt.run()
            metrics, _ = summarize_performance(daily, **metric_kwargs)

            entry_count = (
                int(daily.get("entry_flag", pd.Series(dtype=bool)).sum())
                if "entry_flag" in daily.columns
                else 0
            )
            exit_count = (
                int(daily.get("exit_flag", pd.Series(dtype=bool)).sum())
                if "exit_flag" in daily.columns
                else 0
            )
            final_equity = float(daily["pnl"].iloc[-1]) if not daily.empty else np.nan

            row = {
                **params,
                **metrics,
                "Trades": entry_count,
                "Exits": exit_count,
                "FinalEquity": final_equity,
                "Error": None,
            }
            results.append(row)
            successful_runs.append((params, metrics, daily, fits))
            run_records.append(
                {
                    "params": params.copy(),
                    "metrics": metrics,
                    "daily": daily,
                    "fits": fits,
                    "error": None,
                }
            )
        except Exception as exc:  # noqa: BLE001
            if raise_on_error:
                raise
            row = {
                **params,
                "Error": str(exc),
            }
            results.append(row)
            run_records.append({"params": params.copy(), "error": str(exc)})
            if log_prefix:
                logger.warning(
                    "[%s] Grid combo %d/%d failed: %s",
                    log_prefix,
                    idx,
                    total_combos,
                    exc,
                )
        finally:
            if progress_cb is not None:
                try:
                    progress_cb({"params": params, "index": idx, "total": total_combos})
                except Exception:
                    logger.debug("Progress callback failed", exc_info=True)

    table = pd.DataFrame(results)
    if table.empty:
        return GridSearchSummary(
            table=table,
            best_params=None,
            best_metrics=None,
            best_daily=None,
            best_fits=None,
            runs=run_records,
        )

    if sort_by not in table.columns:
        table[sort_by] = np.nan

    if ascending is None:
        ascending = sort_by in {"MaxDrawdown"}

    table = table.sort_values(sort_by, ascending=ascending).reset_index(drop=True)

    best_params: dict[str, Any] | None = None
    best_metrics: dict[str, Any] | None = None
    best_daily: pd.DataFrame | None = None
    best_fits: pd.DataFrame | None = None

    if successful_runs:
        # match top-ranked row to stored runs
        top = table.iloc[0]
        for params, metrics, daily, fits in successful_runs:
            if (
                np.isclose(params["train_len"], top["train_len"])
                and np.isclose(params["entry_k"], top["entry_k"])
                and np.isclose(params["exit_k"], top["exit_k"])
            ):
                best_params = params
                best_metrics = metrics
                best_daily = daily
                best_fits = fits
                break

    return GridSearchSummary(
        table=table,
        best_params=best_params,
        best_metrics=best_metrics,
        best_daily=best_daily,
        best_fits=best_fits,
        runs=run_records,
    )
