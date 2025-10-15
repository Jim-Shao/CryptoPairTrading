from __future__ import annotations

import logging
import multiprocessing as mp
import os
from dataclasses import dataclass, field
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import pandas as pd

from metrics import summarize_performance
from optimizer import GridSearchSummary, grid_search_parameters
from trade import CointBacktester, Exchange


logger = logging.getLogger(__name__)


@dataclass
class PairSpec:
    """
    Definition of a single trading pair to include in the portfolio grid search.

    Attributes
    ----------
    name: str
        Identifier for the pair (used in result tables).
    exchange: Exchange
        Aligned data source feeding the backtester.
    train_window: Tuple[pd.Timestamp, pd.Timestamp]
        Inclusive start/end timestamps for the training backtest.
    test_window: Tuple[pd.Timestamp, pd.Timestamp]
        Inclusive start/end timestamps for the test backtest.
    grid: Dict[str, Iterable[Any]]
        Parameter grid with keys 'train_len', 'entry_k', 'exit_k' and iterable values.
    backtester_kwargs: Dict[str, Any]
        Additional keyword arguments forwarded to CointBacktester for both train/test phases.
    metrics_kwargs: Dict[str, Any]
        Optional kwargs for summarize_performance.
    """

    name: str
    exchange: Exchange
    train_window: Tuple[pd.Timestamp, pd.Timestamp]
    test_window: Tuple[pd.Timestamp, pd.Timestamp]
    grid: Dict[str, Iterable[Any]]
    backtester_kwargs: Dict[str, Any] = field(default_factory=dict)
    metrics_kwargs: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PairRunResult:
    """Holds training and test artifacts for a single pair."""

    pair: PairSpec
    train_summary: GridSearchSummary
    test_daily: pd.DataFrame | None
    test_fits: pd.DataFrame | None
    test_metrics: Dict[str, Any] | None


def _train_single_pair(
    spec: PairSpec,
    *,
    sort_by: str,
    ascending: Optional[bool],
    raise_on_error: bool,
    progress_fd: Optional[int] = None,
) -> tuple[str, GridSearchSummary]:
    grid = {k: list(v) for k, v in spec.grid.items()}
    backtester_kwargs = dict(spec.backtester_kwargs)
    backtester_kwargs["log_prefix"] = spec.name

    def _progress(info: dict[str, Any]) -> None:
        if progress_fd is not None:
            try:
                os.write(progress_fd, b"\x01")
            except Exception:
                logger.debug("Failed to write progress update", exc_info=True)

    summary = grid_search_parameters(
        exchange=spec.exchange,
        start_time=spec.train_window[0],
        end_time=spec.train_window[1],
        train_len_values=grid["train_len"],
        entry_k_values=grid["entry_k"],
        exit_k_values=grid["exit_k"],
        sort_by=sort_by,
        ascending=ascending,
        backtester_kwargs=backtester_kwargs,
        metrics_kwargs=spec.metrics_kwargs,
        raise_on_error=raise_on_error,
        progress_cb=_progress,
    )
    return spec.name, summary


class Portfolio:
    """
    Coordinate training parameter search and test-period backtests across multiple pairs.

    Capital Allocation
    ------------------
    Each pair receives an equal fraction of the portfolio's initial balance:
        allocation_per_pair = initial_balance / max(num_pairs, 1)
    The allocated balance is handed to the test-phase backtester with trade_frac=1.0,
    meaning every entry will size positions off that dedicated slice. This approximates
    the desired behaviour where each pair taps at most 1/N of total equity when entering.
    """

    def __init__(
        self,
        pairs: Sequence[PairSpec],
        *,
        initial_balance: float,
        sort_by: str = "Sharpe",
        ascending: Optional[bool] = None,
        max_workers: Optional[int] = None,
        mp_context=None,
    ):
        if not pairs:
            raise ValueError("Portfolio requires at least one PairSpec.")
        self.pairs: List[PairSpec] = list(pairs)
        self.initial_balance = float(initial_balance)
        self.sort_by = sort_by
        self.ascending = ascending
        self.max_workers = max_workers
        self.mp_context = mp_context
        self._train_results: Dict[str, GridSearchSummary] = {}
        self._pair_runs: Dict[str, PairRunResult] = {}

    @property
    def allocation_per_pair(self) -> float:
        return self.initial_balance / max(len(self.pairs), 1)

    def train(
        self,
        *,
        raise_on_error: bool = False,
        progress_fd: Optional[int] = None,
    ) -> Dict[str, GridSearchSummary]:
        """
        Run grid search on the training window for every pair.

        Returns
        -------
        dict: pair_name -> GridSearchSummary
        """
        logger.info(
            "Starting grid search for %d pairs (max_workers=%s)...",
            len(self.pairs),
            self.max_workers,
        )
        summaries: Dict[str, GridSearchSummary] = {}

        def _validate(spec: PairSpec) -> None:
            missing = {"train_len", "entry_k", "exit_k"} - set(spec.grid)
            if missing:
                raise ValueError(
                    f"Pair '{spec.name}' grid missing keys: {sorted(missing)}"
                )

        if self.max_workers is None or self.max_workers <= 1:
            for spec in self.pairs:
                _validate(spec)
                logger.info("Training pair %s...", spec.name)
                name, summary = _train_single_pair(
                    spec,
                    sort_by=self.sort_by,
                    ascending=self.ascending,
                    raise_on_error=raise_on_error,
                    progress_fd=progress_fd,
                )
                summaries[name] = summary
                logger.info(
                    "Completed training for %s (%d combinations).",
                    name,
                    len(summary.table),
                )
        else:
            ctx = self.mp_context or mp.get_context()
            with ProcessPoolExecutor(max_workers=self.max_workers, mp_context=ctx) as executor:
                futures = {}
                for spec in self.pairs:
                    _validate(spec)
                    logger.info("Dispatching training job for %s...", spec.name)
                    fut = executor.submit(
                        _train_single_pair,
                        spec,
                        sort_by=self.sort_by,
                        ascending=self.ascending,
                        raise_on_error=raise_on_error,
                        progress_fd=progress_fd,
                    )
                    futures[fut] = spec.name

                for fut in as_completed(futures):
                    name = futures[fut]
                    name_result, summary = fut.result()
                    summaries[name_result] = summary
                    logger.info(
                        "Completed training for %s (%d combinations).",
                        name_result,
                        len(summary.table),
                    )

        self._train_results = summaries
        return summaries

    def run_test(self) -> Dict[str, PairRunResult]:
        """
        Execute test-period backtests using best parameters from the training search.

        Returns
        -------
        dict: pair_name -> PairRunResult
        """
        if not self._train_results:
            raise RuntimeError("Call train() before run_test().")

        logger.info("Running test backtests for %d pairs...", len(self.pairs))
        runs: Dict[str, PairRunResult] = {}
        allocation = self.allocation_per_pair

        for spec in self.pairs:
            summary = self._train_results.get(spec.name)
            if summary is None:
                raise RuntimeError(f"No training result stored for pair '{spec.name}'.")

            best_params = summary.best_params or {}
            if not best_params:
                logger.warning(
                    "Skipping test run for %s (no valid training results).",
                    spec.name,
                )
                runs[spec.name] = PairRunResult(
                    pair=spec,
                    train_summary=summary,
                    test_daily=None,
                    test_fits=None,
                    test_metrics=None,
                )
                continue

            logger.info("Testing pair %s with params %s...", spec.name, best_params)
            bt_kwargs = dict(spec.backtester_kwargs)
            bt_kwargs.update(best_params)
            bt_kwargs.setdefault("trade_frac", 1.0)
            bt_kwargs.setdefault("initial_balance", allocation)

            combo_start = spec.train_window[0] - pd.to_timedelta(
                best_params.get("train_len", 0), unit="h"
            )
            earliest = pd.to_datetime(spec.exchange.time_index().min())
            if combo_start < earliest:
                combo_start = earliest

            bt = CointBacktester(
                exchange=spec.exchange,
                start_time=combo_start,
                end_time=spec.test_window[1],
                **bt_kwargs,
            )
            daily, fits = bt.run()
            mk = spec.metrics_kwargs or {}
            metrics, _ = summarize_performance(daily, **mk)

            runs[spec.name] = PairRunResult(
                pair=spec,
                train_summary=summary,
                test_daily=daily,
                test_fits=fits,
                test_metrics=metrics,
            )
            final_equity = (
                float(daily["pnl"].iloc[-1]) if daily is not None and not daily.empty else float("nan")
            )
            logger.info(
                "Completed test for %s | Sharpe=%s | FinalEquity=%.2f",
                spec.name,
                metrics.get("Sharpe") if metrics else "n/a",
                final_equity,
            )

        self._pair_runs = runs
        return runs

    def aggregate_equity(self) -> pd.DataFrame:
        """
        Combine per-pair daily curves by aligning on timestamp and summing metrics.

        Missing values are forward-filled within each pair before aggregation.
        Returns a DataFrame with columns time, portfolio_equity, portfolio_cash,
        portfolio_margin_used, portfolio_fees (cumulative).
        """
        if not self._pair_runs:
            raise RuntimeError("No pair runs available. Call run_test() first.")

        metrics_map = {
            "pnl": "equity",
            "cash": "cash",
            "margin_used": "margin_used",
            "fees_cumulative": "fees",
            "position": "active",
        }

        frames: list[pd.DataFrame] = []
        for name, result in self._pair_runs.items():
            daily = result.test_daily
            if daily is None or daily.empty:
                continue
            use_cols = [c for c in metrics_map if c in daily.columns]
            if not use_cols:
                continue
            df = daily[["time", *use_cols]].copy()
            df["time"] = pd.to_datetime(df["time"])
            for col in use_cols:
                series = pd.to_numeric(df[col], errors="coerce")
                if col == "position":
                    series = series.abs().clip(upper=1.0)
                    series = series.where(series == 0, 1.0)
                df[f"{name}_{metrics_map[col]}"] = series
            cols_to_keep = ["time", *[f"{name}_{metrics_map[c]}" for c in use_cols]]
            frames.append(df[cols_to_keep])

        if not frames:
            return pd.DataFrame(
                columns=[
                    "time",
                    "portfolio_equity",
                    "portfolio_cash",
                    "portfolio_margin_used",
                    "portfolio_fees",
                ]
            )

        merged = frames[0]
        for df in frames[1:]:
            merged = pd.merge(merged, df, on="time", how="outer")

        merged = merged.sort_values("time").reset_index(drop=True)
        metric_suffixes = {"equity", "cash", "margin_used", "fees", "active"}
        for suffix in metric_suffixes:
            cols = [c for c in merged.columns if c.endswith(f"_{suffix}")]
            if cols:
                merged[cols] = merged[cols].ffill()
                merged[f"portfolio_{suffix}"] = merged[cols].sum(axis=1, min_count=1)

        active_count_col = "portfolio_active"
        if active_count_col in merged.columns:
            merged[active_count_col] = merged[active_count_col].fillna(0.0)
            merged["portfolio_active_ratio"] = merged[active_count_col] / max(len(self.pairs), 1)

        keep_cols = [
            "time",
            "portfolio_equity",
            "portfolio_fees",
            "portfolio_active",
            "portfolio_active_ratio",
        ]
        for col in keep_cols:
            if col not in merged.columns:
                merged[col] = float("nan")
        return merged[keep_cols]

    @property
    def pair_runs(self) -> Dict[str, PairRunResult]:
        return self._pair_runs
