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


def _test_single_pair(
    spec: PairSpec,
    best_params: dict[str, Any] | None,
    allocation: float,
) -> tuple[str, pd.DataFrame | None, pd.DataFrame | None, dict[str, Any] | None]:
    if not best_params:
        return spec.name, None, None, None

    bt_kwargs = dict(spec.backtester_kwargs)
    bt_kwargs.update(best_params)
    bt_kwargs.setdefault("trade_frac", 0.1)
    bt_kwargs.setdefault("initial_balance", allocation)

    train_len_hours = int(best_params.get("train_len", 0))
    combo_start = spec.test_window[0] - pd.to_timedelta(train_len_hours, unit="h")
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
    metric_kwargs = spec.metrics_kwargs or {}
    metrics, _ = summarize_performance(daily, **metric_kwargs)
    return spec.name, daily, fits, metrics


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
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=ctx
            ) as executor:
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

        spec_map = {spec.name: spec for spec in self.pairs}

        if self.max_workers is None or self.max_workers <= 1:
            for spec in self.pairs:
                summary = self._train_results.get(spec.name)
                if summary is None:
                    raise RuntimeError(
                        f"No training result stored for pair '{spec.name}'."
                    )
                logger.info(
                    "Testing pair %s with params %s...", spec.name, summary.best_params
                )
                name, daily, fits, metrics = _test_single_pair(
                    spec, summary.best_params, allocation
                )
                if daily is None:
                    logger.warning(
                        "Skipping test run for %s (no valid training results).",
                        spec.name,
                    )
                runs[name] = PairRunResult(
                    pair=spec,
                    train_summary=summary,
                    test_daily=daily,
                    test_fits=fits,
                    test_metrics=metrics,
                )
                final_equity = (
                    float(daily["pnl"].iloc[-1])
                    if daily is not None and not daily.empty
                    else float("nan")
                )
                logger.info(
                    "Completed test for %s | Sharpe=%s | FinalEquity=%.2f",
                    spec.name,
                    metrics.get("Sharpe") if metrics else "n/a",
                    final_equity,
                )
        else:
            ctx = self.mp_context or mp.get_context()
            with ProcessPoolExecutor(
                max_workers=self.max_workers, mp_context=ctx
            ) as executor:
                futures = {}
                for spec in self.pairs:
                    summary = self._train_results.get(spec.name)
                    if summary is None:
                        raise RuntimeError(
                            f"No training result stored for pair '{spec.name}'."
                        )
                    futures[
                        executor.submit(
                            _test_single_pair, spec, summary.best_params, allocation
                        )
                    ] = spec.name

                for fut in as_completed(futures):
                    name = futures[fut]
                    summary = self._train_results[name]
                    spec = spec_map[name]
                    name_res, daily, fits, metrics = fut.result()
                    if daily is None:
                        logger.warning(
                            "Skipping test run for %s (no valid training results).",
                            name_res,
                        )
                    runs[name_res] = PairRunResult(
                        pair=spec,
                        train_summary=summary,
                        test_daily=daily,
                        test_fits=fits,
                        test_metrics=metrics,
                    )
                    final_equity = (
                        float(daily["pnl"].iloc[-1])
                        if daily is not None and not daily.empty
                        else float("nan")
                    )
                    logger.info(
                        "Completed test for %s | Sharpe=%s | FinalEquity=%.2f",
                        name_res,
                        metrics.get("Sharpe") if metrics else "n/a",
                        final_equity,
                    )

        self._pair_runs = runs
        return runs

    def aggregate_equity(self) -> pd.DataFrame:
        """
        Combine per-pair daily curves by aligning on timestamp and summing metrics.

        Missing values are forward-filled within each pair before aggregation.
        Returns a DataFrame with columns time, portfolio_equity,
        portfolio_fees, portfolio_active.
        """
        if not self._pair_runs:
            raise RuntimeError("No pair runs available. Call run_test() first.")

        metrics_map = {
            "pnl": "equity",
            "fees_cumulative": "fees",
            "position": "active",
            "deactivated": "inactive",
        }

        long_frames: list[pd.DataFrame] = []
        total_trades = 0
        for result in self._pair_runs.values():
            daily = result.test_daily
            if daily is None or daily.empty:
                continue
            entry_series = daily.get("entry_flag")
            if entry_series is not None:
                try:
                    entry_vals = pd.to_numeric(entry_series, errors="coerce").fillna(0.0)
                    total_trades += int(entry_vals.sum())
                except Exception:
                    pass
            use_cols = [c for c in metrics_map if c in daily.columns]
            if not use_cols:
                continue
            df = daily[["time", *use_cols]].copy()
            df["time"] = pd.to_datetime(df["time"])
            df = df.sort_values("time").reset_index(drop=True)
            for col in use_cols:
                series = pd.to_numeric(df[col], errors="coerce")
                if col == "position":
                    series = series.abs().clip(upper=1.0)
                    series = series.where(series == 0, 1.0)
                elif col == "deactivated":
                    series = series.fillna(method="ffill").fillna(0.0)
                df[col] = series
            df[use_cols] = df[use_cols].ffill()
            melted = df.melt(
                id_vars="time",
                value_vars=use_cols,
                var_name="metric",
                value_name="value",
            )
            melted["metric"] = melted["metric"].map(metrics_map)
            long_frames.append(melted)

        pairs_included = len(long_frames)

        if not long_frames:
            empty = pd.DataFrame(
                columns=[
                    "time",
                    "portfolio_equity",
                    "portfolio_fees",
                    "portfolio_active",
                    "portfolio_inactive",
                ]
            )
            empty.attrs["num_pairs"] = 0
            empty.attrs["total_trades"] = total_trades
            return empty

        agg = pd.concat(long_frames, ignore_index=True)
        agg = (
            agg.groupby(["time", "metric"], sort=True)["value"].sum().unstack("metric")
        )
        agg = agg.sort_index().reset_index()

        test_start = max(spec.test_window[0] for spec in self.pairs)
        agg = agg[agg["time"] >= pd.to_datetime(test_start)].reset_index(drop=True)

        agg.rename(
            columns={
                "equity": "portfolio_equity",
                "fees": "portfolio_fees",
                "active": "portfolio_active",
                "inactive": "portfolio_inactive",
            },
            inplace=True,
        )

        numeric_cols = [
            c
            for c in [
                "portfolio_equity",
                "portfolio_fees",
                "portfolio_active",
                "portfolio_inactive",
            ]
            if c in agg.columns
        ]
        agg[numeric_cols] = agg[numeric_cols].fillna(method="ffill")
        if "portfolio_fees" in agg.columns:
            agg["portfolio_fees"] = agg["portfolio_fees"].fillna(0.0)
        if "portfolio_active" in agg.columns:
            agg["portfolio_active"] = agg["portfolio_active"].fillna(0.0)
        if "portfolio_inactive" in agg.columns:
            agg["portfolio_inactive"] = agg["portfolio_inactive"].fillna(0.0)

        for col in [
            "portfolio_equity",
            "portfolio_fees",
            "portfolio_active",
            "portfolio_inactive",
        ]:
            if col not in agg.columns:
                agg[col] = float("nan")

        result = agg[
            [
                "time",
                "portfolio_equity",
                "portfolio_fees",
                "portfolio_active",
                "portfolio_inactive",
            ]
        ].copy()
        result.attrs["num_pairs"] = pairs_included
        result.attrs["total_trades"] = total_trades
        return result

    @property
    def pair_runs(self) -> Dict[str, PairRunResult]:
        return self._pair_runs
