#
# File Name: trade.py
# Create Time: 2025-09-29 23:05:27
# Modified Time: 2025-10-01 14:53:54
#


import numpy as np
import pandas as pd
from utils import fit_coint, residual_on_row
from statsmodels.tsa.stattools import adfuller


class Exchange:
    """Simple exchange that streams aligned bars with fields needed by the strategy."""

    def __init__(self, merged: pd.DataFrame, coint_var: str = "log_close"):
        """
        merged must contain:
          - 'open_time', 'close_1', 'close_2'
          - f'{coint_var}_1', f'{coint_var}_2'
        """
        req_cols = {
            "open_time",
            "close_1",
            "close_2",
            f"{coint_var}_1",
            f"{coint_var}_2",
        }
        missing = req_cols - set(merged.columns)
        if missing:
            raise ValueError(f"merged is missing columns: {missing}")
        self.df = merged.sort_values("open_time").reset_index(drop=True).copy()
        self.coint_var = coint_var

    def time_index(self) -> pd.Series:
        return self.df["open_time"]

    def get_bar(self, i: int) -> dict:
        row = self.df.iloc[i]
        return {
            "time": row["open_time"],
            "y_close": float(row["close_1"]),
            "x_close": float(row["close_2"]),
            "y_val": float(row[f"{self.coint_var}_1"]),
            "x_val": float(row[f"{self.coint_var}_2"]),
        }

    def window(self, end_idx: int, length: int) -> pd.DataFrame:
        """Return trailing window (length bars) ending at end_idx inclusive."""
        start = max(0, end_idx - length + 1)
        return self.df.iloc[start : end_idx + 1].copy()


class CointRegime:
    """Holds the latest cointegration decision and parameters."""

    def __init__(self):
        self.active = False
        self.beta0 = None
        self.beta1 = None  # log-scale beta
        self.sigma = None  # log residual sigma
        self.p_value = None
        self.valid_from = None
        self.valid_to = None  # inclusive index until which this regime is valid

    def fit_from_window(
        self, train: pd.DataFrame, coint_var: str, pval_alpha: float
    ) -> None:
        fit = fit_coint(train, coint_var)
        self.beta0 = fit["beta0"]
        self.beta1 = fit["beta1"]
        self.sigma = fit["sigma"]
        self.p_value = fit["p_value"]
        self.active = bool(self.p_value < pval_alpha)

    def residual(self, y_val: float, x_val: float) -> float | None:
        if self.beta0 is None or self.beta1 is None:
            return None
        return residual_on_row(self.beta0, self.beta1, y_val, x_val)


class Account:
    """
    Tracks position state, margin-capped sizing (single margin rate), MTM, and equity.

    Sizing at entry:
      N = cash * trade_frac  (margin budget)
      h_units = beta1_log * (P1_entry / P2_entry)
      q_y = N / (m * (P1_entry + h_units * P2_entry))
      q_x = h_units * q_y
    """

    def __init__(
        self, initial_balance: float, trade_frac: float, margin_rate: float = 0.10
    ):
        self.cash = float(initial_balance)
        self.trade_frac = float(trade_frac)
        self.m = float(
            margin_rate
        )  # single margin rate applied to both legs' gross notional

        self.state = "flat"  # "flat" | "long" | "short"
        self.notional = 0.0  # margin budget = cash * trade_frac

        self.entry_y = None
        self.entry_x = None
        self.entry_beta1 = None  # log beta at entry (for reference)
        self.entry_sigma = None  # log residual sigma at entry
        self.entry_residual = None  # log residual at entry
        self.entry_time = None

        # fixed quantities after entry
        self.qty_y = 0.0
        self.qty_x = 0.0

        # optional bookkeeping
        self.entry_gross = 0.0
        self.entry_margin = 0.0

    @staticmethod
    def _units_ratio(beta1_log: float, p1: float, p2: float) -> float:
        """Convert log-beta to linear-units hedge ratio at entry prices."""
        return float(beta1_log) * float(p1) / max(float(p2), 1e-9)

    def can_enter(self) -> bool:
        return self.state == "flat"

    def _size_by_margin(
        self, p1: float, p2: float, h_units: float
    ) -> tuple[float, float]:
        """Compute quantities meeting the combined-margin cap."""
        N = self.cash * self.trade_frac
        hu = abs(h_units)  # gross exposure, not net
        den = self.m * (p1 + hu * p2)
        q_y = N / max(den, 1e-12)  # always a positive quantity
        q_x = hu * q_y  # always a positive quantity
        return q_y, q_x

    def open_long(
        self,
        y_price: float,
        x_price: float,
        beta1_log: float,
        sigma_log: float,
        t,
        entry_residual: float,
    ) -> None:
        """Open long spread: +y, -x using margin-capped sizing."""
        self.state = "long"
        self.notional = self.cash * self.trade_frac
        self.entry_y = float(y_price)
        self.entry_x = float(x_price)
        self.entry_beta1 = float(beta1_log)
        self.entry_sigma = float(sigma_log)
        self.entry_time = t
        self.entry_residual = float(entry_residual)

        h_units = self._units_ratio(self.entry_beta1, self.entry_y, self.entry_x)
        self.qty_y, self.qty_x = self._size_by_margin(
            self.entry_y, self.entry_x, h_units
        )

        self.entry_gross = self.qty_y * self.entry_y + self.qty_x * self.entry_x
        self.entry_margin = self.m * self.entry_gross

    def open_short(
        self,
        y_price: float,
        x_price: float,
        beta1_log: float,
        sigma_log: float,
        t,
        entry_residual: float,
    ) -> None:
        """Open short spread: -y, +x using margin-capped sizing."""
        self.state = "short"
        self.notional = self.cash * self.trade_frac
        self.entry_y = float(y_price)
        self.entry_x = float(x_price)
        self.entry_beta1 = float(beta1_log)
        self.entry_sigma = float(sigma_log)
        self.entry_time = t
        self.entry_residual = float(entry_residual)

        h_units = self._units_ratio(self.entry_beta1, self.entry_y, self.entry_x)
        self.qty_y, self.qty_x = self._size_by_margin(
            self.entry_y, self.entry_x, h_units
        )

        self.entry_gross = self.qty_y * self.entry_y + self.qty_x * self.entry_x
        self.entry_margin = self.m * self.entry_gross

    def mark_to_market(self, y_price: float, x_price: float) -> float:
        """Cash + unrealized PnL using fixed quantities."""
        if self.state == "flat":
            return self.cash
        leg_y = self.qty_y * (y_price - self.entry_y)
        leg_x = self.qty_x * (x_price - self.entry_x)
        pnl = (leg_y - leg_x) if self.state == "long" else (-leg_y + leg_x)
        return self.cash + pnl

    def should_exit_cross(
        self, prev_resid: float | None, curr_resid: float | None, exit_k: float
    ) -> bool:
        """Exit when residual crosses the entry-side band."""
        if (
            self.state == "flat"
            or self.entry_sigma is None
            or prev_resid is None
            or curr_resid is None
            or np.isnan(prev_resid)
            or np.isnan(curr_resid)
        ):
            return False

        level = exit_k * self.entry_sigma
        if self.state == "long":
            # cross upward through -level
            return (prev_resid < -level) and (curr_resid >= -level)
        else:  # short
            # cross downward through +level
            return (prev_resid > level) and (curr_resid <= level)

    # Stop-loss: residual moves stop_k * sigma further against the entry residual
    def should_stop_loss(self, curr_resid: float | None, stop_k: float) -> bool:
        if (
            self.state == "flat"
            or self.entry_sigma is None
            or self.entry_residual is None
            or curr_resid is None
        ):
            return False
        delta = stop_k * self.entry_sigma
        if self.state == "long":
            # long: entry residual is negative; more negative than entry by delta -> stop
            return curr_resid <= (self.entry_residual - delta)
        else:
            # short: entry residual is positive; more positive than entry by delta -> stop
            return curr_resid >= (self.entry_residual + delta)

    def close_position(self, y_price: float, x_price: float) -> None:
        equity = self.mark_to_market(y_price, x_price)
        self.cash = equity
        # reset
        self.state = "flat"
        self.notional = 0.0
        self.entry_y = self.entry_x = None
        self.entry_beta1 = None
        self.entry_sigma = None
        self.entry_residual = None
        self.entry_time = None
        self.qty_y = 0.0
        self.qty_x = 0.0
        self.entry_gross = 0.0
        self.entry_margin = 0.0


class CointBacktester:
    """
    Event-driven, OOP-style backtester with:
      - Exchange for bar streaming
      - Cointegration decision every 'gap' bars when flat
      - Account to track positions and equity

    New:
      - On each reset day, re-check cointegration with the current fixed beta over the last train_len bars.
        * If flat: use strict pval_alpha; if it fails, switch to a newly refit beta immediately.
        * If in position: use relaxed_pval_alpha; if it fails, force close the position, then refit/switch beta.
      - Stop-loss stop_k that triggers when residual drifts stop_k * sigma further against the entry residual
        (higher priority than the crossing exit).
    """

    def __init__(
        self,
        exchange: Exchange,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        *,
        coint_var: str = "log_close",
        train_len: int,
        gap: int,
        pval_alpha: float = 0.05,
        entry_k: float = 2.0,
        exit_k: float = 0.5,
        # New:
        relaxed_pval_alpha: float = 0.10,  # looser threshold while holding
        stop_k: float = 3.0,  # stop-loss multiple relative to entry_sigma
        stop_loss_cooling_days: float = 0.0,  # cooling period after stop loss
        initial_balance: float = 1_000_000.0,
        trade_frac: float = 0.10,
        margin_rate: float = 0.10,
    ):
        self.ex = exchange
        times = self.ex.time_index()
        # clip to [start_time, end_time]
        mask = (times >= start_time) & (times <= end_time)
        self.idx_map = np.where(mask)[0]
        if len(self.idx_map) == 0:
            raise ValueError("No bars in [start_time, end_time].")
        self.start_idx = int(self.idx_map[0])
        self.end_idx = int(self.idx_map[-1])

        self.train_len = int(train_len)
        self.gap = int(gap)
        self.coint_var = coint_var
        self.pval_alpha = float(pval_alpha)
        self.entry_k = float(entry_k)
        self.exit_k = float(exit_k)
        # New:
        self.relaxed_pval_alpha = float(relaxed_pval_alpha)
        self.stop_k = float(stop_k)
        self.stop_loss_cooling_days = float(stop_loss_cooling_days)
        self.stop_loss_cooling = (
            pd.Timedelta(days=self.stop_loss_cooling_days)
            if self.stop_loss_cooling_days > 0
            else None
        )

        # prevent rapid re-entry after a stop-loss
        self.cooldown_until_time: pd.Timestamp | None = None

        self.account = Account(
            initial_balance=initial_balance,
            trade_frac=trade_frac,
            margin_rate=margin_rate,
        )
        self.regime = CointRegime()

        # logs
        self.daily = []  # one row per bar
        self.fits = []  # one row per (re)fit

    def _is_decision_day(self, bar_idx: int, first_decision_idx: int) -> bool:
        if bar_idx < first_decision_idx:
            return False
        return ((bar_idx - first_decision_idx) % self.gap) == 0

    # Fixed-beta cointegration re-check using the latest regime beta
    def _pvalue_with_fixed_beta(
        self, bar_idx: int, beta0: float | None, beta1: float | None
    ) -> tuple[float | None, float | None, pd.Timestamp | None, pd.Timestamp | None]:
        """
        Return (p_value, sigma, train_start_time, train_end_time).
        If insufficient data or missing beta, return (None, None, None, None).
        """
        if beta0 is None or beta1 is None:
            return None, None, None, None
        train = self.ex.window(bar_idx, self.train_len)
        if train.empty:
            return None, None, None, None

        y = train[f"{self.coint_var}_1"].astype(float).to_numpy()
        x = train[f"{self.coint_var}_2"].astype(float).to_numpy()
        resid = y - (beta0 + beta1 * x)
        if np.all(np.isfinite(resid)) and len(resid) >= 3:
            try:
                adf = adfuller(resid, autolag="AIC")
                pval = float(adf[1])
            except Exception:
                pval = None
            sigma = float(np.std(resid, ddof=1)) if np.isfinite(resid).all() else None
        else:
            pval, sigma = None, None

        return (
            pval,
            sigma,
            train["open_time"].min() if "open_time" in train else None,
            train["open_time"].max() if "open_time" in train else None,
        )

    def run(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        # first index with enough history
        first_decision_idx = self.start_idx + self.train_len
        if first_decision_idx > self.end_idx:
            raise ValueError("Not enough data for the first decision day.")

        # regime validity window: [d, d+gap-1] after each fit at day d
        regime_valid_to = None  # inclusive bar index
        prev_residual = None  # for crossing exit logic
        # ensure clean start if run() is called multiple times on same instance
        self.cooldown_until_time = None

        for bar_idx in range(self.start_idx, self.end_idx + 1):
            bar = self.ex.get_bar(bar_idx)
            time = bar["time"]
            y_close = bar["y_close"]
            x_close = bar["x_close"]
            y_val = bar["y_val"]  # log-scale
            x_val = bar["x_val"]

            entry_allowed_today = False
            p_value_today = self.regime.p_value
            beta1_today = self.regime.beta1
            sigma_today = self.regime.sigma
            residual_today = None
            h_units_today = np.nan

            # fixed-beta re-check stats
            fixed_beta_p = None
            fixed_beta_sigma = None
            fixed_beta_train_start = None
            fixed_beta_train_end = None
            fixed_beta_check_done = False
            fixed_beta_failed = False
            fixed_beta_threshold_used = None  # pval_alpha or relaxed_pval_alpha
            cooling_active_today = False

            if self.cooldown_until_time is not None:
                if pd.Timestamp(time) < self.cooldown_until_time:
                    cooling_active_today = True
                else:
                    self.cooldown_until_time = None

            # 1) residual based on current regime, if available
            if self.regime.beta0 is not None and self.regime.beta1 is not None:
                residual_today = self.regime.residual(y_val, x_val)

            # 2) on reset day, re-check cointegration with fixed beta over the last train_len bars
            #    - flat: strict threshold pval_alpha; if fail -> switch beta (refit now)
            #    - in position: relaxed threshold; if fail -> force close, then switch beta (refit now)
            if self._is_decision_day(bar_idx, first_decision_idx) and (
                self.regime.beta0 is not None and self.regime.beta1 is not None
            ):
                fixed_beta_check_done = True
                thr = (
                    self.relaxed_pval_alpha
                    if self.account.state != "flat"
                    else self.pval_alpha
                )
                fixed_beta_threshold_used = thr

                (
                    fixed_beta_p,
                    fixed_beta_sigma,
                    fixed_beta_train_start,
                    fixed_beta_train_end,
                ) = self._pvalue_with_fixed_beta(
                    bar_idx, self.regime.beta0, self.regime.beta1
                )

                if (fixed_beta_p is not None) and (fixed_beta_p >= thr):
                    fixed_beta_failed = True

            exit_flag = False

            # 3) if fixed-beta check fails and we are holding a position, force close first
            force_close_reason = None
            equity_view = self.account.mark_to_market(y_close, x_close)
            if fixed_beta_failed and self.account.state != "flat":
                self.account.close_position(y_close, x_close)
                equity_view = self.account.cash
                force_close_reason = "no_longer_coint"
                exit_flag = True

            # 4) if fixed-beta check fails (flat or after force-close), refit and switch to a new beta now
            regime_switched = False
            if fixed_beta_failed:
                train = self.ex.window(bar_idx, self.train_len)
                self.regime.fit_from_window(train, self.coint_var, self.pval_alpha)
                p_value_today = self.regime.p_value
                beta1_today = self.regime.beta1
                sigma_today = self.regime.sigma
                residual_today = self.regime.residual(y_val, x_val)
                regime_valid_to = min(bar_idx + self.gap - 1, self.end_idx)
                self.regime.valid_from = bar_idx
                self.regime.valid_to = regime_valid_to
                regime_switched = True

                # log this fit as a switch triggered by failed fixed-beta check
                self.fits.append(
                    {
                        "fit_time": time,
                        "beta0": self.regime.beta0,
                        "beta1_log": self.regime.beta1,
                        "sigma_log": self.regime.sigma,
                        "p_value": self.regime.p_value,
                        "entry_allowed": self.regime.active,
                        "train_start": train["open_time"].min(),
                        "train_end": train["open_time"].max(),
                        "valid_from_idx": bar_idx,
                        "valid_to_idx": regime_valid_to,
                        "reason": "switch_after_fixed_beta_fail",
                        "fixed_beta_p": fixed_beta_p,
                        "fixed_beta_sigma": fixed_beta_sigma,
                        "fixed_beta_thr": fixed_beta_threshold_used,
                        "fixed_beta_train_start": fixed_beta_train_start,
                        "fixed_beta_train_end": fixed_beta_train_end,
                    }
                )

            # 5) during a position, exit priority: stop-loss first, then crossing-based exit
            if not exit_flag and self.account.state != "flat" and residual_today is not None:
                # (a) stop-loss
                if self.account.should_stop_loss(residual_today, self.stop_k):
                    self.account.close_position(y_close, x_close)
                    equity_view = self.account.cash
                    exit_flag = True
                    force_close_reason = "stop_loss"
                    if self.stop_loss_cooling is not None:
                        base_time = pd.Timestamp(time)
                        self.cooldown_until_time = base_time + self.stop_loss_cooling
                        cooling_active_today = True

                # (b) crossing exit (only if not stopped out)
                elif self.account.should_exit_cross(
                    prev_resid=prev_residual,
                    curr_resid=residual_today,
                    exit_k=self.exit_k,
                ):
                    self.account.close_position(y_close, x_close)
                    equity_view = self.account.cash
                    exit_flag = True

            # 6) if still flat and today is a reset day and we did not just switch above, do the periodic refit
            if (
                self.account.can_enter()
                and self._is_decision_day(bar_idx, first_decision_idx)
                and not regime_switched
            ):
                train = self.ex.window(bar_idx, self.train_len)
                self.regime.fit_from_window(train, self.coint_var, self.pval_alpha)
                p_value_today = self.regime.p_value
                beta1_today = self.regime.beta1
                sigma_today = self.regime.sigma
                residual_today = self.regime.residual(y_val, x_val)

                regime_valid_to = min(bar_idx + self.gap - 1, self.end_idx)
                self.regime.valid_from = bar_idx
                self.regime.valid_to = regime_valid_to

                self.fits.append(
                    {
                        "fit_time": time,
                        "beta0": self.regime.beta0,
                        "beta1_log": self.regime.beta1,
                        "sigma_log": self.regime.sigma,
                        "p_value": self.regime.p_value,
                        "entry_allowed": self.regime.active,
                        "train_start": train["open_time"].min(),
                        "train_end": train["open_time"].max(),
                        "valid_from_idx": bar_idx,
                        "valid_to_idx": regime_valid_to,
                        "reason": "periodic_decision",
                    }
                )

            # 7) determine whether entry is allowed today (must be inside current regime window)
            in_regime_window = (
                regime_valid_to is not None
                and (bar_idx <= regime_valid_to)
                and (self.regime.valid_from is not None)
                and (bar_idx >= self.regime.valid_from)
            )
            entry_allowed_today = bool(
                self.regime.active
                and in_regime_window
                and self.account.can_enter()
                and not cooling_active_today
            )

            # 8) entry logic (unchanged apart from using possibly updated regime params)
            trade_signal = 0
            entry_flag = False
            entry_side = 0

            if (
                entry_allowed_today
                and residual_today is not None
                and sigma_today is not None
                and sigma_today > 0
            ):
                # linear-units hedge ratio used for sizing
                h_units = beta1_today * (y_close / max(x_close, 1e-9))
                h_units_today = h_units

                if residual_today >= self.entry_k * sigma_today:
                    # short spread
                    self.account.open_short(
                        y_close,
                        x_close,
                        beta1_today,
                        sigma_today,
                        time,
                        residual_today,
                    )
                    trade_signal = -1
                    entry_flag = True
                    entry_side = -1
                elif residual_today <= -self.entry_k * sigma_today:
                    # long spread
                    self.account.open_long(
                        y_close,
                        x_close,
                        beta1_today,
                        sigma_today,
                        time,
                        residual_today,
                    )
                    trade_signal = +1
                    entry_flag = True
                    entry_side = +1
                equity_view = self.account.mark_to_market(y_close, x_close)

            # 9) if regime window ended and we are still flat, deactivate until next decision day
            if (
                self.account.can_enter()
                and regime_valid_to is not None
                and bar_idx > regime_valid_to
            ):
                self.regime.active = False

            # 10) derive h_units from account if in position (for logging/plots)
            if np.isnan(h_units_today) and self.account.qty_y != 0:
                h_units_today = self.account.qty_x / self.account.qty_y

            # 11) daily log row (includes new fields for re-checks/forced exits/stops)
            self.daily.append(
                {
                    "time": time,
                    "in_coint_period": bool(self.regime.active and in_regime_window),
                    "p_value": p_value_today,
                    "beta1": beta1_today,  # log beta
                    "h_units": h_units_today,  # units hedge ratio used/held
                    "sigma": sigma_today,  # log residual sigma
                    "residual": residual_today,  # log residual
                    "trade_signal": trade_signal,
                    "entry_flag": entry_flag,
                    "exit_flag": exit_flag,
                    "entry_side": entry_side,
                    "position": (
                        0
                        if self.account.state == "flat"
                        else (1 if self.account.state == "long" else -1)
                    ),
                    "pnl": equity_view,
                    "state": self.account.state,
                    "qty_y": self.account.qty_y,
                    "qty_x": self.account.qty_x,
                    # new: fixed-beta re-check / forced close / stop-loss
                    "fixed_beta_check_done": fixed_beta_check_done,
                    "fixed_beta_p": fixed_beta_p,
                    "fixed_beta_sigma": fixed_beta_sigma,
                    "fixed_beta_thr": fixed_beta_threshold_used,
                    "fixed_beta_fail": fixed_beta_failed,
                    "force_close_reason": force_close_reason,  # None | "stop_loss" | "no_longer_coint"
                    "stop_k": self.stop_k,
                    "exit_k": self.exit_k,
                    "relaxed_pval_alpha": self.relaxed_pval_alpha,
                    "cooldown_active": cooling_active_today,
                    "cooldown_until": self.cooldown_until_time,
                }
            )

            # 12) update previous residual
            prev_residual = residual_today

        daily_df = pd.DataFrame(self.daily).sort_values("time").reset_index(drop=True)
        fits_df = pd.DataFrame(self.fits).sort_values("fit_time").reset_index(drop=True)
        return daily_df, fits_df
