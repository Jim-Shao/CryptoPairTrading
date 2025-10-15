import os
import numpy as np
import pandas as pd
import statsmodels.api as sm
from pandas.api.types import is_numeric_dtype
from statsmodels.tsa.stattools import adfuller


def load_crypto_dir(
    data_dir: str, start_date: str = None, end_date: str = None
) -> dict:
    """
    data_dir/symbol/freq/*.csv -> dict of DataFrames.
    Parses 'open_time' to datetime and filters to [start_date, end_date] if provided.
    """
    sd = pd.to_datetime(start_date) if start_date else None
    ed = pd.to_datetime(end_date) if end_date else None

    out = {}
    for symbol in sorted(os.listdir(data_dir)):
        spath = os.path.join(data_dir, symbol)
        if not os.path.isdir(spath):
            continue
        for freq in sorted(os.listdir(spath)):
            fpath = os.path.join(spath, freq)
            if not os.path.isdir(fpath):
                continue
            frames = []
            for fn in sorted(os.listdir(fpath)):
                if fn.endswith(".csv"):
                    frames.append(pd.read_csv(os.path.join(fpath, fn)))
            if not frames:
                continue

            df = (
                pd.concat(frames, ignore_index=True)
                .drop_duplicates()
                .reset_index(drop=True)
            )

            if "open_time" in df.columns:
                if is_numeric_dtype(df["open_time"]):
                    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
                else:
                    df["open_time"] = pd.to_datetime(df["open_time"], errors="coerce")
                if sd is not None or ed is not None:
                    m = pd.Series(True, index=df.index)
                    if sd is not None:
                        m &= df["open_time"] >= sd
                    if ed is not None:
                        m &= df["open_time"] <= ed
                    df = df.loc[m].reset_index(drop=True)

            if not df.empty:
                out[f"{symbol}_{freq}"] = df
    return out


def preprocess_ohlc(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df.sort_values("open_time").reset_index(drop=True)
    df = df[["open_time", "close", "volume"]]
    df["log_close"] = np.log(df["close"])
    return df


def align_two(
    df1: pd.DataFrame, df2: pd.DataFrame, coint_var: str = "log_close"
) -> pd.DataFrame:
    """Align by time; keep coint_var_1/2 and close_1/2."""
    m = pd.merge(
        df1[["open_time", coint_var, "close"]].rename(
            columns={coint_var: f"{coint_var}_1", "close": "close_1"}
        ),
        df2[["open_time", coint_var, "close"]].rename(
            columns={coint_var: f"{coint_var}_2", "close": "close_2"}
        ),
        on="open_time",
        how="inner",
    )
    return m.dropna().reset_index(drop=True)


def adf_pvalue(series: pd.Series) -> float:
    """Return p-value from ADF test (lower means more likely stationary)."""
    series = pd.Series(series).dropna()
    if len(series) < 3:
        return np.nan
    return float(adfuller(series, autolag="AIC")[1])


def fit_coint(train: pd.DataFrame, coint_var: str) -> dict:
    """OLS y = beta0 + beta1 * x, return params, residual sigma, and ADF p-value."""
    y = train[f"{coint_var}_1"].astype(float)
    x_name = f"{coint_var}_2"
    x = train[x_name].astype(float).to_frame(name=x_name)
    X = sm.add_constant(x, has_constant="add")
    model = sm.OLS(y, X).fit()

    params = model.params
    if hasattr(params, "index"):
        beta0 = float(params.get("const", params.iloc[0]))
        beta1 = float(params.get(x_name, params.iloc[-1]))
    else:
        params_arr = np.asarray(params).flatten()
        if params_arr.size < 2:
            raise ValueError("OLS returned insufficient parameters")
        beta0 = float(params_arr[0])
        beta1 = float(params_arr[1])

    resid = y - model.predict(X)
    sigma = float(resid.std())
    pval = adf_pvalue(resid)
    return {"beta0": beta0, "beta1": beta1, "sigma": sigma, "p_value": pval}


def residual_on_row(beta0: float, beta1: float, y_val: float, x_val: float) -> float:
    """Compute residual for a single bar."""
    return float(y_val - (beta0 + beta1 * x_val))


def drop_warmup_rows(
    daily: pd.DataFrame,
    *,
    activity_columns: tuple[str, ...] = (
        "entry_flag",
        "exit_flag",
        "trade_signal",
        "position",
        "fixed_beta_check_done",
        "in_coint_period",
    ),
) -> pd.DataFrame:
    """
    Remove rows before the first trading activity so warmup bars don't affect stats.

    Activity is detected via the earliest index where any listed column is truthy/non-zero.
    If no activity found, return the DataFrame unchanged (aside from index reset).
    """

    if daily is None or daily.empty:
        return daily

    d = daily.copy()
    first_idx: int | None = None

    for col in activity_columns:
        if col not in d.columns:
            continue
        series = d[col]
        if series.dtype == bool:
            active_mask = series
        else:
            active_mask = pd.to_numeric(series, errors="coerce").fillna(0) != 0
        active_indices = d.index[active_mask]
        if len(active_indices):
            idx = int(active_indices[0])
            first_idx = idx if first_idx is None else min(first_idx, idx)

    if first_idx is None:
        return d.reset_index(drop=True)

    start_idx = max(first_idx - 1, 0)
    return d.loc[start_idx:].reset_index(drop=True)
