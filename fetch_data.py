#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Binance historical klines downloader (Python version of your Bash script).

Features:
- Download monthly klines zips for spot or futures (UM) markets
- Unzip CSV into target folder and remove the zip
- Skip if CSV already exists
- Works across Windows/macOS/Linux
- Optional multithreading
- Uses system proxy env vars by default; can override via CLI

Example:
python download_binance_klines.py \
  --cryptos DOGE DOT OP MINA C98 1INCH CELO KSM BAND \
  --quotes USDT \
  --data-types futures \
  --begin 2022-09 --end 2025-08 \
  --interval 1h \
  --save data \
  --threads 8

Notes:
- Data types: "spot" or "futures" (UM)
- For spot base url:
  https://data.binance.vision/data/spot/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY-MM}.zip
- For UM futures base url:
  https://data.binance.vision/data/futures/um/monthly/klines/{SYMBOL}/{INTERVAL}/{SYMBOL}-{INTERVAL}-{YYYY-MM}.zip
"""

from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass

from urllib.request import Request, urlopen, build_opener, ProxyHandler
from urllib.error import HTTPError, URLError


SPOT_BASE = "https://data.binance.vision/data/spot/monthly/klines"
FUTURES_UM_BASE = "https://data.binance.vision/data/futures/um/monthly/klines"


@dataclass(frozen=True)
class Task:
    data_type: str         # "spot" or "futures"
    symbol: str            # e.g., "BTCUSDT"
    interval: str          # e.g., "1h"
    year_month: str        # "YYYY-MM"
    url: str               # zip url
    csv_path: str          # final csv output path
    zip_path: str          # temp zip path on disk (we'll use in-memory by default)


def month_add(ym: str, k: int = 1) -> str:
    """Add k months to a 'YYYY-MM' string and return 'YYYY-MM'."""
    y, m = map(int, ym.split("-"))
    m_total = m - 1 + k
    y += m_total // 12
    m = (m_total % 12) + 1
    return f"{y:04d}-{m:02d}"


def expand_months(begin: str, end: str) -> list[str]:
    """Inclusive range from begin to end, both 'YYYY-MM'."""
    cur = begin
    out = []
    while True:
        out.append(cur)
        if cur == end:
            break
        cur = month_add(cur, 1)
    return out


def make_url(data_type: str, symbol: str, interval: str, year_month: str) -> str:
    """Build the zip url based on data type."""
    if data_type == "spot":
        base = SPOT_BASE
    elif data_type == "futures":
        base = FUTURES_UM_BASE
    else:
        raise ValueError(f"Unsupported data_type: {data_type}")
    fn = f"{symbol}-{interval}-{year_month}.zip"
    return f"{base}/{symbol}/{interval}/{fn}"


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def default_proxy_handler(explicit_proxy: str | None) -> ProxyHandler:
    """
    Build a ProxyHandler.
    - If explicit_proxy is provided, force that for both http/https.
    - Else use environment variables (HTTP_PROXY/HTTPS_PROXY/ALL_PROXY).
    """
    if explicit_proxy:
        proxies = {
            "http": explicit_proxy,
            "https": explicit_proxy,
        }
    else:
        # Use system env proxies
        # urllib automatically respects env if ProxyHandler(None) not used.
        proxies = None
    return ProxyHandler(proxies)


def download_and_extract(task: Task, proxy: str | None, timeout: int, retries: int) -> str:
    """
    Download the zip to memory, extract CSV to target folder, and return a log string.
    If csv already exists, return a 'skip' message.
    """
    if os.path.isfile(task.csv_path):
        size = human_filesize(task.csv_path)
        return f"{task.symbol}-{task.interval}-{task.year_month}.csv already exists, file size: {size}"

    opener = build_opener(default_proxy_handler(proxy))
    headers = {
        "User-Agent": "Mozilla/5.0 (compatible; BinanceKlinesDownloader/1.0)"
    }

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            req = Request(task.url, headers=headers)
            with opener.open(req, timeout=timeout) as resp:
                if resp.status != 200:
                    raise HTTPError(task.url, resp.status, f"HTTP {resp.status}", hdrs=resp.headers, fp=None)
                data = resp.read()
            # extract from zip (in-memory)
            with zipfile.ZipFile(io.BytesIO(data)) as zf:
                # Usually there is a single csv in each monthly zip
                # Extract all to the directory containing csv_path
                out_dir = os.path.dirname(task.csv_path)
                ensure_dir(out_dir)
                zf.extractall(out_dir)
            size = human_filesize(task.csv_path)
            return f"{task.symbol}-{task.interval}-{task.year_month}.csv downloaded, file size: {size}"
        except HTTPError as e:
            # 404: this month might be missing
            if e.code == 404:
                return f"[404] Not found, skip: {task.url}"
            last_err = e
        except URLError as e:
            last_err = e
        except zipfile.BadZipFile as e:
            last_err = e
        except Exception as e:
            last_err = e

        # brief backoff
        time.sleep(min(1.0 * attempt, 3.0))

    # after retries
    return f"[FAILED] {task.url} -> {type(last_err).__name__}: {last_err}"


def human_filesize(path: str) -> str:
    """Return human-readable file size like '12K', '3.4M'."""
    try:
        n = os.path.getsize(path)
    except OSError:
        return "0B"
    units = ["B", "K", "M", "G", "T"]
    i = 0
    while n >= 1024 and i < len(units) - 1:
        n /= 1024.0
        i += 1
    if i == 0:
        return f"{int(n)}{units[i]}"
    return f"{n:.1f}{units[i]}"


def build_tasks(
    cryptocurrencies: list[str],
    quote_currencies: list[str],
    data_types: list[str],
    months: list[str],
    interval: str,
    save_root: str,
) -> list[Task]:
    tasks: list[Task] = []
    for crypto in cryptocurrencies:
        for quote in quote_currencies:
            symbol = f"{crypto}{quote}"
            for dt in data_types:
                # rows header (like your echo section)
                # we will just build all tasks; header printing happens in run()
                out_dir = os.path.join(save_root, dt, symbol, interval)
                ensure_dir(out_dir)
                for ym in months:
                    url = make_url(dt, symbol, interval, ym)
                    csv_path = os.path.join(out_dir, f"{symbol}-{interval}-{ym}.csv")
                    zip_path = os.path.join(out_dir, f"{symbol}-{interval}-{ym}.zip")
                    tasks.append(Task(dt, symbol, interval, ym, url, csv_path, zip_path))
    return tasks


def run(
    cryptocurrencies: list[str],
    quote_currencies: list[str],
    data_types: list[str],
    begin: str,
    end: str,
    interval: str,
    save_root: str,
    threads: int,
    proxy: str | None,
    timeout: int,
    retries: int,
) -> None:
    months = expand_months(begin, end)
    start_time = time.strftime("%Y-%m-%d %H:%M:%S")
    print(f"====================================")
    print(f"downloading: {','.join(cryptocurrencies)} x {','.join(quote_currencies)} "
          f"{','.join(data_types)} data, from {months[0]} to {months[-1]}, interval: {interval}")
    print(f"current time: {start_time}, saving to: {os.path.abspath(save_root)}")

    tasks = build_tasks(cryptocurrencies, quote_currencies, data_types, months, interval, save_root)

    # Pretty group header like the Bash loop
    # We'll print group by (data_type, symbol)
    groups = {}
    for t in tasks:
        groups.setdefault((t.data_type, t.symbol), []).append(t)

    for (dt, symbol), ts in groups.items():
        print(f"\n=== {symbol} [{dt}] ===")
        if threads <= 1:
            # sequential
            count = 0
            for t in ts:
                msg = download_and_extract(t, proxy=proxy, timeout=timeout, retries=retries)
                if msg.endswith("downloaded") or "downloaded, file size:" in msg:
                    count += 1
                    print(f"{count}: {msg}")
                else:
                    print(msg)
        else:
            # threaded
            count = 0
            with ThreadPoolExecutor(max_workers=threads) as ex:
                fut2task = {ex.submit(download_and_extract, t, proxy, timeout, retries): t for t in ts}
                for fut in as_completed(fut2task):
                    msg = fut.result()
                    if msg.endswith("downloaded") or "downloaded, file size:" in msg:
                        count += 1
                        print(f"{count}: {msg}")
                    else:
                        print(msg)

    print(f"\n====================================")
    print("done.")


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Download Binance monthly kline CSVs (spot/futures UM).")
    p.add_argument("--cryptos", nargs="+", required=True, help="Base currencies, e.g. DOGE DOT OP ...")
    p.add_argument("--quotes", nargs="+", required=True, help="Quote currencies, e.g. USDT BUSD ...")
    p.add_argument("--data-types", nargs="+", required=True, choices=["spot", "futures"],
                   help="Data types: spot or futures (UM).")
    p.add_argument("--begin", required=True, help="Begin year-month (YYYY-MM), inclusive.")
    p.add_argument("--end", required=True, help="End year-month (YYYY-MM), inclusive.")
    p.add_argument("--interval", required=True, help="Kline interval, e.g. 1h / 1m / 1s ...")
    p.add_argument("--save", default="data", help="Root folder to save data.")
    p.add_argument("--threads", type=int, default=1, help="Number of parallel download workers.")
    p.add_argument("--proxy", default=None,
                   help="Override proxy (e.g., http://127.0.0.1:61017 or socks5://127.0.0.1:61017). "
                        "If omitted, system env proxies are used if present.")
    p.add_argument("--timeout", type=int, default=30, help="Per-request timeout (seconds).")
    p.add_argument("--retries", type=int, default=3, help="Download retries per file.")
    return p.parse_args(argv)


def get_pairs_spot(quote: str = "USDT") -> List[str]:
    url = "https://data-api.binance.vision/api/v3/exchangeInfo"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    symbols_info = data.get("symbols", [])
    pairs = {
        s["baseAsset"]
        for s in symbols_info
        if s.get("quoteAsset") == quote and s.get("status") == "TRADING"
    }
    return sorted(pairs)


if __name__ == "__main__":
    # Defaults that mirror your Bash script; can be overridden via CLI.
    # default_cryptos = ["DOGE", "DOT", "OP", "MINA", "C98", "1INCH", "CELO", "KSM", "BAND"]
    default_cryptos = get_pairs_spot("USDT")
    default_quotes = ["USDT"]
    default_data_types = ["futures"]
    default_begin = "2022-09"
    default_end = "2025-08"
    default_interval = "1h"
    default_save = "data"

    if len(sys.argv) == 1:
        # No CLI arguments provided -> run with defaults similar to your Bash script
        run(
            cryptocurrencies=default_cryptos,
            quote_currencies=default_quotes,
            data_types=default_data_types,
            begin=default_begin,
            end=default_end,
            interval=default_interval,
            save_root=default_save,
            threads=64,            # set >1 to enable multithreading
            proxy=None,           # or e.g. "http://127.0.0.1:61017"
            timeout=30,
            retries=3,
        )
    else:
        args = parse_args(sys.argv[1:])
        run(
            cryptocurrencies=args.cryptos,
            quote_currencies=args.quotes,
            data_types=args.data_types,
            begin=args.begin,
            end=args.end,
            interval=args.interval,
            save_root=args.save,
            threads=args.threads,
            proxy=args.proxy,
            timeout=args.timeout,
            retries=args.retries,
        )
