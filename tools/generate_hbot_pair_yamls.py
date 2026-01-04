#!/usr/bin/env python3
"""
Generate Hummingbot controller YAMLs from a pair summary CSV.

Defaults:
- Input CSV: runs/portfolio_1110_no_coint_T1reduced/test_pair_summary.csv
- Output dir: /home/jim/hummingbot/conf/controllers
- Top N: 100

Mapping rules:
- pair format: e.g., "SXPUSDT-ATAUSDT" -> trading_pair_1: "SXP-USDT", trading_pair_2: "ATA-USDT"
- id: "pair_trading_{BASE1}_{BASE2}" (e.g., pair_trading_SXP_ATA)
- z_score_entry_threshold: entry_k
- z_score_exit_threshold: exit_k
- max_records: default 720 (can optionally use train_len via --use-train-len)
- All other fields are fixed to the provided template values.

Usage:
  python tools/generate_hbot_pair_yamls.py \
    --csv runs/portfolio_1110_no_coint_T1reduced/test_pair_summary.csv \
    --out /home/jim/hummingbot/conf/controllers \
    --top 100 \
    --scripts-conf /home/jim/hummingbot/conf/scripts/conf_v2_with_controllers.yml

Notes:
- The script selects up to top N rows, by default sorted by Sharpe desc if present, else by Ret, else by FinalEquity.
- If output directory is outside this workspace, run this script locally on your machine.
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple


STABLE_QUOTES = ("USDT", "USDC", "BUSD", "USD", "TUSD", "FDUSD")


@dataclass
class PairRow:
    pair: str
    train_len: Optional[int]
    entry_k: Optional[float]
    exit_k: Optional[float]
    ret: Optional[float]
    final_equity: Optional[float]
    sharpe: Optional[float]


def parse_float(val: str) -> Optional[float]:
    try:
        return float(val)
    except Exception:
        return None


def parse_int(val: str) -> Optional[int]:
    try:
        return int(float(val))
    except Exception:
        return None


def split_pair_symbol(sym: str) -> Tuple[str, str]:
    """Split a symbol like 'SXPUSDT' into (base, quote) -> ('SXP', 'USDT').

    Chooses the longest matching known quote suffix.
    """
    for q in sorted(STABLE_QUOTES, key=len, reverse=True):
        if sym.endswith(q):
            base = sym[: -len(q)]
            if not base:
                break
            return base, q
    raise ValueError(f"Could not split symbol into base/quote: {sym}")


def parse_pair(pair: str) -> Tuple[str, str, str]:
    """Parse pair string like 'SXPUSDT-ATAUSDT' -> (base1, base2, quote).

    Ensures both legs share the same quote.
    """
    parts = pair.strip().split("-")
    if len(parts) != 2:
        raise ValueError(f"Unexpected pair format: {pair}")
    b1, q1 = split_pair_symbol(parts[0])
    b2, q2 = split_pair_symbol(parts[1])
    if q1 != q2:
        raise ValueError(f"Mismatched quotes in pair '{pair}': {q1} vs {q2}")
    return b1, b2, q1


def to_trading_pair(base: str, quote: str) -> str:
    return f"{base}-{quote}"


def yaml_from_row(
    row: PairRow,
    *,
    leverage: int = 5,
    interval: str = "1h",
    max_records: int = 720,
    connector: str = "binance_perpetual",
    total_amount_quote_str: str = "50",
    pnl_stop_loss_pct: float = 0.2,
    stop_loss_cooldown_days: float = 5.0,
    deactivate_drawdown_pct: float = 0.2,
) -> Tuple[str, str]:
    base1, base2, quote = parse_pair(row.pair)
    tp1 = to_trading_pair(base1, quote)
    tp2 = to_trading_pair(base2, quote)

    entry_k = row.entry_k if row.entry_k is not None else 0.8
    exit_k = row.exit_k if row.exit_k is not None else 0.5

    # Build yaml text (no external dependency on PyYAML).
    yaml_lines = [
        f"id: pair_trading_{base1}_{base2}",
        "controller_name: pair_trading",
        "controller_type: generic",
        f"total_amount_quote: '{total_amount_quote_str}'",
        "manual_kill_switch: false",
        "candles_config: []",
        "initial_positions: []",
        f"connector_name: {connector}",
        f"connector_name_leg_1: {connector}",
        f"connector_name_leg_2: {connector}",
        f"trading_pair_1: {tp1}",
        f"trading_pair_2: {tp2}",
        "position_mode: HEDGE",
        f"leverage: {leverage}",
        f"max_records: {max_records}",
        f"interval: {interval}",
        f"z_score_entry_threshold: {entry_k}",
        f"z_score_exit_threshold: {exit_k}",
        f"pnl_stop_loss_pct: {pnl_stop_loss_pct}",
        f"stop_loss_cooldown_days: {stop_loss_cooldown_days}",
        f"deactivate_drawdown_pct: {deactivate_drawdown_pct}",
        f"total_quote_allocation: {int(float(total_amount_quote_str))}",
        "",
    ]

    filename = f"{base1}_{base2}.yml"
    return filename, "\n".join(yaml_lines)


def load_rows(csv_path: Path) -> List[PairRow]:
    rows: List[PairRow] = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalize fieldnames (strip spaces/BOMs)
        field_map = {name: name.strip() for name in reader.fieldnames or []}
        for raw in reader:
            # Some CSVs may include weird BOMs/spaces; standardize keys
            rec = {field_map.get(k, k).strip(): v.strip() for k, v in raw.items()}
            pair = rec.get("pair") or rec.get("Pair") or rec.get("symbols") or ""
            if not pair:
                continue
            sharpe_val = rec.get("Sharpe") or rec.get("sharpe") or ""
            sharpe = parse_float(sharpe_val)
            if sharpe is None:
                sharpe = 0.0
            rows.append(
                PairRow(
                    pair=pair,
                    train_len=parse_int(rec.get("train_len", "")),
                    entry_k=parse_float(rec.get("entry_k", "")),
                    exit_k=parse_float(rec.get("exit_k", "")),
                    ret=parse_float(rec.get("Ret", "")),
                    final_equity=parse_float(rec.get("FinalEquity", "")),
                    sharpe=sharpe,
                )
            )
    return rows


def pick_top(
    rows: List[PairRow], top: int, *, keep_order: bool = False
) -> List[PairRow]:
    # Prefer sort by Sharpe desc; fallback to Ret desc then FinalEquity desc; else keep original order
    if keep_order:
        return rows[:top]
    scored = []
    have_sharpe = any(r.sharpe is not None for r in rows)
    have_ret = any(r.ret is not None for r in rows)
    have_fe = any(r.final_equity is not None for r in rows)
    if have_sharpe:
        # Treat missing/empty Sharpe as 0, then sort desc
        scored = sorted(rows, key=lambda r: r.sharpe, reverse=True)
    elif have_ret:
        scored = sorted(rows, key=lambda r: (r.ret is None, r.ret), reverse=True)
    elif have_fe:
        scored = sorted(
            rows, key=lambda r: (r.final_equity is None, r.final_equity), reverse=True
        )
    else:
        scored = rows[:]
    return scored[:top]


def _compose_controllers_config_block(names: List[str]) -> str:
    lines = [
        "controllers_config:",
        "  # AUTOGEN-CONTROLLERS-CONFIG-START",
    ]
    lines += [f"  - {n}" for n in names]
    lines += [
        "  # AUTOGEN-CONTROLLERS-CONFIG-END",
    ]
    return "\n".join(lines)


def _compose_full_scripts_yaml(names: List[str]) -> str:
    block = _compose_controllers_config_block(names)
    lines = [
        "markets: {}",
        "candles_config: []",
        block,
        "script_file_name: v2_with_controllers.py",
        "max_global_drawdown_quote: null",
        "max_controller_drawdown_quote: null",
        "",
    ]
    return "\n".join(lines)


def update_scripts_conf(scripts_path: Path, controller_paths: List[Path]) -> None:
    scripts_path.parent.mkdir(parents=True, exist_ok=True)
    names = [p.name for p in controller_paths]
    start_tag = "# AUTOGEN-CONTROLLERS-CONFIG-START"
    end_tag = "# AUTOGEN-CONTROLLERS-CONFIG-END"

    if not scripts_path.exists():
        scripts_path.write_text(_compose_full_scripts_yaml(names), encoding="utf-8")
        return

    text = scripts_path.read_text(encoding="utf-8")

    # Case 1: Replace inside existing markers block
    if start_tag in text and end_tag in text:
        s_idx = text.find(start_tag)
        e_idx = text.find(end_tag)
        # Expand to full lines
        s_line = text.rfind("\n", 0, s_idx) + 1
        e_line = text.find("\n", e_idx)
        if e_line == -1:
            e_line = len(text)
        new_block = _compose_controllers_config_block(names).splitlines()
        # We only want to replace the marker lines and list, not the 'controllers_config:' key if it's above.
        # So rebuild using the same leading indentation: two spaces for list items.
        replacement = (
            "  # AUTOGEN-CONTROLLERS-CONFIG-START\n"
            + "\n".join(f"  - {n}" for n in names)
            + "\n  # AUTOGEN-CONTROLLERS-CONFIG-END"
        )
        new_text = text[:s_line] + replacement + text[e_line:]
        scripts_path.write_text(new_text, encoding="utf-8")
        return

    # Case 2: Replace an existing top-level 'controllers_config:' section
    lines = text.splitlines()
    idx = None
    for i, line in enumerate(lines):
        if line.lstrip() == line and line.strip().startswith("controllers_config:"):
            idx = i
            break
    if idx is not None:
        j = idx + 1
        while j < len(lines):
            ln = lines[j]
            if ln.strip() == "":
                j += 1
                continue
            if ln[0].isspace():
                j += 1
                continue
            break
        new_block = _compose_controllers_config_block(names).splitlines()
        new_lines = lines[:idx] + new_block + lines[j:]
        scripts_path.write_text("\n".join(new_lines) + "\n", encoding="utf-8")
        return

    # Case 3: Append controllers_config block at the end
    if not text.endswith("\n\n"):
        text = text.rstrip("\n") + "\n\n"
    scripts_path.write_text(
        text + _compose_controllers_config_block(names) + "\n", encoding="utf-8"
    )


def main():
    ap = argparse.ArgumentParser(
        description="Generate Hummingbot pair controller YAMLs from CSV"
    )
    ap.add_argument(
        "--csv",
        default="runs/portfolio_1110_no_coint_T1reduced/test_pair_summary.csv",
        help="Path to test_pair_summary.csv",
    )
    ap.add_argument(
        "--out",
        default="/home/jim/hummingbot/conf/controllers",
        help="Output directory for YAML files",
    )
    ap.add_argument(
        "--top", type=int, default=100, help="Number of top rows to generate"
    )
    ap.add_argument("--leverage", type=int, default=5)
    ap.add_argument("--interval", default="1h")
    ap.add_argument("--max-records", type=int, default=720)
    ap.add_argument(
        "--use-train-len",
        action="store_true",
        help="Use train_len as max_records when available",
    )
    ap.add_argument("--connector", default="binance_perpetual")
    ap.add_argument(
        "--total-amount-quote",
        default="50",
        help="As string; will also set total_quote_allocation numeric",
    )
    ap.add_argument("--pnl-stop-loss-pct", type=float, default=0.2)
    ap.add_argument("--stop-loss-cooldown-days", type=float, default=5.0)
    ap.add_argument("--deactivate-drawdown-pct", type=float, default=0.2)
    ap.add_argument(
        "--scripts-conf",
        default="/home/jim/hummingbot/conf/scripts/conf_v2_with_controllers.yml",
        help="Scripts conf YAML to update with controller list",
    )
    ap.add_argument(
        "--no-update-scripts",
        action="store_true",
        help="Do not update scripts conf file",
    )
    ap.add_argument(
        "--keep-order",
        action="store_true",
        help="Keep CSV order (no sorting), take first N",
    )
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows(csv_path)
    if not rows:
        raise SystemExit(f"No rows found in {csv_path}")

    selected = pick_top(rows, args.top, keep_order=args.keep_order)

    written = 0
    written_paths: List[Path] = []
    for r in selected:
        mr = r.train_len if (args.use_train_len and r.train_len) else args.max_records
        fname, text = yaml_from_row(
            r,
            leverage=args.leverage,
            interval=args.interval,
            max_records=int(mr),
            connector=args.connector,
            total_amount_quote_str=args.total_amount_quote,
            pnl_stop_loss_pct=args.pnl_stop_loss_pct,
            stop_loss_cooldown_days=args.stop_loss_cooldown_days,
            deactivate_drawdown_pct=args.deactivate_drawdown_pct,
        )
        out_path = out_dir / fname
        out_path.write_text(text, encoding="utf-8")
        written_paths.append(out_path)
        written += 1

    print(f"Wrote {written} YAMLs to {out_dir}")

    if not args.no_update_scripts:
        update_scripts_conf(Path(args.scripts_conf), written_paths)
        print(f"Updated scripts conf: {args.scripts_conf}")


if __name__ == "__main__":
    main()
