"""Read-only Alpaca FILL+CFEE reconciliation (paper only).

Never writes orders, never prints credentials, fails closed on incomplete activity
history, and writes a durable daily aggregate with an explicit completeness flag.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from reflective_loop_runtime.trading.broker import PaperBroker
from reflective_loop_runtime.trading.cashflow import parse_activities, summary, round_trips
from reflective_loop_runtime.trading.http import _default_http, _default_post


def reconcile(*, output: Path, days: int = 3) -> dict:
    key = os.environ.get("ALPACA_PAPER_KEY", "")
    secret = os.environ.get("ALPACA_PAPER_SECRET", "")
    if not key.startswith("PK") or not secret:
        raise RuntimeError("paper credentials absent")
    after = (datetime.now(timezone.utc) - timedelta(days=days)).replace(
        hour=0, minute=0, second=0, microsecond=0
    ).isoformat()
    broker = PaperBroker(key, secret, http_get=_default_http, http_post=_default_post)
    raw = broker.activities_since(after)
    activities = parse_activities(raw)
    result = summary(activities)
    trips = round_trips(activities)
    crypto_fills = [a for a in activities if a.type == "FILL" and "/" in a.symbol]
    fee_rows = [a for a in activities if a.type == "CFEE"]
    # Paper APIs can omit crypto CFEE activities entirely. A price-only
    # matching result is NOT a fee-verified net result.
    crypto_order_ids = {a.order_id for a in crypto_fills if a.order_id}
    fee_order_ids = {a.order_id for a in fee_rows if a.order_id}
    missing_order_fees = crypto_order_ids - fee_order_ids
    # A handful of CFEE rows is not proof that *each* crypto buy/sell
    # fill has been fully fee reconciled. Missing fee rows fail closed.
    fee_verified = not crypto_fills or (
        bool(fee_rows) and not missing_order_fees
        and all(a.order_id for a in crypto_fills)
        and sum(t.fees_usd for t in trips) > 0
    )
    data = {
        "at_utc": datetime.now(timezone.utc).isoformat(),
        "after": after,
        "source": "alpaca_paper_FILLS_CFEE",
        "strategy_class": "DIAGNOSTIC_NON_ALPHA" if "autonomy-1m" in str(output) else "RESEARCH_PAPER",
        "performance_interpretation": "OPERATIONAL_AUTONOMY_PROBE_ONLY_NOT_ALPHA"
            if "autonomy-1m" in str(output) else "PAPER_RESEARCH_UNVERIFIED_EDGE",
        "complete": not bool(result["incomplete"]) and fee_verified,
        "fills_complete": not bool(result["incomplete"]),
        "fee_evidence_complete": fee_verified,
        "crypto_orders_missing_CFEE": len(missing_order_fees),
        "activity_type_counts": {
            "FILL": sum(a.type == "FILL" for a in activities),
            "CFEE": len(fee_rows),
        },
        "warning": ("Crypto CFEE missing for some orders; fill-derived amounts must not "
                    "be treated as true net P&L") if not fee_verified else None,
        "n_activities": len(activities),
        "round_trips": len(trips),
        "unverified_fill_matched_pnl_usd": str(result["realized_net_usd"]),
        "total_net_usd": str(result["realized_net_usd"]) if fee_verified else None,
        "total_fees_usd": str(result["fees_usd"]) if fee_verified else None,
        "open_lots": {k: str(v) for k, v in result["open_lots"].items()},
        "unmatched_sell_qty": {k: str(v) for k, v in result["unmatched_sell_qty"].items()},
        "trades": [
            {
                "symbol": t.symbol,
                "entry_ts": t.entry_ts,
                "exit_ts": t.exit_ts,
                "qty_sold": str(t.qty_sold),
                "net_usd": str(t.net_usd) if fee_verified else None,
                "fees_usd": str(t.fees_usd) if fee_verified else None,
                "net_pct": str(t.net_pct) if fee_verified else None,
                "fee_unverified_matched_pnl_usd": str(t.net_usd) if not fee_verified else None,
            } for t in trips
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    tmp.replace(output)
    return data


def main() -> int:
    try:
        output = Path(os.environ.get(
            "RLR_NET_PNL_REPORT", str(Path.home() / "rlr" / "trader-solusd" / "net_pnl_report.json")
        ))
        result = reconcile(output=output)
        print(json.dumps({
            "complete": result["complete"],
            "round_trips": result["round_trips"],
            "total_net_usd": result["total_net_usd"],
            "total_fees_usd": result["total_fees_usd"],
        "fee_evidence_complete": result["fee_evidence_complete"],
        "crypto_orders_missing_CFEE": result["crypto_orders_missing_CFEE"],
        }))
        # An incomplete broker report is an observed data condition. Keep the
        # scheduler healthy while keeping every P&L value explicitly unverified.
        return 0
    except Exception as exc:
        print(f"net P&L reconciliation failed: {type(exc).__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
