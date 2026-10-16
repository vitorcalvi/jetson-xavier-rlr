"""Independent Alpaca PAPER-only certifier with strict account/host isolation.

No network call occurs in deterministic-exchange-sim. Live-paper requires
independent certifier credentials and matching expected account id. The
HALT_ENTRIES flag MUST remain present; legacy Stage2 is never touched.
"""

from __future__ import annotations

import argparse
import http.client
import json
import os
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from reflective_loop_runtime.quant.execution.smart_router import (
    Book,
    Level,
    SmartRouter,
    VenueCapabilities,
)

HOST = "paper-api.alpaca.markets"
HALT = Path("/home/vitor/rlr/flags/HALT_ENTRIES")
AUDIT = Path("/home/vitor/rlr/monitoring/alpaca_paper_30_cert.toon")


class PaperRefusal(RuntimeError):
    pass


def validate_host(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme != "https" or parsed.hostname != HOST or parsed.port is not None:
        raise PaperRefusal("paper-only exact HTTPS hostname required")
    if parsed.path not in ("", "/") or parsed.query or parsed.username:
        raise PaperRefusal("extra path/query/userinfo not allowed")
    if not HALT.is_file():
        raise PaperRefusal("FATAL: independent Stage2 HALT_ENTRIES absent")


@dataclass
class PaperREST:
    key: str
    secret: str
    expected_account_id: str
    host: str = HOST

    def __post_init__(self) -> None:
        validate_host("https://" + self.host)
        if not self.key or not self.secret or not self.expected_account_id:
            raise PaperRefusal("isolated certifier credentials/account id missing")

    def call(self, method: str, path: str, body: dict | None = None) -> dict | list:
        if method not in ("GET", "POST", "DELETE") or not path.startswith("/v2/"):
            raise PaperRefusal("unapproved HTTP operation")
        validate_host("https://" + self.host)
        data = json.dumps(body, separators=(",", ":")).encode() if body is not None else None
        connection = http.client.HTTPSConnection(self.host, timeout=12)
        try:
            connection.request(
                method,
                path,
                data,
                {
                    "APCA-API-KEY-ID": self.key,
                    "APCA-API-SECRET-KEY": self.secret,
                    "Content-Type": "application/json",
                },
            )
            response = connection.getresponse()
            raw = response.read(1_000_000)
            if response.status // 100 != 2:
                raise PaperRefusal("paper request failed: HTTP " + str(response.status))
            return json.loads(raw) if raw else {}
        finally:
            connection.close()

    def preflight(self) -> None:
        acct = self.call("GET", "/v2/account")
        if not isinstance(acct, dict) or str(acct.get("id")) != self.expected_account_id:
            raise PaperRefusal("certifier account does not match isolated expected paper account")
        # Safety: must not share Stage2's paper account. Explicit operator attestation
        # needed until a separate audited Stage2 ID is obtainable without reading secrets.
        if os.environ.get("ALPACA_CERT_INDEPENDENT_ACCOUNT_VERIFIED") != "YES":
            raise PaperRefusal("no independent-account attestation")
        if self.call("GET", "/v2/positions"):
            raise PaperRefusal("paper certifier account already holds positions")
        if self.call("GET", "/v2/orders?status=open&limit=100"):
            raise PaperRefusal("paper certifier account already has open orders")


def synthetic_limit(
    bid: float, ask: float, bid_size: float, ask_size: float, side: str, tick: float = 0.01
) -> float:
    if bid <= 0 or ask <= bid or (ask - bid) < 2 * tick:
        raise PaperRefusal("spread too tight for synthetic passive guard")
    book = Book(
        (Level(bid, bid_size), Level(bid - tick, max(0.000001, bid_size))),
        (Level(ask, ask_size), Level(ask + tick, max(0.000001, ask_size))),
        timestamp_ms=int(time.time() * 1000),
        sequence=1,
    )
    router = SmartRouter(
        VenueCapabilities("alpaca-crypto-spot", False, "gtc", tick, 0.000001, True),
        max_spread_bps=1000,
    )
    quote = router.plan(
        symbol="BTC/USD",
        side=side,
        quantity=0.0001,
        book=book,
        now_ms=int(time.time() * 1000),
        cycle_id=str(uuid.uuid4()),
        require_native_post_only=False,
        rtt_ms=50,
        sigma_price_per_sqrt_second=0.0,
    )
    return quote.limit


def simulate_30() -> dict:
    from scripts.certify_alpaca_paper_30 import simulate_30 as original_sim

    return original_sim()


def run_live(
    *, qty: float, max_order_notional_usd: float, cycles: int = 30, audit: Path = AUDIT
) -> dict:
    if not (0 < qty <= 0.001 and max_order_notional_usd <= 25.0 and cycles == 30):
        raise PaperRefusal("notional/quantity/cycle cap")
    api = PaperREST(
        os.environ.get("ALPACA_CERT_KEY_ID", ""),
        os.environ.get("ALPACA_CERT_SECRET", ""),
        os.environ.get("ALPACA_CERT_EXPECTED_ACCOUNT_ID", ""),
    )
    api.preflight()
    # Cannot prove safe position flattening or exchange maker classification from
    # REST snapshot alone: fail closed without sending an order.
    raise PaperRefusal(
        "NO_BROKER_POST: queue-aware execution, maker flags and guaranteed "
        "orphan cleanup are not certified; 30 real fills remain BLOCKED"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mode", choices=("deterministic-exchange-sim", "live-paper"), required=True
    )
    parser.add_argument("--qty", type=float, default=0.0001)
    parser.add_argument("--max-order-notional-usd", type=float, default=25)
    args = parser.parse_args()
    if args.mode == "live-paper":
        result = run_live(qty=args.qty, max_order_notional_usd=args.max_order_notional_usd)
    else:
        result = simulate_30()
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
