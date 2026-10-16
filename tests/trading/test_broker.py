"""Tests for the paper-only Alpaca order adapter."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any
from urllib.parse import parse_qs, urlparse

import pytest

from reflective_loop_runtime.trading.broker import (
    PAPER_BASE_URL,
    BrokerDataError,
    PaperBroker,
    PaperOnlyError,
    canonical_symbol,
    client_order_id,
    make_order_adapter,
)

FIXTURES = Path(__file__).parent / "fixtures"


def _load(name: str) -> Any:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def _make_broker(
    *,
    http_get=None,
    http_post=None,
    key: str = "PK1234567890ABCDEF",
    secret: str = "secret-shhh-do-not-print",
    base_url: str = PAPER_BASE_URL,
) -> PaperBroker:
    return PaperBroker(
        key,
        secret,
        http_get=http_get or (lambda url, headers: {}),
        http_post=http_post or (lambda url, headers, body: {}),
        base_url=base_url,
    )


class FakeHttpError(Exception):
    """Stand-in for an HTTP error raised by the injected transport."""

    def __init__(self, code: int, message: str = "") -> None:
        super().__init__(message)
        self.code = code


class FakeHttp:
    """A double that records calls and serves scripted responses."""

    def __init__(
        self,
        *,
        get_responses: list[Any] | None = None,
        post_response: Any = None,
    ) -> None:
        self.get_calls: list[tuple[str, dict[str, str]]] = []
        self.post_calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self._get_responses = list(get_responses or [])
        self._post_response = post_response
        self._get_index = 0

    def get(self, url: str, headers: dict[str, str]) -> Any:
        self.get_calls.append((url, headers))
        if self._get_index >= len(self._get_responses):
            return self._get_responses[-1]
        value = self._get_responses[self._get_index]
        self._get_index += 1
        if isinstance(value, Exception):
            raise value
        return value

    def post(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> Any:
        self.post_calls.append((url, headers, body))
        return self._post_response


class _CallableHttp:
    """A minimal HTTP double whose ``get`` is driven by a user-supplied function."""

    def __init__(self, get_fn) -> None:
        self.get_calls: list[tuple[str, dict[str, str]]] = []
        self.post_calls: list[tuple[str, dict[str, str], dict[str, Any]]] = []
        self._get_fn = get_fn

    def get(self, url: str, headers: dict[str, str]) -> Any:
        self.get_calls.append((url, headers))
        return self._get_fn(url, headers)

    def post(
        self, url: str, headers: dict[str, str], body: dict[str, Any]
    ) -> Any:
        self.post_calls.append((url, headers, body))
        return {}


# ---------------------------------------------------------------------------
# Fixtures used by the strict-validation tests
# ---------------------------------------------------------------------------

def _order(
    *,
    oid: str,
    client_order_id: str = "rlr-deadbeefcafebabe12345678",
    symbol: str = "BTC/USD",
    side: str = "buy",
    status: str = "filled",
    submitted_at: str = "2024-01-01T00:00:00Z",
    **extra: Any,
) -> dict[str, Any]:
    """Return a fully-populated order dict that passes strict validation.

    Tests can use this to assemble payloads whose primary concern is shape;
    individual tests can then strip or mutate one field to assert the
    exact failure mode.
    """
    payload: dict[str, Any] = {
        "id": oid,
        "client_order_id": client_order_id,
        "symbol": symbol,
        "side": side,
        "status": status,
        "submitted_at": submitted_at,
    }
    payload.update(extra)
    return payload


def _position(
    *,
    symbol: str = "BTC/USD",
    qty: str = "0.25",
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"symbol": symbol, "qty": qty}
    payload.update(extra)
    return payload


def _account(
    *,
    status: str = "ACTIVE",
    crypto_status: str = "ACTIVE",
    **extra: Any,
) -> dict[str, Any]:
    payload: dict[str, Any] = {"status": status, "crypto_status": crypto_status}
    payload.update(extra)
    return payload


class TestPaperGuard:
    def test_refuses_live_endpoint(self) -> None:
        with pytest.raises(PaperOnlyError):
            _make_broker(base_url="https://api.alpaca.markets")

    def test_refuses_non_pk_key(self) -> None:
        with pytest.raises(PaperOnlyError):
            _make_broker(key="AK1234567890ABCDEF")

    def test_accepts_paper_base_url(self) -> None:
        broker = _make_broker()
        assert broker is not None


class TestClientOrderId:
    def test_idempotent_client_order_id(self) -> None:
        first = client_order_id("BTC/USD", "buy", 42)
        second = client_order_id("BTC/USD", "buy", 42)
        assert first == second
        # The id must be namespaced and bounded to <= 24 hex chars after prefix.
        assert first.startswith("rlr-")
        assert len(first) == len("rlr-") + 24
        # Different tick -> different id.
        other = client_order_id("BTC/USD", "buy", 43)
        assert other != first
        # Different symbol or side also changes the id.
        assert client_order_id("ETH/USD", "buy", 42) != first
        assert client_order_id("BTC/USD", "sell", 42) != first


class TestReprHidesSecrets:
    def test_repr_hides_secrets(self) -> None:
        broker = _make_broker()
        text = repr(broker)
        str_text = str(broker)
        for hidden in ("PK1234567890ABCDEF", "secret-shhh-do-not-print"):
            assert hidden not in text
            assert hidden not in str_text
        # Headers must contain the secret but it must never leak through repr.
        headers = broker._headers()
        assert headers["APCA-API-KEY-ID"] == "PK1234567890ABCDEF"
        assert headers["APCA-API-SECRET-KEY"] == "secret-shhh-do-not-print"


class TestSubmitValidation:
    def test_submit_buy_requires_notional_sell_requires_qty(self) -> None:
        http = FakeHttp(post_response=_load("order_submitted.json"))
        broker = _make_broker(http_get=http.get, http_post=http.post)

        # Buy without notional is rejected.
        with pytest.raises(ValueError):
            broker.submit("BTC/USD", "buy", client_order_id="rlr-x", qty=0.01)
        # Sell without qty is rejected.
        with pytest.raises(ValueError):
            broker.submit("BTC/USD", "sell", client_order_id="rlr-x", notional=10.0)
        # Providing both notional and qty is rejected.
        with pytest.raises(ValueError):
            broker.submit(
                "BTC/USD",
                "buy",
                client_order_id="rlr-x",
                notional=10.0,
                qty=0.01,
            )
        # Providing neither is rejected.
        with pytest.raises(ValueError):
            broker.submit("BTC/USD", "buy", client_order_id="rlr-x")
        # No HTTP call should have happened for the validation failures.
        assert http.post_calls == []

        # A valid buy hits the endpoint with type=market, tif=gtc, notional.
        broker.submit(
            "BTC/USD",
            "buy",
            client_order_id="rlr-abcdef1234567890abcdef",
            notional=100.0,
        )
        assert len(http.post_calls) == 1
        url, _headers, body = http.post_calls[0]
        assert url == f"{PAPER_BASE_URL}/v2/orders"
        assert body["symbol"] == "BTC/USD"
        assert body["side"] == "buy"
        assert body["type"] == "market"
        assert body["time_in_force"] == "gtc"
        assert body["notional"] == 100.0
        assert "qty" not in body

        # A valid sell hits the endpoint with qty.
        broker.submit(
            "BTC/USD",
            "sell",
            client_order_id="rlr-zzzz1234567890abcdef",
            qty=0.5,
        )
        assert len(http.post_calls) == 2
        _, _, body = http.post_calls[1]
        assert body["side"] == "sell"
        assert body["qty"] == 0.5
        assert "notional" not in body


class TestWaitFilled:
    def test_wait_filled_timeout(self) -> None:
        # Every get_order call returns a pending order; we therefore expect
        # the loop to exhaust the deadline and raise TimeoutError.
        pending = _load("order_pending.json")
        http = FakeHttp(get_responses=[pending])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        sleeps: list[float] = []

        def fake_sleep(duration: float) -> None:
            sleeps.append(duration)

        with pytest.raises(TimeoutError):
            broker.wait_filled(
                "order-id",
                timeout_s=0.05,
                poll_s=0.01,
                sleep=fake_sleep,
            )
        # Sleep was called with poll_s.
        assert sleeps
        assert all(s == 0.01 for s in sleeps)

    def test_wait_filled_returns_on_filled(self) -> None:
        pending = _load("order_pending.json")
        filled = _load("order_filled.json")
        http = FakeHttp(get_responses=[pending, filled])
        broker = _make_broker(http_get=http.get, http_post=http.post)
        order = broker.wait_filled(
            "order-id",
            timeout_s=1.0,
            poll_s=0.01,
            sleep=lambda _d: None,
        )
        assert order["status"] == "filled"
        assert order["id"] == filled["id"]


class TestOrderAdapter:
    def test_order_adapter_returns_fill(self) -> None:
        submitted = _load("order_submitted.json")
        filled = _load("order_filled.json")
        http = FakeHttp(get_responses=[filled], post_response=submitted)
        broker = _make_broker(http_get=http.get, http_post=http.post)

        adapter = make_order_adapter(broker)
        result = adapter(
            {
                "symbol": "BTC/USD",
                "side": "buy",
                "client_order_id": client_order_id("BTC/USD", "buy", 1),
                "notional": 100.0,
            }
        )
        assert set(result.keys()) == {
            "order_id",
            "status",
            "filled_qty",
            "filled_avg_price",
            "client_order_id",
        }
        assert result["order_id"] == submitted["id"]
        assert result["status"] == "filled"
        assert result["filled_qty"] == filled["filled_qty"]
        assert result["filled_avg_price"] == filled["filled_avg_price"]
        assert result["client_order_id"] == filled["client_order_id"]
        # The adapter submitted with notional, no qty.
        assert len(http.post_calls) == 1
        assert http.post_calls[0][2]["notional"] == 100.0


class TestReadEndpoints:
    def test_account_returns_payload(self) -> None:
        account = _load("account.json")
        http = FakeHttp(get_responses=[account])
        broker = _make_broker(http_get=http.get, http_post=http.post)
        result = broker.account()
        assert result == account
        assert http.get_calls and http.get_calls[0][0].endswith("/v2/account")

    def test_positions_returns_payload(self) -> None:
        positions = _load("positions.json")
        http = FakeHttp(get_responses=[positions])
        broker = _make_broker(http_get=http.get, http_post=http.post)
        result = broker.positions()
        assert result == positions
        assert http.get_calls and http.get_calls[0][0].endswith("/v2/positions")

    def test_get_order_returns_payload(self) -> None:
        filled = _load("order_filled.json")
        http = FakeHttp(get_responses=[filled])
        broker = _make_broker(http_get=http.get, http_post=http.post)
        result = broker.get_order("abc-123")
        assert result == filled
        assert http.get_calls and http.get_calls[0][0].endswith("/v2/orders/abc-123")

    def test_open_orders_query(self) -> None:
        open_orders = [
            _order(oid="o1", status="open", submitted_at="2024-01-01T00:00:00Z"),
            _order(oid="o2", status="open", submitted_at="2024-01-02T00:00:00Z"),
        ]
        http = FakeHttp(get_responses=[open_orders])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        result = broker.open_orders()

        assert result == open_orders
        assert len(http.get_calls) == 1
        url = http.get_calls[0][0]
        assert url.startswith(f"{PAPER_BASE_URL}/v2/orders?")
        assert "status=open" in url
        assert "limit=500" in url
        assert "nested=false" in url

    def test_orders_since_paginates_and_dedupes(self) -> None:
        page1 = [
            _order(oid="o1", submitted_at="2024-01-01T00:00:00Z"),
            _order(oid="o2", submitted_at="2024-01-02T00:00:00Z"),
        ]
        # page2 overlaps page1 by id "o2": the cursor moves to o2 - 1s, so
        # the broker still returns o2 (after is exclusive) plus a fresh o3.
        page2 = [
            _order(oid="o2", submitted_at="2024-01-02T00:00:00Z"),
            _order(oid="o3", submitted_at="2024-01-03T00:00:00Z"),
        ]
        # page3 is below page_limit so the iteration stops here; the overlap
        # in page2 has already been deduped.
        page3 = [_order(oid="o3", submitted_at="2024-01-03T00:00:00Z")]
        http = FakeHttp(get_responses=[page1, page2, page3])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        result = broker.orders_since(
            "2023-12-31T00:00:00Z", page_limit=2, max_pages=10
        )

        assert [o["id"] for o in result] == ["o1", "o2", "o3"]
        assert len(http.get_calls) == 3
        # First call uses the supplied after_iso.
        first_url = http.get_calls[0][0]
        assert first_url.startswith(f"{PAPER_BASE_URL}/v2/orders?")
        assert "status=all" in first_url
        assert "direction=asc" in first_url
        assert "limit=2" in first_url
        assert "after=2023-12-31T00%3A00%3A00Z" in first_url
        # Second call advances after to the last order's submitted_at minus 1s.
        assert "after=2024-01-01T23%3A59%3A59Z" in http.get_calls[1][0]
        # Third call advances again to o3 - 1s.
        assert "after=2024-01-02T23%3A59%3A59Z" in http.get_calls[2][0]

    def test_orders_since_raises_when_incomplete(self) -> None:
        # A single full page that keeps re-paginating would loop forever
        # without max_pages; cap it at 1 to force the incomplete error.
        full_page = [
            _order(oid="o1", submitted_at="2024-01-01T00:00:00Z"),
            _order(oid="o2", submitted_at="2024-01-02T00:00:00Z"),
        ]
        http = FakeHttp(get_responses=[full_page])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(RuntimeError, match="order history incomplete"):
            broker.orders_since(
                "2023-12-31T00:00:00Z", page_limit=2, max_pages=1
            )
        # The single allowed page was fetched before the error was raised.
        assert len(http.get_calls) == 1

    def test_get_order_by_client_id_found(self) -> None:
        order = _order(
            oid="abc-123",
            client_order_id="rlr-xyz",
            status="filled",
            symbol="BTC/USD",
            side="buy",
            submitted_at="2024-01-01T00:00:00Z",
        )
        http = FakeHttp(get_responses=[order])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        result = broker.get_order_by_client_id("rlr-xyz")

        assert result == order
        url = http.get_calls[0][0]
        assert url.startswith(
            f"{PAPER_BASE_URL}/v2/orders:by_client_order_id"
        )
        assert "client_order_id=rlr-xyz" in url

    def test_get_order_by_client_id_404_returns_none(self) -> None:
        err = FakeHttpError(404, "order not found")
        http = FakeHttp(get_responses=[err])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        assert broker.get_order_by_client_id("rlr-missing") is None
        assert len(http.get_calls) == 1


class TestOrdersSinceTieSafe:
    """The tie-safe pagination guarantees on top of the basic pagination."""

    def test_orders_since_tie_group_across_page_boundary_not_skipped(self) -> None:
        # 5 orders, three of which share the same submitted_at. With a
        # page_limit of 2 the naive (last-timestamp) cursor would skip the
        # second and third t1 orders because Alpaca's ``after`` is exclusive.
        # The overlap-by-1s cursor + id-based dedupe must return every id
        # exactly once.
        all_orders = [
            _order(oid="o1", submitted_at="2024-01-01T00:00:00Z", symbol="BTCUSD"),
            _order(oid="o2", submitted_at="2024-01-02T00:00:00Z", symbol="BTCUSD"),
            _order(oid="o3", submitted_at="2024-01-02T00:00:00Z", symbol="BTCUSD"),
            _order(oid="o4", submitted_at="2024-01-02T00:00:00Z", symbol="BTCUSD"),
            _order(oid="o5", submitted_at="2024-01-03T00:00:00Z", symbol="BTCUSD"),
        ]

        def _after_dt(after: str) -> datetime:
            return datetime.fromisoformat(after.rstrip("Z"))

        def fake_get(url: str, headers: dict[str, str]) -> Any:
            qs = parse_qs(urlparse(url).query)
            after = qs.get("after", ["2023-12-31T00:00:00Z"])[0]
            limit = int(qs.get("limit", ["500"])[0])
            after_dt = _after_dt(after)
            filtered = [
                o for o in all_orders
                if _after_dt(o["submitted_at"]) > after_dt
            ]
            # Broker returns orders in ascending submitted_at order, then we
            # honour the page limit.  This matches the real Alpaca behaviour
            # the adapter is being made safe against.
            filtered.sort(key=lambda o: o["submitted_at"])
            return filtered[:limit]

        http = _CallableHttp(fake_get)
        broker = _make_broker(http_get=http.get, http_post=http.post)

        # A tie group smaller than a page is exhausted through the overlap.
        result = broker.orders_since(
            "2023-12-31T00:00:00Z", page_limit=4, max_pages=20
        )

        ids = [o["id"] for o in result]
        assert ids == ["o1", "o2", "o3", "o4", "o5"]

        # A tie group that fills a whole page cannot be exhausted with a
        # timestamp cursor: fail closed instead of returning partial history.
        with pytest.raises(RuntimeError, match="tie group"):
            broker.orders_since("2023-12-31T00:00:00Z", page_limit=2, max_pages=20)
        # Sanity: every id is present exactly once and the result is ordered
        # by submitted_at ascending, matching the broker.
        assert len(set(ids)) == len(ids)
        timestamps = [o["submitted_at"] for o in result]
        assert timestamps == sorted(timestamps)
        # The symbols are canonicalised as a side effect.
        assert all(o["symbol"] == "BTC/USD" for o in result)

    def test_orders_since_full_page_same_timestamp_raises(self) -> None:
        # A full page whose orders all share one timestamp yields the same
        # last-submitted-at as the previous page's last order, so subtracting
        # 1s produces a cursor no later than the current one. That is a
        # pathological tie group and must abort instead of looping forever.
        full_page = [
            _order(oid="o1", submitted_at="2024-01-01T00:00:00Z"),
            _order(oid="o2", submitted_at="2024-01-01T00:00:00Z"),
        ]
        # Two responses: the first page advances the cursor, the second page
        # would advance it by zero (since subtracting 1s from the same
        # timestamp returns the previous cursor), triggering the tie-group
        # error.
        http = FakeHttp(get_responses=[full_page, full_page])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(
            RuntimeError, match="order history incomplete: tie group"
        ):
            broker.orders_since(
                "2023-12-31T00:00:00Z", page_limit=2, max_pages=10
            )
        # We made it to the second page before raising.
        assert len(http.get_calls) == 2

    def test_open_orders_truncation_raises(self) -> None:
        # 500 open orders hits the broker's single-page cap; the adapter must
        # refuse to call that a complete snapshot.
        open_orders = [
            _order(
                oid=f"o{i}",
                status="open",
                submitted_at="2024-01-01T00:00:00Z",
            )
            for i in range(500)
        ]
        http = FakeHttp(get_responses=[open_orders])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(RuntimeError, match="open orders incomplete"):
            broker.open_orders()


class TestCanonicalSymbols:
    def test_canonical_symbol_unit(self) -> None:
        assert canonical_symbol("BTCUSD") == "BTC/USD"
        assert canonical_symbol("BTC/USD") == "BTC/USD"
        assert canonical_symbol("ETHUSD") == "ETH/USD"
        # No slash and not ending in USD -> unchanged.
        assert canonical_symbol("AAPL") == "AAPL"
        # Other quote currencies pass through.
        assert canonical_symbol("BTC/USDT") == "BTC/USDT"

    def test_symbols_canonicalised(self) -> None:
        # positions() rewrites BTCUSD -> BTC/USD.
        positions = [_position(symbol="BTCUSD", qty="0.25")]
        http_pos = FakeHttp(get_responses=[positions])
        broker_pos = _make_broker(
            http_get=http_pos.get, http_post=http_pos.post
        )
        result = broker_pos.positions()
        assert result[0]["symbol"] == "BTC/USD"

        # open_orders() rewrites ETHUSD -> ETH/USD and leaves BTC/USD alone.
        open_orders_payload = [
            _order(oid="o1", symbol="ETHUSD", status="open"),
            _order(oid="o2", symbol="BTC/USD", status="open"),
        ]
        http_open = FakeHttp(get_responses=[open_orders_payload])
        broker_open = _make_broker(
            http_get=http_open.get, http_post=http_open.post
        )
        result = broker_open.open_orders()
        assert [o["symbol"] for o in result] == ["ETH/USD", "BTC/USD"]

        # orders_since() rewrites SOLUSD -> SOL/USD on every returned order.
        page = [
            _order(
                oid="o1",
                submitted_at="2024-01-01T00:00:00Z",
                symbol="SOLUSD",
            ),
        ]
        http_since = FakeHttp(get_responses=[page])
        broker_since = _make_broker(
            http_get=http_since.get, http_post=http_since.post
        )
        result = broker_since.orders_since(
            "2023-12-31T00:00:00Z", page_limit=10, max_pages=10
        )
        assert result[0]["symbol"] == "SOL/USD"

        # get_order_by_client_id() rewrites BTCUSD -> BTC/USD.
        order = _order(
            oid="abc-123",
            client_order_id="rlr-xyz",
            status="filled",
            symbol="BTCUSD",
        )
        http_by = FakeHttp(get_responses=[order])
        broker_by = _make_broker(
            http_get=http_by.get, http_post=http_by.post
        )
        result = broker_by.get_order_by_client_id("rlr-xyz")
        assert result is not None
        assert result["symbol"] == "BTC/USD"


# ---------------------------------------------------------------------------
# Strict schema validation -- N1/N5 re-review.
# ---------------------------------------------------------------------------


class TestBrokerDataError:
    """The broker must reject malformed/null payloads instead of folding
    them into ``[]``."""

    def test_positions_null_payload_raises(self) -> None:
        # The classic N1 defect: a JSON ``null`` from http_get used to be
        # collapsed into ``[]`` via ``payload or []``.  Now it raises.
        http = FakeHttp(get_responses=[None])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(BrokerDataError, match="positions"):
            broker.positions()
        assert len(http.get_calls) == 1

    def test_open_orders_object_payload_raises(self) -> None:
        # A ``{}`` (object) response is not a list and must raise instead
        # of being treated as "no open orders".
        http = FakeHttp(get_responses=[{}])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(BrokerDataError, match="open_orders"):
            broker.open_orders()
        assert len(http.get_calls) == 1

    def test_orders_since_null_page_raises(self) -> None:
        # The very first page being ``null`` used to silently produce ``[]``;
        # the strict validator must abort instead so the caller treats it
        # as "broker unreachable / malformed" rather than "clean history".
        http = FakeHttp(get_responses=[None])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(BrokerDataError, match="orders_since"):
            broker.orders_since(
                "2023-12-31T00:00:00Z", page_limit=10, max_pages=5
            )
        assert len(http.get_calls) == 1

    def test_order_missing_fields_raises(self) -> None:
        # An order payload missing the required ``submitted_at`` field
        # must raise; the broker does not get to decide what "good enough"
        # means.
        incomplete = {
            "id": "o1",
            "client_order_id": "rlr-xyz",
            "symbol": "BTC/USD",
            "side": "buy",
            "status": "open",
            # submitted_at intentionally omitted.
        }
        http = FakeHttp(get_responses=[[incomplete]])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(BrokerDataError, match="missing required field"):
            broker.open_orders()

    def test_duplicate_ids_raise(self) -> None:
        # Two orders in the same response carrying the same ``id`` is a
        # wire-format violation -- the controller cannot dedupe by id
        # meaningfully when the broker already contradicted itself.
        page = [
            _order(oid="dup", submitted_at="2024-01-01T00:00:00Z"),
            _order(oid="dup", submitted_at="2024-01-01T00:00:01Z"),
        ]
        http = FakeHttp(get_responses=[page])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(BrokerDataError, match="duplicate id"):
            broker.open_orders()

    def test_empty_list_is_valid(self) -> None:
        # An empty list is the "complete, nothing to report" signal and
        # must NOT raise.  Both positions() and open_orders() exercise this.
        http_pos = FakeHttp(get_responses=[[]])
        broker_pos = _make_broker(
            http_get=http_pos.get, http_post=http_pos.post
        )
        assert broker_pos.positions() == []

        http_open = FakeHttp(get_responses=[[]])
        broker_open = _make_broker(
            http_get=http_open.get, http_post=http_open.post
        )
        assert broker_open.open_orders() == []

    def test_order_missing_fields_in_get_by_client_id_raises(self) -> None:
        # ``get_order_by_client_id`` returns a single dict; a non-404
        # response missing required fields must raise BrokerDataError.
        incomplete = {
            "id": "o1",
            "client_order_id": "rlr-xyz",
            # symbol / side / status / submitted_at intentionally missing.
        }
        http = FakeHttp(get_responses=[incomplete])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(BrokerDataError, match="missing required field"):
            broker.get_order_by_client_id("rlr-xyz")


class TestAccountStatus:
    """N5: ``account_status()`` exposes a normalised view of the account
    gating flags and refuses malformed payloads."""

    def test_account_status_active(self) -> None:
        http = FakeHttp(get_responses=[_account(status="ACTIVE", crypto_status="ACTIVE")])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        result = broker.account_status()

        assert result == {
            "active": True,
            "status": "ACTIVE",
            "crypto_status": "ACTIVE",
        }
        # The endpoint hit was the canonical /v2/account URL.
        assert http.get_calls and http.get_calls[0][0].endswith("/v2/account")

    def test_account_status_inactive_crypto(self) -> None:
        # A blocked crypto status flips ``active`` to False even when the
        # account itself is ACTIVE -- this is the only signal the runtime
        # has to refuse crypto entries.
        payload = _account(status="ACTIVE", crypto_status="INACTIVE")
        http = FakeHttp(get_responses=[payload])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        result = broker.account_status()

        assert result == {
            "active": False,
            "status": "ACTIVE",
            "crypto_status": "INACTIVE",
        }

    def test_account_status_inactive_account(self) -> None:
        # An account-side deactivation also flips ``active``; the runtime
        # needs both signals anyway, so we surface them faithfully.
        payload = _account(status="INACTIVE", crypto_status="ACTIVE")
        http = FakeHttp(get_responses=[payload])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        result = broker.account_status()

        assert result == {
            "active": False,
            "status": "INACTIVE",
            "crypto_status": "ACTIVE",
        }

    def test_account_status_malformed_raises(self) -> None:
        # JSON ``null`` must raise -- the runtime cannot proceed when the
        # gate signal itself is unreadable.
        http = FakeHttp(get_responses=[None])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(BrokerDataError, match="account"):
            broker.account_status()

    def test_account_status_missing_crypto_status_raises(self) -> None:
        # Missing crypto_status must raise; "active" is meaningless
        # without both signals known.
        http = FakeHttp(get_responses=[{"status": "ACTIVE"}])
        broker = _make_broker(http_get=http.get, http_post=http.post)

        with pytest.raises(BrokerDataError, match="crypto_status"):
            broker.account_status()
