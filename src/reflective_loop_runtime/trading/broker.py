"""Paper-only Alpaca order adapter for the RLR paper trader.

This module is deliberately restricted to the Alpaca *paper* endpoint. The
:class:`PaperBroker` refuses any base URL that points at the live trading host
or uses a non-HTTPS scheme, and refuses any key that does not begin with the
Alpaca paper-key prefix ``PK``. These guards make it impossible to accidentally
route real orders through the paper-trading lane.

HTTP is always injected (``http_get`` / ``http_post`` callables) so tests can
exercise the adapter without performing real network I/O.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from datetime import datetime, timedelta
from hashlib import sha256
from typing import Any
from urllib.parse import urlencode, urlparse

PAPER_BASE_URL = "https://paper-api.alpaca.markets"
_PAPER_HOST = "paper-api.alpaca.markets"
_KEY_HEADER = "APCA-API-KEY-ID"
_SECRET_HEADER = "APCA-API-SECRET-KEY"
_OPEN_ORDERS_LIMIT = 500

# HTTP callable type: ``(url, headers) -> parsed JSON``.
HttpGet = Callable[[str, dict[str, str]], Any]
HttpPost = Callable[[str, dict[str, str], dict[str, Any]], Any]


class PaperOnlyError(RuntimeError):
    """Raised when an attempt is made to point the adapter at a live endpoint."""


class BrokerDataError(RuntimeError):
    """Raised when the broker returns a malformed or incomplete payload.

    Every read endpoint on :class:`PaperBroker` validates the wire format
    (list-of-dicts with required fields, no duplicate ids). An empty list is
    treated as "complete, empty", which is a valid result; a JSON ``null``,
    ``{}``, or any other type raises :class:`BrokerDataError` so the caller
    never confuses "broker is unreachable" with "broker is clean".
    """


# Required id-bearing fields: the value must be a non-empty string.
_ID_FIELDS = frozenset({"id", "client_order_id"})

# All fields required on an order payload (open_orders / orders_since /
# get_order_by_client_id). ``id`` and ``client_order_id`` must be non-empty
# strings; the others must merely be present.
_ORDER_REQUIRED = (
    "id",
    "client_order_id",
    "symbol",
    "side",
    "status",
    "submitted_at",
)

# Required fields on a position payload.
_POSITION_REQUIRED = ("symbol", "qty")


def _validate_base_url(base_url: str) -> None:
    parsed = urlparse(base_url)
    if parsed.scheme.lower() != "https":
        raise PaperOnlyError(
            f"PaperBroker requires https, got scheme={parsed.scheme!r}"
        )
    if (parsed.hostname or "").lower() != _PAPER_HOST:
        raise PaperOnlyError(
            "PaperBroker only accepts the paper-api.alpaca.markets host; "
            f"got host={parsed.hostname!r}"
        )


def _validate_key(key: str) -> None:
    if not isinstance(key, str) or not key.startswith("PK"):
        raise PaperOnlyError(
            "PaperBroker requires an Alpaca paper key starting with 'PK'"
        )


def client_order_id(symbol: str, side: str, tick: int) -> str:
    """Return a deterministic 24-char hex client order id.

    The id is namespaced under ``rlr-`` and hashed from ``symbol|side|tick``
    so that identical inputs produce identical ids (idempotency) while a single
    changed input produces an unrelated id.
    """
    digest = sha256(f"{symbol}|{side}|{tick}".encode()).hexdigest()
    return f"rlr-{digest[:24]}"


def canonical_symbol(s: str) -> str:
    """Return the canonical slash-separated symbol for ``s``.

    Alpaca paper responses sometimes omit the slash (``"BTCUSD"``); the rest of
    the runtime compares against the explicit form ``"BTC/USD"``. This helper
    inserts the slash before a trailing ``USD`` quote suffix when none is
    present, leaves already-canonical symbols alone, and leaves symbols that
    do not end in ``USD`` untouched (so e.g. ``"AAPL"`` passes through).

    >>> canonical_symbol("BTCUSD")
    'BTC/USD'
    >>> canonical_symbol("BTC/USD")
    'BTC/USD'
    >>> canonical_symbol("ETHUSD")
    'ETH/USD'
    >>> canonical_symbol("AAPL")
    'AAPL'
    """
    if not isinstance(s, str) or not s:
        return s
    if "/" in s:
        return s
    if s.endswith("USD"):
        return s[: -len("USD")] + "/USD"
    return s


def _canonicalise_dict(d: dict[str, Any]) -> dict[str, Any]:
    """Rewrite the ``"symbol"`` key of ``d`` through :func:`canonical_symbol`."""
    if isinstance(d, dict) and isinstance(d.get("symbol"), str):
        d["symbol"] = canonical_symbol(d["symbol"])
    return d


def _parse_iso(iso: str) -> datetime:
    """Parse an ISO 8601 timestamp with a trailing ``Z`` into a naive datetime."""
    return datetime.fromisoformat(iso.rstrip("Z"))


def _shift_iso_back(iso: str, seconds: int = 1) -> str:
    """Return ``iso`` shifted backwards by ``seconds`` seconds, formatted ISO+Z."""
    dt = _parse_iso(iso)
    new_dt = dt - timedelta(seconds=seconds)
    return new_dt.strftime("%Y-%m-%dT%H:%M:%SZ")


def _require_list(
    payload: Any,
    what: str,
    required_fields: tuple[str, ...],
) -> list[dict[str, Any]]:
    """Validate ``payload`` is a list of dicts with the required fields.

    ``payload`` must be a :class:`list` (an empty list is allowed and means
    "complete, empty"). Every element must be a :class:`dict` containing
    every name in ``required_fields``. For names in :data:`_ID_FIELDS` the
    value must additionally be a non-empty :class:`str` (so the caller can
    rely on ids being present and unique). Duplicate ids inside one
    response raise :class:`BrokerDataError`. Any deviation raises
    :class:`BrokerDataError` referencing ``what`` (e.g. ``"positions"``,
    ``"open_orders"``) so a misbehaving broker cannot be confused with a
    clean one.
    """
    if not isinstance(payload, list):
        raise BrokerDataError(
            f"{what}: expected list, got {type(payload).__name__}"
        )
    seen_ids: set[str] = set()
    for index, item in enumerate(payload):
        if not isinstance(item, dict):
            raise BrokerDataError(
                f"{what}[{index}]: expected dict, got {type(item).__name__}"
            )
        missing = [name for name in required_fields if name not in item]
        if missing:
            raise BrokerDataError(
                f"{what}[{index}]: missing required field(s) {missing!r}"
            )
        for id_field in _ID_FIELDS:
            if id_field in required_fields:
                value = item.get(id_field)
                if not isinstance(value, str) or not value:
                    raise BrokerDataError(
                        f"{what}[{index}]: {id_field!r} must be a non-empty "
                        f"string, got {value!r}"
                    )
        if "id" in required_fields:
            ident = item["id"]
            if ident in seen_ids:
                raise BrokerDataError(
                    f"{what}: duplicate id {ident!r} within one response"
                )
            seen_ids.add(ident)
    return payload


def _require_order_dict(payload: Any, what: str) -> dict[str, Any]:
    """Validate ``payload`` is an order dict with the required fields."""
    if not isinstance(payload, dict):
        raise BrokerDataError(
            f"{what}: expected dict, got {type(payload).__name__}"
        )
    missing = [name for name in _ORDER_REQUIRED if name not in payload]
    if missing:
        raise BrokerDataError(
            f"{what}: missing required field(s) {missing!r}"
        )
    for id_field in ("id", "client_order_id"):
        value = payload.get(id_field)
        if not isinstance(value, str) or not value:
            raise BrokerDataError(
                f"{what}: {id_field!r} must be a non-empty string, "
                f"got {value!r}"
            )
    return payload


def _require_account_dict(payload: Any) -> dict[str, Any]:
    """Validate ``payload`` is a dict containing ``status`` and ``crypto_status``.

    Both values must be strings (the convention in the rest of the adapter is
    that ``status`` arrives upper-case, e.g. ``"ACTIVE"``). A JSON ``null``,
    a list, or any other type raises :class:`BrokerDataError`.
    """
    if not isinstance(payload, dict):
        raise BrokerDataError(
            f"account: expected dict, got {type(payload).__name__}"
        )
    if "status" not in payload or not isinstance(payload["status"], str):
        raise BrokerDataError(
            f"account: missing or non-string 'status' (got "
            f"{payload.get('status')!r})"
        )
    if "crypto_status" not in payload or not isinstance(
        payload["crypto_status"], str
    ):
        raise BrokerDataError(
            f"account: missing or non-string 'crypto_status' (got "
            f"{payload.get('crypto_status')!r})"
        )
    return payload


class PaperBroker:
    """A strictly paper-only Alpaca order adapter."""

    def __init__(
        self,
        key: str,
        secret: str,
        *,
        http_get: HttpGet,
        http_post: HttpPost,
        base_url: str = PAPER_BASE_URL,
        time_in_force: str = "gtc",
    ) -> None:
        _validate_base_url(base_url)
        _validate_key(key)
        # Crypto uses gtc; US-equity fractional/notional orders must be day.
        if time_in_force not in ("gtc", "day"):
            raise ValueError(f"unsupported time_in_force: {time_in_force!r}")
        self._time_in_force = time_in_force
        # Secret is held but must never be exposed via __repr__/__str__.
        self._key = key
        self._secret = secret
        self._base_url = base_url.rstrip("/")
        self._http_get = http_get
        self._http_post = http_post

    def _headers(self) -> dict[str, str]:
        return {
            _KEY_HEADER: self._key,
            _SECRET_HEADER: self._secret,
            "Accept": "application/json",
        }

    def __repr__(self) -> str:
        return f"PaperBroker(base_url={self._base_url!r})"

    def account(self) -> Any:
        return self._http_get(f"{self._base_url}/v2/account", self._headers())

    def account_status(self) -> dict[str, Any]:
        """Return a summary of the account gating flags.

        Reads the ``/v2/account`` endpoint and validates the payload is a
        dict containing both ``status`` and ``crypto_status``. The
        ``active`` flag is true iff both are exactly ``"ACTIVE"``; any
        other value (or a missing key) makes ``active`` false. A JSON
        ``null``, a list, or any other non-dict payload -- including a
        dict missing either required key -- raises
        :class:`BrokerDataError`.
        """
        payload = self._http_get(f"{self._base_url}/v2/account", self._headers())
        validated = _require_account_dict(payload)
        status = validated["status"]
        crypto_status = validated["crypto_status"]
        return {
            "active": status == "ACTIVE" and crypto_status == "ACTIVE",
            "status": status,
            "crypto_status": crypto_status,
        }

    def market_is_open(self) -> bool:
        """Alpaca's market clock; any error or malformed reply counts as closed."""
        try:
            payload = self._http_get(f"{self._base_url}/v2/clock", self._headers())
            return isinstance(payload, dict) and payload.get("is_open") is True
        except Exception:  # noqa: BLE001 - unknown clock state blocks entries
            return False

    def positions(self) -> list[dict[str, Any]]:
        payload = self._http_get(f"{self._base_url}/v2/positions", self._headers())
        validated = _require_list(payload, "positions", _POSITION_REQUIRED)
        return [_canonicalise_dict(d) for d in validated]

    def activities_since(self, after_iso: str, *, max_pages: int = 30) -> list[dict[str, Any]]:
        """Read-only fill and fee activities, with complete ID-based pagination.

        Returns FILL/CFEE rows. Never silently report partial pages as P&L.
        """
        items: list[dict[str, Any]] = []
        seen: set[str] = set()
        token: str | None = None
        for _ in range(max_pages):
            params = {
                "activity_types": "FILL,CFEE",
                "after": str(after_iso),
                "direction": "asc",
                "page_size": "100",
            }
            if token:
                params["page_token"] = token
            payload = self._http_get(
                f"{self._base_url}/v2/account/activities?{urlencode(params)}",
                self._headers(),
            )
            if not isinstance(payload, list) or any(not isinstance(x, dict) for x in payload):
                raise BrokerDataError("activities: malformed response")
            if len(payload) > 100:
                raise BrokerDataError("activities: overfull page")
            for row in payload:
                rid = str(row.get("id") or "")
                if not rid:
                    raise BrokerDataError("activities: missing id")
                if rid in seen:
                    continue
                seen.add(rid)
                items.append(row)
            if len(payload) < 100:
                return items
            last = str(payload[-1]["id"])
            if last == token:
                raise BrokerDataError("activities: non-advancing pagination")
            token = last
        raise BrokerDataError("activities: pagination incomplete")

    def get_order(self, order_id: str) -> Any:
        return self._http_get(
            f"{self._base_url}/v2/orders/{order_id}", self._headers()
        )

    def open_orders(self) -> list[dict[str, Any]]:
        """Return all currently-open orders from the paper account.

        If the broker returns exactly the page limit the result is by
        definition potentially truncated (Alpaca caps a single page at 500);
        we surface that as a hard error rather than silently handing the
        caller an incomplete snapshot. The payload is validated: a JSON
        ``null`` or ``{}`` raises :class:`BrokerDataError` instead of being
        collapsed into an empty list.
        """
        params = {"status": "open", "limit": str(_OPEN_ORDERS_LIMIT), "nested": "false"}
        url = f"{self._base_url}/v2/orders?{urlencode(params)}"
        payload = self._http_get(url, self._headers())
        validated = _require_list(payload, "open_orders", _ORDER_REQUIRED)
        result = [_canonicalise_dict(d) for d in validated]
        if len(result) >= _OPEN_ORDERS_LIMIT:
            raise RuntimeError("open orders incomplete")
        return result

    def orders_since(
        self,
        after_iso: str,
        *,
        page_limit: int = 500,
        max_pages: int = 50,
    ) -> list[dict[str, Any]]:
        """Return all orders submitted strictly after ``after_iso``.

        Alpaca's ``/v2/orders`` endpoint does not expose an explicit page
        token, so we walk forward in time by feeding ``after=`` query
        parameters. Because the broker's ``after`` is **exclusive**, we
        deliberately overlap each page: the next cursor is the last order's
        ``submitted_at`` minus one second, which guarantees the same order
        (and any other order sharing that timestamp) appears on the next page
        so we can dedupe by ``id``. We stop **only** when the raw API page
        length is strictly below ``page_limit`` -- a full page might contain
        only duplicates of an earlier page and must never be treated as
        terminal. When the cursor does not strictly advance forward in time
        **and** the page contributed no new ids (i.e. it is a full page of
        ids we have already collected at the same timestamp), we abort with
        ``RuntimeError("order history incomplete: tie group")`` because the
        next page would be identical and we would loop forever. After
        ``max_pages`` successive full pages we abort with
        ``RuntimeError("order history incomplete")``.

        Every page is validated via :func:`_require_list`: a JSON ``null``
        or ``{}`` raises :class:`BrokerDataError` rather than being
        silently treated as "no orders on this page". An empty list is a
        valid terminal page.
        """
        seen: dict[str, dict[str, Any]] = {}
        cursor = after_iso
        cursor_dt = _parse_iso(cursor)
        for _ in range(max_pages):
            params = {
                "status": "all",
                "direction": "asc",
                "limit": str(page_limit),
                "after": cursor,
            }
            url = f"{self._base_url}/v2/orders?{urlencode(params)}"
            raw_page = self._http_get(url, self._headers())
            page = _require_list(raw_page, "orders_since", _ORDER_REQUIRED)
            new_ids = 0
            for order in page:
                if order["id"] not in seen:
                    new_ids += 1
                seen[order["id"]] = _canonicalise_dict(order)
            if len(page) < page_limit:
                return list(seen.values())
            new_last = page[-1]["submitted_at"]
            new_cursor = _shift_iso_back(new_last, 1)
            new_cursor_dt = _parse_iso(new_cursor)
            # Tie group: the cursor did not strictly advance forward in time
            # AND the page gave us no new ids.  A full page of orders at the
            # same timestamp as the previous page's last order (all of which
            # were already collected) produces exactly this pattern and
            # would loop forever without an explicit escape.
            if new_cursor_dt <= cursor_dt and new_ids == 0:
                raise RuntimeError("order history incomplete: tie group")
            cursor = new_cursor
            cursor_dt = new_cursor_dt
        raise RuntimeError("order history incomplete")

    def get_order_by_client_id(self, client_order_id: str) -> dict[str, Any] | None:
        """Look up an order by its client-supplied ``client_order_id``.

        Returns ``None`` when the broker reports HTTP 404 (the order does not
        exist). Any other error from the HTTP layer is propagated unchanged.
        A successful (non-404) response must be a dict containing every
        field required for an order payload (id, client_order_id, symbol,
        side, status, submitted_at); any deviation raises
        :class:`BrokerDataError` so the caller cannot accidentally act on a
        half-formed record.
        """
        params = {"client_order_id": client_order_id}
        url = f"{self._base_url}/v2/orders:by_client_order_id?{urlencode(params)}"
        try:
            payload = self._http_get(url, self._headers())
        except Exception as exc:  # noqa: BLE001 - narrowed by .code below
            if getattr(exc, "code", None) == 404:
                return None
            raise
        if payload is None:
            return None
        validated = _require_order_dict(payload, "get_order_by_client_id")
        return _canonicalise_dict(validated)

    def submit(
        self,
        symbol: str,
        side: str,
        *,
        client_order_id: str,
        notional: float | None = None,
        qty: float | None = None,
    ) -> Any:
        if notional is None and qty is None:
            raise ValueError("submit requires exactly one of notional or qty")
        if notional is not None and qty is not None:
            raise ValueError("submit accepts exactly one of notional or qty")
        normalized_side = side.lower()
        if normalized_side == "buy" and notional is None:
            raise ValueError("buy orders require notional")
        if normalized_side == "sell" and qty is None:
            raise ValueError("sell orders require qty")
        body: dict[str, Any] = {
            "symbol": symbol,
            "side": normalized_side,
            "type": "market",
            "time_in_force": self._time_in_force,
            "client_order_id": client_order_id,
        }
        if notional is not None:
            body["notional"] = float(notional)
        if qty is not None:
            body["qty"] = float(qty)
        return self._http_post(
            f"{self._base_url}/v2/orders", self._headers(), body
        )

    def wait_filled(
        self,
        order_id: str,
        *,
        timeout_s: float = 20.0,
        poll_s: float = 1.0,
        sleep: Callable[[float], None] = time.sleep,
    ) -> dict[str, Any]:
        deadline = time.monotonic() + float(timeout_s)
        while True:
            order = self.get_order(order_id)
            status = (order or {}).get("status")
            if status == "filled":
                return order
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"order {order_id} did not fill within {timeout_s}s "
                    f"(last status={status!r})"
                )
            sleep(float(poll_s))


def make_order_adapter(
    broker: PaperBroker,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """Return a callable that submits then waits for an order to fill."""

    def _adapter(args: dict[str, Any]) -> dict[str, Any]:
        order = broker.submit(
            args["symbol"],
            args["side"],
            client_order_id=args["client_order_id"],
            notional=args.get("notional"),
            qty=args.get("qty"),
        )
        order_id = order["id"]
        filled = broker.wait_filled(order_id)
        return {
            "order_id": order_id,
            "status": filled.get("status"),
            "filled_qty": filled.get("filled_qty"),
            "filled_avg_price": filled.get("filled_avg_price"),
            "client_order_id": filled.get("client_order_id"),
        }

    return _adapter
