"""Trading data, signals, and paper-trading utilities for the RLR runtime.

This package owns the data ingestion side of the paper trader: Alpaca crypto
market snapshots and the deterministic feature pipeline that other lanes
consume. The interfaces in :mod:`reflective_loop_runtime.trading.types` are
the shared contract every dependent lane relies on; see that module for the
authoritative shape.
"""

from __future__ import annotations
