"""Read-only latency/thermal harness; no unmeasured TensorRT or power claims.

Can benchmark a passed model callable through a local Python import. Default
bench measures only CPU policy encoding, not ONNX/TRT model inference.
"""

from __future__ import annotations

import json
import platform
import resource
import statistics
import time

from reflective_loop_runtime.quant.edge_jetson.runtime_opt import toon_encode
from reflective_loop_runtime.quant.edge_jetson.thermal_v2 import ThermalGovernor


def benchmark(n: int = 250) -> dict:
    if n < 10:
        raise ValueError("min 10 iterations")
    payload = {
        "W1": {"symbol": "SPY", "position": 0, "time": 0},
        "W2": {"state": "UNSAFE", "obi": 0.2},
    }
    laps = []
    gov = ThermalGovernor()
    for i in range(n):
        payload["W1"]["time"] = i
        t = time.perf_counter_ns()
        toon_encode(payload)
        laps.append((time.perf_counter_ns() - t) / 1e6)
    laps.sort()
    return dict(
        platform=platform.platform(),
        measurement="CPU-only TOON serialization; NOT model inference",
        n=n,
        p50_ms=statistics.median(laps),
        p95_ms=laps[int(0.95 * (n - 1))],
        p99_ms=laps[int(0.99 * (n - 1))],
        max_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        thermal_ok_to_train=gov.check(),
        thermal_reason=gov.reason,
        tensorrt_model_latency=None,
        onnx_model_latency=None,
        torch_model_latency=None,
        tick_to_broker_latency=None,
        promotion_gate=False,
    )


if __name__ == "__main__":
    print(json.dumps(benchmark(), indent=2, sort_keys=True))
