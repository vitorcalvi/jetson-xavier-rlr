"""Reflective Loop Runtime (RLR).

A persistent closed loop for bounded autonomous decision making.

The runtime owns the loop; OpenDecider is the primary fast decision cortex on
normal structured ticks; LFM2.5-350M is an optional semantic normalizer used
only for unstructured evidence; Qwen is an optional slow reasoning brain that
advises/reframes but never executes; and a deterministic Governor is the final
authority that may veto any side-effecting action while keeping abstain always
allowed.
"""

from .config import Config
from .runtime import ReflectiveLoopRuntime

__all__ = ["Config", "ReflectiveLoopRuntime", "__version__"]

__version__ = "0.1.0"
