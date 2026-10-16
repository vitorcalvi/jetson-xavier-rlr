"""Measure agreement between the first author's labels and the second annotation.

Prints ``AGREE <n>/<total>`` followed by one ``DISAGREE <name> author=<x> b=<y>``
line per mismatch. Disagreements are data, not failure: this script always
exits 0.
"""

from __future__ import annotations

import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_ROOT = _HERE.parents[1]
for _path in (str(_ROOT), str(_HERE)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

try:  # normal layout: repo root on sys.path
    from bench.heldout.cases_heldout import HELDOUT_CASES
except ImportError:  # run as a plain script from inside bench/heldout/
    from cases_heldout import HELDOUT_CASES  # type: ignore[no-redef]

try:
    from bench.heldout.label_b import label
except ImportError:
    from label_b import label  # type: ignore[no-redef]


def main() -> int:
    total = len(HELDOUT_CASES)
    agree = 0
    lines = []
    for case in HELDOUT_CASES:
        mine = label(case)
        author = getattr(case, "expected_action", None)
        if mine == author:
            agree += 1
        else:
            lines.append(
                f"DISAGREE {case.name} author={author} b={mine}"
            )

    print(f"AGREE {agree}/{total}")
    for line in lines:
        print(line)
    return 0


if __name__ == "__main__":
    sys.exit(main())