"""Round-3 diagnostic harness switches. Importing this module runs nothing."""
import argparse


def run_discrimination(loop, b2n=False):
    return loop.self_report_discrimination(no_anchor=bool(b2n))


def parser():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--b2n",
        action="store_true",
        default=False,
        help="disable goal anchoring before the withheld self-report",
    )
    return p
