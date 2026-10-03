"""Fail-closed preflight hashes and append-only run ledger."""
import hashlib
import json
import os
from pathlib import Path


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_hashes(expected, paths):
    actual = {name: sha256_file(paths[name]) for name in sorted(paths)}
    mismatch = {
        key: {"expected": expected.get(key), "actual": actual.get(key)}
        for key in sorted(set(expected) | set(actual))
        if expected.get(key) != actual.get(key)
    }
    if mismatch:
        raise RuntimeError("run_hash_mismatch:" + json.dumps(mismatch, sort_keys=True))
    return actual


def append_ledger(path, record):
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd = os.open(str(target), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
    try:
        os.write(fd, (json.dumps(record, sort_keys=True) + "\n").encode())
    finally:
        os.close(fd)


def preflight_then_append_start(expected, paths, ledger_path, record):
    """Refuse before any START row is written if a frozen hash differs."""
    actual = verify_hashes(expected, paths)
    start = dict(record)
    start["event"] = "START"
    start["verified_hashes"] = actual
    append_ledger(ledger_path, start)
    return start
