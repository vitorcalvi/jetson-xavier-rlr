"""Command line entry point for the standalone RLR process."""
import argparse
import os
import signal
import threading

from .llm import LlamaServer
from .runtime import Config, ReflectiveLoop


def main():
    parser = argparse.ArgumentParser(description="Standalone Reflective Loop Runtime")
    parser.add_argument("--db", default=os.environ.get("RLR_DB", "/models/rlr/brain.db"))
    parser.add_argument("--data-dir", default=os.environ.get("RLR_DATA_DIR", "/models/rlr"))
    parser.add_argument("--server", default=os.environ.get("RLR_LLM_URL", "http://127.0.0.1:8080"))
    parser.add_argument("--model", default=os.environ.get("RLR_MODEL", "Qwen2.5-1.5B"))
    parser.add_argument("--tick", type=float, default=float(os.environ.get("RLR_TICK_SECONDS", "10")))
    parser.add_argument("--once", action="store_true", help="run one observe/decision cycle")
    args = parser.parse_args()
    config = Config(data_dir=args.data_dir, db_path=args.db, tick_seconds=args.tick)
    loop = ReflectiveLoop(LlamaServer(args.server, model=args.model), config=config)
    if args.once:
        try:
            print(loop.tick())
        finally:
            loop.close()
        return
    stop = threading.Event()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    signal.signal(signal.SIGINT, lambda *_: stop.set())
    try:
        loop.run_forever(stop)
    finally:
        loop.close()


if __name__ == "__main__":
    main()
