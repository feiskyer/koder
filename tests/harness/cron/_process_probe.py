"""Isolated subprocess probe for cron ownership and recovery tests."""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("storage", type=Path)
    parser.add_argument("minute")
    parser.add_argument("--mode", choices=("fire", "crash", "hold"), default="fire")
    args = parser.parse_args()
    root = args.storage.parent
    real_home = Path.home()
    repository = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository))
    original_expanduser = os.path.expanduser

    def isolated_expanduser(value):
        text = os.fsdecode(value)
        if text == "~" or text.startswith("~/"):
            result = str(root / text.removeprefix("~").lstrip("/"))
            return os.fsencode(result) if isinstance(value, bytes) else result
        return original_expanduser(value)

    def guard(event, values):
        if event != "open" or not values or not isinstance(values[0], (str, bytes, os.PathLike)):
            return
        path = Path(os.path.abspath(os.fsdecode(values[0])))
        if (
            path.is_relative_to(real_home / ".koder")
            or path.is_relative_to(repository / ".koder")
            or path.name == ".env"
            or path.name.startswith(".env.")
        ):
            raise PermissionError("Private configuration is excluded from this probe")

    sys.addaudithook(guard)
    with (
        patch.object(Path, "home", return_value=root),
        patch.object(os.path, "expanduser", side_effect=isolated_expanduser),
    ):
        from koder_agent.harness.cron.scheduler import CronScheduler
        from koder_agent.harness.cron.storage import CronStorage

        events = []

        def fire(prompt):
            if args.mode == "crash":
                # Deliberately abandon only this synthetic probe process.
                os._exit(23)
            if args.mode == "hold":
                (root / "entered").write_text("claimed", encoding="utf-8")
                deadline = time.monotonic() + 15
                while not (root / "release").exists():
                    if time.monotonic() > deadline:
                        raise TimeoutError("test did not release the probe")
                    time.sleep(0.01)
            events.append(prompt)

        class Clock:
            @staticmethod
            def now():
                return datetime.fromisoformat(args.minute)

        scheduler = CronScheduler(CronStorage(args.storage), on_fire=fire)
        with patch("koder_agent.harness.cron.scheduler.datetime", Clock):
            asyncio.run(scheduler._tick())
        print("CRON_EVENTS:" + json.dumps(events), flush=True)


if __name__ == "__main__":
    main()
