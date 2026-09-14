"""Synthetic task writer used to exercise real process exit and file locks."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from unittest.mock import patch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("mode", choices=("add-crash", "delete-crash", "create"))
    parser.add_argument("--writer", default="worker")
    args = parser.parse_args()
    real_home = Path.home()
    repository = Path(__file__).resolve().parents[3]
    sys.path.insert(0, str(repository))
    original_expanduser = os.path.expanduser

    def isolated_expanduser(value):
        text = os.fsdecode(value)
        if text == "~" or text.startswith("~/"):
            result = str(args.root / text.removeprefix("~").lstrip("/"))
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
        patch.object(Path, "home", return_value=args.root),
        patch.object(os.path, "expanduser", side_effect=isolated_expanduser),
    ):
        from koder_agent.harness.tasks.storage import TaskStorage

        storage = TaskStorage(args.root)
        if args.mode == "create":
            (args.root / f"{args.writer}.ready").write_text("ready", encoding="utf-8")
            deadline = time.monotonic() + 15
            while not (args.root / "start").exists():
                if time.monotonic() > deadline:
                    raise TimeoutError("test did not release the writer")
                time.sleep(0.01)
            ids = [storage.create(f"{args.writer}-{index}").id for index in range(12)]
            print("TASK_IDS:" + json.dumps(ids), flush=True)
            return

        real_write = storage._write_task

        def crash_after_write(task):
            real_write(task)
            # Exit this test-only worker between two durable record operations.
            os._exit(24)

        with patch.object(storage, "_write_task", side_effect=crash_after_write):
            if args.mode == "add-crash":
                storage.add_block(blocker_id="1", blocked_id="2")
            else:
                storage.delete("1")
        raise AssertionError("probe did not reach the crash point")


if __name__ == "__main__":
    main()
