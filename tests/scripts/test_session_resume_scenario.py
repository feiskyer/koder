"""The untitled-resume terminal contract accepts missing optional metadata."""

import json
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from scripts import tmux_feature_scenarios as scenarios


@pytest.mark.parametrize(
    ("has_metadata", "title", "should_fail"),
    [
        (False, None, False),
        (True, None, False),
        (True, "", False),
        (True, "Unexpected generated title", True),
    ],
    ids=["no-metadata", "null-title", "empty-title", "unwanted-title"],
)
def test_untitled_resume_postconditions(tmp_path, has_metadata, title, should_fail):
    home = tmp_path / "home"
    repo = tmp_path / "repo"
    (home / ".koder").mkdir(parents=True)
    repo.mkdir()
    database = home / ".koder" / "koder.db"
    with closing(sqlite3.connect(database)) as connection:
        connection.execute(
            "CREATE TABLE agent_messages (id INTEGER PRIMARY KEY, session_id TEXT, message_data TEXT)"
        )
        connection.executemany(
            "INSERT INTO agent_messages VALUES (?, ?, ?)",
            [
                (
                    1,
                    "resume-untitled-target",
                    json.dumps({"role": "user", "content": "Untitled user message"}),
                ),
                (
                    2,
                    "resume-untitled-target",
                    json.dumps({"role": "assistant", "content": "Untitled assistant reply"}),
                ),
            ],
        )
        connection.execute(
            "CREATE TABLE session_metadata (session_id TEXT PRIMARY KEY, title TEXT)"
        )
        if has_metadata:
            connection.execute(
                "INSERT INTO session_metadata VALUES (?, ?)", ("resume-untitled-target", title)
            )
        connection.commit()

    manifest = json.loads(
        (Path(__file__).resolve().parents[1] / "e2e" / "tui_feature_scenarios.json").read_text(
            encoding="utf-8"
        )
    )
    case = scenarios.ScenarioRef(
        "features", "resume-untitled-history", manifest["features"]["resume-untitled-history"]
    )
    failures = scenarios._run_post_assertions(case, home=home, repo=repo)
    assert bool(failures) is should_fail, failures
    if should_fail:
        assert len(failures) == 1
        assert "title-remains-optional" in failures[0]
