"""Snapshot dispatch retains the ordinary runtime's ownership and rules."""

import json
import threading
from contextlib import contextmanager

import pytest

from koder_agent.harness.hooks import runtime as hooks


def _scope(tmp_path, *, source="user_settings", disabled=False, once=False):
    return hooks.HookScope(
        source=source,
        file_path=tmp_path / ".koder/settings.json",
        disable_all_hooks=disabled,
        hooks={
            "ConfigChange": [
                {
                    "matcher": "^user_settings$",
                    "hooks": [{"type": "command", "command": "old", "once": once}],
                }
            ]
        },
    )


def _capture(monkeypatch, scopes):
    owned = []

    @contextmanager
    def load(cwd, **kwargs):
        owned.append(True)
        try:
            yield scopes
        finally:
            owned.pop()

    monkeypatch.setattr(hooks, "load_hook_scopes", load)
    return owned


def _dispatch(tmp_path, snapshot, match="user_settings"):
    return hooks.dispatch_command_hooks(
        cwd=tmp_path,
        event_name="ConfigChange",
        match_value=match,
        payload={"event": "ConfigChange", "source": match, "file_path": str(tmp_path / "config")},
        snapshot=snapshot,
    )


def test_snapshot_detaches_mutable_definitions_and_keeps_resources(tmp_path, monkeypatch):
    scope = _scope(tmp_path)
    owned = _capture(monkeypatch, [scope])
    seen = []

    def run(**kwargs):
        assert owned
        seen.append(kwargs["command"])
        return 0, "", ""

    monkeypatch.setattr(hooks, "_run_command_hook", run)
    with hooks.snapshot_command_hooks(cwd=tmp_path) as snapshot:
        scope.hooks["ConfigChange"][0]["hooks"][0]["command"] = "new"
        assert _dispatch(tmp_path, snapshot).matched_hooks == 1
        assert _dispatch(tmp_path, snapshot, "project_settings").matched_hooks == 0
        with pytest.raises(ValueError, match="different cwd"):
            _dispatch(tmp_path / "other", snapshot)
    assert seen == ["old"]
    assert not owned
    with pytest.raises(ValueError, match="closed"):
        _dispatch(tmp_path, snapshot)


@pytest.mark.parametrize(
    "source, expected",
    [
        ("user_settings", ["managed"]),
        ("policy_settings", []),
    ],
)
def test_snapshot_retains_disable_precedence(tmp_path, monkeypatch, source, expected):
    scopes = [_scope(tmp_path, source=source, disabled=True)]
    managed = _scope(tmp_path, source="policy_settings")
    managed.hooks["ConfigChange"][0]["hooks"][0]["command"] = "managed"
    scopes.append(managed)
    _capture(monkeypatch, scopes)
    seen = []
    monkeypatch.setattr(
        hooks,
        "_run_command_hook",
        lambda **kwargs: (seen.append(kwargs["command"]) or 0, "", ""),
    )
    with hooks.snapshot_command_hooks(cwd=tmp_path) as snapshot:
        scopes.clear()
        _dispatch(tmp_path, snapshot)
    assert seen == expected


def test_snapshot_preserves_dedup_once_and_reentrancy(tmp_path, monkeypatch):
    scope = _scope(tmp_path, once=True)
    _capture(monkeypatch, [scope, scope])
    monkeypatch.setattr(hooks, "_once_fired", set())
    seen = []

    def run(**kwargs):
        seen.append(json.loads(kwargs["payload_text"]))
        assert _dispatch(tmp_path, snapshot).matched_hooks == 0
        return 0, "", ""

    monkeypatch.setattr(hooks, "_run_command_hook", run)
    with hooks.snapshot_command_hooks(cwd=tmp_path) as snapshot:
        assert _dispatch(tmp_path, snapshot).matched_hooks == 1
        assert _dispatch(tmp_path, snapshot).matched_hooks == 0
    assert len(seen) == 1


def test_snapshot_keeps_async_and_nonblocking_exit_semantics(tmp_path, monkeypatch):
    scope = _scope(tmp_path)
    scope.hooks["ConfigChange"][0]["hooks"].append(
        {"type": "command", "command": "background", "async": True}
    )
    _capture(monkeypatch, [scope])
    background = []
    monkeypatch.setattr(hooks, "_run_command_hook", lambda **kwargs: (1, "", "nonblocking"))
    monkeypatch.setattr(hooks, "_run_async_command", lambda **kwargs: background.append(kwargs))
    with hooks.snapshot_command_hooks(cwd=tmp_path) as snapshot:
        result = _dispatch(tmp_path, snapshot)
    assert result.matched_hooks == 2
    assert not result.blocked
    assert [call["command"] for call in background] == ["background"]


def test_snapshot_obeys_foreground_cancellation_and_releases_ownership(tmp_path, monkeypatch):
    owned = _capture(monkeypatch, [_scope(tmp_path)])
    cancellation = threading.Event()
    cancellation.set()
    token = hooks._foreground_cancellation.set(cancellation)
    try:
        with pytest.raises(hooks.HookCommandCancelledError):
            with hooks.snapshot_command_hooks(cwd=tmp_path) as snapshot:
                _dispatch(tmp_path, snapshot)
    finally:
        hooks._foreground_cancellation.reset(token)
    assert not owned
