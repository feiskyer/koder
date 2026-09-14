"""Whole commands should be pasted atomically into an active terminal reader."""

from types import SimpleNamespace

import pytest

from scripts import tmux_feature_scenarios as scenarios


def test_send_waits_for_reader_and_uses_one_bracketed_paste(monkeypatch):
    calls = []
    readiness = iter(["0", "1"])

    def tmux(*args, **kwargs):
        calls.append(args)
        if args[0] == "display-message":
            return SimpleNamespace(returncode=0, stdout=next(readiness))
        return SimpleNamespace(returncode=0, stdout="")

    monkeypatch.setattr(scenarios, "_tmux", tmux)
    monkeypatch.setattr(scenarios.time, "sleep", lambda _seconds: None)
    command = "!printf 'one\\ntwo'\n# a multiline command"
    scenarios._send("synthetic-session", command)
    reads = [call for call in calls if call[0] == "display-message"]
    assert len(reads) == 2
    assert all("#{bracket_paste_flag}" in call for call in reads)
    writes = [call for call in calls if call[0] == "set-buffer"]
    assert len(writes) == 1 and writes[0][-1] == command
    pastes = [call for call in calls if call[0] == "paste-buffer"]
    assert len(pastes) == 1 and "-p" in pastes[0] and "-d" in pastes[0]
    assert calls[-1] == ("send-keys", "-t", "synthetic-session", "Enter")


def test_send_does_not_write_before_an_input_reader_is_ready(monkeypatch):
    calls = []
    clock = iter([0.0, 100.0])
    monkeypatch.setattr(scenarios.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(
        scenarios,
        "_tmux",
        lambda *args, **_kwargs: calls.append(args) or SimpleNamespace(returncode=0, stdout="0"),
    )
    with pytest.raises(RuntimeError, match="input"):
        scenarios._send("synthetic-session", "/status")
    assert not any(call[0] in {"set-buffer", "paste-buffer", "send-keys"} for call in calls)
