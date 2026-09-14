"""Keychain statuses/deadlines, with all native operations replaced."""

import json
import subprocess

import pytest

from koder_agent.auth import keychain_backend, secure_storage, token_storage
from koder_agent.auth.base import OAuthTokens


@pytest.mark.parametrize("method", ["store", "retrieve", "delete"])
def test_keychain_operations_have_finite_timeout(keychain_pipe, method):
    args = ("service", "account", "synthetic") if method == "store" else ("service", "account")
    getattr(keychain_pipe.storage, method)(*args)
    timeout = keychain_pipe.calls[-1][1]["timeout"]
    assert isinstance(timeout, (float, int))
    assert 0 < timeout <= 10


@pytest.mark.parametrize("failure", [1, 36, 128, "timeout"])
def test_ambiguous_keychain_delete_cannot_claim_file_logout(
    keychain_pipe, tmp_path, monkeypatch, failure
):
    storage = token_storage.TokenStorage(tmp_path / "tokens")
    original = OAuthTokens("google", "synthetic", "synthetic-refresh", 4000000000000)
    keychain_pipe.state.status = -25308
    storage.save(original)
    if failure == "timeout":
        keychain_pipe.state.transport_error = subprocess.TimeoutExpired("synthetic", 5)
    else:
        monkeypatch.setattr(
            secure_storage.subprocess,
            "run",
            lambda args, **_: subprocess.CompletedProcess(args, failure, b"", b""),
        )
    with pytest.raises(OSError, match="Keychain"):
        storage.delete("google")
    assert storage.load("google") == original


def test_confirmed_keychain_absence_allows_file_logout(keychain_pipe, tmp_path):
    storage = token_storage.TokenStorage(tmp_path / "tokens")
    keychain_pipe.state.status = -25308
    storage.save(OAuthTokens("google", "synthetic", "synthetic-refresh", 1))
    keychain_pipe.state.status = keychain_backend.ERR_ITEM_NOT_FOUND
    assert storage.delete("google")
    assert storage.load("google") is None


@pytest.mark.parametrize(
    "method, expected", [("store", False), ("retrieve", None), ("delete", False)]
)
def test_timeout_preserves_secure_storage_compatibility(keychain_pipe, method, expected):
    keychain_pipe.state.transport_error = subprocess.TimeoutExpired("synthetic-command-secret", 5)
    args = ("service", "account", "synthetic") if method == "store" else ("service", "account")
    assert getattr(keychain_pipe.storage, method)(*args) is expected


@pytest.mark.parametrize("method", ["retrieve_checked", "delete_checked"])
@pytest.mark.parametrize("failure", [1, 36, 128, "timeout", "unavailable"])
def test_checked_keychain_errors_are_not_absence_or_secret_output(
    keychain_pipe, monkeypatch, method, failure
):
    storage = keychain_pipe.storage
    if failure == "timeout":
        keychain_pipe.state.transport_error = subprocess.TimeoutExpired(
            "synthetic-command-secret", 5
        )
    elif failure == "unavailable":
        monkeypatch.setattr(storage, "is_available", lambda: False)
    else:
        keychain_pipe.state.status = failure
    with pytest.raises(OSError) as error:
        getattr(storage, method)("service", "account")
    assert "synthetic" not in str(error.value)
    assert error.value.__cause__ is None


def test_cas_read_failure_does_not_overwrite_unmarked_legacy_file(keychain_pipe, tmp_path):
    storage = token_storage.TokenStorage(tmp_path / "tokens")
    original = OAuthTokens("google", "synthetic-old", "synthetic-refresh", 1)
    path = storage.base_dir / "google.json"
    path.write_text(json.dumps(original.to_dict()))
    keychain_pipe.state.status = -25308
    assert storage.load("google") == original
    keychain_pipe.calls.clear()
    with pytest.raises(OSError, match="Keychain"):
        storage.save_if_current(
            original, OAuthTokens("google", "synthetic-new", "rotated", 4000000000000)
        )
    assert len(keychain_pipe.calls) == 1
    assert json.loads(keychain_pipe.calls[0][1]["input"])["operation"] == "retrieve"
    assert json.loads(path.read_text()) == original.to_dict()
