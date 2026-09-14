"""Public storage contracts over a fully synthetic pipe/native boundary."""

import base64
import io
import json
import subprocess
import sys
from types import SimpleNamespace

import pytest

from koder_agent.auth import keychain_backend, secure_storage
from koder_agent.auth.base import OAuthTokens
from koder_agent.auth.token_storage import TokenStorage


@pytest.mark.parametrize(
    "value", ["", "  leading and trailing  ", "\n\tvalue\r\n", "合成 λ 🔑", "a\x00b", "deadbeef"]
)
def test_generic_secret_round_trip_preserves_exact_text(keychain_pipe, value):
    storage = keychain_pipe.storage
    assert storage.store_checked("service", "account", value)
    assert storage.retrieve_checked("service", "account") == value
    assert storage.delete_checked("service", "account")
    assert storage.retrieve_checked("service", "account") is None
    assert not storage.delete_checked("service", "account")


def test_secret_and_inherited_credentials_never_enter_argv_or_child_environment(
    keychain_pipe, monkeypatch
):
    secret = "synthetic-secret-argv-canary"
    monkeypatch.setenv("OPENAI_API_KEY", "synthetic-parent-key")
    monkeypatch.setenv("KODER_API_KEY", "synthetic-parent-koder-key")
    monkeypatch.setenv("PYTHONPATH", "synthetic-untrusted-python-path")
    monkeypatch.setenv("DYLD_INSERT_LIBRARIES", "synthetic-library")
    assert keychain_pipe.storage.store_checked("private-service", "private-account", secret)
    arguments, options = keychain_pipe.calls[-1]
    assert arguments == [sys.executable, "-I", str(secure_storage._HELPER)]
    assert all(value not in arguments for value in (secret, "private-service", "private-account"))
    assert not {"OPENAI_API_KEY", "KODER_API_KEY", "PYTHONPATH", "DYLD_INSERT_LIBRARIES"} & set(
        options["env"]
    )
    request = json.loads(options["input"])
    assert base64.b64decode(request["data_b64"]) == secret.encode()
    assert options["timeout"] == 5
    assert options["stderr"] == subprocess.DEVNULL
    assert options["close_fds"] is True


@pytest.mark.parametrize("status", [-25293, -25291, -25308, 44, -25556])
@pytest.mark.parametrize("operation", ["retrieve_checked", "delete_checked"])
def test_only_full_native_item_not_found_status_means_absence(keychain_pipe, status, operation):
    keychain_pipe.state.status = status
    with pytest.raises(OSError, match="Keychain"):
        getattr(keychain_pipe.storage, operation)("service", "account")


@pytest.mark.parametrize("operation", ["retrieve_checked", "delete_checked"])
def test_process_exit_44_is_not_native_absence(keychain_pipe, monkeypatch, operation):
    monkeypatch.setattr(
        secure_storage.subprocess,
        "run",
        lambda args, **_: subprocess.CompletedProcess(args, 44, b"", b"synthetic-secret"),
    )
    with pytest.raises(OSError) as error:
        getattr(keychain_pipe.storage, operation)("service", "account")
    assert "synthetic" not in str(error.value)
    assert error.value.__cause__ is None


@pytest.mark.parametrize(
    "reply",
    [
        {},
        {"version": True, "operation": "retrieve", "status": -25300},
        {"version": 2, "operation": "retrieve", "status": -25300},
        {"version": 1, "operation": "delete", "status": -25300},
        {"version": 1, "operation": "retrieve", "status": False},
        {"version": 1, "operation": "retrieve", "status": -(2**32)},
        {"version": 1, "operation": "retrieve", "status": 0},
        {"version": 1, "operation": "retrieve", "status": 0, "data_b64": "invalid!"},
        {"version": 1, "operation": "retrieve", "status": -25300, "data_b64": ""},
    ],
)
def test_malformed_or_mismatched_response_is_not_absence(keychain_pipe, reply):
    keychain_pipe.state.reply_override = lambda _request: reply
    with pytest.raises(OSError):
        keychain_pipe.storage.retrieve_checked("service", "account")


@pytest.mark.parametrize("operation", ["store_checked", "delete_checked"])
def test_mutation_timeout_is_explicitly_uncertain_without_secret_output(keychain_pipe, operation):
    keychain_pipe.state.transport_error = subprocess.TimeoutExpired("synthetic-secret-command", 5)
    args = (
        ("service", "account", "synthetic-data")
        if operation == "store_checked"
        else ("service", "account")
    )
    with pytest.raises(secure_storage.KeychainMutationUncertainError) as error:
        getattr(keychain_pipe.storage, operation)(*args)
    assert "synthetic" not in str(error.value)
    assert error.value.__cause__ is None


def test_uncertain_write_preserves_existing_authoritative_fallback(keychain_pipe, tmp_path):
    storage = TokenStorage(tmp_path / "tokens")
    old = OAuthTokens("google", "synthetic-old", "refresh", 4000000000000)
    keychain_pipe.state.status = -25308
    storage.save(old)
    path = storage.base_dir / "google.json"
    before = path.read_bytes()
    keychain_pipe.state.transport_error = subprocess.TimeoutExpired("synthetic-secret", 5)
    with pytest.raises(secure_storage.KeychainMutationUncertainError):
        storage.save(OAuthTokens("google", "synthetic-new", "refresh", 4000000000000))
    assert path.read_bytes() == before
    assert storage.load("google") == old


def test_native_absence_allows_file_logout_but_native_denial_does_not(keychain_pipe, tmp_path):
    storage = TokenStorage(tmp_path / "tokens")
    original = OAuthTokens("google", "synthetic-file", "refresh", 4000000000000)
    keychain_pipe.state.status = -25308
    storage.save(original)
    with pytest.raises(OSError):
        storage.delete("google")
    assert storage.load("google") == original
    keychain_pipe.state.status = keychain_backend.ERR_ITEM_NOT_FOUND
    assert storage.delete("google")
    assert storage.load("google") is None


@pytest.mark.parametrize(
    "payload",
    [
        None,
        [],
        {"version": 1, "operation": "synthetic-secret"},
        {"version": 1, "operation": ["synthetic-secret"]},
        {"version": True, "operation": "retrieve", "service": "s", "account": "a"},
        {"version": 1, "operation": "delete", "service": "", "account": "a"},
        {"version": 1, "operation": "delete", "service": "s", "account": ""},
        {"version": 1, "operation": "retrieve", "service": "s\x00other", "account": "a"},
        {"version": 1, "operation": "retrieve", "service": "s", "account": "a\x00other"},
        {
            "version": 1,
            "operation": "store",
            "service": "s",
            "account": "a",
            "data_b64": "invalid!",
        },
    ],
)
def test_invalid_request_never_constructs_native_backend_or_echoes_secret(payload):
    def forbidden():
        pytest.fail("Native backend must not be constructed")

    result = keychain_backend.handle_request(payload, backend_factory=forbidden)
    assert result["error"] == "invalid_request"
    assert "synthetic-secret" not in json.dumps(result)


def test_lost_mutation_result_does_not_claim_rollback_or_overwrite_fallback(
    keychain_pipe, tmp_path, monkeypatch
):
    storage = TokenStorage(tmp_path / "tokens")
    old = OAuthTokens("google", "synthetic-old", "refresh", 4000000000000)
    new = OAuthTokens("google", "synthetic-new", "refresh", 4000000000000)
    keychain_pipe.state.status = -25308
    storage.save(old)
    path = storage.base_dir / "google.json"
    before = path.read_bytes()
    keychain_pipe.state.status = None

    def committed_but_reply_lost(arguments, **options):
        keychain_pipe.run(arguments, **options)
        raise subprocess.TimeoutExpired("synthetic-secret-command", 5)

    monkeypatch.setattr(secure_storage.subprocess, "run", committed_but_reply_lost)
    with pytest.raises(secure_storage.KeychainMutationUncertainError):
        storage.save(new)
    # A timeout does not establish that the native mutation was undone.
    assert (
        json.loads(keychain_pipe.entries[("koder-oauth", "google")])["access_token"]
        == "synthetic-new"
    )
    assert path.read_bytes() == before
    assert storage.load("google") == old


def test_helper_main_uses_bounded_byte_streams_and_complete_json(monkeypatch):
    value = b" \x00synthetic\n "
    request = {"version": 1, "operation": "retrieve", "service": "s", "account": "a"}
    output = io.BytesIO()
    fake = SimpleNamespace(perform=lambda *_args: keychain_backend.NativeResult(0, value))
    with monkeypatch.context() as patch:
        patch.setattr(keychain_backend, "NativeKeychain", lambda: fake)
        patch.setattr(
            sys, "stdin", SimpleNamespace(buffer=io.BytesIO(json.dumps(request).encode()))
        )
        patch.setattr(sys, "stdout", SimpleNamespace(buffer=output))
        assert keychain_backend.main() == 0
    response = json.loads(output.getvalue())
    assert response["operation"] == "retrieve"
    assert base64.b64decode(response["data_b64"]) == value


@pytest.mark.parametrize(
    "raw", [b"not-json-secret", b"x" * (keychain_backend.MAX_PROTOCOL_BYTES + 1)]
)
def test_helper_main_rejects_bad_input_before_native_access(monkeypatch, raw):
    output = io.BytesIO()
    with monkeypatch.context() as patch:
        patch.setattr(keychain_backend, "NativeKeychain", lambda: pytest.fail("No native access"))
        patch.setattr(sys, "stdin", SimpleNamespace(buffer=io.BytesIO(raw)))
        patch.setattr(sys, "stdout", SimpleNamespace(buffer=output))
        assert keychain_backend.main() == 0
    assert json.loads(output.getvalue())["error"] == "invalid_request"
    assert b"secret" not in output.getvalue()


def test_isolation_guard_blocks_native_helper_and_framework_before_execution():
    from scripts.run_isolated_tests import _audit

    state = {"configurations": []}
    with pytest.raises(PermissionError, match="Native credential"):
        _audit(
            "subprocess.Popen",
            (sys.executable, [sys.executable, "-I", str(secure_storage._HELPER)], None, {}),
            state,
        )
    with pytest.raises(PermissionError, match="Native Keychain"):
        _audit("ctypes.dlopen", (keychain_backend.SECURITY_FRAMEWORK,), state)
