"""Compatibility wrappers and fixed pipe invocation for secure storage."""

import base64
import json
import subprocess
import sys

from koder_agent.auth import secure_storage
from koder_agent.auth.secure_storage import SecureStorage, get_storage


class TestSecureStorage:
    def test_is_available_returns_true_on_macos(self, monkeypatch):
        monkeypatch.setattr(secure_storage.platform, "system", lambda: "Darwin")
        assert SecureStorage().is_available() is True

    def test_is_available_returns_false_when_security_not_found(self, monkeypatch, tmp_path):
        """A missing native helper is unavailable without probing a keychain."""
        monkeypatch.setattr(secure_storage.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(secure_storage, "_HELPER", tmp_path / "missing-helper.py")
        assert SecureStorage().is_available() is False

    def test_is_available_returns_false_on_non_macos(self, monkeypatch):
        monkeypatch.setattr(secure_storage.platform, "system", lambda: "Linux")
        assert SecureStorage().is_available() is False

    def test_store_calls_subprocess_with_correct_args(self, keychain_pipe):
        assert keychain_pipe.storage.store("koder", "api_key", "synthetic-key")
        arguments, options = keychain_pipe.calls[-1]
        assert arguments == [sys.executable, "-I", str(secure_storage._HELPER)]
        assert not {"koder", "api_key", "synthetic-key"} & set(arguments)
        payload = json.loads(options["input"])
        assert payload["operation"] == "store"
        assert payload["service"] == "koder"
        assert payload["account"] == "api_key"
        assert base64.b64decode(payload["data_b64"]) == b"synthetic-key"

    def test_store_returns_false_on_error(self, keychain_pipe):
        keychain_pipe.state.status = -25308
        assert keychain_pipe.storage.store("koder", "api_key", "synthetic-key") is False

    def test_retrieve_calls_subprocess_with_correct_args(self, keychain_pipe):
        keychain_pipe.entries[("koder", "api_key")] = b"synthetic-key"
        assert keychain_pipe.storage.retrieve("koder", "api_key") == "synthetic-key"
        arguments, options = keychain_pipe.calls[-1]
        assert arguments == [sys.executable, "-I", str(secure_storage._HELPER)]
        payload = json.loads(options["input"])
        assert payload == {
            "version": 1,
            "operation": "retrieve",
            "service": "koder",
            "account": "api_key",
        }

    def test_retrieve_returns_none_on_error(self, keychain_pipe):
        keychain_pipe.state.transport_error = subprocess.CalledProcessError(1, "synthetic")
        assert keychain_pipe.storage.retrieve("koder", "api_key") is None

    def test_retrieve_returns_none_when_not_found(self, keychain_pipe):
        assert keychain_pipe.storage.retrieve("koder", "nonexistent") is None

    def test_delete_calls_subprocess_with_correct_args(self, keychain_pipe):
        keychain_pipe.entries[("koder", "api_key")] = b"synthetic-key"
        assert keychain_pipe.storage.delete("koder", "api_key")
        arguments, options = keychain_pipe.calls[-1]
        assert arguments == [sys.executable, "-I", str(secure_storage._HELPER)]
        assert json.loads(options["input"]) == {
            "version": 1,
            "operation": "delete",
            "service": "koder",
            "account": "api_key",
        }
        assert ("koder", "api_key") not in keychain_pipe.entries

    def test_delete_returns_false_on_error(self, keychain_pipe):
        keychain_pipe.state.status = -25308
        assert keychain_pipe.storage.delete("koder", "api_key") is False


class TestGetStorage:
    def test_get_storage_returns_secure_storage_on_macos(self, monkeypatch):
        monkeypatch.setattr(secure_storage.platform, "system", lambda: "Darwin")
        storage = get_storage()
        assert isinstance(storage, SecureStorage)
        assert storage.is_available()

    def test_get_storage_returns_none_on_linux(self, monkeypatch):
        monkeypatch.setattr(secure_storage.platform, "system", lambda: "Linux")
        assert get_storage() is None

    def test_get_storage_returns_none_on_windows(self, monkeypatch):
        monkeypatch.setattr(secure_storage.platform, "system", lambda: "Windows")
        assert get_storage() is None

    def test_get_storage_returns_none_when_security_unavailable(self, monkeypatch, tmp_path):
        monkeypatch.setattr(secure_storage.platform, "system", lambda: "Darwin")
        monkeypatch.setattr(secure_storage, "_HELPER", tmp_path / "missing-helper.py")
        assert get_storage() is None
