"""Synthetic Keychain protocol fixtures; no native framework or account access."""

import json
import subprocess
from types import SimpleNamespace

import pytest

from koder_agent.auth import keychain_backend, secure_storage


@pytest.fixture
def keychain_pipe(monkeypatch):
    entries = {}
    calls = []
    state = SimpleNamespace(status=None, transport_error=None, reply_override=None)

    class Backend:
        def perform(self, operation, service, account, data):
            if state.status is not None:
                return keychain_backend.NativeResult(state.status)
            key = (service.decode("utf-8"), account.decode("utf-8"))
            if operation == "store":
                entries[key] = data
                return keychain_backend.NativeResult(0)
            if key not in entries:
                return keychain_backend.NativeResult(keychain_backend.ERR_ITEM_NOT_FOUND)
            if operation == "retrieve":
                return keychain_backend.NativeResult(0, entries[key])
            entries.pop(key)
            return keychain_backend.NativeResult(0)

    def run(arguments, **options):
        calls.append((arguments, options))
        if state.transport_error is not None:
            raise state.transport_error
        request = json.loads(options["input"])
        response = keychain_backend.handle_request(request, backend_factory=Backend)
        if state.reply_override is not None:
            response = state.reply_override(request)
        return subprocess.CompletedProcess(arguments, 0, json.dumps(response).encode(), b"")

    monkeypatch.setattr(secure_storage.SecureStorage, "is_available", lambda _self: True)
    monkeypatch.setattr(secure_storage.subprocess, "run", run)
    monkeypatch.setattr(
        keychain_backend.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: pytest.fail("Native framework access is forbidden"),
    )
    return SimpleNamespace(
        storage=secure_storage.SecureStorage(), entries=entries, calls=calls, state=state, run=run
    )
