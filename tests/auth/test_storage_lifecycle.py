"""Token lifecycle with a fake Keychain and temporary file canaries only."""

import json
from unittest.mock import Mock

import pytest

from koder_agent.auth import token_storage
from koder_agent.auth.base import OAuthTokens


@pytest.fixture
def keychain_storage(tmp_path, monkeypatch):
    entries = {}
    keychain = Mock()
    keychain.is_available.return_value = True

    def store(service, account, data):
        entries[account] = data
        return True

    keychain.store_checked.side_effect = store
    keychain.retrieve.side_effect = lambda service, account: entries.get(account)
    keychain.delete.side_effect = lambda service, account: entries.pop(account, None) is not None
    keychain.delete_checked.side_effect = keychain.delete.side_effect
    monkeypatch.setattr(token_storage, "SecureStorage", lambda: keychain)
    return token_storage.TokenStorage(tmp_path / "tokens"), keychain


def tokens(access="synthetic-access"):
    return OAuthTokens("google", access, "synthetic-refresh", 4000000000000)


def test_keychain_only_credentials_are_listed_and_revoked(keychain_storage):
    storage, _keychain = keychain_storage
    storage.save(tokens())
    assert storage.list_providers() == ["google"]
    assert storage.delete("google")
    assert storage.load("google") is None


def test_delete_removes_both_copies(keychain_storage):
    storage, _keychain = keychain_storage
    storage.save(tokens())
    (storage.base_dir / "google.json").write_text(json.dumps(tokens("old-file").to_dict()))
    assert storage.delete("google")
    assert storage.load("google") is None


def test_keychain_save_removes_stale_plaintext_copy(keychain_storage):
    storage, _keychain = keychain_storage
    (storage.base_dir / "google.json").write_text(json.dumps(tokens("old-file").to_dict()))
    storage.save(tokens("new-keychain"))
    assert not (storage.base_dir / "google.json").exists()
    assert storage.load("google").access_token == "new-keychain"


def test_legacy_dual_storage_keeps_keychain_precedence(keychain_storage):
    storage, _keychain = keychain_storage
    storage.save(tokens("current-keychain"))
    # Older releases left a plaintext copy behind after moving to Keychain.
    (storage.base_dir / "google.json").write_text(json.dumps(tokens("legacy-file").to_dict()))
    assert storage.load("google").access_token == "current-keychain"


def test_fallback_refresh_does_not_resurrect_stale_keychain_token(keychain_storage):
    storage, keychain = keychain_storage
    storage.save(tokens("old-keychain"))
    keychain.store_checked.side_effect = lambda *_args: False
    storage.save(tokens("new-file"))
    assert storage.load("google").access_token == "new-file"


def test_failed_serialization_preserves_previous_file(keychain_storage):
    storage, keychain = keychain_storage
    keychain.store_checked.side_effect = lambda *_args: False
    storage.save(tokens("previous"))
    invalid = tokens("replacement")
    invalid.extra["not_json"] = object()
    with pytest.raises(TypeError):
        storage.save(invalid)
    assert storage.load("google").access_token == "previous"


@pytest.mark.parametrize("provider", ["../escape", "/absolute", "nested/provider", "a\\b"])
def test_provider_path_is_validated_before_keychain_access(keychain_storage, provider):
    storage, keychain = keychain_storage
    invalid = tokens()
    invalid.provider = provider
    with pytest.raises(ValueError, match="provider"):
        storage.save(invalid)
    keychain.store_checked.assert_not_called()
