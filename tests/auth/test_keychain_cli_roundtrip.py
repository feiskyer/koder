"""OAuth JSON and previously encoded records survive the native pipe boundary."""

import json
from dataclasses import replace

import pytest

from koder_agent.auth.base import OAuthTokens
from koder_agent.auth.token_storage import TokenStorage


@pytest.fixture
def keychain_cli(keychain_pipe):
    """Keep legacy test seeds readable while the backing store holds raw bytes."""

    class TextEntries:
        def __getitem__(self, key):
            return keychain_pipe.entries[key].decode("utf-8")

        def __setitem__(self, key, value):
            keychain_pipe.entries[key] = value.encode("utf-8")

    return TextEntries()


@pytest.fixture
def token():
    return OAuthTokens(
        provider="google",
        access_token="synthetic-access",
        refresh_token="synthetic-refresh",
        expires_at=2_000_000_000_000,
        extra={"note": "synthetic\nmultiline", "display": "合成测试 λ"},
        models=["synthetic-model"],
        models_fetched_at=1_999_999_000_000,
    )


def test_new_keychain_save_is_readable_after_restart(tmp_path, keychain_cli, token):
    store = TokenStorage(tmp_path / "tokens")
    store.save(token)
    assert not (store.base_dir / "google.json").exists()
    assert TokenStorage(store.base_dir).load("google") == token


def test_preexisting_multiline_keychain_value_remains_readable(tmp_path, keychain_cli, token):
    raw = json.dumps(token.to_dict(), indent=2)
    keychain_cli[("koder-oauth", "google")] = raw
    store = TokenStorage(tmp_path / "tokens")

    assert store.load("google") == token
    assert keychain_cli[("koder-oauth", "google")] == raw
    assert not (store.base_dir / "google.json").exists()


def test_refresh_can_compare_existing_hex_rendered_credentials(tmp_path, keychain_cli, token):
    keychain_cli[("koder-oauth", "google")] = json.dumps(token.to_dict(), indent=2)
    store = TokenStorage(tmp_path / "tokens")
    refreshed = replace(token, access_token="synthetic-updated")

    assert store.save_if_current(token, refreshed)
    assert store.load("google") == refreshed
    assert not (store.base_dir / "google.json").exists()


def test_existing_printable_keychain_json_is_preserved(tmp_path, keychain_cli, token):
    raw = json.dumps(token.to_dict(), separators=(",", ":"))
    keychain_cli[("koder-oauth", "google")] = raw
    store = TokenStorage(tmp_path / "tokens")

    assert store.load("google") == token
    assert keychain_cli[("koder-oauth", "google")] == raw


@pytest.mark.parametrize("raw", ["", "not-json", "f", "ff", "7b7d", "3132", "5b5d"])
def test_malformed_keychain_data_does_not_authorize_refresh(tmp_path, keychain_cli, token, raw):
    keychain_cli[("koder-oauth", "google")] = raw
    store = TokenStorage(tmp_path / "tokens")
    refreshed = replace(token, access_token="must-not-publish")

    assert store.load("google") is None
    with pytest.raises((ValueError, KeyError, TypeError)):
        store.save_if_current(token, refreshed)
    assert keychain_cli[("koder-oauth", "google")] == raw
    assert not (store.base_dir / "google.json").exists()


def test_hex_keychain_record_must_match_requested_provider(tmp_path, keychain_cli, token):
    raw = json.dumps(replace(token, provider="claude").to_dict(), indent=2)
    keychain_cli[("koder-oauth", "google")] = raw
    store = TokenStorage(tmp_path / "tokens")

    assert store.load("google") is None
    assert not store.save_if_current(token, replace(token, access_token="must-not-publish"))
    assert keychain_cli[("koder-oauth", "google")] == raw


def test_hex_keychain_record_does_not_relax_snapshot_comparison(tmp_path, keychain_cli, token):
    raw = json.dumps(token.to_dict(), indent=2)
    keychain_cli[("koder-oauth", "google")] = raw
    store = TokenStorage(tmp_path / "tokens")

    assert not store.save_if_current(
        replace(token, access_token="stale-input"),
        replace(token, access_token="must-not-publish"),
    )
    assert keychain_cli[("koder-oauth", "google")] == raw
