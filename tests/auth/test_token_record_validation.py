"""Serialized OAuth records must not create malformed runtime credentials."""

import json
import time
from types import SimpleNamespace

import pytest

from koder_agent.auth import token_storage
from koder_agent.auth.base import OAuthTokens

BAD_FIELDS = [
    ("provider", ""),
    ("access_token", ""),
    ("access_token", 7),
    ("refresh_token", ["not-a-token"]),
    ("expires_at", "tomorrow"),
    ("expires_at", None),
    ("expires_at", True),
    ("expires_at", float("nan")),
    ("expires_at", float("inf")),
    ("models", "not-a-list"),
    ("models", [7]),
    ("extra", ["not-an-object"]),
    ("email", ["not-an-address"]),
    ("models_fetched_at", "yesterday"),
]


def record():
    return {
        "provider": "google",
        "access_token": "synthetic-access",
        "refresh_token": "synthetic-refresh",
        "expires_at": int(time.time() * 1000) + 3600000,
        "models": ["synthetic-model"],
        "extra": {"synthetic": True},
    }


@pytest.mark.parametrize(("field", "value"), BAD_FIELDS)
def test_from_dict_rejects_malformed_credential_fields(field, value):
    data = {**record(), field: value}
    with pytest.raises(TypeError):
        OAuthTokens.from_dict(data)


@pytest.mark.parametrize(("field", "value"), BAD_FIELDS)
def test_file_storage_rejects_malformed_credential_records(tmp_path, monkeypatch, field, value):
    monkeypatch.setattr(
        token_storage, "SecureStorage", lambda: SimpleNamespace(is_available=lambda: False)
    )
    storage = token_storage.TokenStorage(base_dir=tmp_path)
    path = tmp_path / "google.json"
    original = json.dumps({**record(), field: value})
    path.write_text(original, encoding="utf-8")
    assert storage.load("google") is None
    assert not storage.has_valid_token("google")
    assert path.read_text() == original


def test_optional_legacy_nulls_have_explicit_empty_defaults():
    data = {**record(), "refresh_token": None, "models": None, "extra": None}
    tokens = OAuthTokens.from_dict(data)
    assert tokens.refresh_token == ""
    assert tokens.models == []
    assert tokens.extra == {}


def test_expired_but_structurally_valid_records_remain_readable():
    tokens = OAuthTokens.from_dict({**record(), "expires_at": 0})
    assert tokens.is_expired()
    assert tokens.access_token == "synthetic-access"


def test_invalid_utf8_cache_is_rejected_without_rewriting(tmp_path, monkeypatch):
    monkeypatch.setattr(
        token_storage, "SecureStorage", lambda: SimpleNamespace(is_available=lambda: False)
    )
    storage = token_storage.TokenStorage(base_dir=tmp_path)
    path = tmp_path / "google.json"
    path.write_bytes(b"\xffnot-json")
    assert storage.load("google") is None
    assert path.read_bytes() == b"\xffnot-json"
