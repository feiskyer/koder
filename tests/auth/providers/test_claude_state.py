"""Manual authorization responses must retain the current request's state."""

from unittest.mock import Mock
from urllib.parse import parse_qs, urlsplit

import pytest

from koder_agent.auth.providers.claude import ClaudeOAuthProvider


@pytest.mark.parametrize("mode", ["max", "console"])
def test_manual_state_matches_the_authorization_request(mode):
    provider = ClaudeOAuthProvider(mode=mode)
    url, verifier = provider.get_authorization_url()
    state = parse_qs(urlsplit(url).query)["state"][0]
    assert state == verifier
    request = provider._build_token_request(f"synthetic-code#{state}", verifier)
    assert request["code"] == "synthetic-code"
    assert request["state"] == state
    assert request["code_verifier"] == verifier


def test_code_without_suffix_retains_the_local_state_compatibility():
    request = ClaudeOAuthProvider()._build_token_request("synthetic-code", "local-verifier")
    assert request["state"] == request["code_verifier"] == "local-verifier"


@pytest.mark.parametrize(
    "code", ["synthetic#other", "synthetic#", "synthetic#local-verifier#extra"]
)
@pytest.mark.asyncio
async def test_mismatched_state_is_rejected_before_creating_http_client(monkeypatch, code):
    client = Mock(side_effect=AssertionError("Must not create an HTTP client"))
    monkeypatch.setattr("aiohttp.ClientSession", client)
    result = await ClaudeOAuthProvider().exchange_code(code, "local-verifier")
    assert not result.success
    assert "state" in result.error.lower()
    client.assert_not_called()


@pytest.mark.parametrize(
    ("code", "verifier"), [("", "verifier"), ("#verifier", "verifier"), ("code", "")]
)
def test_empty_exchange_fields_are_rejected_locally(code, verifier):
    with pytest.raises(ValueError):
        ClaudeOAuthProvider()._build_token_request(code, verifier)
