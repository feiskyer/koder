"""OAuth registration must not discard or silently hijack other providers."""

from __future__ import annotations

import importlib
from types import SimpleNamespace

import pytest

from koder_agent.auth import providers
from koder_agent.utils import client


@pytest.fixture
def registry(monkeypatch):
    foreign_handler = object()
    foreign_entry = {
        "provider": "independent",
        "custom_handler": foreign_handler,
        "metadata": {"fixture": True},
    }
    sdk = SimpleNamespace(
        custom_provider_map=[foreign_entry],
        provider_list=["openai", "independent"],
        _custom_providers=["independent"],
        open_ai_chat_completion_models={"gpt-4.1", "gpt-5.1", "codex-fixture"},
    )
    monkeypatch.setattr(providers, "litellm", sdk)
    if hasattr(client, "_oauth_providers_registered"):
        monkeypatch.setattr(client, "_oauth_providers_registered", False)
    return sdk, foreign_entry


def _owned_entries(sdk):
    return {
        item["provider"]: item
        for item in sdk.custom_provider_map
        if item["provider"] in providers.OAUTH_PROVIDER_IDS
    }


def test_registration_preserves_unrelated_handlers_and_metadata(registry):
    sdk, foreign_entry = registry
    providers.register_oauth_providers()
    assert any(item is foreign_entry for item in sdk.custom_provider_map)
    assert foreign_entry["metadata"] == {"fixture": True}
    assert "independent" in sdk.provider_list
    assert "independent" in sdk._custom_providers
    assert set(_owned_entries(sdk)) == set(providers.OAUTH_PROVIDER_IDS)


def test_repeated_registration_is_a_noop_for_current_registry_objects(registry):
    sdk, _ = registry
    providers.register_oauth_providers()
    lists = (sdk.custom_provider_map, sdk.provider_list, sdk._custom_providers)
    entries = list(sdk.custom_provider_map)
    providers.register_oauth_providers()
    assert (sdk.custom_provider_map, sdk.provider_list, sdk._custom_providers) == lists
    assert all(current is old for current, old in zip(sdk.custom_provider_map, entries))
    assert sdk.custom_provider_map is lists[0]
    assert sdk.provider_list is lists[1]
    assert sdk._custom_providers is lists[2]
    for name in providers.OAUTH_PROVIDER_IDS:
        assert sum(item["provider"] == name for item in sdk.custom_provider_map) == 1
        assert sdk.provider_list.count(name) == 1
        assert sdk._custom_providers.count(name) == 1


@pytest.mark.parametrize("name", providers.OAUTH_PROVIDER_IDS)
def test_foreign_same_name_handler_is_not_silently_replaced(registry, name):
    sdk, _ = registry
    collision = {"provider": name, "custom_handler": object()}
    sdk.custom_provider_map.append(collision)
    before = (
        list(sdk.custom_provider_map),
        list(sdk.provider_list),
        list(sdk._custom_providers),
        set(sdk.open_ai_chat_completion_models),
    )
    with pytest.raises(ValueError, match="conflict"):
        providers.register_oauth_providers()
    assert (
        sdk.custom_provider_map,
        sdk.provider_list,
        sdk._custom_providers,
        sdk.open_ai_chat_completion_models,
    ) == before
    assert sdk.custom_provider_map[-1] is collision


def test_owned_handler_refresh_preserves_unrelated_registrations(registry, monkeypatch):
    sdk, foreign_entry = registry
    providers.register_oauth_providers()
    old = _owned_entries(sdk)["google"]["custom_handler"]
    replacement = object()
    monkeypatch.setattr(providers, "_google_oauth_llm", replacement)
    providers.register_oauth_providers()
    assert _owned_entries(sdk)["google"]["custom_handler"] is replacement
    assert replacement is not old
    assert any(item is foreign_entry for item in sdk.custom_provider_map)


@pytest.mark.parametrize(
    "reset_field", ["custom_provider_map", "provider_list", "_custom_providers"]
)
def test_client_reestablishes_registration_after_sdk_state_reset(registry, reset_field):
    sdk, _ = registry
    client._ensure_oauth_providers_registered()
    setattr(sdk, reset_field, [])
    client._ensure_oauth_providers_registered()
    assert set(_owned_entries(sdk)) == set(providers.OAUTH_PROVIDER_IDS)
    assert all(name in sdk.provider_list for name in providers.OAUTH_PROVIDER_IDS)
    assert all(name in sdk._custom_providers for name in providers.OAUTH_PROVIDER_IDS)


def test_in_place_replacement_after_initial_registration_is_a_conflict(registry):
    sdk, _ = registry
    client._ensure_oauth_providers_registered()
    own = _owned_entries(sdk)["chatgpt"]
    outsider = object()
    own["custom_handler"] = outsider
    with pytest.raises(ValueError, match="conflict"):
        client._ensure_oauth_providers_registered()
    assert _owned_entries(sdk)["chatgpt"]["custom_handler"] is outsider


def test_duplicate_owned_entries_are_coalesced_without_touching_external_entries(registry):
    sdk, foreign = registry
    providers.register_oauth_providers()
    own = _owned_entries(sdk)["google"]
    sdk.custom_provider_map.append(dict(own))
    sdk.provider_list.append("google")
    sdk._custom_providers.append("google")
    providers.register_oauth_providers()
    assert sum(item["provider"] == "google" for item in sdk.custom_provider_map) == 1
    assert sdk.provider_list.count("google") == 1
    assert sdk._custom_providers.count("google") == 1
    assert any(item is foreign for item in sdk.custom_provider_map)


@pytest.mark.parametrize(
    "field, value",
    [
        ("custom_provider_map", "invalid"),
        ("custom_provider_map", [{"provider": "incomplete"}]),
        ("provider_list", "invalid"),
        ("_custom_providers", "invalid"),
    ],
)
def test_invalid_registry_does_not_partially_publish(registry, field, value):
    sdk, _ = registry
    setattr(sdk, field, value)
    before = dict(vars(sdk))
    snapshots = {
        key: list(item) if isinstance(item, list) else set(item) if isinstance(item, set) else item
        for key, item in before.items()
    }
    with pytest.raises(ValueError, match="registry"):
        providers.register_oauth_providers()
    for key, original in before.items():
        assert getattr(sdk, key) is original
        assert getattr(sdk, key) == snapshots[key]


def test_real_sdk_custom_handler_lookup_preserves_foreign_provider(monkeypatch):
    """Exercise the SDK's actual custom-handler lookup, without transport or auth."""
    sdk = providers.litellm
    main = importlib.import_module("litellm.main")
    foreign = object()
    monkeypatch.setattr(
        sdk, "custom_provider_map", [{"provider": "independent", "custom_handler": foreign}]
    )
    monkeypatch.setattr(sdk, "provider_list", list(sdk.provider_list))
    monkeypatch.setattr(sdk, "_custom_providers", ["independent"])
    monkeypatch.setattr(
        sdk, "open_ai_chat_completion_models", set(sdk.open_ai_chat_completion_models)
    )
    monkeypatch.setattr(sdk, "_koder_oauth_handlers", {}, raising=False)
    selected = []

    def router(*, custom_llm, **_kwargs):
        selected.append(custom_llm)
        return lambda **_request: "LOCAL_HANDLER_RESULT"

    monkeypatch.setattr(main, "custom_chat_llm_router", router)
    monkeypatch.setattr(main, "_get_encoding", lambda: None)
    providers.register_oauth_providers()
    context = SimpleNamespace(
        acompletion=False,
        api_base=None,
        api_key=None,
        client=None,
        custom_llm_provider="independent",
        custom_prompt_dict={},
        headers={},
        litellm_params={},
        logger_fn=None,
        logging=None,
        messages=[{"role": "user", "content": "synthetic"}],
        model="synthetic",
        model_response=None,
        optional_params={},
        stream=False,
        timeout=1,
    )
    assert main._complete_custom_providers(context) == "LOCAL_HANDLER_RESULT"
    assert selected == [foreign]
