from __future__ import annotations

import importlib
import os
import threading
import types
from concurrent.futures import ThreadPoolExecutor

import koder_agent
from koder_agent import litellm_cost_map as costs
from koder_agent.litellm_cost_map import (
    LITELLM_LOCAL_MODEL_COST_MAP_ENV,
    configure_litellm_local_model_cost_map,
    get_litellm_cost_map_debug_lines,
    install_vendored_litellm_model_cost_map,
    load_vendored_model_cost_map,
)


def test_package_init_forces_litellm_local_cost_map(monkeypatch):
    monkeypatch.setenv(LITELLM_LOCAL_MODEL_COST_MAP_ENV, "false")

    importlib.reload(koder_agent)

    assert koder_agent.__version__
    assert os.environ[LITELLM_LOCAL_MODEL_COST_MAP_ENV] == "true"


def test_configure_litellm_local_model_cost_map(monkeypatch):
    monkeypatch.delenv(LITELLM_LOCAL_MODEL_COST_MAP_ENV, raising=False)

    configure_litellm_local_model_cost_map()

    assert os.environ[LITELLM_LOCAL_MODEL_COST_MAP_ENV] == "true"


def test_load_vendored_model_cost_map_has_known_model_metadata():
    model_cost_map = load_vendored_model_cost_map()

    assert len(model_cost_map) > 1000
    assert "gpt-4o" in model_cost_map
    assert "max_input_tokens" in model_cost_map["gpt-4o"]


def test_install_vendored_litellm_model_cost_map_merges_custom_entries():
    fake_litellm = types.SimpleNamespace(
        model_cost={
            "custom-model": {"max_input_tokens": 123},
            "gpt-4o": {"max_input_tokens": 1},
        }
    )

    installed = install_vendored_litellm_model_cost_map(fake_litellm)

    vendored = load_vendored_model_cost_map()
    assert installed["custom-model"]["max_input_tokens"] == 123
    assert installed["gpt-4o"]["max_input_tokens"] == vendored["gpt-4o"]["max_input_tokens"]
    assert "gpt-4o" in fake_litellm.model_cost


def test_install_vendored_litellm_model_cost_map_is_idempotent():
    fake_litellm = types.SimpleNamespace(model_cost={})

    installed = install_vendored_litellm_model_cost_map(fake_litellm)
    reinstalled = install_vendored_litellm_model_cost_map(fake_litellm)

    assert reinstalled is installed


def test_install_vendored_litellm_model_cost_map_handles_litellm_reinit():
    fake_litellm = types.SimpleNamespace(model_cost={})
    install_vendored_litellm_model_cost_map(fake_litellm)

    fake_litellm.model_cost = {"custom-model": {"input_cost_per_token": 0.1}}
    reinstalled = install_vendored_litellm_model_cost_map(fake_litellm)

    assert reinstalled["custom-model"]["input_cost_per_token"] == 0.1
    assert "gpt-4o" in reinstalled


def test_koder_litellm_entrypoint_installs_vendored_model_cost_map():
    import litellm

    import koder_agent.utils.client  # noqa: F401

    vendored = load_vendored_model_cost_map()
    assert len(litellm.model_cost) >= len(vendored)
    assert (
        litellm.model_cost["gpt-4o"]["max_input_tokens"] == vendored["gpt-4o"]["max_input_tokens"]
    )


def test_litellm_cost_map_debug_lines_include_init_process(monkeypatch):
    monkeypatch.setenv(LITELLM_LOCAL_MODEL_COST_MAP_ENV, "true")
    fake_litellm = types.SimpleNamespace(model_cost={"gpt-4o": {"max_input_tokens": 128000}})

    lines = get_litellm_cost_map_debug_lines(fake_litellm)

    rendered = "\n".join(lines)
    assert "LiteLLM cost data init:" in rendered
    assert "local_mode_env: true" in rendered
    assert "vendored_entries:" in rendered
    assert "active_entries: 1" in rendered
    assert "events:" in rendered


def test_litellm_cost_map_debug_lines_include_source_info_errors(monkeypatch):
    fake_litellm = types.SimpleNamespace(model_cost={})

    def fake_source_info():
        return {"error": "missing LiteLLM source info"}

    monkeypatch.setattr(
        "koder_agent.litellm_cost_map._get_litellm_model_cost_map_source_info",
        fake_source_info,
    )

    rendered = "\n".join(get_litellm_cost_map_debug_lines(fake_litellm))

    assert "source_info_error: missing LiteLLM source info" in rendered


def test_explicit_sdk_bootstrap_configures_before_import_and_handles_reinit(monkeypatch):
    module = types.ModuleType("litellm")
    module.model_cost = {"custom": {"max_input_tokens": 123}}
    real_import = importlib.import_module
    imports = []

    def controlled_import(name, *args, **kwargs):
        if name != "litellm":
            return real_import(name, *args, **kwargs)
        imports.append(os.environ[LITELLM_LOCAL_MODEL_COST_MAP_ENV])
        return module

    monkeypatch.setenv(LITELLM_LOCAL_MODEL_COST_MAP_ENV, "false")
    monkeypatch.setattr(costs.importlib, "import_module", controlled_import)
    assert costs.get_litellm() is module
    first = module.model_cost
    assert costs.get_litellm().model_cost is first
    assert first["custom"]["max_input_tokens"] == 123
    module.model_cost = {"second-custom": {"max_input_tokens": 456}}
    assert costs.get_litellm().model_cost["second-custom"]["max_input_tokens"] == 456
    assert module.model_cost["gpt-4o"] == load_vendored_model_cost_map()["gpt-4o"]
    assert imports == ["true", "true", "true"]


def test_concurrent_installers_share_one_published_map(monkeypatch):
    entered, release = threading.Event(), threading.Event()
    second_started = threading.Event()
    second_load = threading.Event()
    module = types.SimpleNamespace(model_cost={"custom": {"max_input_tokens": 12}})
    calls = []

    def load():
        calls.append("load")
        if len(calls) > 1:
            second_load.set()
        entered.set()
        assert release.wait(3)
        return {"vendored": {"max_input_tokens": 99}}

    def second():
        second_started.set()
        return costs.install_vendored_litellm_model_cost_map(module)

    monkeypatch.setattr(costs, "load_vendored_model_cost_map", load)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(costs.install_vendored_litellm_model_cost_map, module)
        other = None
        try:
            assert entered.wait(3)
            other = pool.submit(second)
            assert second_started.wait(3)
            assert not second_load.wait(0.15)
            assert not other.done()
        finally:
            release.set()
        assert other is not None
        assert first.result(timeout=3) is other.result(timeout=3)
    assert calls == ["load"]
    assert module.model_cost["custom"]["max_input_tokens"] == 12


def test_repeated_bootstrap_does_not_report_a_false_late_configuration_warning(monkeypatch):
    monkeypatch.setattr(costs, "_INIT_EVENTS", [])
    monkeypatch.setenv(LITELLM_LOCAL_MODEL_COST_MAP_ENV, "true")
    monkeypatch.setitem(costs.sys.modules, "litellm", types.ModuleType("litellm"))
    costs.configure_litellm_local_model_cost_map()
    assert not any("warning:" in event for event in costs._INIT_EVENTS)
    monkeypatch.setenv(LITELLM_LOCAL_MODEL_COST_MAP_ENV, "false")
    costs.configure_litellm_local_model_cost_map()
    assert any("warning:" in event for event in costs._INIT_EVENTS)
