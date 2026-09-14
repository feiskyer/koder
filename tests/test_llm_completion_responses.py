import asyncio
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from agents import ModelBehaviorError

from koder_agent.config import reset_config_manager
from koder_agent.config.manager import ConfigManager
from koder_agent.harness.memory.budget import ContextPreflightError, estimate_messages_tokens
from koder_agent.utils.client import LLMCompletionResult, llm_completion
from koder_agent.utils.model_info import UnknownModelContextWindowError


def _write_config(tmp_path, data: dict) -> None:
    config_dir = tmp_path / ".koder"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "config.yaml").write_text(yaml.safe_dump(data), encoding="utf-8")


@pytest.fixture(autouse=True)
def isolate_config(monkeypatch, tmp_path):
    config_path = Path(tmp_path) / ".koder" / "config.yaml"
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(ConfigManager, "DEFAULT_CONFIG_PATH", config_path)
    monkeypatch.delenv("KODER_MODEL", raising=False)
    monkeypatch.delenv("KODER_BASE_URL", raising=False)
    monkeypatch.delenv("KODER_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("KODER_SMALL_MODEL", raising=False)
    monkeypatch.delenv("KODER_SMALL_MODEL_CONTEXT_WINDOW", raising=False)
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    monkeypatch.delenv("ANTHROPIC_BASE_URL", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    reset_config_manager()
    yield
    reset_config_manager()


def test_llm_completion_uses_aresponses_for_copilot_codex(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-5.1-codex", "provider": "github_copilot"}},
    )

    calls = {"aresponses": 0, "acompletion": 0}
    captured: dict[str, object] = {}

    async def fake_aresponses(**kwargs):
        calls["aresponses"] += 1
        captured.update(kwargs)
        return {
            "id": "resp_123",
            "output": [
                {
                    "type": "message",
                    "content": [{"type": "output_text", "text": "ok"}],
                }
            ],
        }

    async def fake_acompletion(**kwargs):
        calls["acompletion"] += 1
        raise AssertionError("acompletion should not be called for copilot codex")

    import koder_agent.utils.client as client_mod

    # In some test environments litellm may be stubbed without these attrs.
    monkeypatch.setattr(client_mod.litellm, "aresponses", fake_aresponses, raising=False)
    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)

    text = asyncio.run(
        llm_completion(
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert text == "ok"
    assert calls["aresponses"] == 1
    assert "api_key" not in captured


@pytest.mark.parametrize("status", ["failed", "incomplete", "cancelled"])
@pytest.mark.parametrize("as_mapping", [False, True])
def test_auxiliary_responses_terminal_failure_rejects_partial_text(
    monkeypatch, tmp_path, status, as_mapping
):
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-5.1-codex", "provider": "github_copilot"}},
    )
    payload = {
        "status": status,
        "output_text": "partial-output-canary",
        "output": [
            {
                "type": "message",
                "content": [{"type": "output_text", "text": "partial-output-canary"}],
            }
        ],
        "error": {"message": "private-provider-error-canary"},
    }
    calls = []

    async def fake_aresponses(**kwargs):
        calls.append(kwargs)
        return payload if as_mapping else SimpleNamespace(**payload)

    monkeypatch.setattr("koder_agent.utils.client.litellm.aresponses", fake_aresponses)

    with pytest.raises(ModelBehaviorError, match=status) as caught:
        asyncio.run(
            llm_completion(
                [{"role": "user", "content": "hello"}],
                return_metadata=True,
            )
        )

    assert len(calls) == 1
    assert "partial-output-canary" not in str(caught.value)
    assert "private-provider-error-canary" not in str(caught.value)


@pytest.mark.parametrize("as_mapping", [False, True])
def test_auxiliary_responses_completed_text_is_preserved(monkeypatch, tmp_path, as_mapping):
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-5.1-codex", "provider": "github_copilot"}},
    )
    payload = {
        "status": "completed",
        "output": [
            {"type": "message", "content": [{"type": "output_text", "text": "complete text"}]}
        ],
    }

    async def fake_aresponses(**_kwargs):
        return payload if as_mapping else SimpleNamespace(**payload)

    monkeypatch.setattr("koder_agent.utils.client.litellm.aresponses", fake_aresponses)

    result = asyncio.run(
        llm_completion([{"role": "user", "content": "hello"}], return_metadata=True)
    )

    assert isinstance(result, LLMCompletionResult)
    assert result.text == "complete text"
    assert result.truncation is None


def test_llm_completion_uses_override_provider_credentials_and_base_url(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4.1", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-anthropic")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://anthropic.example.local")

    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    import koder_agent.utils.client as client_mod

    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)

    text = asyncio.run(
        llm_completion(
            messages=[{"role": "user", "content": "hi"}],
            model="anthropic/claude-opus-4-1",
        )
    )
    assert text == "ok"
    assert str(captured["model"]).endswith("anthropic/claude-opus-4-1")
    assert captured["api_key"] == "sk-anthropic"
    assert captured["base_url"] == "https://anthropic.example.local"


def test_llm_completion_prefers_koder_base_url(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-4.1", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-openai")
    monkeypatch.setenv("OPENAI_BASE_URL", "https://openai.example.local")
    monkeypatch.setenv("KODER_BASE_URL", "https://koder.example.local/v1")

    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    import koder_agent.utils.client as client_mod

    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)

    text = asyncio.run(llm_completion(messages=[{"role": "user", "content": "hi"}]))

    assert text == "ok"
    assert captured["base_url"] == "https://koder.example.local/v1"


def test_llm_completion_prefixes_native_openai_model_for_litellm(monkeypatch, tmp_path):
    """Newer OpenAI models (e.g. gpt-5.4) must be prefixed with openai/ for litellm."""
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-5.4", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    import koder_agent.utils.client as client_mod

    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)

    text = asyncio.run(
        llm_completion(
            messages=[{"role": "user", "content": "hi"}],
        )
    )
    assert text == "ok"
    assert captured["model"] == "openai/gpt-5.4"
    assert captured["api_key"] == "sk-test"


def test_llm_completion_truncates_oversized_auxiliary_input(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "tiny-model", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    import koder_agent.utils.client as client_mod

    monkeypatch.setattr(client_mod, "get_context_window_size", lambda _model: 120)
    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)

    result = asyncio.run(
        llm_completion(
            messages=[{"role": "user", "content": "large " * 500}],
            response_reserve=20,
            overflow_policy="truncate",
            return_metadata=True,
        )
    )

    assert isinstance(result, LLMCompletionResult)
    assert result.text == "ok"
    assert result.truncation is not None
    sent_messages = captured["messages"]
    assert estimate_messages_tokens(sent_messages) <= 100
    assert "truncated to fit context" in sent_messages[0]["content"]


def test_llm_completion_skips_when_preserved_system_prompt_cannot_fit(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "tiny-model", "provider": "openai"}},
    )
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    calls = 0

    async def fake_acompletion(**_kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("provider must not be called")

    import koder_agent.utils.client as client_mod

    monkeypatch.setattr(client_mod, "get_context_window_size", lambda _model: 80)
    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)

    with pytest.raises(ContextPreflightError, match="cannot fit") as exc_info:
        asyncio.run(
            llm_completion(
                messages=[
                    {"role": "system", "content": "fixed " * 200},
                    {"role": "user", "content": "hello"},
                ],
                response_reserve=20,
                overflow_policy="truncate",
                return_metadata=True,
            )
        )

    assert calls == 0
    assert "response reserve=20" in str(exc_info.value)


def test_unknown_custom_model_requires_context_window(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "custom-tiny-model", "provider": "custom"}},
    )
    monkeypatch.setenv("KODER_API_KEY", "sk-test")
    calls = 0

    async def fake_acompletion(**_kwargs):
        nonlocal calls
        calls += 1

    import koder_agent.utils.client as client_mod

    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)

    with pytest.raises(UnknownModelContextWindowError, match="model.context_window"):
        asyncio.run(llm_completion([{"role": "user", "content": "hello"}]))

    assert calls == 0


def test_custom_context_window_applies_to_scheduler_and_auxiliary_calls(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {
            "model": {
                "name": "custom-main-model",
                "provider": "custom",
                "context_window": 256,
            }
        },
    )
    monkeypatch.setenv("KODER_API_KEY", "sk-test")
    captured: dict[str, object] = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    import koder_agent.utils.client as client_mod
    from koder_agent.core.scheduler import AgentScheduler

    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)
    text = asyncio.run(
        llm_completion(
            [{"role": "user", "content": "hello"}],
            response_reserve=32,
        )
    )
    assert text == "ok"
    assert captured["max_tokens"] == 32

    scheduler = AgentScheduler.__new__(AgentScheduler)
    scheduler._auto_compact = None
    scheduler._context_model_name = "custom-main-model"
    scheduler._estimate_instruction_context_tokens = lambda: 0
    scheduler._estimate_tool_schema_tokens = lambda: 0
    scheduler._estimate_run_input_tokens = lambda _value: 1
    scheduler._estimate_session_tokens = lambda: None

    estimate = asyncio.run(scheduler._estimate_main_call_preflight("hello", history_tokens=0))
    assert estimate.context_window == 256


def test_small_model_uses_its_own_window(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {
            "model": {
                "name": "custom-main-model",
                "provider": "custom",
                "context_window": 10_000,
                "small_model": "custom-small-model",
                "small_model_context_window": 80,
            }
        },
    )
    monkeypatch.setenv("KODER_API_KEY", "sk-test")
    calls = 0

    async def fake_acompletion(**_kwargs):
        nonlocal calls
        calls += 1

    import koder_agent.utils.client as client_mod

    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)

    with pytest.raises(ContextPreflightError) as exc_info:
        asyncio.run(
            llm_completion(
                [{"role": "user", "content": "too large " * 100}],
                use_small=True,
                response_reserve=20,
            )
        )

    assert exc_info.value.estimate.context_window == 80
    assert calls == 0


def test_llm_completion_forwards_response_reserve_as_max_tokens(monkeypatch, tmp_path):
    _write_config(tmp_path, {"model": {"name": "gpt-4.1", "provider": "openai"}})
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    captured = {}

    async def fake_acompletion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))])

    import koder_agent.utils.client as client_mod

    monkeypatch.setattr(client_mod.litellm, "acompletion", fake_acompletion, raising=False)
    asyncio.run(
        llm_completion(
            [{"role": "user", "content": "hello"}],
            response_reserve=321,
        )
    )
    assert captured["max_tokens"] == 321


def test_llm_responses_forwards_response_reserve_as_max_output_tokens(monkeypatch, tmp_path):
    _write_config(
        tmp_path,
        {"model": {"name": "gpt-5.1-codex", "provider": "github_copilot"}},
    )
    captured = {}

    async def fake_aresponses(**kwargs):
        captured.update(kwargs)
        return {"output_text": "ok"}

    import koder_agent.utils.client as client_mod

    monkeypatch.setattr(client_mod.litellm, "aresponses", fake_aresponses, raising=False)
    asyncio.run(
        llm_completion(
            [{"role": "user", "content": "hello"}],
            response_reserve=654,
        )
    )
    assert captured["max_output_tokens"] == 654
