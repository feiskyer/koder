"""Vendored LiteLLM model cost map loading."""

from __future__ import annotations

import importlib
import json
import os
import sys
import threading
from functools import lru_cache
from importlib.resources import files
from types import ModuleType
from typing import Any

LITELLM_LOCAL_MODEL_COST_MAP_ENV = "LITELLM_LOCAL_MODEL_COST_MAP"
MODEL_COST_MAP_PACKAGE = "koder_agent.data"
MODEL_COST_MAP_FILENAME = "model_prices_and_context_window.json"
_INSTALLED_MAP_ID_ATTR = "_koder_vendored_model_cost_map_id"
_INIT_EVENTS: list[str] = []
_INSTALL_LOCK = threading.RLock()


def _reset_install_lock_after_fork() -> None:
    global _INSTALL_LOCK
    _INSTALL_LOCK = threading.RLock()


if hasattr(os, "register_at_fork"):
    os.register_at_fork(after_in_child=_reset_install_lock_after_fork)


def _record_init_event(message: str) -> None:
    if message not in _INIT_EVENTS:
        _INIT_EVENTS.append(message)


def configure_litellm_local_model_cost_map() -> None:
    """Prevent LiteLLM from fetching its model cost map at import time."""
    if (
        "litellm" in sys.modules
        and os.environ.get(LITELLM_LOCAL_MODEL_COST_MAP_ENV, "").strip().lower() != "true"
    ):
        _record_init_event(
            "warning: litellm was already imported before local cost-map env was configured"
        )
    os.environ[LITELLM_LOCAL_MODEL_COST_MAP_ENV] = "true"
    _record_init_event(f"set {LITELLM_LOCAL_MODEL_COST_MAP_ENV}=true before LiteLLM import")


def get_litellm() -> ModuleType:
    """Initialize the SDK and Koder's price map only at a model-using boundary."""
    configure_litellm_local_model_cost_map()
    module = importlib.import_module("litellm")
    install_vendored_litellm_model_cost_map(module)
    return module


@lru_cache(maxsize=1)
def load_vendored_model_cost_map() -> dict[str, Any]:
    """Load Koder's packaged LiteLLM model cost map."""
    resource = files(MODEL_COST_MAP_PACKAGE).joinpath(MODEL_COST_MAP_FILENAME)
    raw_content = resource.read_text(encoding="utf-8")
    content = json.loads(raw_content)
    if not isinstance(content, dict):
        raise RuntimeError(f"{MODEL_COST_MAP_FILENAME} must contain a JSON object")
    _record_init_event(f"loaded vendored cost map entries={len(content)} resource={resource}")
    return content


def install_vendored_litellm_model_cost_map(litellm_module: Any) -> dict[str, Any]:
    """Install the packaged cost map onto an imported LiteLLM module.

    LiteLLM owns the global ``model_cost`` map, so keep this installation idempotent.
    If LiteLLM replaces the map later, a subsequent call will merge the vendored data
    into the new map while preserving custom entries that are not in Koder's copy.
    """
    # Different model-facing modules can be imported by different threads.
    # Join one publication before returning its shared map to any caller.
    with _INSTALL_LOCK:
        return _install_vendored_model_cost_map(litellm_module)


def _install_vendored_model_cost_map(litellm_module: Any) -> dict[str, Any]:
    active_model_cost = getattr(litellm_module, "model_cost", None)
    installed_map_id = getattr(litellm_module, _INSTALLED_MAP_ID_ATTR, None)
    if isinstance(active_model_cost, dict) and installed_map_id == id(active_model_cost):
        _record_init_event(
            f"vendored cost map already installed in LiteLLM entries={len(active_model_cost)}"
        )
        return active_model_cost

    vendored_model_cost = load_vendored_model_cost_map()
    existing_model_cost = active_model_cost if isinstance(active_model_cost, dict) else {}
    merged_model_cost = dict(existing_model_cost)
    preserved_custom_entries = len(set(merged_model_cost) - set(vendored_model_cost))
    merged_model_cost.update(vendored_model_cost)

    litellm_module.model_cost = merged_model_cost
    setattr(litellm_module, _INSTALLED_MAP_ID_ATTR, id(merged_model_cost))
    _record_init_event(
        "installed vendored cost map into LiteLLM "
        f"entries={len(merged_model_cost)} preserved_custom_entries={preserved_custom_entries}"
    )
    return merged_model_cost


def _get_litellm_model_cost_map_source_info() -> dict[str, Any]:
    try:
        from litellm.litellm_core_utils.get_model_cost_map import get_model_cost_map_source_info
    except (ImportError, AttributeError) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}

    try:
        return get_model_cost_map_source_info()
    except Exception as exc:  # pragma: no cover - defensive around LiteLLM internals
        return {"error": f"{type(exc).__name__}: {exc}"}


def get_litellm_cost_map_debug_lines(litellm_module: Any | None = None) -> list[str]:
    """Render debug details for LiteLLM cost-map initialization."""
    if litellm_module is None:
        litellm_module = get_litellm()

    source_info = _get_litellm_model_cost_map_source_info()

    vendored_entries = len(load_vendored_model_cost_map())
    active_model_cost = getattr(litellm_module, "model_cost", {}) or {}

    lines = [
        "LiteLLM cost data init:",
        f"  local_mode_env: {os.environ.get(LITELLM_LOCAL_MODEL_COST_MAP_ENV, 'unset')}",
        f"  source: {source_info.get('source', 'unknown')}",
        f"  source_url: {source_info.get('url')}",
        f"  env_forced: {source_info.get('is_env_forced')}",
        f"  fallback_reason: {source_info.get('fallback_reason')}",
        f"  vendored_entries: {vendored_entries}",
        f"  active_entries: {len(active_model_cost)}",
        "  events:",
    ]
    if "error" in source_info:
        lines.insert(6, f"  source_info_error: {source_info['error']}")
    lines.extend(f"    - {event}" for event in _INIT_EVENTS)
    return lines
