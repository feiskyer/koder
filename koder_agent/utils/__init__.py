"""Utilities, with public exports loaded only when requested.

Importing a low-level helper must not initialize provider/client dependencies.
In particular, cancellation helpers are also used by the client module.
"""

from importlib import import_module

_EXPORT_MODULES = {
    "AsyncMessageQueue": ".queue",
    "KODER_SYSTEM_PROMPT": ".prompts",
    "default_session_local_ms": ".sessions",
    "get_model_name": ".client",
    "parse_session_dt": ".sessions",
    "picker_arrows": ".sessions",
    "picker_arrows_with_titles": ".sessions",
    "setup_openai_client": ".client",
    "sort_sessions_desc": ".sessions",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str):
    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value
