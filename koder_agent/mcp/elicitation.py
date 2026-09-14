"""Owned, asynchronous MCP form elicitation for non-sensitive information."""

from __future__ import annotations

import asyncio
import logging
import math
import weakref
from contextlib import suppress
from typing import Any

from jsonschema import Draft202012Validator
from mcp.types import ElicitRequestFormParams, ElicitRequestParams, ElicitResult
from referencing import Registry
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from koder_agent.harness.execution_context import get_execution_cwd
from koder_agent.utils.async_tasks import await_owned_task

logger = logging.getLogger(__name__)

# Keep simultaneous server requests from reading the same terminal. Weak values
# retain neither idle locks nor closed event loops after the owning calls finish.
_input_locks: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()


def _input_lock() -> asyncio.Lock:
    loop = asyncio.get_running_loop()
    reference = _input_locks.get(loop)
    lock = reference() if reference is not None else None
    if lock is None:
        lock = asyncio.Lock()
        _input_locks[loop] = weakref.ref(lock)
    return lock


class ElicitationHandler:
    """Collect schema-valid form answers without blocking the MCP event loop.

    User-configured Elicitation hooks can answer in headless mode. Unsupported
    URL requests are declined without opening a browser. EOF dismisses a form;
    task cancellation joins the active input reader before propagating.
    """

    def __init__(self, console: Console | None = None) -> None:
        self._console = console or Console()

    async def __call__(self, context: Any, params: ElicitRequestParams) -> ElicitResult:
        source = "client"
        if not isinstance(params, ElicitRequestFormParams):
            result = ElicitResult(action="decline")
        else:
            try:
                schema = params.model_dump(by_alias=True)["requestedSchema"]
                Draft202012Validator.check_schema(schema)
                # An explicit empty registry forbids remote $ref retrieval.
                validator = Draft202012Validator(schema, registry=Registry())
                result = await self._try_hook_auto_response(params)
                if result is not None:
                    source = "hook"
                elif not self._console.is_terminal:
                    result = ElicitResult(action="cancel")
                else:
                    source = "user"
                    async with _input_lock():
                        result = await self._handle_form(params)
                if result.action == "accept":
                    validator.validate(result.content if result.content is not None else {})
                else:
                    result = ElicitResult(action=result.action)
            except (EOFError, KeyboardInterrupt):
                result = ElicitResult(action="cancel")
            except Exception:
                # Do not log submitted values or schema error messages containing
                # them. Unsupported/invalid input is not permission to send data.
                logger.warning("MCP form response cancelled: unsupported or invalid form data")
                result = ElicitResult(action="cancel")
        await self._dispatch_result_hook(params, result, source)
        return result

    async def _read_input(self, label: str, *, default: str = "") -> str:
        from prompt_toolkit import PromptSession

        prompt = PromptSession(
            enable_open_in_editor=False,
            enable_system_prompt=False,
            enable_suspend=False,
        )

        async def read() -> str:
            try:
                return await prompt.prompt_async(label, default=default)
            except KeyboardInterrupt as exc:
                # Convert inside the child task, before KeyboardInterrupt can
                # stop the event loop and unrelated MCP work.
                raise EOFError from exc

        task = asyncio.create_task(read(), name="mcp-elicitation-input")
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            task.cancel()
            with suppress(Exception, asyncio.CancelledError):
                await await_owned_task(task)
            raise

    async def _confirm(self, label: str, *, default: bool = True) -> bool:
        while True:
            raw = await self._read_input(f"{label} [y/n]: ", default="y" if default else "n")
            choice = raw.strip().lower()
            if choice in {"y", "yes"}:
                return True
            if choice in {"n", "no"}:
                return False
            if not choice:
                return default
            self._console.print("Enter yes or no.")

    async def _handle_form(self, params: ElicitRequestFormParams) -> ElicitResult:
        schema = params.model_dump(by_alias=True)["requestedSchema"]
        if schema.get("type", "object") != "object":
            raise ValueError("form must describe an object")
        properties = schema.get("properties", {})
        required_fields = schema.get("required", [])
        self._console.print(
            Panel(Text(params.message), title="MCP Server Request", border_style="cyan")
        )
        if not properties:
            return ElicitResult(action="accept" if await self._confirm("Accept?") else "decline")

        content: dict[str, Any] = {}
        for name, field_schema in properties.items():
            value = await self._prompt_field(name, field_schema, required=name in required_fields)
            if value is not None:
                content[name] = value
        if not await self._confirm("Submit this response?"):
            return ElicitResult(action="decline")
        return ElicitResult(action="accept", content=content)

    async def _prompt_field(
        self, name: str, schema: dict[str, Any], *, required: bool = False
    ) -> Any:
        field_type = schema.get("type", "string")
        if field_type not in {"string", "integer", "number", "boolean", "array"}:
            raise ValueError("unsupported form field type")
        if schema.get("format") == "password":
            raise ValueError("sensitive input is not supported in form mode")
        default = schema.get("default")
        enum_values = schema.get("enum")
        description = schema.get("description")
        label = name + (f" ({description})" if description else "") + (" *" if required else "")
        if enum_values:
            self._console.print(Text(label))
            for index, value in enumerate(enum_values, 1):
                self._console.print(Text(f"  {index}. {value}"))
        elif field_type == "boolean":
            return await self._confirm(
                label, default=bool(default) if default is not None else False
            )

        default_text = (
            ", ".join(default)
            if field_type == "array" and isinstance(default, list)
            else str(default)
            if default is not None
            else ""
        )
        while True:
            raw = await self._read_input(
                f"{label}{' (comma-separated)' if field_type == 'array' else ''}: ",
                default=default_text,
            )
            if not raw:
                raw = default_text
            if not raw and not required:
                return None
            try:
                if enum_values:
                    if raw in enum_values:
                        return raw
                    index = int(raw) - 1
                    if 0 <= index < len(enum_values):
                        return enum_values[index]
                    raise ValueError("invalid selection")
                if field_type == "integer":
                    return int(raw)
                if field_type == "number":
                    value = float(raw)
                    if not math.isfinite(value):
                        raise ValueError("number must be finite")
                    return value
                if field_type == "array":
                    return [item.strip() for item in raw.split(",") if item.strip()]
                if raw:
                    return raw
            except (TypeError, ValueError):
                pass
            self._console.print("Enter a valid value for this field.")

    @staticmethod
    async def _dispatch_result_hook(
        params: ElicitRequestParams, result: ElicitResult, source: str
    ) -> None:
        try:
            from koder_agent.harness.hooks.runtime import dispatch_command_hooks_async

            await dispatch_command_hooks_async(
                cwd=get_execution_cwd(),
                event_name="ElicitationResult",
                payload={
                    "event": "ElicitationResult",
                    "message": params.message,
                    "action": result.action,
                    "source": source,
                    "field_names": sorted(result.content) if result.content else [],
                },
            )
        except ImportError:
            pass
        except Exception:
            logger.debug("ElicitationResult hook dispatch failed")

    async def _try_hook_auto_response(self, params: ElicitRequestFormParams) -> ElicitResult | None:
        try:
            from koder_agent.harness.hooks.runtime import dispatch_command_hooks_async

            result = await dispatch_command_hooks_async(
                cwd=get_execution_cwd(),
                event_name="Elicitation",
                payload={
                    "event": "Elicitation",
                    "message": params.message,
                    "mode": "form",
                    "requestedSchema": params.model_dump(by_alias=True)["requestedSchema"],
                },
            )
            if result.elicitation_action in {"accept", "decline", "cancel"}:
                return ElicitResult(
                    action=result.elicitation_action,
                    content=(
                        result.elicitation_content
                        if result.elicitation_action == "accept"
                        else None
                    ),
                )
        except ImportError:
            pass
        except Exception:
            logger.debug("Elicitation hook dispatch failed")
        return None


_handler: ElicitationHandler | None = None


def get_elicitation_handler(console: Console | None = None) -> ElicitationHandler:
    """Return the legacy console handler; active prompt ownership is per call."""
    global _handler
    if _handler is None:
        _handler = ElicitationHandler(console=console)
    return _handler
