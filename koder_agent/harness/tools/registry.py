"""Tool registry for the harness runtime."""

from __future__ import annotations

import _thread
import importlib
import logging
import sys
import threading
from contextlib import contextmanager
from dataclasses import dataclass, replace
from functools import wraps
from importlib.abc import Loader, MetaPathFinder
from importlib.machinery import ModuleSpec, PathFinder
from importlib.util import module_from_spec, spec_from_loader
from types import ModuleType
from typing import Any, Awaitable, Callable, Iterable, Iterator

from koder_agent.harness.execution_context import capture_tool_directory, get_execution_cwd
from koder_agent.harness.hooks.runtime import dispatch_command_hooks
from koder_agent.harness.permissions.results import PermissionEvaluationResult

ToolInvoke = Callable[[dict[str, Any]], Awaitable[dict[str, Any]]]

logger = logging.getLogger(__name__)
_MODULE_LOAD_LOCK = threading.RLock()
_PROXY_TARGET_KEY = "_tool_registry_proxy_target"
_PROXY_ORIGINAL_CLASS_KEY = "_tool_registry_proxy_original_class"
_PROXY_SELF_TARGET = object()
_PROXY_UNPUBLISHED_TARGET = object()
_PROXY_STAGING_TARGETS = threading.local()
_CANDIDATE_EXECUTION = threading.local()
_MISSING = object()
_MODULE_METADATA_KEYS = ("__spec__", "__loader__", "__package__", "__cached__")
_CANDIDATE_THREAD_START_EVENTS = frozenset(
    {"_thread.start_joinable_thread", "_thread.start_new_thread"}
)
_RETAINED_PROXY_STORAGE_KEYS = frozenset({"__name__", _PROXY_TARGET_KEY})
DEFAULT_REPLACEMENT_HISTORY_LIMIT = 100

CORE_TOOL_GROUPS: dict[str, set[str]] = {
    "code": {"code_intelligence"},
    "file": {"read_file", "write_file", "edit_file"},
    "search": {"glob_search", "grep_search"},
    "web": {"web_fetch", "web_search"},
    "mcp": {"list_mcp_resources", "read_mcp_resource", "tool_search"},
}

GROUP_MODULES = {
    "code": "code_intelligence_ops",
    "file": "file_ops",
    "search": "search_ops",
    "web": "web_ops",
    "mcp": "mcp_ops",
}

NAME_TO_GROUP = {
    tool_name: group_name
    for group_name, tool_names in CORE_TOOL_GROUPS.items()
    for tool_name in tool_names
}


@dataclass(frozen=True)
class ToolSpec:
    """Runtime-facing tool descriptor."""

    name: str
    enabled: bool = True
    invoke: ToolInvoke | None = None
    category: str | None = None
    aliases: tuple[str, ...] = ()
    source: str | None = None


@dataclass(frozen=True, slots=True)
class ToolSpecSummary:
    """Lightweight tool metadata safe to retain in replacement history."""

    name: str
    enabled: bool
    category: str | None
    aliases: tuple[str, ...]
    source: str | None

    @classmethod
    def from_spec(cls, spec: ToolSpec) -> "ToolSpecSummary":
        return cls(
            name=spec.name,
            enabled=spec.enabled,
            category=spec.category,
            aliases=tuple(spec.aliases),
            source=spec.source,
        )


@dataclass(frozen=True, slots=True)
class ToolReplacement:
    """Bounded, metadata-only record of an explicit tool replacement."""

    name: str
    existing_source: str
    incoming_source: str
    existing_spec: ToolSpecSummary
    incoming_spec: ToolSpecSummary


@dataclass(frozen=True)
class _RegistryState:
    """Complete registry state prepared before an atomic publication."""

    tools: dict[str, ToolSpec]
    raw_tools: dict[str, ToolSpec]
    sources: dict[str, str]
    aliases: dict[str, str]
    replacement_history: tuple[ToolReplacement, ...]
    module_owners: dict[str, str]


@dataclass(frozen=True)
class _ModulePublicationSnapshot:
    """Exact public module surfaces to restore after failed initial staging."""

    sys_module: object
    parent_child: object


@dataclass
class _CandidateExecutionState:
    """Thread-local candidate transaction state, including fatal violations."""

    qualified_name: str
    fatal_violation: RuntimeError | None = None


class CandidateThreadStartError(RuntimeError):
    """Raised when a candidate attempts to start work on a child thread."""


class ReentrantToolModuleRegistrationError(RuntimeError):
    """Raised when candidate collection tries to publish another tool module."""


class UnpublishedToolModuleAccessError(RuntimeError):
    """Raised when another thread reaches a first-generation staging proxy."""


def _active_candidate_name() -> str | None:
    state = getattr(_CANDIDATE_EXECUTION, "state", None)
    return state.qualified_name if state is not None else None


def _mark_candidate_fatal(error: RuntimeError) -> RuntimeError:
    state = getattr(_CANDIDATE_EXECUTION, "state", None)
    if state is not None and state.fatal_violation is None:
        state.fatal_violation = error
    return error


@contextmanager
def _candidate_execution(qualified_name: str) -> Iterator[None]:
    previous = getattr(_CANDIDATE_EXECUTION, "state", _MISSING)
    if previous is not _MISSING:
        raise _mark_candidate_fatal(
            ReentrantToolModuleRegistrationError(
                f"Tool module candidate {qualified_name!r} cannot run inside active "
                f"candidate {previous.qualified_name!r}"
            )
        )
    state = _CandidateExecutionState(qualified_name=qualified_name)
    _CANDIDATE_EXECUTION.state = state
    try:
        with _legacy_candidate_thread_guard():
            yield
        if state.fatal_violation is not None:
            raise state.fatal_violation
    finally:
        delattr(_CANDIDATE_EXECUTION, "state")


def _reject_candidate_thread_start(event: str, _args: tuple[Any, ...]) -> None:
    qualified_name = _active_candidate_name()
    if qualified_name is None or event not in _CANDIDATE_THREAD_START_EVENTS:
        return
    raise CandidateThreadStartError(
        f"Tool module candidate {qualified_name!r} cannot start child threads "
        "while import and tool collection are staged"
    )


@contextmanager
def _legacy_candidate_thread_guard() -> Iterator[None]:
    """Cover standard Python launchers before CPython's thread-start audit event.

    The module-load lock serializes these temporary replacements. The policy is
    thread-local, so unrelated callers can still start threads. This does not
    cover retained native function references or arbitrary extension code and
    is not an OS sandbox.
    """
    if sys.version_info >= (3, 12):
        yield
        return

    def checked_start(original):
        @wraps(original)
        def start(*args, **kwargs):
            _reject_candidate_thread_start("_thread.start_new_thread", ())
            return original(*args, **kwargs)

        return start

    replacements = []
    try:
        for module, name in (
            (_thread, "start_new_thread"),
            (_thread, "start_new"),
            (threading, "_start_new_thread"),
        ):
            original = getattr(module, name)
            guarded = checked_start(original)
            replacements.append((module, name, original, guarded))
            setattr(module, name, guarded)
        yield
    finally:
        for module, name, original, guarded in reversed(replacements):
            if getattr(module, name) is guarded:
                setattr(module, name, original)


sys.addaudithook(_reject_candidate_thread_start)


class _PublishedModuleProxy(ModuleType):
    """Stable module identity with a thread-local target during staging."""

    def __getattribute__(self, name: str) -> Any:
        target = _effective_proxy_target(self)
        if target is _PROXY_SELF_TARGET:
            return ModuleType.__getattribute__(self, name)
        if target is _PROXY_UNPUBLISHED_TARGET:
            raise _unpublished_proxy_access(self, f"read attribute {name!r}")
        if name == "__dict__":
            return target.__dict__
        return getattr(target, name)

    def __setattr__(self, name: str, value: Any) -> None:
        target = _effective_proxy_target(self)
        if target is _PROXY_SELF_TARGET:
            ModuleType.__setattr__(self, name, value)
            return
        if target is _PROXY_UNPUBLISHED_TARGET:
            raise _unpublished_proxy_access(self, f"set attribute {name!r}")
        setattr(target, name, value)

    def __delattr__(self, name: str) -> None:
        target = _effective_proxy_target(self)
        if target is _PROXY_SELF_TARGET:
            ModuleType.__delattr__(self, name)
            return
        if target is _PROXY_UNPUBLISHED_TARGET:
            raise _unpublished_proxy_access(self, f"delete attribute {name!r}")
        delattr(target, name)

    def __dir__(self) -> list[str]:
        target = _effective_proxy_target(self)
        if target is _PROXY_SELF_TARGET:
            return ModuleType.__dir__(self)
        if target is _PROXY_UNPUBLISHED_TARGET:
            raise _unpublished_proxy_access(self, "list attributes")
        return dir(target)


def _proxy_storage(proxy: _PublishedModuleProxy) -> dict[str, Any]:
    return ModuleType.__getattribute__(proxy, "__dict__")


def _unpublished_proxy_access(
    proxy: _PublishedModuleProxy,
    action: str,
) -> UnpublishedToolModuleAccessError:
    qualified_name = ModuleType.__getattribute__(proxy, "__name__")
    return UnpublishedToolModuleAccessError(
        f"Tool module {qualified_name!r} is not published yet; cannot {action}"
    )


def _staging_targets() -> dict[_PublishedModuleProxy, ModuleType]:
    targets = getattr(_PROXY_STAGING_TARGETS, "targets", None)
    if targets is None:
        targets = {}
        _PROXY_STAGING_TARGETS.targets = targets
    return targets


def _effective_proxy_target(proxy: _PublishedModuleProxy) -> ModuleType | object:
    targets = getattr(_PROXY_STAGING_TARGETS, "targets", None)
    if targets is not None:
        staged_target = targets.get(proxy)
        if staged_target is not None:
            return staged_target
    return _proxy_storage(proxy).get(_PROXY_TARGET_KEY, _PROXY_SELF_TARGET)


def _published_proxy_target(proxy: _PublishedModuleProxy) -> ModuleType | object:
    return _proxy_storage(proxy).get(_PROXY_TARGET_KEY, _PROXY_SELF_TARGET)


def _published_proxy_namespace(proxy: _PublishedModuleProxy) -> dict[str, Any]:
    target = _published_proxy_target(proxy)
    if target is _PROXY_SELF_TARGET:
        return _proxy_storage(proxy)
    if target is _PROXY_UNPUBLISHED_TARGET:
        raise _unpublished_proxy_access(proxy, "access its namespace")
    return target.__dict__


def _set_proxy_target(
    proxy: _PublishedModuleProxy,
    target: ModuleType | object,
) -> None:
    _proxy_storage(proxy)[_PROXY_TARGET_KEY] = target


def _new_module_proxy(
    qualified_name: str,
    target: ModuleType | object,
) -> _PublishedModuleProxy:
    proxy = _PublishedModuleProxy(qualified_name)
    _set_proxy_target(proxy, target)
    return proxy


def _module_publication_snapshot(
    qualified_name: str,
    parent_module: ModuleType,
    child_name: str,
) -> _ModulePublicationSnapshot:
    return _ModulePublicationSnapshot(
        sys_module=sys.modules.get(qualified_name, _MISSING),
        parent_child=parent_module.__dict__.get(child_name, _MISSING),
    )


def _restore_module_publication(
    qualified_name: str,
    parent_module: ModuleType,
    child_name: str,
    snapshot: _ModulePublicationSnapshot,
) -> None:
    if snapshot.parent_child is _MISSING:
        parent_module.__dict__.pop(child_name, None)
    else:
        parent_module.__dict__[child_name] = snapshot.parent_child

    if snapshot.sys_module is _MISSING:
        sys.modules.pop(qualified_name, None)
    else:
        sys.modules[qualified_name] = snapshot.sys_module


def _assert_module_proxy_publication(
    qualified_name: str,
    parent_module: ModuleType,
    child_name: str,
    proxy: _PublishedModuleProxy,
) -> None:
    if sys.modules.get(qualified_name) is not proxy:
        raise RuntimeError(f"sys.modules publication for {qualified_name!r} changed during staging")
    if getattr(parent_module, child_name) is not proxy:
        raise RuntimeError(
            f"Parent package lookup for {qualified_name!r} did not resolve "
            "the sys.modules publication"
        )


def _publish_module_proxy(
    qualified_name: str,
    parent_module: ModuleType,
    child_name: str,
    proxy: _PublishedModuleProxy,
) -> None:
    """Publish through one sys.modules write plus the package's shared lookup."""
    snapshot = _module_publication_snapshot(
        qualified_name,
        parent_module,
        child_name,
    )
    sys.modules[qualified_name] = proxy
    try:
        _assert_module_proxy_publication(
            qualified_name,
            parent_module,
            child_name,
            proxy,
        )
    except BaseException:
        _restore_module_publication(
            qualified_name,
            parent_module,
            child_name,
            snapshot,
        )
        raise


def _adopt_module_proxy(module: ModuleType) -> _PublishedModuleProxy:
    if isinstance(module, _PublishedModuleProxy):
        return module
    original_class = type(module)
    try:
        module.__class__ = _PublishedModuleProxy
    except TypeError as exc:
        raise TypeError(
            f"Cannot preserve stable identity for module {module.__name__!r} "
            f"with custom class {original_class!r}"
        ) from exc
    proxy = module
    storage = _proxy_storage(proxy)
    storage[_PROXY_ORIGINAL_CLASS_KEY] = original_class
    storage[_PROXY_TARGET_KEY] = _PROXY_SELF_TARGET
    return proxy


def _undo_module_proxy_adoption(proxy: _PublishedModuleProxy) -> None:
    storage = _proxy_storage(proxy)
    if storage.get(_PROXY_TARGET_KEY) is not _PROXY_SELF_TARGET:
        return
    original_class = storage.pop(_PROXY_ORIGINAL_CLASS_KEY, ModuleType)
    storage.pop(_PROXY_TARGET_KEY, None)
    ModuleType.__setattr__(proxy, "__class__", original_class)


def _discard_obsolete_adopted_namespace(proxy: _PublishedModuleProxy) -> None:
    """Drop the pre-adoption generation while retaining stable proxy identity."""
    storage = _proxy_storage(proxy)
    for key in tuple(storage):
        if key not in _RETAINED_PROXY_STORAGE_KEYS:
            storage.pop(key, None)


@contextmanager
def _stage_proxy_target(
    proxy: _PublishedModuleProxy | None,
    target: ModuleType,
) -> Iterator[None]:
    if proxy is None:
        yield
        return

    targets = _staging_targets()
    previous = targets.get(proxy, _MISSING)
    targets[proxy] = target
    try:
        yield
    finally:
        if previous is _MISSING:
            targets.pop(proxy, None)
        else:
            targets[proxy] = previous
        if not targets:
            delattr(_PROXY_STAGING_TARGETS, "targets")


class UnmanagedToolModuleReloadError(RuntimeError):
    """Raised when importlib.reload bypasses a registry-managed transaction."""


class _RejectManagedReloadLoader(Loader):
    def __init__(
        self,
        qualified_name: str,
        namespace: dict[str, Any],
        metadata: dict[str, Any],
    ):
        self._qualified_name = qualified_name
        self._namespace = namespace
        self._metadata = metadata

    def exec_module(self, _module: ModuleType) -> None:
        for key, value in self._metadata.items():
            if value is _MISSING:
                self._namespace.pop(key, None)
            else:
                self._namespace[key] = value
        raise UnmanagedToolModuleReloadError(
            f"Tool module {self._qualified_name!r} is managed by ToolRegistry; "
            "use registry.register_module(..., replace=True) instead of importlib.reload()"
        )


class _ManagedModuleReloadFinder(MetaPathFinder):
    def find_spec(
        self,
        fullname: str,
        path: Any = None,
        target: ModuleType | None = None,
    ) -> ModuleSpec | None:
        del path
        if not isinstance(target, _PublishedModuleProxy):
            return None
        namespace = _published_proxy_namespace(target)
        metadata = {key: namespace.get(key, _MISSING) for key in _MODULE_METADATA_KEYS}
        loader = _RejectManagedReloadLoader(fullname, namespace, metadata)
        return spec_from_loader(fullname, loader, origin="tool-registry-managed")


_MANAGED_MODULE_RELOAD_FINDER = _ManagedModuleReloadFinder()
if _MANAGED_MODULE_RELOAD_FINDER not in sys.meta_path:
    sys.meta_path.insert(0, _MANAGED_MODULE_RELOAD_FINDER)


class DuplicateToolError(ValueError):
    """Raised when a tool name or alias is registered more than once."""

    def __init__(
        self,
        key: str,
        *,
        existing_spec: ToolSpec,
        existing_source: str,
        incoming_spec: ToolSpec,
        incoming_source: str,
    ):
        self.key = key
        self.existing_spec = existing_spec
        self.existing_source = existing_source
        self.incoming_spec = incoming_spec
        self.incoming_source = incoming_source
        super().__init__(
            f"Tool name or alias {key!r} is already registered; "
            f"existing source={existing_source!r}, spec={existing_spec!r}; "
            f"incoming source={incoming_source!r}, spec={incoming_spec!r}. "
            "Use replace=True only to replace the same canonical tool registration."
        )


def _string_looks_like_error(content: Any, error_markers: tuple[str, ...]) -> bool:
    if not isinstance(content, str):
        return False
    return any(content.startswith(marker) for marker in error_markers)


def build_tool_result(
    name: str,
    content: Any,
    *,
    error_markers: tuple[str, ...] = (),
    status: str | None = None,
) -> dict[str, Any]:
    """Build a stable result envelope for runtime-owned tool transcripts."""
    resolved_status = status
    if resolved_status is None:
        resolved_status = "error" if _string_looks_like_error(content, error_markers) else "success"
    return {
        "tool": name,
        "status": resolved_status,
        "content": content,
    }


class ToolRegistry:
    """Registry of tool descriptors."""

    def __init__(
        self,
        tools: dict[str, ToolSpec] | None = None,
        permission_service=None,
        *,
        replacement_history_limit: int = DEFAULT_REPLACEMENT_HISTORY_LIMIT,
    ):
        if replacement_history_limit < 0:
            raise ValueError("replacement_history_limit must be non-negative")
        self._tools: dict[str, ToolSpec] = {}
        self._raw_tools: dict[str, ToolSpec] = {}
        self._sources: dict[str, str] = {}
        self._aliases: dict[str, str] = {}
        self._replacement_history: tuple[ToolReplacement, ...] = ()
        self._replacement_history_limit = replacement_history_limit
        self._module_owners: dict[str, str] = {}
        self._permission_service = permission_service
        self._lock = threading.RLock()
        if tools:
            self.register_many(tools.values(), source="initial registry")

    @classmethod
    def empty(
        cls,
        permission_service=None,
        *,
        replacement_history_limit: int = DEFAULT_REPLACEMENT_HISTORY_LIMIT,
    ) -> "ToolRegistry":
        return cls(
            {},
            permission_service=permission_service,
            replacement_history_limit=replacement_history_limit,
        )

    @classmethod
    def with_permission_service(
        cls,
        permission_service,
        *,
        replacement_history_limit: int = DEFAULT_REPLACEMENT_HISTORY_LIMIT,
    ) -> "ToolRegistry":
        return cls.empty(
            permission_service=permission_service,
            replacement_history_limit=replacement_history_limit,
        )

    @classmethod
    def with_core_tools(
        cls,
        *,
        categories: set[str] | None = None,
        permission_service=None,
        replacement_history_limit: int = DEFAULT_REPLACEMENT_HISTORY_LIMIT,
    ) -> "ToolRegistry":
        """Create a registry with core tool modules registered."""
        registry = cls.empty(
            permission_service=permission_service,
            replacement_history_limit=replacement_history_limit,
        )
        target_categories = categories or set(GROUP_MODULES.keys())
        for category in sorted(target_categories):
            module_name = GROUP_MODULES.get(category)
            if module_name:
                registry.register_module(module_name)
        return registry

    @staticmethod
    def _resolve_source(spec: ToolSpec, source: str | None) -> str:
        if source:
            return source
        if spec.source:
            return spec.source
        if spec.invoke is not None:
            module = getattr(spec.invoke, "__module__", None)
            qualname = getattr(spec.invoke, "__qualname__", None)
            if module and qualname:
                return f"{module}.{qualname}"
        return "<unspecified>"

    @staticmethod
    def _raw_invoke(invoke: ToolInvoke | None) -> ToolInvoke | None:
        if invoke is None:
            return None
        return getattr(invoke, "__tool_registry_raw_invoke__", invoke)

    def _normalize_spec(self, spec: ToolSpec, source: str | None) -> ToolSpec:
        resolved_source = self._resolve_source(spec, source)
        return replace(
            spec,
            invoke=self._raw_invoke(spec.invoke),
            aliases=tuple(spec.aliases),
            source=resolved_source,
        )

    def _wrap_spec(self, spec: ToolSpec) -> ToolSpec:
        if spec.invoke is not None and self._permission_service is not None:
            raw_invoke = spec.invoke

            async def guarded_invoke(
                arguments: dict[str, Any], *, _name=spec.name, _raw=raw_invoke
            ):
                decision: PermissionEvaluationResult = (
                    await self._permission_service.evaluate_tool_call_async(
                        _name,
                        arguments,
                    )
                )
                if decision.requires_approval:
                    hook_result = dispatch_command_hooks(
                        cwd=get_execution_cwd(),
                        event_name="PermissionRequest",
                        match_value=_name,
                        payload={
                            "event": "PermissionRequest",
                            "tool_name": _name,
                            "tool_input": arguments,
                            "reason": decision.reason,
                        },
                    )
                    dispatch_command_hooks(
                        cwd=get_execution_cwd(),
                        event_name="Notification",
                        match_value="permission_prompt",
                        payload={
                            "event": "Notification",
                            "notification_type": "permission_prompt",
                            "tool_name": _name,
                            "reason": decision.reason,
                        },
                    )
                    if hook_result.permission_request_result:
                        behavior = hook_result.permission_request_result.get("behavior")
                        if behavior == "allow":
                            updated = hook_result.permission_request_result.get("updatedInput")
                            next_arguments = updated if isinstance(updated, dict) else arguments
                            result = await _raw(next_arguments)
                            updates = hook_result.permission_request_result.get(
                                "updatedPermissions"
                            )
                            if isinstance(updates, list):
                                result["permission_updates"] = updates
                            return result
                        if behavior == "deny":
                            return {
                                "tool": _name,
                                "status": "error",
                                "content": hook_result.permission_request_result.get("message")
                                or decision.reason,
                                "permission": decision.to_dict(),
                            }
                    return {
                        "tool": _name,
                        "status": "approval_required",
                        "content": decision.reason,
                        "permission": decision.to_dict(),
                    }
                if not decision.allowed:
                    denied_result = dispatch_command_hooks(
                        cwd=get_execution_cwd(),
                        event_name="PermissionDenied",
                        match_value=_name,
                        payload={
                            "event": "PermissionDenied",
                            "tool_name": _name,
                            "tool_input": arguments,
                            "reason": decision.reason,
                        },
                    )
                    denied_response: dict[str, Any] = {
                        "tool": _name,
                        "status": "error",
                        "content": decision.reason,
                        "permission": decision.to_dict(),
                    }
                    if denied_result.retry:
                        denied_response["retry"] = True
                    return denied_response
                try:
                    result = await _raw(arguments)
                except Exception as exc:
                    failure_result = dispatch_command_hooks(
                        cwd=get_execution_cwd(),
                        event_name="PostToolUseFailure",
                        match_value=_name,
                        payload={
                            "event": "PostToolUseFailure",
                            "tool_name": _name,
                            "tool_input": arguments,
                            "error": str(exc),
                        },
                    )
                    if failure_result.blocked:
                        return {
                            "tool": _name,
                            "status": "error",
                            "content": failure_result.block_reason or str(exc),
                        }
                    raise
                if isinstance(result, dict) and result.get("status") == "error":
                    failure_result = dispatch_command_hooks(
                        cwd=get_execution_cwd(),
                        event_name="PostToolUseFailure",
                        match_value=_name,
                        payload={
                            "event": "PostToolUseFailure",
                            "tool_name": _name,
                            "tool_input": arguments,
                            "error": result.get("content"),
                        },
                    )
                    if failure_result.blocked:
                        result = {
                            **result,
                            "content": failure_result.block_reason or result.get("content"),
                        }
                else:
                    post_result = dispatch_command_hooks(
                        cwd=get_execution_cwd(),
                        event_name="PostToolUse",
                        match_value=_name,
                        payload={
                            "event": "PostToolUse",
                            "tool_name": _name,
                            "tool_input": arguments,
                            "result": result.get("content") if isinstance(result, dict) else result,
                        },
                    )
                    if post_result.blocked:
                        return {
                            "tool": _name,
                            "status": "error",
                            "content": post_result.block_reason or "Blocked by PostToolUse hook",
                        }
                return result

            async def scoped_invoke(arguments: dict[str, Any]):
                with capture_tool_directory():
                    return await guarded_invoke(arguments)

            scoped_invoke.__tool_registry_raw_invoke__ = raw_invoke
            return replace(spec, invoke=scoped_invoke)

        return spec

    @staticmethod
    def _claims(spec: ToolSpec) -> tuple[str, ...]:
        return (spec.name, *spec.aliases)

    @staticmethod
    def _raise_duplicate(
        key: str,
        *,
        existing_spec: ToolSpec,
        incoming_spec: ToolSpec,
    ) -> None:
        raise DuplicateToolError(
            key,
            existing_spec=existing_spec,
            existing_source=existing_spec.source or "<unspecified>",
            incoming_spec=incoming_spec,
            incoming_source=incoming_spec.source or "<unspecified>",
        )

    def _validate_batch(
        self,
        specs: list[ToolSpec],
        *,
        replace_existing: bool,
        existing_tools: dict[str, ToolSpec] | None = None,
    ) -> None:
        current_tools = self._raw_tools if existing_tools is None else existing_tools
        incoming_names: dict[str, ToolSpec] = {}
        for spec in specs:
            previous = incoming_names.get(spec.name)
            if previous is not None:
                self._raise_duplicate(
                    spec.name,
                    existing_spec=previous,
                    incoming_spec=spec,
                )
            incoming_names[spec.name] = spec

        replaced_names = (
            {spec.name for spec in specs if spec.name in current_tools}
            if replace_existing
            else set()
        )
        existing_claims: dict[str, ToolSpec] = {}
        for name, existing_spec in current_tools.items():
            if name in replaced_names:
                continue
            for key in self._claims(existing_spec):
                existing_claims[key] = existing_spec

        incoming_claims: dict[str, ToolSpec] = {}
        for spec in specs:
            for key in self._claims(spec):
                existing_spec = existing_claims.get(key)
                if existing_spec is not None:
                    self._raise_duplicate(
                        key,
                        existing_spec=existing_spec,
                        incoming_spec=spec,
                    )
                previous = incoming_claims.get(key)
                if previous is not None:
                    self._raise_duplicate(
                        key,
                        existing_spec=previous,
                        incoming_spec=spec,
                    )
                incoming_claims[key] = spec

    def _build_alias_map(self, specs: Iterable[ToolSpec]) -> dict[str, str]:
        claims: dict[str, ToolSpec] = {}
        aliases: dict[str, str] = {}
        for spec in specs:
            for key in self._claims(spec):
                existing_spec = claims.get(key)
                if existing_spec is not None:
                    self._raise_duplicate(
                        key,
                        existing_spec=existing_spec,
                        incoming_spec=spec,
                    )
                claims[key] = spec
            for alias in spec.aliases:
                aliases[alias] = spec.name
        return aliases

    @staticmethod
    def _replacement_record(
        existing_spec: ToolSpec,
        incoming_spec: ToolSpec,
    ) -> ToolReplacement:
        return ToolReplacement(
            name=incoming_spec.name,
            existing_source=existing_spec.source or "<unspecified>",
            incoming_source=incoming_spec.source or "<unspecified>",
            existing_spec=ToolSpecSummary.from_spec(existing_spec),
            incoming_spec=ToolSpecSummary.from_spec(incoming_spec),
        )

    def _extend_replacement_history(
        self,
        replacements: Iterable[ToolReplacement],
    ) -> tuple[ToolReplacement, ...]:
        combined = (*self._replacement_history, *replacements)
        if self._replacement_history_limit == 0:
            return ()
        return combined[-self._replacement_history_limit :]

    @staticmethod
    def _log_replacements(replacements: Iterable[ToolReplacement]) -> None:
        for replacement in replacements:
            logger.warning(
                "Replacing tool %r: existing source=%r, spec=%r; incoming source=%r, spec=%r",
                replacement.name,
                replacement.existing_source,
                replacement.existing_spec,
                replacement.incoming_source,
                replacement.incoming_spec,
            )

    def _snapshot_state(self) -> _RegistryState:
        return _RegistryState(
            tools=self._tools,
            raw_tools=self._raw_tools,
            sources=self._sources,
            aliases=self._aliases,
            replacement_history=self._replacement_history,
            module_owners=self._module_owners,
        )

    def _publish_state(self, state: _RegistryState) -> None:
        (
            self._tools,
            self._raw_tools,
            self._sources,
            self._aliases,
            self._replacement_history,
            self._module_owners,
        ) = (
            state.tools,
            state.raw_tools,
            state.sources,
            state.aliases,
            state.replacement_history,
            state.module_owners,
        )

    def _prepare_module_state(
        self,
        specs: list[ToolSpec],
        *,
        module_name: str,
        replace_existing: bool,
    ) -> tuple[_RegistryState, list[ToolReplacement]]:
        owned_names = (
            {name for name, owner in self._module_owners.items() if owner == module_name}
            if replace_existing
            else set()
        )
        validation_tools = (
            {name: spec for name, spec in self._raw_tools.items() if name not in owned_names}
            if replace_existing
            else self._raw_tools
        )
        self._validate_batch(
            specs,
            replace_existing=False,
            existing_tools=validation_tools,
        )

        wrapped_by_name = {spec.name: self._wrap_spec(spec) for spec in specs}
        raw_by_name = {spec.name: spec for spec in specs}
        replacements = [
            self._replacement_record(self._raw_tools[spec.name], spec)
            for spec in specs
            if spec.name in owned_names
        ]

        next_tools: dict[str, ToolSpec] = {}
        next_raw_tools: dict[str, ToolSpec] = {}
        next_sources: dict[str, str] = {}
        published_names: set[str] = set()

        for name, existing_spec in self._raw_tools.items():
            if name in owned_names:
                incoming_spec = raw_by_name.get(name)
                if incoming_spec is None:
                    continue
                next_tools[name] = wrapped_by_name[name]
                next_raw_tools[name] = incoming_spec
                next_sources[name] = incoming_spec.source or "<unspecified>"
                published_names.add(name)
                continue

            next_tools[name] = self._tools[name]
            next_raw_tools[name] = existing_spec
            next_sources[name] = self._sources[name]

        for spec in specs:
            if spec.name in published_names:
                continue
            next_tools[spec.name] = wrapped_by_name[spec.name]
            next_raw_tools[spec.name] = spec
            next_sources[spec.name] = spec.source or "<unspecified>"

        next_module_owners = {
            name: owner for name, owner in self._module_owners.items() if owner != module_name
        }
        next_module_owners.update({spec.name: module_name for spec in specs})

        return (
            _RegistryState(
                tools=next_tools,
                raw_tools=next_raw_tools,
                sources=next_sources,
                aliases=self._build_alias_map(next_raw_tools.values()),
                replacement_history=self._extend_replacement_history(replacements),
                module_owners=next_module_owners,
            ),
            replacements,
        )

    @staticmethod
    def _create_isolated_module(qualified_name: str) -> tuple[ModuleType, Loader]:
        """Create a candidate module without publishing it in ``sys.modules``."""
        importlib.invalidate_caches()
        parent_name, _, _ = qualified_name.rpartition(".")
        parent_module = importlib.import_module(parent_name)
        search_path = getattr(parent_module, "__path__", None)
        spec = PathFinder.find_spec(qualified_name, search_path)
        if spec is None or spec.loader is None:
            raise ModuleNotFoundError(f"No module named {qualified_name!r}")
        module = module_from_spec(spec)
        return module, spec.loader

    @staticmethod
    def _public_module_proxy(
        qualified_name: str,
    ) -> tuple[ModuleType, str, _PublishedModuleProxy | None, bool]:
        parent_name, _, child_name = qualified_name.rpartition(".")
        parent_module = importlib.import_module(parent_name)
        sys_module = sys.modules.get(qualified_name, _MISSING)
        parent_child = parent_module.__dict__.get(child_name, _MISSING)

        if sys_module is _MISSING and parent_child is not _MISSING:
            raise RuntimeError(
                f"Incoherent module publication for {qualified_name!r}: "
                "parent package has a child binding but sys.modules does not"
            )
        if (
            sys_module is not _MISSING
            and parent_child is not _MISSING
            and sys_module is not parent_child
        ):
            raise RuntimeError(
                f"Incoherent module publication for {qualified_name!r}: "
                "sys.modules and the parent package expose different objects"
            )
        if sys_module is _MISSING:
            return parent_module, child_name, None, False
        if not isinstance(sys_module, ModuleType):
            raise TypeError(
                f"Public module entry {qualified_name!r} must be a module, got {type(sys_module)!r}"
            )
        if isinstance(sys_module, _PublishedModuleProxy):
            return parent_module, child_name, sys_module, False
        return parent_module, child_name, _adopt_module_proxy(sys_module), True

    def register(
        self,
        spec: ToolSpec,
        *,
        source: str | None = None,
        replace: bool = False,
    ) -> None:
        """Register one tool, rejecting duplicate names and aliases by default."""
        self.register_many([spec], source=source, replace=replace)

    def register_many(
        self,
        specs: Iterable[ToolSpec],
        *,
        source: str | None = None,
        replace: bool = False,
    ) -> None:
        """Atomically register multiple tools.

        ``replace=True`` may replace an existing registration with the same
        canonical name. It never steals a name or alias owned by another tool.
        """
        normalized_specs = [self._normalize_spec(spec, source) for spec in specs]
        if not normalized_specs:
            return

        replacements: list[ToolReplacement] = []
        with self._lock:
            self._validate_batch(normalized_specs, replace_existing=replace)
            wrapped_specs = [self._wrap_spec(spec) for spec in normalized_specs]

            next_tools = dict(self._tools)
            next_raw_tools = dict(self._raw_tools)
            next_sources = dict(self._sources)
            next_module_owners = dict(self._module_owners)

            for raw_spec, wrapped_spec in zip(normalized_specs, wrapped_specs, strict=True):
                existing_spec = self._raw_tools.get(raw_spec.name)
                if existing_spec is not None:
                    replacement_record = self._replacement_record(existing_spec, raw_spec)
                    replacements.append(replacement_record)

                next_tools[raw_spec.name] = wrapped_spec
                next_raw_tools[raw_spec.name] = raw_spec
                next_sources[raw_spec.name] = raw_spec.source or "<unspecified>"
                next_module_owners.pop(raw_spec.name, None)

            next_aliases = self._build_alias_map(next_raw_tools.values())
            next_replacement_history = self._extend_replacement_history(replacements)

            self._publish_state(
                _RegistryState(
                    tools=next_tools,
                    raw_tools=next_raw_tools,
                    sources=next_sources,
                    aliases=next_aliases,
                    replacement_history=next_replacement_history,
                    module_owners=next_module_owners,
                )
            )

        self._log_replacements(replacements)

    def register_module(
        self,
        module_name: str,
        *,
        source: str | None = None,
        replace: bool = False,
    ) -> None:
        """Atomically register a tool module, with opt-in hot replacement."""
        qualified_name = f"{__package__}.{module_name}"
        active_candidate = _active_candidate_name()
        if active_candidate is not None:
            raise _mark_candidate_fatal(
                ReentrantToolModuleRegistrationError(
                    f"Cannot register tool module {qualified_name!r} while candidate "
                    f"{active_candidate!r} is importing or collecting tools"
                )
            )

        with _MODULE_LOAD_LOCK:
            parent_module, child_name, published_proxy, adopted_proxy = self._public_module_proxy(
                qualified_name
            )
            initial_publication_snapshot: _ModulePublicationSnapshot | None = None
            initial_publication_active = False
            try:
                module, loader = self._create_isolated_module(qualified_name)
                if published_proxy is None:
                    published_proxy = _new_module_proxy(
                        qualified_name,
                        _PROXY_UNPUBLISHED_TARGET,
                    )
                    initial_publication_snapshot = _module_publication_snapshot(
                        qualified_name,
                        parent_module,
                        child_name,
                    )
                    initial_publication_active = True
                    _publish_module_proxy(
                        qualified_name,
                        parent_module,
                        child_name,
                        published_proxy,
                    )

                with _stage_proxy_target(published_proxy, module):
                    with _candidate_execution(qualified_name):
                        loader.exec_module(module)
                        register_tools = getattr(module, "register_tools", None)
                        if not callable(register_tools):
                            raise TypeError(
                                f"Tool module {qualified_name!r} must define callable register_tools"
                            )

                        staging_registry = ToolRegistry.empty()
                        register_tools(staging_registry)
                normalized_specs = [
                    self._normalize_spec(spec, source or qualified_name)
                    for spec in staging_registry._raw_specs()
                ]

                replacements: list[ToolReplacement]
                with self._lock:
                    next_state, replacements = self._prepare_module_state(
                        normalized_specs,
                        module_name=qualified_name,
                        replace_existing=replace,
                    )
                    previous_state = self._snapshot_state()
                    previous_target = _published_proxy_target(published_proxy)
                    self._publish_state(next_state)
                    try:
                        _assert_module_proxy_publication(
                            qualified_name,
                            parent_module,
                            child_name,
                            published_proxy,
                        )
                        _set_proxy_target(published_proxy, module)
                        if adopted_proxy:
                            _discard_obsolete_adopted_namespace(published_proxy)
                    except BaseException:
                        self._publish_state(previous_state)
                        _set_proxy_target(published_proxy, previous_target)
                        raise
                    initial_publication_active = False
            except BaseException:
                if initial_publication_active and initial_publication_snapshot is not None:
                    _restore_module_publication(
                        qualified_name,
                        parent_module,
                        child_name,
                        initial_publication_snapshot,
                    )
                if adopted_proxy and published_proxy is not None:
                    _undo_module_proxy_adoption(published_proxy)
                raise

            self._log_replacements(replacements)

    def _raw_specs(self) -> list[ToolSpec]:
        with self._lock:
            return list(self._raw_tools.values())

    def list_names(self) -> list[str]:
        with self._lock:
            return list(self._tools.keys())

    def get(self, name: str) -> ToolSpec | None:
        with self._lock:
            canonical_name = self._aliases.get(name, name)
            return self._tools.get(canonical_name)

    def source_for(self, name: str) -> str | None:
        """Return registration source metadata for a tool name or alias."""
        with self._lock:
            canonical_name = self._aliases.get(name, name)
            return self._sources.get(canonical_name)

    def replacement_history(self) -> tuple[ToolReplacement, ...]:
        """Return the bounded immutable metadata history of explicit replacements."""
        with self._lock:
            return tuple(self._replacement_history)
