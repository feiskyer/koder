import _thread
import asyncio
import gc
import importlib
import logging
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType

import pytest

from koder_agent.harness import tools as tools_package
from koder_agent.harness.permissions.modes import PermissionMode
from koder_agent.harness.permissions.results import PermissionEvaluationResult
from koder_agent.harness.tools import registry as tool_registry_module
from koder_agent.harness.tools.registry import (
    CandidateThreadStartError,
    DuplicateToolError,
    ReentrantToolModuleRegistrationError,
    ToolRegistry,
    ToolSpec,
    UnmanagedToolModuleReloadError,
)


async def _invoke_old(_arguments):
    return {"tool": "shared", "status": "success", "content": "old"}


async def _invoke_new(_arguments):
    return {"tool": "shared", "status": "success", "content": "new"}


class _AllowingPermissionService:
    def __init__(self):
        self.calls = []

    async def evaluate_tool_call_async(self, name, arguments):
        self.calls.append((name, arguments))
        return PermissionEvaluationResult.allow(
            tool_name=name,
            mode=PermissionMode.DEFAULT,
        )


def _tool_module_source(version: str, exports: tuple[tuple[str, str], ...]) -> str:
    lines = [
        "from koder_agent.harness.tools.registry import ToolSpec",
        "",
        f"VERSION = {version!r}",
        "",
    ]
    for name, _alias in exports:
        lines.extend(
            [
                f"async def invoke_{name}(_arguments):",
                f"    return {{'tool': {name!r}, 'status': 'success', 'content': VERSION}}",
                "",
            ]
        )
    lines.extend(["def register_tools(registry):", "    registry.register_many(["])
    for name, alias in exports:
        lines.extend(
            [
                "        ToolSpec(",
                f"            name={name!r},",
                f"            aliases=({alias!r},),",
                f"            invoke=invoke_{name},",
                "        ),",
            ]
        )
    lines.extend(["    ])", ""])
    return "\n".join(lines)


@pytest.fixture
def reload_module_fixture(tmp_path, monkeypatch, request):
    module_name = "_tool_registry_hot_reload_fixture"
    qualified_name = f"{tools_package.__name__}.{module_name}"
    module_path = tmp_path / f"{module_name}.py"
    missing = object()
    previous_package_attribute = getattr(tools_package, module_name, missing)
    monkeypatch.setattr(
        tools_package,
        "__path__",
        [str(tmp_path), *list(tools_package.__path__)],
    )
    monkeypatch.delitem(sys.modules, qualified_name, raising=False)

    def restore_module_state() -> None:
        sys.modules.pop(qualified_name, None)
        if previous_package_attribute is missing:
            if hasattr(tools_package, module_name):
                delattr(tools_package, module_name)
        else:
            setattr(tools_package, module_name, previous_package_attribute)

    request.addfinalizer(restore_module_state)

    def write_module(source: str) -> None:
        previous_mtime_ns = module_path.stat().st_mtime_ns if module_path.exists() else None
        module_path.write_text(source, encoding="utf-8")
        if previous_mtime_ns is not None:
            current_stat = module_path.stat()
            os.utime(
                module_path,
                ns=(
                    current_stat.st_atime_ns,
                    max(
                        current_stat.st_mtime_ns,
                        previous_mtime_ns + 2_000_000_000,
                    ),
                ),
            )

    return module_name, qualified_name, write_module


def test_empty_registry_has_no_tools():
    registry = ToolRegistry.empty()
    assert registry.list_names() == []


def test_register_adds_tool_and_source_metadata():
    registry = ToolRegistry.empty()
    registry.register(ToolSpec(name="test_tool", enabled=True), source="core.tests")

    assert registry.get("test_tool") is not None
    assert registry.get("test_tool").enabled is True
    assert registry.source_for("test_tool") == "core.tests"


def test_register_rejects_duplicate_with_existing_and_incoming_details():
    registry = ToolRegistry.empty()
    existing = ToolSpec(name="shared", invoke=_invoke_old, source="core.file")
    incoming = ToolSpec(name="shared", invoke=_invoke_new, source="plugin.demo")
    registry.register(existing)

    with pytest.raises(DuplicateToolError) as exc_info:
        registry.register(incoming)

    message = str(exc_info.value)
    assert "shared" in message
    assert "core.file" in message
    assert "plugin.demo" in message
    assert repr(existing) in message
    assert repr(incoming) in message
    assert registry.get("shared").invoke is _invoke_old


def test_register_many_is_atomic_when_one_name_already_exists():
    registry = ToolRegistry.empty()
    registry.register(ToolSpec(name="existing", source="core"))

    with pytest.raises(DuplicateToolError):
        registry.register_many(
            [
                ToolSpec(name="new_tool", source="plugin.demo"),
                ToolSpec(name="existing", source="plugin.demo"),
            ]
        )

    assert registry.list_names() == ["existing"]
    assert registry.get("new_tool") is None


def test_register_many_rejects_duplicate_aliases_without_partial_mutation():
    registry = ToolRegistry.empty()

    with pytest.raises(DuplicateToolError) as exc_info:
        registry.register_many(
            [
                ToolSpec(name="first", aliases=("shared-alias",), source="plugin.one"),
                ToolSpec(name="second", aliases=("shared-alias",), source="plugin.two"),
            ]
        )

    assert "shared-alias" in str(exc_info.value)
    assert "plugin.one" in str(exc_info.value)
    assert "plugin.two" in str(exc_info.value)
    assert registry.list_names() == []
    assert registry.get("shared-alias") is None


def test_explicit_replace_updates_aliases_source_and_observability(caplog):
    registry = ToolRegistry.empty()
    registry.register(
        ToolSpec(name="shared", aliases=("old-alias",), invoke=_invoke_old),
        source="plugin.demo@1",
    )

    with caplog.at_level(logging.WARNING):
        registry.register(
            ToolSpec(name="shared", aliases=("new-alias",), invoke=_invoke_new),
            source="plugin.demo@2",
            replace=True,
        )

    assert registry.get("shared").invoke is _invoke_new
    assert registry.get("new-alias") is registry.get("shared")
    assert registry.get("old-alias") is None
    assert registry.source_for("shared") == "plugin.demo@2"
    assert "plugin.demo@1" in caplog.text
    assert "plugin.demo@2" in caplog.text
    replacement = registry.replacement_history()[-1]
    assert replacement.name == "shared"
    assert replacement.existing_source == "plugin.demo@1"
    assert replacement.incoming_source == "plugin.demo@2"


def test_replace_cannot_steal_another_tools_alias_or_partly_mutate_registry():
    registry = ToolRegistry.empty()
    registry.register_many(
        [
            ToolSpec(name="first", aliases=("first-alias",), source="plugin.one"),
            ToolSpec(name="second", aliases=("second-alias",), source="plugin.two"),
        ]
    )

    with pytest.raises(DuplicateToolError):
        registry.register(
            ToolSpec(
                name="first",
                aliases=("second-alias",),
                source="plugin.one@2",
            ),
            replace=True,
        )

    assert registry.source_for("first") == "plugin.one"
    assert registry.get("first-alias") is registry.get("first")
    assert registry.get("second-alias") is registry.get("second")
    assert registry.replacement_history() == ()


def test_register_many_replace_is_atomic_when_later_spec_conflicts():
    registry = ToolRegistry.empty()
    registry.register_many(
        [
            ToolSpec(name="first", aliases=("first-alias",), source="plugin.one@1"),
            ToolSpec(name="second", aliases=("second-alias",), source="plugin.two@1"),
        ]
    )

    with pytest.raises(DuplicateToolError):
        registry.register_many(
            [
                ToolSpec(name="first", aliases=("first-v2",), source="plugin.one@2"),
                ToolSpec(
                    name="third",
                    aliases=("second-alias",),
                    source="plugin.three@1",
                ),
            ],
            replace=True,
        )

    assert registry.source_for("first") == "plugin.one@1"
    assert registry.get("first-alias") is registry.get("first")
    assert registry.get("first-v2") is None
    assert registry.get("second-alias") is registry.get("second")
    assert registry.get("third") is None
    assert registry.replacement_history() == ()


def test_register_many_replace_atomically_swaps_two_tool_aliases():
    registry = ToolRegistry.empty()
    registry.register_many(
        [
            ToolSpec(name="a", aliases=("alias-a",), source="plugin.a@1"),
            ToolSpec(name="b", aliases=("alias-b",), source="plugin.b@1"),
        ]
    )

    registry.register_many(
        [
            ToolSpec(name="a", aliases=("alias-b",), source="plugin.a@2"),
            ToolSpec(name="b", aliases=("alias-a",), source="plugin.b@2"),
        ],
        replace=True,
    )

    assert registry.get("alias-a") is registry.get("b")
    assert registry.get("alias-b") is registry.get("a")
    assert registry.source_for("alias-a") == "plugin.b@2"
    assert registry.source_for("alias-b") == "plugin.a@2"
    assert [record.name for record in registry.replacement_history()] == ["a", "b"]


def test_register_many_replace_atomically_rotates_larger_alias_cycle():
    registry = ToolRegistry.empty()
    names = ("a", "b", "c", "d", "e")
    aliases = tuple(f"alias-{name}" for name in names)
    registry.register_many(
        [
            ToolSpec(name=name, aliases=(alias,), source=f"plugin.{name}@1")
            for name, alias in zip(names, aliases, strict=True)
        ]
    )

    registry.register_many(
        [
            ToolSpec(
                name=name,
                aliases=(aliases[(index + 1) % len(aliases)],),
                source=f"plugin.{name}@2",
            )
            for index, name in enumerate(names)
        ],
        replace=True,
    )

    for index, alias in enumerate(aliases):
        expected_owner = names[(index - 1) % len(names)]
        assert registry.get(alias) is registry.get(expected_owner)
        assert registry.source_for(alias) == f"plugin.{expected_owner}@2"
    assert len(registry.replacement_history()) == len(names)


def test_replace_rebuilds_permission_wrapper_around_new_invoke():
    permission_service = _AllowingPermissionService()
    registry = ToolRegistry.with_permission_service(permission_service)
    registry.register(ToolSpec(name="shared", invoke=_invoke_old), source="core")

    old_wrapped = registry.get("shared").invoke
    registry.register(
        ToolSpec(name="shared", invoke=_invoke_new),
        source="hot-reload",
        replace=True,
    )
    new_wrapped = registry.get("shared").invoke

    assert new_wrapped is not old_wrapped
    result = asyncio.run(new_wrapped({"value": 1}))
    assert result["content"] == "new"
    assert permission_service.calls == [("shared", {"value": 1})]


def test_replace_from_wrapped_spec_does_not_stack_permission_wrappers():
    permission_service = _AllowingPermissionService()
    registry = ToolRegistry.with_permission_service(permission_service)
    registry.register(ToolSpec(name="shared", invoke=_invoke_old), source="core")

    wrapped_spec = registry.get("shared")
    registry.register(wrapped_spec, source="hot-reload", replace=True)

    result = asyncio.run(registry.get("shared").invoke({"value": 1}))
    assert result["content"] == "old"
    assert permission_service.calls == [("shared", {"value": 1})]


def test_register_module_requires_explicit_replace_and_atomically_publishes_module(
    reload_module_fixture, caplog
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("plugin_tool", "plugin-alias"),)))
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="plugin.demo@v1")
    external_v1_reference = sys.modules[qualified_name]
    assert asyncio.run(registry.get("plugin_tool").invoke({}))["content"] == "v1"

    write_module(_tool_module_source("version-v2", (("plugin_tool", "plugin-alias"),)))

    with pytest.raises(DuplicateToolError) as exc_info:
        registry.register_module(module_name, source="plugin.demo@v2")
    assert "plugin.demo@v1" in str(exc_info.value)
    assert "plugin.demo@v2" in str(exc_info.value)
    assert asyncio.run(registry.get("plugin_tool").invoke({}))["content"] == "v1"

    with caplog.at_level(logging.WARNING):
        registry.register_module(module_name, source="plugin.demo@v2", replace=True)

    published_v2_module = sys.modules[qualified_name]
    assert published_v2_module is external_v1_reference
    assert getattr(tools_package, module_name) is published_v2_module
    assert published_v2_module.VERSION == "version-v2"
    assert external_v1_reference.VERSION == "version-v2"
    assert asyncio.run(registry.get("plugin_tool").invoke({}))["content"] == "version-v2"
    assert registry.source_for("plugin-alias") == "plugin.demo@v2"
    assert "Replacing tool 'plugin_tool'" in caplog.text
    replacement = registry.replacement_history()[-1]
    assert replacement.name == "plugin_tool"
    assert replacement.existing_source == "plugin.demo@v1"
    assert replacement.incoming_source == "plugin.demo@v2"
    assert replacement.existing_spec.name == "plugin_tool"
    assert replacement.existing_spec.aliases == ("plugin-alias",)
    assert replacement.incoming_spec.name == "plugin_tool"
    assert replacement.incoming_spec.aliases == ("plugin-alias",)
    assert not hasattr(replacement.existing_spec, "invoke")
    assert not hasattr(replacement.incoming_spec, "invoke")


def test_register_module_adopts_preimported_module_identity_before_first_registration(
    reload_module_fixture,
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))

    preimport_reference = importlib.import_module(qualified_name)
    assert sys.modules[qualified_name] is preimport_reference
    assert getattr(tools_package, module_name) is preimport_reference

    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")

    assert sys.modules[qualified_name] is preimport_reference
    assert getattr(tools_package, module_name) is preimport_reference
    assert preimport_reference.VERSION == "v1"

    write_module(_tool_module_source("v2", (("one", "one-new"),)))
    registry.register_module(module_name, source="module@v2", replace=True)

    assert sys.modules[qualified_name] is preimport_reference
    assert getattr(tools_package, module_name) is preimport_reference
    assert preimport_reference.VERSION == "v2"
    assert asyncio.run(registry.get("one").invoke({}))["content"] == "v2"
    assert registry.get("one-old") is None
    assert registry.get("one-new") is registry.get("one")


def test_first_publication_gate_never_splits_sys_modules_and_parent_surface(
    reload_module_fixture, monkeypatch
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))
    assert qualified_name not in sys.modules
    assert module_name not in tools_package.__dict__

    published = threading.Event()
    release = threading.Event()
    original_publish = tool_registry_module._publish_module_proxy

    def gated_publish(qualified, parent, child, proxy):
        sys.modules[qualified] = proxy
        published.set()
        if not release.wait(timeout=5):
            raise RuntimeError("publication gate timed out")
        assert getattr(parent, child) is proxy

    monkeypatch.setattr(tool_registry_module, "_publish_module_proxy", gated_publish)
    registry = ToolRegistry.empty()

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(registry.register_module, module_name, source="module@v1")
        assert published.wait(timeout=5)
        public_from_sys = sys.modules[qualified_name]
        public_from_parent = getattr(tools_package, module_name)
        assert public_from_sys is public_from_parent
        assert module_name not in tools_package.__dict__
        release.set()
        future.result(timeout=5)

    monkeypatch.setattr(tool_registry_module, "_publish_module_proxy", original_publish)
    assert sys.modules[qualified_name] is getattr(tools_package, module_name)
    assert asyncio.run(registry.get("one").invoke({}))["content"] == "v1"


def test_first_registration_self_import_rolls_back_exactly_and_retries(
    reload_module_fixture,
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(
        "\n".join(
            [
                "import importlib",
                "import sys",
                "from koder_agent.harness import tools as tools_package",
                "SELF_FROM_IMPORT = importlib.import_module(__name__)",
                "SELF_FROM_SYS = sys.modules[__name__]",
                f"SELF_FROM_PARENT = getattr(tools_package, {module_name!r})",
                "assert SELF_FROM_IMPORT is SELF_FROM_SYS is SELF_FROM_PARENT",
                "SELF_FROM_IMPORT.CANDIDATE_ONLY = 'failed-first-generation'",
                "raise RuntimeError('first self import failed')",
                "",
            ]
        )
    )
    registry = ToolRegistry.empty()

    with pytest.raises(RuntimeError, match="first self import failed"):
        registry.register_module(module_name, source="module@failed")

    assert qualified_name not in sys.modules
    assert module_name not in tools_package.__dict__
    with pytest.raises(AttributeError):
        getattr(tools_package, module_name)
    assert registry.list_names() == []
    assert registry.replacement_history() == ()

    write_module(
        "\n".join(
            [
                "import importlib",
                "import sys",
                "from koder_agent.harness import tools as tools_package",
                "from koder_agent.harness.tools.registry import ToolSpec",
                "SELF_FROM_IMPORT = importlib.import_module(__name__)",
                "SELF_FROM_SYS = sys.modules[__name__]",
                f"SELF_FROM_PARENT = getattr(tools_package, {module_name!r})",
                "assert SELF_FROM_IMPORT is SELF_FROM_SYS is SELF_FROM_PARENT",
                "VERSION = 'retry-v1'",
                "async def invoke(_arguments):",
                "    return {'tool': 'one', 'status': 'success', 'content': VERSION}",
                "def register_tools(registry):",
                "    registry.register(ToolSpec(name='one', invoke=invoke))",
                "",
            ]
        )
    )
    registry.register_module(module_name, source="module@retry")

    public_module = sys.modules[qualified_name]
    assert getattr(tools_package, module_name) is public_module
    assert public_module.SELF_FROM_IMPORT is public_module
    assert public_module.SELF_FROM_SYS is public_module
    assert public_module.SELF_FROM_PARENT is public_module
    assert not hasattr(public_module, "CANDIDATE_ONLY")
    assert asyncio.run(registry.get("one").invoke({}))["content"] == "retry-v1"
    assert registry.source_for("one") == "module@retry"


@pytest.mark.parametrize(
    "start_worker",
    [
        "threading.Thread(target=mutate_public_module).start()",
        "_thread.start_new_thread(mutate_public_module, ())",
        "_thread.start_new(mutate_public_module, ())",
    ],
)
def test_failed_candidate_import_rejects_child_thread_before_live_module_access(
    reload_module_fixture, start_worker
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)) + "\nSTATE = {'items': []}\n")
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    public_module = sys.modules[qualified_name]
    original_spec = registry.get("one")
    thread_starters = (_thread.start_new_thread, _thread.start_new)

    write_module(
        "\n".join(
            [
                "import _thread",
                "import sys",
                "import threading",
                "from koder_agent.harness.tools.registry import ToolSpec",
                "VERSION = 'failed-import-child'",
                "STATE = {'items': []}",
                "finished = threading.Event()",
                "def mutate_public_module():",
                "    public_module = sys.modules[__name__]",
                "    public_module.STATE['items'].append('import-child-thread')",
                "    public_module.VERSION = 'import-child-thread-leak'",
                "    finished.set()",
                start_worker,
                "assert finished.wait(timeout=2)",
                "def register_tools(registry):",
                "    registry.register(ToolSpec(name='one'))",
                "",
            ]
        )
    )

    with pytest.raises(CandidateThreadStartError, match="cannot start child threads"):
        registry.register_module(module_name, source="module@v2", replace=True)

    assert sys.modules[qualified_name] is public_module
    assert getattr(tools_package, module_name) is public_module
    assert public_module.VERSION == "v1"
    assert public_module.STATE == {"items": []}
    assert registry.get("one") is original_spec
    assert asyncio.run(registry.get("one").invoke({}))["content"] == "v1"
    assert registry.source_for("one") == "module@v1"
    assert registry.replacement_history() == ()
    assert (_thread.start_new_thread, _thread.start_new) == thread_starters


def test_failed_reload_import_preserves_registry_and_external_module_reference(
    reload_module_fixture,
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)) + "\nSTATE = {'items': []}\n")
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    external_reference = sys.modules[qualified_name]
    external_namespace = external_reference.__dict__
    external_state = dict(external_reference.__dict__)
    old_nested_state = external_reference.STATE
    original_spec = registry.get("one")

    write_module(
        "\n".join(
            [
                "import sys",
                "from koder_agent.harness import tools as tools_package",
                "STATE = {'items': []}",
                "old_from_sys = sys.modules[__name__]",
                f"old_from_parent = getattr(tools_package, {module_name!r})",
                "old_from_sys.STATE['items'].append('import-through-sys')",
                "old_from_parent.STATE['items'].append('import-through-parent')",
                "old_from_sys.__dict__['VERSION'] = 'leaked-through-sys-modules'",
                "old_from_sys.__dict__['LEAKED_FROM_SYS_MODULES'] = True",
                "old_from_parent.VERSION = 'leaked-through-parent'",
                "old_from_parent.LEAKED_FROM_PARENT = True",
                "raise RuntimeError('import failed')",
                "",
            ]
        )
    )

    with pytest.raises(RuntimeError, match="import failed"):
        registry.register_module(module_name, source="module@v2", replace=True)

    assert sys.modules[qualified_name] is external_reference
    assert getattr(tools_package, module_name) is external_reference
    assert external_reference.__dict__ is external_namespace
    assert external_reference.__dict__ == external_state
    assert external_reference.STATE is old_nested_state
    assert old_nested_state == {"items": []}
    assert registry.get("one") is original_spec
    assert registry.get("one-old") is original_spec
    assert registry.source_for("one") == "module@v1"
    assert registry.replacement_history() == ()


def test_failed_reload_registration_preserves_registry_and_external_module_reference(
    reload_module_fixture,
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)) + "\nSTATE = {'items': []}\n")
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    external_reference = sys.modules[qualified_name]
    external_namespace = external_reference.__dict__
    external_state = dict(external_reference.__dict__)
    old_nested_state = external_reference.STATE
    original_spec = registry.get("one")

    write_module(
        "\n".join(
            [
                "from koder_agent.harness.tools.registry import ToolSpec",
                "import sys",
                "from koder_agent.harness import tools as tools_package",
                "VERSION = 'failed-v2'",
                "STATE = {'items': []}",
                "def register_tools(registry):",
                "    registry.register(ToolSpec(name='staged'))",
                "    old_from_sys = sys.modules[__name__]",
                f"    old_from_parent = getattr(tools_package, {module_name!r})",
                "    old_from_sys.STATE['items'].append('register-through-sys')",
                "    old_from_parent.STATE['items'].append('register-through-parent')",
                "    old_from_sys.__dict__['VERSION'] = 'leaked-through-sys-modules'",
                "    old_from_sys.__dict__['LEAKED_FROM_SYS_MODULES'] = True",
                "    old_from_parent.VERSION = 'leaked-through-parent'",
                "    old_from_parent.LEAKED_FROM_PARENT = True",
                "    raise RuntimeError('registration failed')",
                "",
            ]
        )
    )

    with pytest.raises(RuntimeError, match="registration failed"):
        registry.register_module(module_name, source="module@v2", replace=True)

    assert sys.modules[qualified_name] is external_reference
    assert getattr(tools_package, module_name) is external_reference
    assert external_reference.__dict__ is external_namespace
    assert external_reference.__dict__ == external_state
    assert external_reference.STATE is old_nested_state
    assert old_nested_state == {"items": []}
    assert registry.list_names() == ["one"]
    assert registry.get("one") is original_spec
    assert registry.get("staged") is None
    assert registry.source_for("one") == "module@v1"
    assert registry.replacement_history() == ()


@pytest.mark.parametrize(
    "start_worker",
    [
        "threading.Thread(target=mutate_public_module).start()",
        "_thread.start_new_thread(mutate_public_module, ())",
        "_thread.start_new(mutate_public_module, ())",
    ],
)
def test_failed_tool_collection_rejects_child_thread_before_live_module_access(
    reload_module_fixture, start_worker
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)) + "\nSTATE = {'items': []}\n")
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    public_module = sys.modules[qualified_name]
    original_spec = registry.get("one")
    thread_starters = (_thread.start_new_thread, _thread.start_new)

    write_module(
        "\n".join(
            [
                "import _thread",
                "import sys",
                "import threading",
                "from koder_agent.harness.tools.registry import ToolSpec",
                "VERSION = 'failed-register-child'",
                "STATE = {'items': []}",
                "finished = threading.Event()",
                "def mutate_public_module():",
                "    public_module = sys.modules[__name__]",
                "    public_module.STATE['items'].append('register-child-thread')",
                "    public_module.VERSION = 'register-child-thread-leak'",
                "    finished.set()",
                "def register_tools(registry):",
                "    registry.register(ToolSpec(name='staged'))",
                f"    {start_worker}",
                "    assert finished.wait(timeout=2)",
                "",
            ]
        )
    )

    with pytest.raises(CandidateThreadStartError, match="cannot start child threads"):
        registry.register_module(module_name, source="module@v2", replace=True)

    assert sys.modules[qualified_name] is public_module
    assert getattr(tools_package, module_name) is public_module
    assert public_module.VERSION == "v1"
    assert public_module.STATE == {"items": []}
    assert registry.get("one") is original_spec
    assert registry.get("staged") is None
    assert asyncio.run(registry.get("one").invoke({}))["content"] == "v1"
    assert registry.source_for("one") == "module@v1"
    assert registry.replacement_history() == ()
    assert (_thread.start_new_thread, _thread.start_new) == thread_starters


def test_reentrant_register_module_during_collection_is_fail_closed_and_rolls_back(
    reload_module_fixture, request
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    public_module = sys.modules[qualified_name]
    original_spec = registry.get("one")

    gate_name = "_tool_registry_reentrant_publication_gate"
    gate = ModuleType(gate_name)
    gate.attempted = False
    gate.caught = None
    sys.modules[gate_name] = gate
    request.addfinalizer(lambda: sys.modules.pop(gate_name, None))
    write_module(
        "\n".join(
            [
                "from koder_agent.harness.tools.registry import (",
                "    ReentrantToolModuleRegistrationError,",
                "    ToolSpec,",
                ")",
                f"import {gate_name} as gate",
                "VERSION = 'v2-reentrant'",
                "async def invoke(_arguments):",
                "    return {'tool': 'one', 'status': 'success', 'content': VERSION}",
                "def register_tools(registry):",
                "    if not gate.attempted:",
                "        gate.attempted = True",
                "        try:",
                f"            registry.register_module({module_name!r}, source='nested', replace=True)",
                "        except ReentrantToolModuleRegistrationError as exc:",
                "            gate.caught = str(exc)",
                "    registry.register(ToolSpec(name='one', invoke=invoke))",
                "",
            ]
        )
    )

    with pytest.raises(
        ReentrantToolModuleRegistrationError,
        match="while candidate",
    ):
        registry.register_module(module_name, source="module@v2", replace=True)

    assert gate.attempted is True
    assert "while candidate" in gate.caught
    assert sys.modules[qualified_name] is public_module
    assert getattr(tools_package, module_name) is public_module
    assert public_module.VERSION == "v1"
    assert registry.get("one") is original_spec
    assert asyncio.run(registry.get("one").invoke({}))["content"] == "v1"
    assert registry.source_for("one") == "module@v1"
    assert registry.replacement_history() == ()


def test_failed_candidate_is_never_transiently_visible_to_lock_free_readers(
    reload_module_fixture, request
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    public_module = sys.modules[qualified_name]

    gate_name = "_tool_registry_transient_visibility_gate"
    gate = ModuleType(gate_name)
    gate.started = threading.Event()
    gate.release = threading.Event()
    sys.modules[gate_name] = gate
    request.addfinalizer(lambda: sys.modules.pop(gate_name, None))
    write_module(
        "\n".join(
            [
                "import sys",
                "from koder_agent.harness.tools.registry import ToolSpec",
                f"import {gate_name} as gate",
                "VERSION = 'failed-candidate-visible'",
                "def register_tools(registry):",
                "    sys.modules[__name__].VERSION = VERSION",
                "    gate.started.set()",
                "    if not gate.release.wait(timeout=5):",
                "        raise RuntimeError('gate timed out')",
                "    registry.register(ToolSpec(name='one'))",
                "    raise RuntimeError('candidate failed after visibility gate')",
                "",
            ]
        )
    )

    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(
            registry.register_module,
            module_name,
            source="module@v2",
            replace=True,
        )
        assert gate.started.wait(timeout=5)
        assert sys.modules[qualified_name] is public_module
        assert getattr(tools_package, module_name) is public_module
        assert public_module.VERSION == "v1"
        assert asyncio.run(registry.get("one").invoke({}))["content"] == "v1"
        # A candidate in the executor thread must not block unrelated callers
        # from creating their own threads while it is staged.
        observed_versions = []
        reader = threading.Thread(target=lambda: observed_versions.append(public_module.VERSION))
        reader.start()
        reader.join(timeout=2)
        assert not reader.is_alive()
        assert observed_versions == ["v1"]
        gate.release.set()
        with pytest.raises(RuntimeError, match="candidate failed after visibility gate"):
            future.result(timeout=5)

    assert public_module.VERSION == "v1"
    assert registry.source_for("one") == "module@v1"
    assert registry.replacement_history() == ()


def test_successful_reload_keeps_lock_free_public_readers_on_one_stable_proxy(
    reload_module_fixture, request
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    stable_proxy = sys.modules[qualified_name]

    gate_name = "_tool_registry_atomic_publication_gate"
    gate = ModuleType(gate_name)
    gate.version = "v2-0"
    sys.modules[gate_name] = gate
    request.addfinalizer(lambda: sys.modules.pop(gate_name, None))
    write_module(
        "\n".join(
            [
                "from koder_agent.harness.tools.registry import ToolSpec",
                f"import {gate_name} as gate",
                "VERSION = gate.version",
                "async def invoke(_arguments):",
                "    return {'tool': 'one', 'status': 'success', 'content': VERSION}",
                "def register_tools(registry):",
                "    registry.register(ToolSpec(name='one', invoke=invoke))",
                "",
            ]
        )
    )

    stop = threading.Event()
    reader_started = threading.Event()
    failures = []

    def read_public_surfaces():
        checks = 0
        reader_started.set()
        while not stop.is_set() or checks < 100_000:
            sys_surface = sys.modules[qualified_name]
            parent_surface = getattr(tools_package, module_name)
            if (
                sys_surface is not parent_surface
                or sys_surface is not stable_proxy
                or parent_surface is not stable_proxy
            ):
                failures.append((sys_surface, parent_surface))
                break
            checks += 1
        return checks

    with ThreadPoolExecutor(max_workers=2) as executor:
        reader_future = executor.submit(read_public_surfaces)
        assert reader_started.wait(timeout=5)
        try:
            for index in range(250):
                gate.version = f"v2-{index}"
                registry.register_module(
                    module_name,
                    source=f"module@v2-{index}",
                    replace=True,
                )
        finally:
            stop.set()
        checks = reader_future.result(timeout=10)

    assert checks >= 100_000
    assert failures == []
    assert sys.modules[qualified_name] is stable_proxy
    assert getattr(tools_package, module_name) is stable_proxy
    assert stable_proxy.VERSION == "v2-249"


def test_reload_registry_commit_failure_preserves_module_and_registry(
    reload_module_fixture,
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    registry.register(ToolSpec(name="outside", aliases=("reserved",)), source="direct")
    external_reference = sys.modules[qualified_name]
    original_one = registry.get("one")
    original_outside = registry.get("outside")

    write_module(_tool_module_source("v2", (("one", "reserved"),)))

    with pytest.raises(DuplicateToolError) as exc_info:
        registry.register_module(module_name, source="module@v2", replace=True)

    assert "module@v2" in str(exc_info.value)
    assert "direct" in str(exc_info.value)
    assert sys.modules[qualified_name] is external_reference
    assert getattr(tools_package, module_name) is external_reference
    assert external_reference.VERSION == "v1"
    assert registry.get("one") is original_one
    assert registry.get("outside") is original_outside
    assert registry.get("reserved") is original_outside
    assert registry.replacement_history() == ()


def test_reload_authoritatively_removes_missing_exports_aliases_and_sources(
    reload_module_fixture,
):
    module_name, _qualified_name, write_module = reload_module_fixture
    write_module(
        _tool_module_source(
            "v1",
            (("one", "one-old"), ("two", "two-old")),
        )
    )
    registry = ToolRegistry.empty()
    registry.register(ToolSpec(name="prefix"), source="direct")
    registry.register_module(module_name, source="module@v1")

    write_module(_tool_module_source("v2", (("one", "one-new"),)))
    registry.register_module(module_name, source="module@v2", replace=True)

    assert registry.list_names() == ["prefix", "one"]
    assert registry.get("one-new") is registry.get("one")
    assert registry.get("one-old") is None
    assert registry.get("two") is None
    assert registry.get("two-old") is None
    assert registry.source_for("two") is None
    assert registry.source_for("two-old") is None
    assert [record.name for record in registry.replacement_history()] == ["one"]


def test_reload_without_callable_register_tools_fails_instead_of_reusing_stale_function(
    reload_module_fixture,
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    external_reference = sys.modules[qualified_name]
    original_spec = registry.get("one")

    write_module("VERSION = 'v2-without-register-tools'\n")

    with pytest.raises(TypeError, match="must define callable register_tools"):
        registry.register_module(module_name, source="module@v2", replace=True)

    assert sys.modules[qualified_name] is external_reference
    assert getattr(tools_package, module_name) is external_reference
    assert external_reference.VERSION == "v1"
    assert registry.get("one") is original_spec
    assert registry.source_for("one") == "module@v1"
    assert registry.replacement_history() == ()


def test_unmanaged_importlib_reload_is_rejected_without_partial_registry_update(
    reload_module_fixture,
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    public_module = sys.modules[qualified_name]
    original_spec = public_module.__spec__
    original_loader = public_module.__loader__

    write_module(_tool_module_source("v2", (("one", "one-new"),)))

    with pytest.raises(
        UnmanagedToolModuleReloadError,
        match=r"registry\.register_module\(\.\.\., replace=True\)",
    ):
        importlib.reload(public_module)

    assert sys.modules[qualified_name] is public_module
    assert getattr(tools_package, module_name) is public_module
    assert public_module.VERSION == "v1"
    assert public_module.__spec__ is original_spec
    assert public_module.__loader__ is original_loader
    assert asyncio.run(registry.get("one").invoke({}))["content"] == "v1"
    assert registry.get("one-old") is registry.get("one")
    assert registry.get("one-new") is None
    assert registry.source_for("one") == "module@v1"
    assert registry.replacement_history() == ()


def test_concurrent_failed_reload_does_not_rollback_direct_registration(
    reload_module_fixture, request
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    external_reference = sys.modules[qualified_name]

    gate_name = "_tool_registry_reload_failure_gate"
    gate = ModuleType(gate_name)
    gate.started = threading.Event()
    gate.release = threading.Event()
    sys.modules[gate_name] = gate
    request.addfinalizer(lambda: sys.modules.pop(gate_name, None))
    write_module(
        "\n".join(
            [
                "from koder_agent.harness.tools.registry import ToolSpec",
                f"import {gate_name} as gate",
                "VERSION = 'failed-v2'",
                "def register_tools(registry):",
                "    gate.started.set()",
                "    if not gate.release.wait(timeout=5):",
                "        raise RuntimeError('gate timed out')",
                "    registry.register(ToolSpec(name='staged'))",
                "    raise RuntimeError('concurrent reload failed')",
                "",
            ]
        )
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        reload_future = executor.submit(
            registry.register_module,
            module_name,
            source="module@v2",
            replace=True,
        )
        assert gate.started.wait(timeout=5)
        runtime_write = object()
        external_reference.RUNTIME_WRITE = runtime_write
        registry.register(ToolSpec(name="direct"), source="concurrent-direct")
        gate.release.set()
        with pytest.raises(RuntimeError, match="concurrent reload failed"):
            reload_future.result(timeout=5)

    assert sys.modules[qualified_name] is external_reference
    assert getattr(tools_package, module_name) is external_reference
    assert external_reference.VERSION == "v1"
    assert external_reference.RUNTIME_WRITE is runtime_write
    assert registry.list_names() == ["one", "direct"]
    assert registry.get("staged") is None
    assert registry.source_for("one") == "module@v1"
    assert registry.source_for("direct") == "concurrent-direct"
    assert registry.replacement_history() == ()


def test_concurrent_reload_and_direct_replacement_commit_in_serial_order(
    reload_module_fixture, request
):
    module_name, qualified_name, write_module = reload_module_fixture
    write_module(_tool_module_source("v1", (("one", "one-old"),)))
    registry = ToolRegistry.empty()
    registry.register_module(module_name, source="module@v1")
    external_reference = sys.modules[qualified_name]

    gate_name = "_tool_registry_reload_serialization_gate"
    gate = ModuleType(gate_name)
    gate.started = threading.Event()
    gate.release = threading.Event()
    sys.modules[gate_name] = gate
    request.addfinalizer(lambda: sys.modules.pop(gate_name, None))
    write_module(
        "\n".join(
            [
                "from koder_agent.harness.tools.registry import ToolSpec",
                f"import {gate_name} as gate",
                "VERSION = 'v2'",
                "async def invoke(_arguments):",
                "    return {'tool': 'one', 'status': 'success', 'content': VERSION}",
                "def register_tools(registry):",
                "    gate.started.set()",
                "    if not gate.release.wait(timeout=5):",
                "        raise RuntimeError('gate timed out')",
                "    registry.register(ToolSpec(name='one', invoke=invoke))",
                "",
            ]
        )
    )

    direct_spec = ToolSpec(name="one", source="concurrent-direct")
    with ThreadPoolExecutor(max_workers=2) as executor:
        reload_future = executor.submit(
            registry.register_module,
            module_name,
            source="module@v2",
            replace=True,
        )
        assert gate.started.wait(timeout=5)
        registry.register(direct_spec, replace=True)
        gate.release.set()
        with pytest.raises(DuplicateToolError):
            reload_future.result(timeout=5)

    assert sys.modules[qualified_name] is external_reference
    assert getattr(tools_package, module_name) is external_reference
    assert external_reference.VERSION == "v1"
    assert registry.get("one").source == "concurrent-direct"
    assert registry.get("one-old") is None
    assert registry.source_for("one") == "concurrent-direct"
    assert [record.incoming_source for record in registry.replacement_history()] == [
        "concurrent-direct"
    ]


def test_concurrent_duplicate_registration_has_one_winner():
    registry = ToolRegistry.empty()
    specs = [
        ToolSpec(name="shared", source="worker.one"),
        ToolSpec(name="shared", source="worker.two"),
    ]

    def register(spec):
        try:
            registry.register(spec)
            return "registered"
        except DuplicateToolError:
            return "duplicate"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(register, specs))

    assert sorted(outcomes) == ["duplicate", "registered"]
    assert registry.list_names() == ["shared"]
    assert registry.source_for("shared") in {"worker.one", "worker.two"}


def test_replacement_history_is_bounded_and_does_not_retain_module_generations(
    reload_module_fixture, request
):
    module_name, qualified_name, write_module = reload_module_fixture
    gate_name = "_tool_registry_generation_retention_gate"
    gate = ModuleType(gate_name)
    gate.resource_refs = []
    sys.modules[gate_name] = gate
    request.addfinalizer(lambda: sys.modules.pop(gate_name, None))

    def generation_source(version: str) -> str:
        return "\n".join(
            [
                "import weakref",
                "from koder_agent.harness.tools.registry import ToolSpec",
                f"import {gate_name} as gate",
                f"VERSION = {version!r}",
                "class Resource:",
                "    pass",
                "RESOURCE = Resource()",
                "gate.resource_refs.append(weakref.ref(RESOURCE))",
                "async def invoke(_arguments):",
                "    return {'tool': 'one', 'status': 'success', 'content': VERSION}",
                "def register_tools(registry):",
                "    registry.register(ToolSpec(name='one', invoke=invoke))",
                "",
            ]
        )

    write_module(generation_source("v0"))
    registry = ToolRegistry.empty(replacement_history_limit=8)
    registry.register_module(module_name, source="module@v0")

    for generation in range(1, 31):
        write_module(generation_source(f"v{generation}"))
        registry.register_module(
            module_name,
            source=f"module@v{generation}",
            replace=True,
        )

    public_module = sys.modules[qualified_name]
    gc.collect()

    assert len(gate.resource_refs) == 31
    assert all(resource_ref() is None for resource_ref in gate.resource_refs[:-1])
    assert gate.resource_refs[-1]() is public_module.RESOURCE
    history = registry.replacement_history()
    assert len(history) == 8
    assert history[-1].incoming_source == "module@v30"
    assert all(not hasattr(record.existing_spec, "invoke") for record in history)
    assert all(not hasattr(record.incoming_spec, "invoke") for record in history)


def test_preimport_adoption_discards_original_namespace_and_all_old_generations(
    reload_module_fixture, request
):
    module_name, qualified_name, write_module = reload_module_fixture
    gate_name = "_tool_registry_preimport_generation_retention_gate"
    gate = ModuleType(gate_name)
    gate.resource_refs = []
    sys.modules[gate_name] = gate
    request.addfinalizer(lambda: sys.modules.pop(gate_name, None))

    def generation_source(version: str) -> str:
        return "\n".join(
            [
                "import weakref",
                "from koder_agent.harness.tools.registry import ToolSpec",
                f"import {gate_name} as gate",
                f"VERSION = {version!r}",
                "class Resource:",
                "    pass",
                "RESOURCE = Resource()",
                "gate.resource_refs.append(weakref.ref(RESOURCE))",
                "async def invoke(_arguments):",
                "    return {'tool': 'one', 'status': 'success', 'content': VERSION}",
                "def register_tools(registry):",
                "    registry.register(ToolSpec(name='one', invoke=invoke))",
                "",
            ]
        )

    write_module(generation_source("preimport"))
    preimport_reference = importlib.import_module(qualified_name)
    assert getattr(tools_package, module_name) is preimport_reference

    write_module(generation_source("v1"))
    registry = ToolRegistry.empty(replacement_history_limit=2)
    registry.register_module(module_name, source="module@v1")

    for generation in range(2, 7):
        write_module(generation_source(f"v{generation}"))
        registry.register_module(
            module_name,
            source=f"module@v{generation}",
            replace=True,
        )

    public_module = sys.modules[qualified_name]
    hidden_proxy_storage = ModuleType.__getattribute__(public_module, "__dict__")
    gc.collect()

    assert public_module is preimport_reference
    assert getattr(tools_package, module_name) is public_module
    assert len(gate.resource_refs) == 7
    assert all(resource_ref() is None for resource_ref in gate.resource_refs[:-1])
    assert gate.resource_refs[-1]() is public_module.RESOURCE
    assert "RESOURCE" not in hidden_proxy_storage
    assert "register_tools" not in hidden_proxy_storage
    assert tool_registry_module._PROXY_ORIGINAL_CLASS_KEY not in hidden_proxy_storage
    assert set(hidden_proxy_storage) <= {
        "__name__",
        tool_registry_module._PROXY_TARGET_KEY,
    }
    history = registry.replacement_history()
    assert len(history) == 2
    assert history[-1].incoming_source == "module@v6"


def test_with_core_tools_registers_file_tools():
    registry = ToolRegistry.with_core_tools(categories={"file"})
    names = set(registry.list_names())
    assert {"read_file", "write_file", "edit_file"} <= names


def test_with_core_tools_registers_code_tools():
    registry = ToolRegistry.with_core_tools(categories={"code"})
    names = set(registry.list_names())
    assert "code_intelligence" in names
