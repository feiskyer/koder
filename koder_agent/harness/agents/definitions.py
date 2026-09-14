"""Agent definition loading for the harness runtime."""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import yaml

from koder_agent.harness.paths import (
    local_agent_memory_dir,
    project_agent_memory_dir,
    settings_path,
    user_agent_memory_dir,
    user_agents_dir,
)
from koder_agent.harness.permissions.modes import FILE_WRITE_TOOLS
from koder_agent.harness.plugins.context import get_plugin_root, normalize_plugin_root
from koder_agent.mcp.server_config import MCPServerConfig, MCPServerScope, MCPServerType
from koder_agent.mcp.server_manager import MCPServerManager

AgentSource = Literal[
    "built-in",
    "plugin",
    "userSettings",
    "projectSettings",
    "flagSettings",
]

FRONTMATTER_RE = re.compile(r"^---\s*\n(?P<yaml>.*?)\n---\s*\n?(?P<body>.*)$", re.DOTALL)
DISPLAY_SOURCE_GROUPS: tuple[tuple[AgentSource, str], ...] = (
    ("userSettings", "User agents"),
    ("projectSettings", "Project agents"),
    ("plugin", "Plugin agents"),
    ("flagSettings", "CLI arg agents"),
    ("built-in", "Built-in agents"),
)


@dataclass(frozen=True)
class AgentDefinition:
    """Source-compatible definition for a built-in or custom agent."""

    agent_type: str
    when_to_use: str
    system_prompt: str
    source: AgentSource
    filename: str | None = None
    base_dir: str | None = None
    source_path: str | None = None
    project_root: str | None = None
    execution_cwd: str | None = None
    plugin: str | None = None
    tools: list[str] | None = None
    disallowed_tools: list[str] | None = None
    skills: list[str] | None = None
    model: str | None = None
    permission_mode: str | None = None
    max_turns: int | None = None
    background: bool | None = None
    memory: str | None = None
    isolation: str | None = None
    hooks: dict[str, Any] | None = None
    mcp_servers: list[Any] | None = None
    effort: str | int | None = None
    color: str | None = None
    initial_prompt: str | None = None
    allowed_agent_types: list[str] | None = None


@dataclass(frozen=True)
class AgentDefinitionsResult:
    """Loaded active and inactive agent definitions."""

    active_agents: list[AgentDefinition]
    all_agents: list[AgentDefinition]
    failed_files: list[dict[str, str]] | None = None


def _general_purpose_prompt() -> str:
    return (
        "You are an agent for Koder, a terminal-based AI coding assistant. "
        "Given the user's message, you should use the tools available to complete the task. "
        "Complete the task fully—don't gold-plate, but don't leave it half-done."
    )


def _explore_prompt() -> str:
    return (
        "You are a file search specialist for Koder. This is a read-only exploration "
        "task for searching and analyzing existing code."
    )


def _plan_prompt() -> str:
    return (
        "You are a software architect and planning specialist for Koder. "
        "This is a read-only planning task for exploring the codebase and designing implementation plans."
    )


def _statusline_setup_prompt() -> str:
    return """You are a status line setup agent for Koder. Your job is to create or update the statusLine command in the user's Koder settings.

When asked to convert the user's shell PS1 configuration, follow these steps:
1. Read the user's shell configuration files in this order of preference:
   - ~/.zshrc
   - ~/.bashrc
   - ~/.bash_profile
   - ~/.profile
2. Extract the PS1 value using this regex pattern:
   /(?:^|\\n)\\s*(?:export\\s+)?PS1\\s*=\\s*["']([^"']+)["']/m
3. Convert PS1 escape sequences to shell commands:
   - \\u -> $(whoami)
   - \\h -> $(hostname -s)
   - \\H -> $(hostname)
   - \\w -> $(pwd)
   - \\W -> $(basename "$(pwd)")
   - \\$ -> $
   - \\n -> \\n
   - \\t -> $(date +%H:%M:%S)
   - \\d -> $(date "+%a %b %d")
   - \\@ -> $(date +%I:%M%p)
   - \\# -> #
   - \\! -> !
4. When using ANSI color codes, be sure to use printf. Do not remove colors.
5. If the imported prompt would have trailing "$" or ">" characters in the output, remove them.
6. If no PS1 is found and the user did not provide other instructions, ask for further instructions.

How to configure Koder's statusLine command:
1. The statusLine command receives JSON on stdin with runtime fields such as:
   - session_id
   - session_name
   - cwd
   - model.id
   - model.display_name
   - workspace.current_dir
   - workspace.project_dir
   - workspace.added_dirs
   - version
   - output_style.name
   - cost.total_cost_usd
   - context_window.total_input_tokens
   - context_window.total_output_tokens
   - context_window.context_window_size
   - context_window.current_usage
   - context_window.used_percentage
   - context_window.remaining_percentage
2. You may reference that JSON using stdin helpers such as:
   - input=$(cat); echo "$(echo "$input" | jq -r '.model.display_name')"
   - python -c 'import json,sys; data=json.load(sys.stdin); print(data["workspace"]["current_dir"])'
3. For longer commands, you can save a helper script in ~/.koder/, such as ~/.koder/statusline-command.sh.
4. Update ~/.koder/settings.json with:
   {
     "statusLine": {
       "type": "command",
       "command": "your_command_here"
     }
   }
5. If ~/.koder/settings.json is a symlink, update the target file instead.

Guidelines:
- Preserve existing settings when updating ~/.koder/settings.json.
- Koder's current terminal UI renders the status line as a single-line bar, so keep output concise even if the user asks for something longer.
- Return a summary of what was configured, including the name of any script file you created.
- At the end of your response, tell the parent agent that the "statusline-setup" agent should be used for future status line changes.
- Also tell the user they can ask Koder to continue refining the status line."""


BUILTIN_AGENT_DEFINITIONS: tuple[AgentDefinition, ...] = (
    AgentDefinition(
        agent_type="general-purpose",
        when_to_use=(
            "General-purpose agent for researching complex questions, searching for code, "
            "and executing multi-step tasks."
        ),
        system_prompt=_general_purpose_prompt(),
        source="built-in",
        base_dir="built-in",
        tools=["*"],
    ),
    AgentDefinition(
        agent_type="Explore",
        when_to_use=(
            "Fast agent specialized for exploring codebases. Use this when you need to "
            "quickly find files, search code, or answer codebase questions."
        ),
        system_prompt=_explore_prompt(),
        source="built-in",
        base_dir="built-in",
        disallowed_tools=["AgentTool", "ExitPlanMode", "Edit", "Write"],
        model="inherit",
    ),
    AgentDefinition(
        agent_type="Plan",
        when_to_use=("Software architect agent for designing implementation plans."),
        system_prompt=_plan_prompt(),
        source="built-in",
        base_dir="built-in",
        disallowed_tools=["AgentTool", "ExitPlanMode", "Edit", "Write"],
        model="inherit",
    ),
    AgentDefinition(
        agent_type="statusline-setup",
        when_to_use="Use this agent to configure the user's Koder status line setting.",
        system_prompt=_statusline_setup_prompt(),
        source="built-in",
        base_dir="built-in",
        tools=["Read", "Edit"],
        model="inherit",
        color="orange",
    ),
    AgentDefinition(
        agent_type="verification",
        when_to_use=(
            "Use this agent to adversarially test whether a task was completed correctly. "
            "Verifies claims by running tests, checking edge cases, and trying to break "
            "the implementation. Use when about to claim work is complete."
        ),
        system_prompt=(
            "You are an adversarial verification agent. Your job is to rigorously test "
            "whether the claimed implementation actually works. Don't take anything at "
            "face value — run the tests yourself, check edge cases, try to break it. "
            "Report PASS only if you've independently verified correctness. Report FAIL "
            "with specific evidence of what's broken."
        ),
        source="built-in",
        base_dir="built-in",
        tools=["*"],
        disallowed_tools=["Edit", "Write", "NotebookEdit"],
        model="inherit",
    ),
)


def _split_tool_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [str(item).strip() for item in value if str(item).strip()]
    if isinstance(value, str):
        parts: list[str] = []
        current: list[str] = []
        depth = 0
        for char in value:
            if char == "," and depth == 0:
                item = "".join(current).strip()
                if item:
                    parts.append(item)
                current = []
                continue
            if char == "(":
                depth += 1
            elif char == ")" and depth > 0:
                depth -= 1
            current.append(char)
        item = "".join(current).strip()
        if item:
            parts.append(item)
        return parts
    return None


def _parse_frontmatter(raw: str) -> tuple[dict[str, Any], str]:
    text = raw.lstrip("\ufeff")
    match = FRONTMATTER_RE.match(text)
    if not match:
        return {}, text.strip()
    loaded = yaml.safe_load(match.group("yaml")) or {}
    if not isinstance(loaded, dict):
        loaded = {}
    return loaded, match.group("body").strip()


def _parse_agent_values(frontmatter: dict[str, Any]) -> dict[str, Any]:
    raw_tools = _split_tool_list(frontmatter.get("tools"))
    allowed_agent_types = None
    if raw_tools:
        for tool_spec in raw_tools:
            alias = tool_spec.strip()
            if alias == "Agent" or alias == "Task":
                allowed_agent_types = []
                break
            if alias.startswith("Agent(") and alias.endswith(")"):
                inside = alias[len("Agent(") : -1]
                allowed_agent_types = [part.strip() for part in inside.split(",") if part.strip()]
                break
            if alias.startswith("Task(") and alias.endswith(")"):
                inside = alias[len("Task(") : -1]
                allowed_agent_types = [part.strip() for part in inside.split(",") if part.strip()]
                break

    model_raw = frontmatter.get("model")
    model = None
    if isinstance(model_raw, str) and model_raw.strip():
        model = "inherit" if model_raw.strip().lower() == "inherit" else model_raw.strip()

    permission_mode = frontmatter.get("permissionMode")
    if permission_mode is not None and not isinstance(permission_mode, str):
        permission_mode = None

    max_turns = frontmatter.get("maxTurns")
    if max_turns is not None:
        try:
            max_turns = int(max_turns)
        except (TypeError, ValueError):
            max_turns = None
        if max_turns is not None and max_turns <= 0:
            max_turns = None

    background = frontmatter.get("background")
    if not isinstance(background, bool):
        background = None

    hooks = frontmatter.get("hooks")
    if not isinstance(hooks, dict):
        hooks = None

    mcp_servers = frontmatter.get("mcpServers")
    if not isinstance(mcp_servers, list):
        mcp_servers = None

    effort = frontmatter.get("effort")
    if effort is not None and not isinstance(effort, (str, int)):
        effort = None

    return {
        "tools": raw_tools,
        "disallowed_tools": _split_tool_list(frontmatter.get("disallowedTools")),
        "skills": _split_tool_list(frontmatter.get("skills")),
        "model": model,
        "permission_mode": permission_mode,
        "max_turns": max_turns,
        "background": background,
        "memory": frontmatter.get("memory") if isinstance(frontmatter.get("memory"), str) else None,
        "isolation": (
            frontmatter.get("isolation") if isinstance(frontmatter.get("isolation"), str) else None
        ),
        "hooks": hooks,
        "mcp_servers": mcp_servers,
        "effort": effort,
        "color": frontmatter.get("color") if isinstance(frontmatter.get("color"), str) else None,
        "initial_prompt": (
            frontmatter.get("initialPrompt")
            if isinstance(frontmatter.get("initialPrompt"), str)
            else None
        ),
        "allowed_agent_types": allowed_agent_types,
    }


def parse_agent_markdown_file(
    file_path: Path,
    *,
    base_dir: Path,
    source: AgentSource,
    plugin_name: str | None = None,
    project_root: str | Path | None = None,
    execution_cwd: str | Path | None = None,
) -> AgentDefinition | None:
    resolved_project_root = None
    resolved_execution_cwd = None
    if source == "projectSettings":
        project_root_path = (
            Path(project_root).expanduser().resolve(strict=True)
            if project_root is not None
            else MCPServerManager.project_boundary(base_dir)
        )
        execution_cwd_path = (
            Path(execution_cwd).expanduser().resolve(strict=True)
            if execution_cwd is not None
            else project_root_path
        )
        file_path = MCPServerManager.validate_project_source_path(file_path, project_root_path)
        resolved_project_root = str(project_root_path)
        resolved_execution_cwd = str(execution_cwd_path)
    raw = file_path.read_text(encoding="utf-8")
    frontmatter, body = _parse_frontmatter(raw)
    agent_name = frontmatter.get("name")
    description = frontmatter.get("description")
    if not isinstance(agent_name, str) or not agent_name.strip():
        return None
    if not isinstance(description, str) or not description.strip():
        return None

    agent_type = agent_name.strip()
    if source == "plugin" and plugin_name:
        relative_parts = file_path.relative_to(base_dir).parts[:-1]
        namespace = ":".join(part for part in relative_parts if part)
        agent_type = (
            f"{plugin_name}:{namespace}:{agent_type}"
            if namespace
            else f"{plugin_name}:{agent_type}"
        )

    values = _parse_agent_values(frontmatter)
    if source == "plugin":
        values["hooks"] = None
        values["mcp_servers"] = None
        values["permission_mode"] = None
    return AgentDefinition(
        agent_type=agent_type,
        when_to_use=description.strip().replace("\\n", "\n"),
        system_prompt=body,
        source=source,
        filename=file_path.stem,
        base_dir=str(base_dir),
        source_path=str(file_path.resolve()),
        project_root=resolved_project_root,
        execution_cwd=resolved_execution_cwd,
        plugin=plugin_name,
        **values,
    )


def parse_agent_from_json(
    name: str,
    definition: Any,
    *,
    source: AgentSource = "flagSettings",
) -> AgentDefinition | None:
    if not isinstance(definition, dict):
        return None
    description = definition.get("description")
    prompt = definition.get("prompt")
    if not isinstance(description, str) or not description.strip():
        return None
    if not isinstance(prompt, str) or not prompt.strip():
        return None

    values = _parse_agent_values(definition)
    return AgentDefinition(
        agent_type=name,
        when_to_use=description.strip(),
        system_prompt=prompt.strip(),
        source=source,
        **values,
    )


def parse_agents_from_json(
    agents_json: dict[str, Any] | None,
    *,
    source: AgentSource = "flagSettings",
) -> list[AgentDefinition]:
    if not isinstance(agents_json, dict):
        return []
    parsed: list[AgentDefinition] = []
    for name, definition in agents_json.items():
        agent = parse_agent_from_json(str(name), definition, source=source)
        if agent is not None:
            parsed.append(agent)
    return parsed


def get_active_agents_from_list(
    all_agents: list[AgentDefinition],
) -> list[AgentDefinition]:
    grouped: dict[AgentSource, list[AgentDefinition]] = {
        "built-in": [],
        "plugin": [],
        "userSettings": [],
        "projectSettings": [],
        "flagSettings": [],
    }
    for agent in all_agents:
        grouped.setdefault(agent.source, []).append(agent)

    precedence_order: tuple[AgentSource, ...] = (
        "built-in",
        "plugin",
        "userSettings",
        "projectSettings",
        "flagSettings",
    )
    active: dict[str, AgentDefinition] = {}
    for source in precedence_order:
        for agent in grouped.get(source, []):
            active[agent.agent_type] = agent
    return list(active.values())


def resolve_agent_model(
    agent_definition: AgentDefinition | None,
    *,
    main_model: str | None = None,
) -> str | None:
    subagent_override = os.environ.get("KODER_SUBAGENT_MODEL")
    if subagent_override:
        return subagent_override
    if agent_definition is None:
        return main_model
    if agent_definition.model and agent_definition.model != "inherit":
        return agent_definition.model
    return main_model


def resolve_agent_mcp_server_configs(agent_definition: AgentDefinition) -> list[MCPServerConfig]:
    if not agent_definition.mcp_servers:
        return []
    configs: list[MCPServerConfig] = []
    existing = {}
    try:
        config_path = Path.home() / ".koder" / "config.yaml"
        loaded = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        raw_servers = loaded.get("mcp_servers") or []
        existing = {
            str(server["name"]): MCPServerConfig(
                name=str(server["name"]),
                transport_type=MCPServerType(str(server["transport_type"])),
                command=server.get("command"),
                args=server.get("args") or [],
                env_vars=server.get("env_vars") or {},
                url=server.get("url"),
                headers=server.get("headers") or {},
                headers_helper=server.get("headers_helper"),
                oauth=server.get("oauth"),
                cache_tools_list=bool(server.get("cache_tools_list", False)),
                allowed_tools=server.get("allowed_tools"),
                blocked_tools=server.get("blocked_tools"),
                scope=MCPServerScope.USER,
                source_path=str(config_path),
            )
            for server in raw_servers
            if isinstance(server, dict) and "name" in server and "transport_type" in server
        }
    except Exception:
        existing = {}

    inline_mappings: dict[str, dict[str, Any]] = {}
    for spec in agent_definition.mcp_servers:
        if isinstance(spec, str):
            config = existing.get(spec)
            if config is not None:
                configs.append(config)
            continue
        if not isinstance(spec, dict):
            continue
        for name, value in spec.items():
            if not isinstance(value, dict):
                continue
            if agent_definition.source == "projectSettings":
                inline_mappings[str(name)] = value
                continue
            transport = value.get("transport_type") or value.get("type") or "stdio"
            env_vars = value.get("env_vars") or value.get("env") or {}
            headers = value.get("headers") or {}
            try:
                configs.append(
                    MCPServerConfig(
                        name=name,
                        transport_type=MCPServerType(str(transport)),
                        command=value.get("command"),
                        args=value.get("args") or [],
                        env_vars=env_vars if isinstance(env_vars, dict) else {},
                        url=value.get("url"),
                        headers=headers if isinstance(headers, dict) else {},
                        cache_tools_list=bool(value.get("cache_tools_list", False)),
                        allowed_tools=value.get("allowed_tools"),
                        blocked_tools=value.get("blocked_tools"),
                        scope=MCPServerScope.USER,
                        source_path=agent_definition.source_path,
                    )
                )
            except Exception:
                continue

    if inline_mappings:
        if not (
            agent_definition.source_path
            and agent_definition.project_root
            and agent_definition.execution_cwd
        ):
            return configs
        try:
            configs.extend(
                MCPServerManager().build_project_source_configs(
                    inline_mappings,
                    source_path=agent_definition.source_path,
                    project_root=agent_definition.project_root,
                    execution_cwd=agent_definition.execution_cwd,
                )
            )
        except (OSError, ValueError):
            pass
    return configs


def load_agent_settings(cwd: str | Path) -> dict[str, Any]:
    current = Path(cwd).resolve()
    home = Path.home().resolve()
    while True:
        candidate = settings_path(current)
        if candidate.exists():
            try:
                loaded = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    return loaded
            except Exception:
                return {}
        if current == home or current.parent == current:
            break
        if (current / ".git").exists():
            break
        current = current.parent
    return {}


def get_configured_agent_name(cwd: str | Path) -> str | None:
    settings = load_agent_settings(cwd)
    agent = settings.get("agent")
    return agent if isinstance(agent, str) and agent.strip() else None


def get_denied_agent_names(cwd: str | Path) -> set[str]:
    settings = load_agent_settings(cwd)
    permissions = settings.get("permissions")
    if not isinstance(permissions, dict):
        return set()
    deny = permissions.get("deny")
    if not isinstance(deny, list):
        return set()
    denied: set[str] = set()
    for item in deny:
        if not isinstance(item, str):
            continue
        stripped = item.strip()
        if stripped.startswith("Agent(") and stripped.endswith(")"):
            denied.add(stripped[len("Agent(") : -1].strip())
    return denied


def _project_agent_dirs(cwd: Path) -> list[Path]:
    dirs: list[Path] = []
    current = cwd.resolve()
    home = Path.home().resolve()
    while True:
        candidate = current / ".koder" / "agents"
        if candidate.is_dir():
            dirs.append(candidate)
        if current == home or current.parent == current:
            break
        git_dir = current / ".git"
        if git_dir.exists():
            break
        current = current.parent
    return dirs


def _load_markdown_agents_from_dir(
    directory: Path,
    *,
    source: AgentSource,
    plugin_name: str | None = None,
    project_root: Path | None = None,
    execution_cwd: Path | None = None,
) -> tuple[list[AgentDefinition], list[dict[str, str]]]:
    agents: list[AgentDefinition] = []
    failed: list[dict[str, str]] = []
    if not directory.exists():
        return agents, failed

    for file_path in sorted(directory.rglob("*.md")):
        try:
            agent = parse_agent_markdown_file(
                file_path,
                base_dir=directory,
                source=source,
                plugin_name=plugin_name,
                project_root=project_root,
                execution_cwd=execution_cwd,
            )
        except Exception as exc:  # pragma: no cover - defensive
            failed.append({"path": str(file_path), "error": str(exc)})
            continue
        if agent is not None:
            agents.append(agent)
    return agents, failed


def _load_plugin_agents(plugin_root: Path) -> tuple[list[AgentDefinition], list[dict[str, str]]]:
    from koder_agent.harness.plugins.lifecycle import PluginLifecycleService
    from koder_agent.harness.plugins.path_safety import (
        PluginPathError,
        open_plugin_component,
    )

    agents: list[AgentDefinition] = []
    failed: list[dict[str, str]] = []
    try:
        lifecycle = PluginLifecycleService(plugin_root)
    except (OSError, ValueError):
        return agents, failed
    failed.extend(
        {"path": str(path), "error": error} for path, error in lifecycle.manifest_errors()
    )
    for manifest, state in lifecycle.installed_plugins():
        if not state.enabled:
            continue
        plugin_dir = lifecycle.resolve_plugin_target(manifest.name)
        try:
            with open_plugin_component(
                plugin_dir,
                manifest.agents,
                default="agents",
                field_name="agents",
                expect="directory",
            ) as agents_dir:
                if agents_dir is None:
                    continue
                plugin_agents, plugin_failed = _load_markdown_agents_from_dir(
                    agents_dir,
                    source="plugin",
                    plugin_name=manifest.name,
                )
        except PluginPathError as exc:
            failed.append({"path": str(plugin_dir), "error": str(exc)})
            continue
        agents.extend(plugin_agents)
        failed.extend(plugin_failed)
    return agents, failed


def get_agent_definitions(
    *,
    cwd: str | Path,
    plugin_root: str | Path | None = None,
    cli_agents_json: dict[str, Any] | None = None,
) -> AgentDefinitionsResult:
    cwd_path = Path(cwd).resolve()
    plugin_root_path = (
        normalize_plugin_root(plugin_root) if plugin_root is not None else get_plugin_root()
    )

    builtins = list(BUILTIN_AGENT_DEFINITIONS)
    user_agents, user_failed = _load_markdown_agents_from_dir(
        user_agents_dir(),
        source="userSettings",
    )

    project_agents: list[AgentDefinition] = []
    project_failed: list[dict[str, str]] = []
    git_project_root = MCPServerManager.project_boundary(cwd_path)
    has_git_project_root = (git_project_root / ".git").exists()
    for directory in _project_agent_dirs(cwd_path):
        project_root = (
            git_project_root if has_git_project_root else directory.parent.parent.resolve()
        )
        agents, failed = _load_markdown_agents_from_dir(
            directory,
            source="projectSettings",
            project_root=project_root,
            execution_cwd=cwd_path,
        )
        project_agents.extend(agents)
        project_failed.extend(failed)

    plugin_agents, plugin_failed = _load_plugin_agents(plugin_root_path)
    flag_agents = parse_agents_from_json(cli_agents_json, source="flagSettings")

    all_agents = [*builtins, *plugin_agents, *user_agents, *project_agents, *flag_agents]
    active_agents = get_active_agents_from_list(all_agents)
    denied_names = get_denied_agent_names(cwd_path)
    if denied_names:
        active_agents = [agent for agent in active_agents if agent.agent_type not in denied_names]
    failed_files = user_failed + project_failed + plugin_failed
    return AgentDefinitionsResult(
        active_agents=active_agents,
        all_agents=all_agents,
        failed_files=failed_files or None,
    )


def render_agent_profiles(agents: list[AgentDefinition] | None = None) -> str:
    agent_list = list(agents) if agents is not None else list(BUILTIN_AGENT_DEFINITIONS)
    lines = ["profiles:"]
    for source, label in DISPLAY_SOURCE_GROUPS:
        scoped = [agent for agent in agent_list if agent.source == source]
        if not scoped:
            continue
        lines.append(f"{label}:")
        for agent in sorted(scoped, key=lambda item: item.agent_type.lower()):
            lines.append(f"- {agent.agent_type}")
    return "\n".join(lines)


def filter_tools_for_agent_definition(agent_definition: AgentDefinition, tools) -> list:
    allowed_specs = agent_definition.tools
    if not allowed_specs:
        filtered = list(tools)
    else:
        allowed_map = {
            "Read": {"read_file"},
            "Write": {"write_file", "append_file"},
            "Edit": {"edit_file", "notebook_edit"},
            "Glob": {"glob_search"},
            "Grep": {"grep_search"},
            "Bash": {"run_shell", "shell_output", "shell_kill", "git_command"},
            "WebFetch": {"web_fetch"},
            "WebSearch": {"web_search"},
            "Skill": {"get_skill"},
            "Task": {"task_delegate"},
            "Agent": {"agent_tool"},
            "SendMessage": {"send_message"},
            "TeamCreate": {"team_create"},
            "TeamDelete": {"team_delete"},
        }
        remaining = list(tools)
        if agent_definition.disallowed_tools:
            denied_names: set[str] = set()
            for spec in agent_definition.disallowed_tools:
                base = spec.split("(", 1)[0]
                denied_names.update(allowed_map.get(base, {base}))
            remaining = [tool for tool in remaining if tool.name not in denied_names]
        if "*" in allowed_specs:
            filtered = remaining
        else:
            allowed_names: set[str] = set()
            for spec in allowed_specs:
                base = spec.split("(", 1)[0]
                allowed_names.update(allowed_map.get(base, {base}))
            filtered = [tool for tool in remaining if tool.name in allowed_names]

    if agent_definition.memory and filtered:
        names = {tool.name for tool in filtered}
        for tool in tools:
            if (
                tool.name in {"read_file", "write_file", "append_file", "edit_file"}
                and tool.name not in names
            ):
                filtered.append(tool)
                names.add(tool.name)

    if agent_definition.permission_mode == "plan":
        filtered = [tool for tool in filtered if tool.name not in FILE_WRITE_TOOLS]
    return filtered


def build_agent_system_prompt(agent_definition: AgentDefinition, *, cwd: str | Path) -> str:
    parts = [agent_definition.system_prompt.strip()]

    if agent_definition.skills:
        from koder_agent.tools.skill import SkillLoader

        project_dir = Path(cwd) / ".koder" / "skills"
        user_dir = Path.home() / ".koder" / "skills"
        merged: dict[str, str] = {}
        if user_dir.exists():
            for skill in SkillLoader(user_dir).discover_skills():
                merged[skill.name] = skill.content
        if project_dir.exists():
            for skill in SkillLoader(project_dir).discover_skills():
                merged[skill.name] = skill.content
        loaded = [merged[name] for name in agent_definition.skills if name in merged]
        if loaded:
            parts.append("Preloaded skills:\n\n" + "\n\n".join(loaded))

    if agent_definition.memory:
        memory_root = {
            "user": user_agent_memory_dir(agent_definition.agent_type),
            "project": project_agent_memory_dir(cwd, agent_definition.agent_type),
            "local": local_agent_memory_dir(cwd, agent_definition.agent_type),
        }.get(agent_definition.memory)
        if memory_root is not None:
            memory_root.mkdir(parents=True, exist_ok=True)
            memory_file = memory_root / "MEMORY.md"
            snippet = ""
            if memory_file.exists():
                text = memory_file.read_text(encoding="utf-8", errors="ignore")
                lines = text.splitlines()[:200]
                snippet = "\n".join(lines)[:25_000]
            parts.append(
                "Agent memory directory: "
                f"{memory_root}\n"
                "Use this directory to store durable learnings for future runs.\n"
                + (f"Existing MEMORY.md excerpt:\n{snippet}" if snippet else "No MEMORY.md yet.")
            )

    return "\n\n".join(part for part in parts if part)


def extract_agent_mention(user_input: str) -> tuple[str, str] | None:
    stripped = user_input.strip()
    if stripped.startswith("@agent-"):
        remainder = stripped[len("@agent-") :]
        if " " in remainder:
            agent_name, prompt = remainder.split(" ", 1)
            return agent_name.strip(), prompt.strip()
        return remainder.strip(), ""
    if stripped.startswith('@"'):
        closing = stripped.find('"', 2)
        if closing != -1:
            label = stripped[2:closing]
            if label.endswith(" (agent)"):
                agent_name = label[: -len(" (agent)")].strip()
                prompt = stripped[closing + 1 :].strip()
                return agent_name, prompt
    return None
