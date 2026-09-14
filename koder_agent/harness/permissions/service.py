"""Permission service for runtime tool calls."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from ..execution_context import (
    execution_workspace_root,
    get_execution_cwd,
    scoped_execution_cwd,
)
from ..sandbox.enforcement import (
    autoapproval_blockers,
    backend_capability_digest,
    canonical_workspace_path,
    sandbox_degradation_reason,
    sandbox_policy_digest,
)
from ..sandbox.workspace import protected_write_violation, read_only_violation
from ..sandbox_settings import is_excluded_command, resolve_sandbox_settings
from .denial_log import DenialLog
from .modes import FILE_WRITE_TOOLS, PermissionMode
from .path_policy import evaluate_path_access
from .persistence import PermissionStore
from .powershell_classifier import classify_powershell_command
from .results import PermissionEvaluationResult
from .rules import (
    derive_path_prefix_rule,
    derive_shell_prefix_rule,
    match_permission_rule,
    parse_permission_rule,
)
from .shell_classifier import (
    WRITE_REDIRECTION_PATTERN,
    classify_shell_command,
    normalize_segment_for_rule,
)
from .shell_segments import parse_shell_segments
from .tool_arguments import (
    ToolArgumentError,
    extract_canonical_tool_target,
    normalize_tool_arguments,
)

if TYPE_CHECKING:
    from .ai_classifier import AiShellClassifier
    from .rule_sources import RuleHierarchy


def _empty_rules() -> dict[str, dict[str, list[str]]]:
    return {}


def _normalize_git_command(command: str) -> str:
    """Normalize a git_command argument to ``git <subcommand> ...`` form."""
    rest = command.strip()
    if rest.startswith("git"):
        rest = rest[len("git") :].strip()
    return f"git {rest}"


# Markers for command/process substitution. A prefix allow rule cannot reason
# about what runs inside these, so a command containing any of them is never
# auto-allowed by a rule (mirrors the same guard in the skill loaders).
_COMMAND_SUBSTITUTION_MARKERS = ("$(", "`", "<(", ">(", "${")


def _contains_command_substitution(command: str) -> bool:
    return any(marker in command for marker in _COMMAND_SUBSTITUTION_MARKERS)


@dataclass(frozen=True)
class _ShellRuleTarget:
    raw: str
    allow_normalized: str | None
    deny_normalized: str | None


@dataclass
class PermissionService:
    """Evaluates whether tool calls should run, ask, or deny."""

    mode: PermissionMode = PermissionMode.DEFAULT
    owner: str = "main"
    workspace_root: Path = field(default_factory=lambda: Path.cwd().resolve())
    additional_roots: list[Path] = field(default_factory=list)
    store: PermissionStore | None = None
    denial_log: DenialLog = field(default_factory=DenialLog)
    rules: dict[str, dict[str, list[str]]] = field(default_factory=_empty_rules)
    rule_hierarchy: "RuleHierarchy | None" = None
    _ai_classifier: "AiShellClassifier | None" = None

    def __post_init__(self) -> None:
        # Load rules from hierarchy if provided
        if self.rule_hierarchy is not None:
            effective_rules = self.rule_hierarchy.get_effective_rules()
            # Merge with existing rules (hierarchy takes precedence)
            for tool_name, behaviors in effective_rules.items():
                tool_rules = self.rules.setdefault(tool_name, {})
                for behavior, rule_list in behaviors.items():
                    bucket = tool_rules.setdefault(behavior, [])
                    for rule in rule_list:
                        if rule not in bucket:
                            bucket.append(rule)
        # Also merge persisted store rules (e.g. "always allow" decisions from a
        # prior session). Previously this was an ``elif`` that only ran when NO
        # hierarchy existed, so with the production setup (hierarchy + store) the
        # persisted rules were written but NEVER reloaded — always-allow silently
        # failed to survive across sessions. Merge them in addition to the
        # hierarchy so persisted decisions are honored.
        if self.store is not None:
            persisted = self.store.load().get("rules", {})
            if isinstance(persisted, dict):
                for tool_name, behaviors in persisted.items():
                    if not isinstance(behaviors, dict):
                        continue
                    tool_rules = self.rules.setdefault(tool_name, {})
                    for behavior, rule_list in behaviors.items():
                        if not isinstance(rule_list, list):
                            continue
                        bucket = tool_rules.setdefault(behavior, [])
                        for rule in rule_list:
                            if rule not in bucket:
                                bucket.append(rule)

    @classmethod
    def default(
        cls,
        *,
        mode: PermissionMode = PermissionMode.DEFAULT,
        store: PermissionStore | None = None,
        denial_log: DenialLog | None = None,
        workspace_root: Path | str | None = None,
        owner: str = "main",
        rule_hierarchy: "RuleHierarchy | None" = None,
        ai_classifier: "AiShellClassifier | None" = None,
    ) -> "PermissionService":
        return cls(
            mode=mode,
            owner=owner,
            workspace_root=(
                Path(workspace_root).resolve() if workspace_root else Path.cwd().resolve()
            ),
            store=store,
            denial_log=denial_log or DenialLog(),
            rule_hierarchy=rule_hierarchy,
            _ai_classifier=ai_classifier,
        )

    def export_rules(self) -> dict[str, dict[str, list[str]]]:
        return deepcopy(self.rules)

    def add_working_directory(self, path: Path | str) -> Path:
        normalized = Path(path).expanduser().resolve()
        if normalized != self.workspace_root and normalized not in self.additional_roots:
            self.additional_roots.append(normalized)
        return normalized

    def list_working_directories(self) -> list[Path]:
        return [self.workspace_root, *self.additional_roots]

    def add_rule(self, tool_name: str, behavior: str, rule_content: str) -> None:
        tool_rules = self.rules.setdefault(tool_name, {})
        bucket = tool_rules.setdefault(behavior, [])
        if rule_content not in bucket:
            bucket.append(rule_content)
        if self.store is not None:
            self.store.save({"rules": self.rules})

    def add_approval_rule(self, tool_name: str, arguments: dict) -> str | None:
        """Persist an "always allow" decision as a generalized rule.

        Rather than keying on the exact command/target string (so a path or flag
        change re-prompts — the permission-fatigue bug), this derives a sensible
        PREFIX rule and persists it via :meth:`add_rule` so it also survives
        across sessions through the store:

          * ``run_shell`` / ``run_powershell`` / ``git_command`` — a clearly safe
            verb widens to ``<prefix>:*`` (``npm test`` -> ``npm test:*``), so
            ``npm test --watch`` reuses it. Destructive/privileged/chained
            commands are NOT widened; they persist the exact command string.
          * file tools — a per-directory rule (``/proj/src/``) is persisted so a
            sibling edit in the same directory is auto-allowed while a file in a
            different directory still prompts.

        Returns the rule string that was persisted, or ``None`` if no rule
        target could be derived (nothing is persisted in that case).
        """
        try:
            normalized_arguments = normalize_tool_arguments(tool_name, arguments)
        except ToolArgumentError:
            return None
        target = self._extract_rule_target(tool_name, normalized_arguments)
        if not target:
            return None

        rule_content: str | None = None
        if tool_name in {"run_shell", "run_powershell", "git_command"}:
            # git_command's target is normalized to a leading "git " token by
            # _extract_rule_target, so the shell derivation sees a real command.
            rule_content = derive_shell_prefix_rule(target)
        elif tool_name in ({"read_file"} | FILE_WRITE_TOOLS):
            rule_content = derive_path_prefix_rule(target)

        # Fall back to the exact target when no safe prefix could be derived, so
        # the decision is still remembered (just not widened).
        rule_content = rule_content or target
        self.add_rule(tool_name, "allow", rule_content)
        return rule_content

    def load_settings_rules(self, settings: dict, source: str) -> None:
        """Load rules from settings dict via rule hierarchy.

        If no hierarchy exists, this is a no-op.
        """
        if self.rule_hierarchy is not None:
            self.rule_hierarchy.load_from_settings(settings, source=source)
            # Re-sync rules from hierarchy
            effective_rules = self.rule_hierarchy.get_effective_rules()
            for tool_name, behaviors in effective_rules.items():
                tool_rules = self.rules.setdefault(tool_name, {})
                for behavior, rule_list in behaviors.items():
                    bucket = tool_rules.setdefault(behavior, [])
                    for rule in rule_list:
                        if rule not in bucket:
                            bucket.append(rule)

    def _extract_rule_target(self, tool_name: str, arguments: dict) -> str | None:
        if tool_name == "git_command":
            command = arguments.get("command")
            if not isinstance(command, str):
                return None
            # git_command accepts the command with or without a leading 'git'
            # token; normalize so allow/deny rules can match a stable "git ..."
            # target regardless of how the model phrased it.
            return command if command.strip().startswith("git") else f"git {command}"
        return extract_canonical_tool_target(tool_name, arguments)

    def _match_rule(self, tool_name: str, behavior: str, target: str | None) -> str | None:
        if not target:
            return None
        targets = [target]
        if tool_name in {"read_file", *FILE_WRITE_TOOLS}:
            try:
                path = Path(target).expanduser()
                if not path.is_absolute():
                    path = (scoped_execution_cwd() or self.workspace_root) / path
                targets.append(str(path.resolve()))
            except (OSError, RuntimeError, ValueError):
                # Invalid paths are still rejected by the path/tool validators.
                pass
        for rule_content in self.rules.get(tool_name, {}).get(behavior, []):
            rule = parse_permission_rule(rule_content)
            if any(match_permission_rule(rule, candidate) for candidate in targets):
                return rule_content
        return None

    def _path_workspace_root(self) -> Path:
        # Sharing policy must not share cwd: concurrent agents keep the current
        # permission rules, but authorize paths against their own workspace.
        scoped = execution_workspace_root()
        return scoped if scoped is not None else self.workspace_root

    @staticmethod
    def _shell_segment_targets(command: str) -> list[_ShellRuleTarget] | None:
        """Split a shell command into per-segment rule targets.

        Returns raw and separate allow/deny projections per segment, using the
        classifier's quote-aware tokenizer, so allow/deny rules are matched
        against each command in a chain individually. This prevents an
        ``echo:*`` allow rule from matching ``echo hi; rm -rf ~`` and an
        ``rm:*`` deny rule from being skipped in ``ls && rm -rf x``.

        Allow normalization preserves executable paths; conservative deny
        normalization can additionally recognize basenames. Both reuse the
        canonical runner resolver. Parsing failure returns ``None``: raw deny
        rules can still veto, but no automatic allow is inferred.
        """
        if not command or not command.strip():
            return None
        try:
            segments = parse_shell_segments(command)
        except ValueError:
            return None
        targets = [
            _ShellRuleTarget(
                raw=segment.raw,
                allow_normalized=normalize_segment_for_rule(list(segment.tokens)),
                deny_normalized=normalize_segment_for_rule(list(segment.tokens), for_deny=True),
            )
            for segment in segments
        ]
        return targets or None

    def _match_shell_rules(
        self, tool_name: str, command: str, target: str | None
    ) -> PermissionEvaluationResult | None:
        """Evaluate allow/deny rules per shell segment (deny takes precedence).

        - A DENY fires if ANY segment matches a deny rule.
        - An ALLOW auto-approves ONLY if EVERY segment matches some allow rule.
        - Single-segment commands behave exactly like whole-string matching.

        Returns a decision when a rule is dispositive, else ``None`` so the
        caller continues to static classification. Read-only auto-allow is left
        to the static classifier (see ``evaluate_tool_call``).
        """
        segment_targets = self._shell_segment_targets(command)
        parsed = segment_targets is not None
        if segment_targets is None:
            segment_targets = [_ShellRuleTarget(target, None, None)] if target else []
        elif len(segment_targets) == 1 and target:
            # Retain original quoting/spacing for the raw single-command match,
            # using the already-computed projections rather than parsing again.
            original = segment_targets[0]
            segment_targets = [
                _ShellRuleTarget(target, original.allow_normalized, original.deny_normalized)
            ]

        # Deny takes precedence: any segment whose RAW or NORMALIZED form matches
        # a deny rule denies the whole command. Matching the normalized form too
        # only makes deny STRICTER — a wrapper (``env rm -rf x``) can never
        # smuggle its inner command past an ``rm`` deny.
        for segment in segment_targets:
            deny_rule = self._match_rule(tool_name, "deny", segment.raw) or self._match_rule(
                tool_name, "deny", segment.deny_normalized
            )
            if deny_rule:
                self.denial_log.record(tool_name, f"Denied by rule: {deny_rule}")
                return PermissionEvaluationResult.deny(
                    tool_name=tool_name,
                    reason=f"Denied by rule: {deny_rule}",
                    mode=self.mode,
                    matched_rule=deny_rule,
                )

        if not parsed:
            return None

        # Command/process substitution smuggles an arbitrary inner command inside
        # an otherwise benign-looking segment: ``make $(rm -rf ~)`` still starts
        # with ``make`` so a ``make:*`` allow rule would greenlight it, yet the
        # real work is the hidden ``rm -rf ~``. A prefix allow rule only vouches
        # for the VISIBLE outer command, never for whatever runs inside ``$(...)``
        # / backticks / ``<(...)``. So a command containing substitution can never
        # be auto-allowed by a rule — it falls through to static classification
        # (which already flags substitution as requires_approval). Deny is
        # unaffected (checked above), so this only ever makes the gate stricter.
        if _contains_command_substitution(command):
            return None

        # Write redirections smuggle filesystem writes behind an innocent-looking
        # segment (e.g. `echo hi > /etc/passwd` after `>` is stripped); never
        # auto-allow via a rule — fall through to static classification.
        if WRITE_REDIRECTION_PATTERN.search(command):
            return None

        # Allow only when EVERY segment is individually allowed by a rule. A
        # segment counts as allowed when its RAW or safe NORMALIZED form matches.
        # Only known argument-preserving wrappers can borrow an inner allow;
        # environment assignments and opaque runners retain their own identity.
        matched_rules: list[str] = []
        for segment in segment_targets:
            allow_rule = self._match_rule(tool_name, "allow", segment.raw) or self._match_rule(
                tool_name, "allow", segment.allow_normalized
            )
            if not allow_rule:
                matched_rules = []
                break
            matched_rules.append(allow_rule)
        if matched_rules and segment_targets:
            return PermissionEvaluationResult.allow(
                tool_name=tool_name,
                mode=self.mode,
                reason=f"Allowed by rule: {matched_rules[0]}",
                matched_rule=matched_rules[0],
            )
        return None

    def _apply_mode_override(
        self, result: PermissionEvaluationResult
    ) -> PermissionEvaluationResult:
        """Apply mode-specific overrides to results (e.g., DONT_ASK mode)."""
        if result.requires_approval and self.mode == PermissionMode.DONT_ASK:
            self.denial_log.record(result.tool_name, "denied by dontAsk mode")
            return PermissionEvaluationResult.deny(
                tool_name=result.tool_name,
                reason="dontAsk mode: approval auto-denied",
                mode=self.mode,
            )
        return result

    async def _consult_ai_classifier(self, command: str) -> PermissionEvaluationResult | None:
        """Consult AI classifier for shell command evaluation.

        The AI classifier is only consulted for commands the static classifier
        already flagged as requiring approval. It is therefore capped so it can
        only make the verdict *stricter* (DENY) or leave it unchanged
        (approval-required); it must NEVER convert a static approval requirement
        into an auto-run. Otherwise a single "safe" hallucination would let the
        model silently execute an approval-gated command.

        Returns None when the classifier is unavailable or errors, so callers
        fall back to the static classification result (an approval request)
        instead of treating classifier downtime as a denial.
        """
        if self._ai_classifier is None:
            return None

        try:
            classification = await self._ai_classifier.classify(command)

            if classification.error:
                return None

            if not classification.allowed:
                self.denial_log.record("run_shell", f"AI classifier: {classification.reason}")
                return PermissionEvaluationResult.deny(
                    tool_name="run_shell",
                    reason=f"AI classifier denied: {classification.reason}",
                    mode=self.mode,
                )

            # SAFE or MODERATE: the classifier may not downgrade a static
            # approval requirement to auto-run, so both keep approval required.
            # (The static classifier already decided this command is not
            # auto-allowable; the AI can only deny it, not release it.)
            return PermissionEvaluationResult.approval_required(
                tool_name="run_shell",
                reason=f"AI classifier requires approval: {classification.reason}",
                mode=self.mode,
            )

        except Exception:
            # Classifier crashed: fall back to the static result.
            return None

    def _evaluate_file_tool(self, tool_name: str, arguments: dict) -> PermissionEvaluationResult:
        operation = {
            "read_file": "read",
            "write_file": "write",
            "edit_file": "write",
            "append_file": "write",
            "notebook_edit": "write",
        }.get(tool_name)
        target = self._extract_rule_target(tool_name, arguments)
        if not operation or not target:
            return PermissionEvaluationResult.allow(
                tool_name=tool_name,
                mode=self.mode,
                reason="non-file tool allowed",
            )

        decision = evaluate_path_access(
            target,
            operation=operation,
            workspace_root=self._path_workspace_root(),
            additional_roots=self.additional_roots,
            working_directory=scoped_execution_cwd(),
        )
        if decision.requires_approval and self.mode == PermissionMode.ACCEPT_EDITS:
            # acceptEdits only auto-approves an ordinary workspace write. A path
            # policy that flagged the target as a dangerous file/directory (e.g.
            # .git/, .vscode/, .idea/, .koder/, dotfiles like .gitconfig) still
            # returns allowed=True + requires_approval=True; those must keep
            # their approval prompt so acceptEdits cannot silently rewrite git
            # hooks or editor configs.
            if (
                tool_name in FILE_WRITE_TOOLS
                and decision.allowed
                and decision.reason != "dangerous file or directory"
            ):
                return PermissionEvaluationResult.allow(
                    tool_name=tool_name,
                    mode=self.mode,
                    reason="acceptEdits: workspace write auto-allowed",
                )
        if decision.requires_approval and self.mode != PermissionMode.BYPASS:
            return PermissionEvaluationResult.approval_required(
                tool_name=tool_name,
                reason=decision.reason,
                mode=self.mode,
            )
        if not decision.allowed:
            self.denial_log.record(tool_name, decision.reason)
            return PermissionEvaluationResult.deny(
                tool_name=tool_name,
                reason=decision.reason,
                mode=self.mode,
            )
        return PermissionEvaluationResult.allow(
            tool_name=tool_name,
            mode=self.mode,
            reason=decision.reason,
        )

    def evaluate_tool_call(self, tool_name: str, arguments: dict) -> PermissionEvaluationResult:
        try:
            arguments = normalize_tool_arguments(tool_name, arguments)
        except ToolArgumentError as exc:
            reason = f"invalid tool arguments: {exc}"
            self.denial_log.record(tool_name, reason)
            return PermissionEvaluationResult.deny(
                tool_name=tool_name,
                reason=reason,
                mode=self.mode,
            )
        target = self._extract_rule_target(tool_name, arguments)

        # For shell tools, match allow/deny rules PER SEGMENT so an ``echo:*``
        # allow cannot green-light ``echo hi; rm -rf ~`` and an ``rm:*`` deny is
        # not skipped in ``ls && rm -rf x``. Deny wins; allow needs every
        # segment covered. Other tools keep whole-string matching.
        if tool_name in {"run_shell", "run_powershell", "git_command"} and isinstance(
            arguments.get("command"), str
        ):
            # git_command needs normalization to "git ..." for segment matching
            rule_command = arguments["command"]
            if tool_name == "git_command":
                rule_command = _normalize_git_command(rule_command)
            shell_rule_result = self._match_shell_rules(tool_name, rule_command, target)
            if shell_rule_result is not None:
                return shell_rule_result
        else:
            deny_rule = self._match_rule(tool_name, "deny", target)
            if deny_rule:
                self.denial_log.record(tool_name, f"Denied by rule: {deny_rule}")
                return PermissionEvaluationResult.deny(
                    tool_name=tool_name,
                    reason=f"Denied by rule: {deny_rule}",
                    mode=self.mode,
                    matched_rule=deny_rule,
                )

            allow_rule = self._match_rule(tool_name, "allow", target)
            if allow_rule:
                return PermissionEvaluationResult.allow(
                    tool_name=tool_name,
                    mode=self.mode,
                    reason=f"Allowed by rule: {allow_rule}",
                    matched_rule=allow_rule,
                )

        if self.mode == PermissionMode.PLAN:
            from .modes import READ_ONLY_TOOLS

            if tool_name in READ_ONLY_TOOLS:
                return PermissionEvaluationResult.allow(
                    tool_name=tool_name,
                    mode=self.mode,
                    reason="read-only tool allowed in plan mode",
                )
            return PermissionEvaluationResult.deny(
                tool_name=tool_name,
                reason="plan mode: mutations not allowed",
                mode=self.mode,
            )

        if self.mode == PermissionMode.BYPASS:
            return PermissionEvaluationResult.allow(
                tool_name=tool_name,
                mode=self.mode,
                reason="bypass mode",
            )

        if tool_name in {"run_shell", "run_powershell", "git_command"}:
            command = arguments.get("command")
            if not isinstance(command, str) or not command.strip():
                return PermissionEvaluationResult.allow(
                    tool_name=tool_name,
                    mode=self.mode,
                    reason="shell command validation deferred until invocation",
                )
            # git_command is analyzed with the bash classifier (never the
            # PowerShell one, even on Windows): its git subcommand / write-flag
            # analysis lives in classify_shell_command. Normalize the target to a
            # leading "git " token so read-only git stays allowed while
            # push --force / reset --hard become requires_approval.
            classify_command = command
            if tool_name == "git_command":
                classify_command = _normalize_git_command(command)
            current_cwd = get_execution_cwd()
            state = resolve_sandbox_settings(current_cwd)
            excluded_from_sandbox = is_excluded_command(classify_command, cwd=current_cwd)
            decision = (
                classify_powershell_command(command)
                if tool_name == "run_powershell"
                else classify_shell_command(classify_command)
            )
            if state.enabled and not excluded_from_sandbox:
                if tool_name == "run_powershell":
                    reason = (
                        "sandbox protection is unavailable for PowerShell; unsandboxed fallback "
                        "loses host-process, workspace, protected-path, and network protections "
                        "and requires explicit approval"
                    )
                    return PermissionEvaluationResult.approval_required(
                        tool_name=tool_name,
                        reason=reason,
                        mode=self.mode,
                    )
                if bool(arguments.get("run_in_background")):
                    reason = (
                        "sandbox is enabled, but background sandbox execution is not implemented "
                        "for model shell commands; run the command in the foreground, add a "
                        "sandbox exclusion with /sandbox exclude, or run /sandbox disable"
                    )
                    self.denial_log.record(tool_name, reason)
                    return PermissionEvaluationResult.deny(
                        tool_name=tool_name,
                        reason=reason,
                        mode=self.mode,
                    )

                if not state.backend_available or not state.platform_enabled:
                    reason = (
                        "sandbox backend is unavailable; approving this request permits only a "
                        "sandbox attempt. A second explicit approval is required before any "
                        "unsandboxed fallback because host-process, workspace, protected-path, "
                        f"and network protections would be lost; backend={state.backend}; "
                        f"reason={state.backend_reason}"
                    )
                    return PermissionEvaluationResult.approval_required(
                        tool_name=tool_name,
                        reason=reason,
                        mode=self.mode,
                    )
                elif tool_name in {"run_shell", "git_command"} and state.policy is not None:
                    violation = read_only_violation(classify_command, policy=state.policy)
                    if violation is None:
                        violation = protected_write_violation(
                            classify_command,
                            policy=state.policy,
                            repo_root=current_cwd,
                        )
                    if violation is not None:
                        self.denial_log.record(tool_name, violation)
                        return PermissionEvaluationResult.deny(
                            tool_name=tool_name,
                            reason=violation,
                            mode=self.mode,
                        )
                    backend_status = next(
                        (
                            status
                            for status in state.backend_statuses
                            if status.backend_id == state.backend and status.available
                        ),
                        None,
                    )
                    blockers = (
                        autoapproval_blockers(
                            state.policy,
                            backend_status.capabilities,
                        )
                        if backend_status is not None and backend_status.available
                        else ()
                    )
                    if (
                        decision.requires_approval
                        and state.auto_allow_bash_if_sandboxed
                        and backend_status is not None
                        and backend_status.available
                        and not blockers
                    ):
                        canonical_cwd = canonical_workspace_path(current_cwd)
                        return PermissionEvaluationResult.allow(
                            tool_name=tool_name,
                            mode=self.mode,
                            reason=(
                                f"sandboxed shell command auto-allowed; backend={state.backend}"
                            ),
                            sandbox_backend=state.backend,
                            sandbox_cwd=canonical_cwd,
                            sandbox_policy_digest=sandbox_policy_digest(state.policy),
                            sandbox_capability_digest=backend_capability_digest(
                                backend_status.capabilities
                            ),
                        )
                    if decision.requires_approval and blockers:
                        degradation = sandbox_degradation_reason(state.backend, blockers)
                        return self._apply_mode_override(
                            PermissionEvaluationResult.approval_required(
                                tool_name=tool_name,
                                reason=f"{decision.reason}; {degradation}",
                                mode=self.mode,
                            )
                        )
            if not decision.allowed:
                self.denial_log.record(tool_name, decision.reason)
                return PermissionEvaluationResult.deny(
                    tool_name=tool_name,
                    reason=decision.reason,
                    mode=self.mode,
                )
            if decision.requires_approval:
                return self._apply_mode_override(
                    PermissionEvaluationResult.approval_required(
                        tool_name=tool_name,
                        reason=decision.reason,
                        mode=self.mode,
                    )
                )
            return self._apply_mode_override(
                PermissionEvaluationResult.allow(
                    tool_name=tool_name,
                    mode=self.mode,
                    reason=decision.reason,
                )
            )

        if tool_name in ({"read_file"} | FILE_WRITE_TOOLS):
            return self._apply_mode_override(self._evaluate_file_tool(tool_name, arguments))

        return self._apply_mode_override(
            PermissionEvaluationResult.allow(
                tool_name=tool_name,
                mode=self.mode,
                reason="tool allowed by default",
            )
        )

    async def evaluate_tool_call_async(
        self, tool_name: str, arguments: dict
    ) -> PermissionEvaluationResult:
        """Async version of evaluate_tool_call that can consult AI classifier.

        For shell commands, if static classifier returns ambiguous (requires_approval),
        and AI classifier is available, consult it.
        """
        # First run static evaluation
        static_result = self.evaluate_tool_call(tool_name, arguments)

        # If it's a shell command that requires approval and AI classifier is available
        if (
            tool_name == "run_shell"
            and static_result.requires_approval
            and self._ai_classifier is not None
            and not static_result.matched_rule  # No explicit rule matched
        ):
            command = arguments.get("command", "")
            ai_result = await self._consult_ai_classifier(command)
            if ai_result is not None:
                return self._apply_mode_override(ai_result)

        return static_result
