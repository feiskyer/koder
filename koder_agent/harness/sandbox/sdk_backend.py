"""OpenAI Agents SDK sandbox execution adapter."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from dataclasses import dataclass, replace
from pathlib import Path

from koder_agent.harness.session_env import is_probably_secret_env_name
from koder_agent.utils.async_tasks import await_owned_task

from .backend import SandboxExecutionContext, SandboxExecutionResult
from .enforcement import autoapproval_blockers, sandbox_degradation_reason
from .policy import SandboxPolicy
from .registry import create_backend_client_and_options, get_backend_status, select_backend_id
from .workspace import protected_write_violation, read_only_violation

logger = logging.getLogger(__name__)


def _decode(value: bytes | str | None) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return value


def _scrub_env(env: dict[str, str]) -> dict[str, str]:
    """Drop secret-looking vars from the env before it enters the sandbox.

    Callers assemble the sandbox env from the host process (which carries API
    keys and tokens). Forwarding those wholesale would leak host credentials
    into sandboxed commands (finding #2), so strip anything that looks like a
    secret here regardless of how the caller built the env.
    """
    return {key: value for key, value in env.items() if not is_probably_secret_env_name(key)}


DELETE_TIMEOUT_SECONDS = 5.0


def _build_manifest(root: Path, env: dict[str, str], *, backend_id: str):
    from agents.sandbox.manifest import Environment, Manifest

    manifest_root = "/workspace" if backend_id == "cloudflare" else str(root)
    return Manifest(root=manifest_root, environment=Environment(value=_scrub_env(env)))


async def _delete_created_session(client, session) -> str | None:
    """Apply one deletion deadline and settle the provider's cancellation.

    The owning request shields cleanup from repeated caller cancellation.
    As with asyncio.wait_for, a provider that ignores cancellation can outlast
    this deadline; abandoning its still-running task would lose ownership.
    """
    try:
        await asyncio.wait_for(client.delete(session), timeout=DELETE_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        return "sandbox cleanup timed out; provider resources may remain"
    except Exception:
        # Provider exceptions can contain credential-bearing request details.
        return "sandbox cleanup failed; provider resources may remain"
    return None


@dataclass
class _SandboxLifetime:
    """Request-local cancellation and cleanup state; no cross-request globals."""

    cancellation_requested: bool = False
    closing: bool = False
    cleanup_error: str | None = None

    def check_cancellation(self) -> None:
        if self.cancellation_requested:
            raise asyncio.CancelledError


async def execute_with_sdk_backend(
    context: SandboxExecutionContext,
) -> SandboxExecutionResult:
    """Run a foreground shell command through the selected SDK sandbox backend."""

    policy = context.policy
    if not isinstance(policy, SandboxPolicy):
        return SandboxExecutionResult(
            status="error",
            reason="invalid sandbox policy",
            sandboxed=False,
        )

    backend_id = select_backend_id(policy.backend)
    status = get_backend_status(backend_id, selected=True)
    if not status.available:
        return SandboxExecutionResult(
            status="unavailable",
            backend_id=backend_id,
            sandboxed=False,
            reason="; ".join(status.unavailable_reasons) or status.reason,
        )

    if context.background:
        return SandboxExecutionResult(
            status="unsupported",
            backend_id=backend_id,
            sandboxed=False,
            reason="background sandbox execution is not implemented for this backend",
        )

    violation = read_only_violation(context.command, policy=policy)
    if violation is None:
        violation = protected_write_violation(
            context.command,
            policy=policy,
            repo_root=context.repo_root,
        )
    if violation is not None:
        return SandboxExecutionResult(
            status="policy_violation",
            backend_id=backend_id,
            sandboxed=False,
            created=False,
            executed=False,
            violation=violation,
            reason=violation,
        )

    blockers = autoapproval_blockers(policy, status.capabilities)
    if blockers and not context.degradation_approved:
        reason = sandbox_degradation_reason(backend_id, blockers)
        return SandboxExecutionResult(
            status="policy_violation",
            backend_id=backend_id,
            sandboxed=False,
            created=False,
            executed=False,
            violation=reason,
            reason=reason,
        )

    lifetime = _SandboxLifetime()
    work = asyncio.create_task(_run_owned_sandbox(context, backend_id, lifetime))
    try:
        result = await asyncio.shield(work)
    except asyncio.CancelledError as original:
        lifetime.cancellation_requested = True
        if not lifetime.closing:
            work.cancel()
        with contextlib.suppress(Exception, asyncio.CancelledError):
            await await_owned_task(work)
        raise original

    if lifetime.cleanup_error is not None:
        result = replace(
            result,
            status="error",
            reason="\n".join(filter(None, (result.reason, lifetime.cleanup_error))),
            stderr="\n".join(filter(None, (result.stderr, lifetime.cleanup_error))),
        )
    return result


async def _run_owned_sandbox(
    context: SandboxExecutionContext,
    backend_id: str,
    lifetime: _SandboxLifetime,
) -> SandboxExecutionResult:
    """Keep SDK entry, execution and context exit in the same owned task."""

    client = None
    session = None
    created = False
    executed = False
    outcome = None
    try:
        lifetime.check_cancellation()
        client, options = create_backend_client_and_options(backend_id, policy=context.policy)
        manifest = _build_manifest(context.cwd, context.env, backend_id=backend_id)
        session = await client.create(manifest=manifest, options=options)
        created = True
        # Some provider coroutines finish successfully after receiving cancel.
        # A returned handle must be deleted, not used to start the command.
        lifetime.check_cancellation()
        async with session:
            try:
                lifetime.check_cancellation()
                executed = True
                result = await session.exec(context.command, timeout=context.timeout, shell=True)
                lifetime.check_cancellation()
                # Freeze the command result before context exit, which may
                # itself fail while persisting state or closing dependencies.
                outcome = SandboxExecutionResult(
                    status="success" if result.exit_code == 0 else "error",
                    stdout=_decode(result.stdout),
                    stderr=_decode(result.stderr),
                    exit_code=result.exit_code,
                    backend_id=backend_id,
                    sandboxed=True,
                    created=True,
                    executed=True,
                )
            finally:
                # __aexit__ may persist workspace state and release scoped
                # dependencies. Further caller cancels must not interrupt it.
                lifetime.closing = True
        lifetime.check_cancellation()
        if outcome is None:
            # A context manager suppressing an execution exception cannot
            # manufacture a successful command result.
            return SandboxExecutionResult(
                status="error",
                backend_id=backend_id,
                sandboxed=created,
                created=created,
                executed=executed,
                reason="sandbox command produced no result",
            )
        return outcome
    except Exception as exc:
        if outcome is not None:
            error = "sandbox context cleanup failed"
            logger.warning(error)
            return replace(
                outcome,
                status="error",
                reason=error,
                stderr="\n".join(filter(None, (outcome.stderr, error))),
            )
        return SandboxExecutionResult(
            status="error",
            backend_id=backend_id,
            sandboxed=created,
            created=created,
            executed=executed,
            reason=f"{type(exc).__name__}: {exc}",
            stderr=f"{type(exc).__name__}: {exc}",
        )
    finally:
        lifetime.closing = True
        if client is not None and session is not None:
            lifetime.cleanup_error = await _delete_created_session(client, session)
            if lifetime.cleanup_error is not None:
                logger.warning(lifetime.cleanup_error)
