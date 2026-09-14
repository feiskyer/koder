"""Lazy sandbox backend registry for OpenAI Agents SDK sandboxes."""

from __future__ import annotations

import importlib
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Any, Callable

from .backend import SandboxBackendCapabilities, SandboxBackendStatus

DEFAULT_BACKEND_ID = "unix-local"
BACKEND_ALIASES = {
    "local": "unix-local",
    "sdk-unix-local": "unix-local",
    "unix-local": "unix-local",
    "sdk-docker": "docker",
    "docker": "docker",
    "sdk-cloudflare": "cloudflare",
    "cloudflare": "cloudflare",
    "sdk-e2b": "e2b",
    "e2b": "e2b",
    "sdk-modal": "modal",
    "modal": "modal",
    "sdk-vercel": "vercel",
    "vercel": "vercel",
}


@dataclass(frozen=True)
class BackendSpec:
    backend_id: str
    module: str
    client_class: str
    options_class: str | None = None
    credential_groups: tuple[tuple[str, ...], ...] = ()
    setup_hint: str | None = None
    validation_tier: str = "mocked-unit"
    capabilities: SandboxBackendCapabilities = SandboxBackendCapabilities()
    availability_check: Callable[[], tuple[bool, str | None]] | None = None


def _check_unix_local() -> tuple[bool, str | None]:
    if sys.platform == "win32":
        return False, "Unix local sandbox is not supported on native Windows"
    if sys.platform == "darwin":
        if shutil.which("sandbox-exec") is None:
            return False, "missing /usr/bin/sandbox-exec"
        return True, None
    # On non-darwin POSIX hosts (Linux, *BSD) the SDK unix_local backend returns
    # the command unchanged — it applies NO kernel confinement. Reporting the
    # backend as available here would let Koder claim sandboxed:true with zero
    # confinement (finding #1). Fail closed: mark it unavailable so callers must
    # pick a real sandbox backend or disable the sandbox explicitly.
    return (
        False,
        "unix-local backend provides no kernel confinement on this platform; "
        "use a real sandbox backend (docker/e2b/modal/…) or disable the sandbox",
    )


def _check_docker_daemon() -> tuple[bool, str | None]:
    docker = shutil.which("docker")
    if docker is None:
        return False, "docker CLI not found"
    try:
        subprocess.run(
            [docker, "info", "--format", "{{.ServerVersion}}"],
            check=True,
            capture_output=True,
            text=True,
            timeout=3,
        )
    except Exception as exc:
        return False, f"docker daemon unavailable: {exc}"
    return True, None


def _modal_auth_configured() -> bool:
    try:
        modal_config = importlib.import_module("modal.config").config
        return bool(modal_config.get("token_id") and modal_config.get("token_secret"))
    except Exception:
        return False


def _check_modal_auth() -> tuple[bool, str | None]:
    if _modal_auth_configured():
        return True, None
    return False, "missing Modal credentials; run `python -m modal setup` or `modal token new`"


BACKEND_SPECS: tuple[BackendSpec, ...] = (
    BackendSpec(
        backend_id="unix-local",
        module="agents.sandbox.sandboxes.unix_local",
        client_class="UnixLocalSandboxClient",
        options_class="UnixLocalSandboxClientOptions",
        setup_hint="Built into openai-agents; Koder requires macOS with /usr/bin/sandbox-exec.",
        validation_tier="required-local-smoke",
        availability_check=_check_unix_local,
        capabilities=SandboxBackendCapabilities(
            supports_shell=True,
            supports_filesystem=True,
            supports_pty="yes",
            supports_background=False,
            supports_host_process_isolation="unsupported",
            supports_workspace_isolation="not-proven",
            supports_repository_sync="not-applicable-host-workspace",
            supports_read_only_filesystem="not-proven",
            supports_network_policy="unsupported",
            supports_domain_policy="unsupported",
            supports_protected_paths="unsupported",
        ),
    ),
    BackendSpec(
        backend_id="docker",
        module="agents.sandbox.sandboxes.docker",
        client_class="DockerSandboxClient",
        options_class="DockerSandboxClientOptions",
        setup_hint="Install the Python docker package and start a reachable Docker daemon.",
        validation_tier="required-when-available",
        availability_check=_check_docker_daemon,
        capabilities=SandboxBackendCapabilities(
            supports_shell=True,
            supports_filesystem=True,
            supports_pty="yes",
            supports_background=False,
            supports_host_process_isolation="container",
            supports_workspace_isolation="not-proven",
            supports_repository_sync="unsupported",
            supports_read_only_filesystem="not-proven",
            # Network isolation depends on how the Docker container is launched.
            # The SDK docker backend does NOT enforce --network=none by default;
            # containers use Docker's default bridge network unless the caller
            # explicitly configures network mode.  Do not claim enforcement.
            supports_network_policy="not-enforced-by-default",
            supports_domain_policy="unsupported",
            supports_protected_paths="unsupported",
        ),
    ),
    BackendSpec(
        backend_id="cloudflare",
        module="agents.extensions.sandbox.cloudflare",
        client_class="CloudflareSandboxClient",
        options_class="CloudflareSandboxClientOptions",
        credential_groups=(("CLOUDFLARE_SANDBOX_WORKER_URL",),),
        setup_hint="Set CLOUDFLARE_SANDBOX_WORKER_URL and optional CLOUDFLARE_SANDBOX_API_KEY.",
        validation_tier="credential-gated-live-smoke",
    ),
    BackendSpec(
        backend_id="e2b",
        module="agents.extensions.sandbox.e2b",
        client_class="E2BSandboxClient",
        options_class="E2BSandboxClientOptions",
        credential_groups=(("E2B_API_KEY",),),
        setup_hint="Set E2B_API_KEY and optional KODER_SANDBOX_E2B_TYPE.",
        validation_tier="credential-gated-live-smoke",
        capabilities=SandboxBackendCapabilities(
            supports_shell=True,
            supports_filesystem=True,
            supports_pty="unknown",
            supports_background=False,
            supports_host_process_isolation="remote-sandbox",
            supports_workspace_isolation="not-proven",
            supports_repository_sync="unsupported",
            supports_read_only_filesystem="not-proven",
            supports_network_policy="enforced",
            supports_domain_policy="unsupported",
            supports_protected_paths="unsupported",
        ),
    ),
    BackendSpec(
        backend_id="modal",
        module="agents.extensions.sandbox.modal",
        client_class="ModalSandboxClient",
        options_class="ModalSandboxClientOptions",
        setup_hint=(
            "Install the Modal sandbox extra and run `python -m modal setup` or `modal token new`; "
            "optional KODER_SANDBOX_MODAL_APP_NAME sets the Modal app name."
        ),
        validation_tier="credential-gated-live-smoke",
        availability_check=_check_modal_auth,
    ),
    BackendSpec(
        backend_id="vercel",
        module="agents.extensions.sandbox.vercel",
        client_class="VercelSandboxClient",
        options_class="VercelSandboxClientOptions",
        credential_groups=(("VERCEL_TOKEN",),),
        setup_hint=(
            "Install the Vercel sandbox extra and set VERCEL_TOKEN; optional "
            "KODER_SANDBOX_VERCEL_PROJECT_ID and KODER_SANDBOX_VERCEL_TEAM_ID scope execution."
        ),
        validation_tier="credential-gated-live-smoke",
    ),
)

BACKEND_IDS = tuple(spec.backend_id for spec in BACKEND_SPECS)
_SPECS_BY_ID = {spec.backend_id: spec for spec in BACKEND_SPECS}


def normalize_backend_id(backend_id: str | None) -> str:
    normalized = str(backend_id or DEFAULT_BACKEND_ID).strip().lower()
    if not normalized or normalized == "auto":
        return DEFAULT_BACKEND_ID
    return BACKEND_ALIASES.get(normalized, normalized)


def get_backend_spec(backend_id: str) -> BackendSpec | None:
    return _SPECS_BY_ID.get(normalize_backend_id(backend_id))


def _import_backend(spec: BackendSpec) -> tuple[bool, tuple[str, ...]]:
    try:
        module = importlib.import_module(spec.module)
        getattr(module, spec.client_class)
        if spec.options_class:
            getattr(module, spec.options_class)
    except Exception as exc:
        return False, (f"{spec.module}: {type(exc).__name__}: {exc}",)
    return True, ()


def _credential_errors(spec: BackendSpec) -> tuple[str, ...]:
    errors: list[str] = []
    for group in spec.credential_groups:
        if not any(os.environ.get(name) for name in group):
            errors.append("missing " + " or ".join(group))
    return tuple(errors)


def get_backend_status(backend_id: str, *, selected: bool = False) -> SandboxBackendStatus:
    backend_id = normalize_backend_id(backend_id)
    spec = get_backend_spec(backend_id)
    if spec is None:
        return SandboxBackendStatus(
            backend_id=backend_id,
            selected=selected,
            available=False,
            reason="unknown backend",
            setup_hint="Choose one of: " + ", ".join(BACKEND_IDS),
        )

    imported, dependency_errors = _import_backend(spec)
    credential_errors = _credential_errors(spec) if imported else ()
    availability_error: str | None = None
    if imported and spec.availability_check is not None:
        ok, availability_error = spec.availability_check()
        if not ok and availability_error:
            dependency_errors = (*dependency_errors, availability_error)

    available = imported and not dependency_errors and not credential_errors
    if available:
        reason = "available"
    elif dependency_errors:
        reason = dependency_errors[0]
    elif credential_errors:
        reason = credential_errors[0]
    else:
        reason = availability_error or "unavailable"

    return SandboxBackendStatus(
        backend_id=backend_id,
        selected=selected,
        available=available,
        reason=reason,
        dependency_errors=dependency_errors,
        credential_errors=credential_errors,
        capabilities=spec.capabilities,
        validation_tier=spec.validation_tier,
        setup_hint=spec.setup_hint,
    )


def select_backend_id(requested: str | None) -> str:
    return normalize_backend_id(requested)


def get_backend_statuses(requested: str | None = None) -> list[SandboxBackendStatus]:
    selected = select_backend_id(requested)
    return [
        get_backend_status(backend_id, selected=backend_id == selected)
        for backend_id in BACKEND_IDS
    ]


def create_backend_client_and_options(backend_id: str, *, policy=None) -> tuple[Any, Any]:
    """Instantiate an SDK sandbox client and options for execution."""

    spec = get_backend_spec(backend_id)
    if spec is None:
        raise ValueError(f"unknown sandbox backend: {backend_id}")
    module = importlib.import_module(spec.module)
    client_class = getattr(module, spec.client_class)
    options_class = getattr(module, spec.options_class) if spec.options_class else None

    backend_id = normalize_backend_id(backend_id)
    if backend_id == "unix-local":
        return client_class(), options_class() if options_class else None
    if backend_id == "docker":
        docker = importlib.import_module("docker")
        image = os.environ.get("KODER_SANDBOX_DOCKER_IMAGE", "python:3.13-slim")
        return client_class(docker.from_env()), options_class(image=image)
    if backend_id == "cloudflare":
        worker_url = os.environ.get("CLOUDFLARE_SANDBOX_WORKER_URL", "")
        api_key = os.environ.get("CLOUDFLARE_SANDBOX_API_KEY")
        return client_class(), options_class(worker_url=worker_url, api_key=api_key)
    if backend_id == "e2b":
        sandbox_type = os.environ.get("KODER_SANDBOX_E2B_TYPE", "e2b_code_interpreter")
        allow_internet_access = True if policy is None else bool(policy.network_access)
        return client_class(), options_class(
            sandbox_type=sandbox_type,
            allow_internet_access=allow_internet_access,
        )
    if backend_id == "modal":
        app_name = os.environ.get("KODER_SANDBOX_MODAL_APP_NAME", "koder-sandbox")
        return client_class(), options_class(app_name=app_name)
    if backend_id == "vercel":
        token = os.environ.get("VERCEL_TOKEN")
        project_id = os.environ.get("KODER_SANDBOX_VERCEL_PROJECT_ID")
        team_id = os.environ.get("KODER_SANDBOX_VERCEL_TEAM_ID")
        return (
            client_class(token=token),
            options_class(project_id=project_id, team_id=team_id),
        )

    raise ValueError(f"unsupported sandbox backend: {backend_id}")
