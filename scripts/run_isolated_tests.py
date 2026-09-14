"""Run non-E2E pytest checks without using the developer's profile or API keys.

This is an accidental-access guard for trusted tests, not an OS sandbox.
Python audit hooks do not mediate arbitrary native code, inherited file
descriptors, or interpreters that deliberately disable site initialization.
"""

from __future__ import annotations

import argparse
import asyncio
import inspect
import ipaddress
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

PROJECT_ROOT = Path(__file__).resolve().parents[1]
_STATE_ATTRIBUTE = "_koder_isolated_test_state"
_SAFE_ENVIRONMENT_KEYS = {
    "PATH",
    "LANG",
    "LC_ALL",
    "LC_COLLATE",
    "LC_CTYPE",
    "LC_MESSAGES",
    "LC_MONETARY",
    "LC_NUMERIC",
    "LC_TIME",
    "TERM",
    "TZ",
    "SYSTEMROOT",
    "SystemRoot",
    "WINDIR",
    "COMSPEC",
    "PATHEXT",
}
_MANAGED_ARGUMENTS = {"--basetemp", "--rootdir", "--manual", "--confcutdir", "-c"}


def _runtime_environment() -> dict[str, str]:
    return {
        "UV_PROJECT_ENVIRONMENT": sys.prefix,
        "UV_PYTHON": sys.executable,
        "UV_NO_SYNC": "1",
        "UV_NO_ENV_FILE": "1",
        "UV_NO_CONFIG": "1",
        "UV_OFFLINE": "1",
        "UV_PYTHON_DOWNLOADS": "never",
        # Newer LiteLLM imports call load_dotenv() themselves. uv's flag only
        # controls uv, not dependency-level stack-based .env discovery.
        "PYTHON_DOTENV_DISABLED": "1",
    }


def _isolated_environment(workspace: Path, bootstrap: Path) -> dict[str, str]:
    # Read only explicitly retained values; unknown vendor credentials are not
    # copied into the new environment and cannot evade a provider-prefix list.
    environment = {key: os.environ[key] for key in os.environ if key in _SAFE_ENVIRONMENT_KEYS}
    environment.update(_runtime_environment())
    environment.update(
        HOME=str(workspace / "home"),
        USERPROFILE=str(workspace / "home"),
        TMPDIR=str(workspace / "tmp"),
        TEMP=str(workspace / "tmp"),
        TMP=str(workspace / "tmp"),
        XDG_CONFIG_HOME=str(workspace / "config"),
        XDG_CACHE_HOME=str(workspace / "cache"),
        XDG_DATA_HOME=str(workspace / "data"),
        APPDATA=str(workspace / "config"),
        LOCALAPPDATA=str(workspace / "data"),
        PYTHONPATH=os.pathsep.join((str(bootstrap), str(PROJECT_ROOT))),
        PYTHONUTF8="1",
        PYTHONDONTWRITEBYTECODE="1",
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_CONFIG_SYSTEM=os.devnull,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_TERMINAL_PROMPT="0",
        PIP_CONFIG_FILE=os.devnull,
        AWS_EC2_METADATA_DISABLED="true",
        OPENAI_AGENTS_DISABLE_TRACING="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        OTEL_SDK_DISABLED="true",
    )
    return environment


def _audit_path(value) -> tuple[Path, Path] | None:
    if not isinstance(value, (str, bytes, os.PathLike)):
        return None
    lexical = Path(os.path.abspath(os.fsdecode(value)))
    return lexical, lexical.resolve()


def _check_private_path(value, configurations: list[dict[str, str]]) -> None:
    paths = _audit_path(value)
    if paths is None:
        return
    for config in configurations:
        workspace = Path(config["workspace"])
        if all(path.is_relative_to(workspace) for path in paths):
            continue
        home = Path(config["home"])
        private_roots = (
            home / ".koder",
            home / ".askllm",
            home / ".aws",
            home / ".ssh",
            home / ".config/gcloud",
            Path(config["repository"]) / ".koder",
        )
        for path in paths:
            if (
                path.name == ".env"
                or path.name.startswith(".env.")
                or path in {home / ".netrc", home / ".git-credentials"}
                or any(path.is_relative_to(root) for root in private_roots)
            ):
                raise PermissionError("Private user configuration is excluded from isolated tests")


def _loopback(host) -> bool:
    text = os.fsdecode(host).rstrip(".").lower()
    if text == "localhost":
        return True
    try:
        address = ipaddress.ip_address(text)
    except ValueError:
        return False
    mapped = getattr(address, "ipv4_mapped", None)
    return address.is_loopback or (mapped is not None and mapped.is_loopback)


def _audit(event, arguments, state) -> None:
    if event in {"open", "os.listdir", "os.scandir", "os.remove", "os.rmdir", "os.mkdir"}:
        if arguments:
            _check_private_path(arguments[0], state["configurations"])
    elif event in {"os.rename", "os.link", "os.symlink"}:
        for path in arguments[:2]:
            _check_private_path(path, state["configurations"])
    elif event == "sqlite3.connect" and arguments:
        database = os.fsdecode(arguments[0])
        if database.startswith("file:"):
            database = unquote(urlsplit(database).path)
        if database != ":memory:":
            _check_private_path(database, state["configurations"])
    elif event == "ctypes.dlopen" and arguments and arguments[0]:
        library = os.fsdecode(arguments[0])
        if "Security.framework" in library or Path(library).name == "Security":
            raise PermissionError("Native Keychain access is excluded from isolated tests")
    elif event == "subprocess.Popen" and arguments:
        executable = Path(os.fsdecode(arguments[0])).name
        command = arguments[1] if len(arguments) > 1 else []
        if isinstance(command, (str, bytes)):
            command = [command]
        native_helper = any(
            "koder_agent.auth.keychain_backend" in os.fsdecode(part)
            or Path(os.fsdecode(part)).name == "keychain_backend.py"
            for part in command
            if isinstance(part, (str, bytes, os.PathLike))
        )
        if executable in {"security", "osascript"} or native_helper:
            raise PermissionError("Native credential helpers are excluded from isolated tests")
    elif event == "socket.connect" and len(arguments) > 1:
        address = arguments[1]
        if isinstance(address, tuple) and not _loopback(address[0]):
            raise PermissionError("Remote sockets are excluded from isolated tests")
        if isinstance(address, (str, bytes)):
            _check_private_path(address, state["configurations"])
    elif event in {"socket.getaddrinfo", "socket.gethostbyname"} and arguments:
        host = arguments[0]
        if host is not None and not _loopback(host):
            try:
                ipaddress.ip_address(os.fsdecode(host))
            except ValueError:
                raise PermissionError("Remote DNS is excluded from isolated tests") from None


def install_child_guards(configurations, bootstrap: str, runtime: dict[str, str]) -> None:
    """Install once per interpreter, including normally initialized Python children.

    Keep nested runner policies rather than letting a synthetic parent HOME
    replace the original real-profile restriction.
    """
    state = getattr(sys, _STATE_ATTRIBUTE, None)
    if state is None:
        state = {"configurations": [], "bootstrap": bootstrap, "runtime": runtime}
        setattr(sys, _STATE_ATTRIBUTE, state)
        sys.addaudithook(lambda event, arguments: _audit(event, arguments, state))
        original_popen = subprocess.Popen
        signature = inspect.signature(original_popen)

        class GuardedPopen(original_popen):
            def __init__(self, *args, **kwargs):
                bound = signature.bind(*args, **kwargs)
                supplied = bound.arguments.get("env")
                environment = {
                    os.fsdecode(key): os.fsdecode(value)
                    for key, value in (os.environ if supplied is None else supplied).items()
                }
                environment.update(state["runtime"])
                # Supplying env={} must not make Python fall back to the
                # account database's real home directory in a child.
                workspace = Path(state["configurations"][-1]["workspace"])
                environment.setdefault("HOME", os.environ.get("HOME") or str(workspace / "home"))
                environment.setdefault("USERPROFILE", environment["HOME"])
                for key in ("TMPDIR", "TEMP", "TMP"):
                    environment.setdefault(key, os.environ.get(key) or str(workspace / "tmp"))
                paths = [
                    state["bootstrap"],
                    str(PROJECT_ROOT),
                    *environment.get("PYTHONPATH", "").split(os.pathsep),
                ]
                environment["PYTHONPATH"] = os.pathsep.join(dict.fromkeys(filter(None, paths)))
                bound.arguments["env"] = environment
                super().__init__(*bound.args, **bound.kwargs)

        subprocess.Popen = GuardedPopen
    for config in configurations:
        if config not in state["configurations"]:
            state["configurations"].append(config)
    state["bootstrap"] = bootstrap
    state["runtime"] = runtime


def _prepare_bootstrap(workspace: Path, original_home: Path) -> Path:
    bootstrap = workspace / "bootstrap"
    bootstrap.mkdir()
    previous = getattr(sys, _STATE_ATTRIBUTE, None)
    configurations = list(previous["configurations"]) if previous is not None else []
    configurations.append(
        {
            "home": str(original_home.resolve()),
            "repository": str(PROJECT_ROOT),
            "workspace": str(workspace),
        }
    )
    runtime = _runtime_environment()
    (bootstrap / "sitecustomize.py").write_text(
        "from scripts.run_isolated_tests import install_child_guards\n"
        f"install_child_guards({configurations!r}, {str(bootstrap)!r}, {runtime!r})\n",
        encoding="utf-8",
    )
    install_child_guards(configurations, str(bootstrap), runtime)
    return bootstrap


def _cleanup_workspace(workspace: Path, original_cwd: Path) -> None:
    # rmtree uses descriptor-relative names, but Python's open audit event
    # omits dir_fd. Keep those names anchored inside the owned workspace while
    # the standard library performs its symlink-safe deletion.
    cleanup_cwd = Path(tempfile.mkdtemp(prefix="cleanup-", dir=workspace))
    try:
        os.chdir(cleanup_cwd)
        for path in list(workspace.iterdir()):
            if path == cleanup_cwd:
                continue
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink()
    finally:
        os.chdir(original_cwd)
    cleanup_cwd.rmdir()
    workspace.rmdir()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--test", action="append", default=[], help="Test path/node ID; repeatable")
    parser.add_argument("--junitxml", type=Path, help="Write the result outside disposable HOME")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep synthetic test files")
    options, pytest_options = parser.parse_known_args(argv)
    if any(arg.split("=")[0] in _MANAGED_ARGUMENTS for arg in pytest_options):
        parser.error(
            "Workspace, root configuration and live mode are managed by the isolated runner"
        )
    if sys.prefix == sys.base_prefix:
        parser.error("Use an existing synced virtualenv (uv run --no-env-file python ...)")

    targets = []
    for selector in options.test or [str(PROJECT_ROOT / "tests")]:
        path, separator, node_id = selector.partition("::")
        target = Path(path).expanduser().resolve()
        if not target.exists():
            parser.error(f"Test path does not exist: {path}")
        if target.is_relative_to(PROJECT_ROOT / "tests/e2e"):
            parser.error("TUI tests use the separate real-terminal scenario runner")
        targets.append(str(target) + (separator + node_id if separator else ""))
    junit = options.junitxml.expanduser().resolve() if options.junitxml else None
    original_cwd, original_home = Path.cwd(), Path.home()
    workspace = Path(tempfile.mkdtemp(prefix="koder-isolated-tests-")).resolve()
    try:
        for name in ("home", "tmp", "config", "cache", "data", "work"):
            (workspace / name).mkdir()
        bootstrap = _prepare_bootstrap(workspace, original_home)
        environment = _isolated_environment(workspace, bootstrap)
        os.environ.clear()
        os.environ.update(environment)
        tempfile.tempdir = None
        sys.dont_write_bytecode = True
        os.chdir(workspace / "work")
        sys.path.insert(0, str(PROJECT_ROOT))
        import pytest

        arguments = [
            "-c",
            str(PROJECT_ROOT / "pyproject.toml"),
            "--rootdir",
            str(PROJECT_ROOT),
            "--basetemp",
            str(workspace / "pytest"),
            "-o",
            f"cache_dir={workspace / 'cache/pytest'}",
            "-p",
            "pytest_asyncio.plugin",
            "-p",
            "anyio.pytest_plugin",
            "-W",
            "error::pytest.PytestUnhandledThreadExceptionWarning",
            f"--ignore={PROJECT_ROOT / 'tests/e2e'}",
            *targets,
            *pytest_options,
        ]
        if junit is not None:
            arguments.extend(["--junitxml", str(junit)])
        print(f"Isolated tests: {workspace}", flush=True)
        # pytest-asyncio saves the default loop before creating its own runners.
        # Own that initial loop explicitly so mixed asyncio.run() tests cannot
        # orphan an implicitly created loop on Python 3.11.
        with asyncio.Runner():
            return int(pytest.main(arguments))
    finally:
        if options.keep_workspace:
            os.chdir(original_cwd)
            print(f"Test workspace retained: {workspace}", flush=True)
        else:
            _cleanup_workspace(workspace, original_cwd)


if __name__ == "__main__":
    raise SystemExit(main())
