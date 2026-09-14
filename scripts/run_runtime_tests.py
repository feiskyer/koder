"""Test checkout code against the dependencies of the koder command on PATH.

The application environment is never synced or modified. uv adds only the test
tools in a disposable overlay, constrained to the versions read from the actual
runtime on this invocation. Verify the overlay before collecting any tests.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = Path(__file__).resolve()


def runtime_snapshot() -> dict:
    names = {
        re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        for dist in importlib.metadata.distributions()
        if dist.metadata["Name"]
    }
    return {
        "python": list(sys.version_info[:3]),
        "implementation": sys.implementation.name,
        "executable": sys.executable,
        # Resolve in import precedence order, including uv's test-tool overlay.
        "packages": {name: importlib.metadata.version(name) for name in sorted(names)},
    }


def version_differences(expected: dict, actual: dict) -> list[str]:
    differences = []
    for key in ("python", "implementation"):
        if expected[key] != actual[key]:
            differences.append(f"{key}: runtime={expected[key]!r}, tests={actual[key]!r}")
    for name, version in expected["packages"].items():
        observed = actual["packages"].get(name)
        if observed != version:
            differences.append(f"{name}: runtime={version}, tests={observed or 'missing'}")
    return differences


def resolve_runtime_python(koder: str | None, explicit: str | None) -> str:
    if explicit:
        # Do not resolve the interpreter's symlink to its base Python: doing so
        # would silently discard the application's virtualenv.
        interpreter = Path(explicit).expanduser().absolute()
    else:
        command = shutil.which(koder or "koder")
        if command is None:
            raise ValueError("koder is not on PATH; supply --runtime-python /path/to/bin/python")
        entrypoint = Path(command).resolve()
        if entrypoint.suffix.lower() == ".exe":
            interpreter = entrypoint.with_name("python.exe")
        else:
            with entrypoint.open("rb") as handle:
                first_line = handle.readline(4096).decode("utf-8").strip()
            arguments = shlex.split(first_line[2:]) if first_line.startswith("#!") else []
            if (
                len(arguments) != 1
                or not Path(arguments[0]).is_absolute()
                or not re.fullmatch(r"(?:python|pypy)[\d.]*", Path(arguments[0]).name)
            ):
                raise ValueError(
                    "Cannot identify koder's Python from its launcher; supply --runtime-python"
                )
            interpreter = Path(arguments[0])
    if not interpreter.is_file():
        raise ValueError(f"Runtime interpreter does not exist: {interpreter}")
    return str(interpreter)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument("--koder", help="Installed koder launcher (default: koder on PATH)")
    parser.add_argument("--runtime-python", help="Explicit application virtualenv interpreter")
    parser.add_argument("--runtime-report", type=Path, help="Save verified dependency versions")
    parser.add_argument("--describe-runtime", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--verify-runtime", type=Path, help=argparse.SUPPRESS)
    options, test_args = parser.parse_known_args(argv)
    if options.describe_runtime:
        print(json.dumps(runtime_snapshot()))
        return 0
    if options.verify_runtime:
        expected = json.loads(options.verify_runtime.read_text(encoding="utf-8"))
        actual = runtime_snapshot()
        differences = version_differences(expected, actual)
        if differences:
            print("Runtime dependency mismatch; tests were NOT collected:", file=sys.stderr)
            print("\n".join(differences), file=sys.stderr)
            return 2
        if options.runtime_report:
            options.runtime_report.write_text(
                json.dumps({"runtime": expected, "tests": actual}, indent=2) + "\n",
                encoding="utf-8",
            )
        print(f"Runtime parity verified: {expected['executable']}", flush=True)
        print(
            "Versions: "
            + ", ".join(
                f"{name}={actual['packages'].get(name, 'missing')}"
                for name in ("openai-agents", "mcp", "litellm", "openai", "anyio")
            ),
            flush=True,
        )
        sys.path.insert(0, str(ROOT))
        from scripts.run_isolated_tests import main as run_isolated

        return run_isolated(test_args)

    if not options.koder and not options.runtime_python:
        parser.error(
            "Use bash scripts/test.sh, or select --koder/--runtime-python explicitly. "
            "uv may prepend a different koder to PATH before this Python script starts."
        )
    try:
        interpreter = resolve_runtime_python(options.koder, options.runtime_python)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    uv = shutil.which("uv")
    if uv is None:
        parser.error("uv is required")
    environment = dict(os.environ)
    for key in (
        "VIRTUAL_ENV",
        "PYTHONPATH",
        "PYTHONHOME",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "UV_PROJECT",
        "UV_FROZEN",
        "UV_LOCKED",
        "UV_ENV_FILE",
        "UV_NO_SYNC",
    ):
        environment.pop(key, None)
    base = [
        uv,
        "run",
        "--no-project",
        "--no-config",
        "--no-env-file",
        "--python",
        interpreter,
    ]
    probe = subprocess.run(
        [*base, "python", str(SCRIPT), "--describe-runtime"],
        env=environment,
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    if probe.returncode:
        print(probe.stderr, file=sys.stderr)
        return probe.returncode
    snapshot = json.loads(probe.stdout)
    if "koder" not in snapshot["packages"]:
        parser.error("Selected interpreter does not contain an installed koder distribution")
    with tempfile.TemporaryDirectory(prefix="koder-runtime-tests-") as directory:
        root = Path(directory)
        manifest = root / "runtime.json"
        manifest.write_text(json.dumps(snapshot), encoding="utf-8")
        constraints = root / "runtime-constraints.txt"
        constraints.write_text(
            "".join(f"{name}=={version}\n" for name, version in snapshot["packages"].items()),
            encoding="utf-8",
        )
        requirements = root / "test-tools.txt"
        requirements.write_text(
            f"-c {constraints}\npytest>=9.0.3\npytest-asyncio>=1.3.0\n", encoding="utf-8"
        )
        command = [
            *base,
            "--with-requirements",
            str(requirements),
            "python",
            str(SCRIPT),
            "--verify-runtime",
            str(manifest),
        ]
        if options.runtime_report:
            command.extend(["--runtime-report", str(options.runtime_report.resolve())])
        return subprocess.run([*command, *test_args], env=environment, check=False).returncode


if __name__ == "__main__":
    raise SystemExit(main())
