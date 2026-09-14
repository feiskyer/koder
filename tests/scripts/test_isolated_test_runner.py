"""The reusable validation entrypoint must isolate before test collection."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).resolve().parents[2]
RUNNER = REPOSITORY / "scripts/run_isolated_tests.py"


def _write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


def _invoke(tmp_path, probe, *arguments):
    ambient_home = tmp_path / "ambient-home"
    original = _write(ambient_home / ".koder/config.yaml", "synthetic-private-profile\n")
    env_file = _write(ambient_home / ".env", "SYNTHETIC_PRIVATE=must-not-be-loaded\n")
    run_parent = tmp_path / "runs"
    run_parent.mkdir(exist_ok=True)
    environment = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(ambient_home),
        "USERPROFILE": str(ambient_home),
        "TMPDIR": str(run_parent),
        "OPENAI_API_KEY": "synthetic-inherited-credential",
        "UNKNOWN_VENDOR_API_KEY": "synthetic-inherited-credential",
        "LC_UNKNOWN_VENDOR_API_KEY": "synthetic-inherited-credential",
        "KODER_BASE_URL": "https://synthetic.invalid",
        "HTTP_PROXY": "http://synthetic.invalid",
        "PYTEST_ADDOPTS": "--not-a-real-pytest-option",
        "PYTHONPATH": str(REPOSITORY),
    }
    completed = subprocess.run(
        [sys.executable, str(RUNNER), "--test", str(probe), *arguments],
        cwd=REPOSITORY,
        env=environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )
    assert original.read_text() == "synthetic-private-profile\n"
    assert env_file.read_text() == "SYNTHETIC_PRIVATE=must-not-be-loaded\n"
    assert "synthetic-inherited-credential" not in completed.stdout + completed.stderr
    return completed


@pytest.mark.parametrize("preserve_child_home", [False, True])
@pytest.mark.parametrize("clear_parent_environment", [False, True])
def test_collection_and_child_python_are_isolated_with_explicit_child_environment(
    tmp_path, preserve_child_home, clear_parent_environment
):
    ambient = tmp_path / "ambient-home"
    proof = tmp_path / "proof.json"
    junit = tmp_path / "junit.xml"
    child_source = (
        "import os\nfrom pathlib import Path\n"
        f"ambient = Path({str(ambient)!r})\n"
        "for filename in (ambient / '.koder/config.yaml', ambient / '.env'):\n"
        "    try:\n"
        "        filename.read_text()\n"
        "    except PermissionError:\n"
        "        pass\n"
        "    else:\n"
        "        raise AssertionError('child accessed the ambient profile')\n"
        "assert os.environ['UV_NO_SYNC'] == '1'\n"
        "assert os.environ['UV_NO_ENV_FILE'] == '1'\n"
        "assert os.environ['PYTHON_DOTENV_DISABLED'] == '1'\n"
        "from dotenv import load_dotenv\n"
        "assert not load_dotenv(ambient / '.env')\n"
        "print('CHILD_ISOLATED')\n"
    )
    probe = _write(
        tmp_path / "probe" / "test_isolation.py",
        textwrap.dedent(f"""\
            import json
            import os
            import subprocess
            import sys
            from contextlib import nullcontext
            from pathlib import Path
            from unittest.mock import patch

            import pytest

            # These checks execute during collection, before autouse fixtures.
            ambient = Path({str(ambient)!r})
            assert Path.home() != ambient
            assert Path.cwd() != Path({str(REPOSITORY)!r})
            for key in ("OPENAI_API_KEY", "UNKNOWN_VENDOR_API_KEY", "LC_UNKNOWN_VENDOR_API_KEY",
                        "KODER_BASE_URL", "HTTP_PROXY", "PYTEST_ADDOPTS"):
                assert key not in os.environ, key
            for value in (ambient / ".koder/config.yaml", ambient / ".env"):
                with pytest.raises(PermissionError):
                    value.read_text()

            def test_child():
                profile = Path.home() / ".koder" / "fixture.txt"
                profile.parent.mkdir(parents=True, exist_ok=True)
                profile.write_text("writable synthetic profile")
                (Path.home() / ".env").write_text("SYNTHETIC_LOCAL=fixture")
                source = {child_source!r}
                source += "assert str(Path.home()) == " + repr(str(Path.home())) + "\\n"
                child_environment = {{"PATH": os.environ["PATH"]}}
                if {preserve_child_home!r}:
                    child_environment["HOME"] = str(Path.home())
                environment_context = (
                    patch.dict(os.environ, {{}}, clear=True)
                    if {clear_parent_environment!r}
                    else nullcontext()
                )
                with environment_context:
                    completed = subprocess.run(
                        [sys.executable, "-c", source],
                        env=child_environment,
                        capture_output=True, text=True, timeout=10, check=False,
                    )
                assert completed.returncode == 0, completed.stdout + completed.stderr
                assert completed.stdout.strip() == "CHILD_ISOLATED"
                Path({str(proof)!r}).write_text(json.dumps({{
                    "home": str(Path.home()), "cwd": str(Path.cwd()),
                    "runtime": os.environ["UV_PROJECT_ENVIRONMENT"],
                    "child": True,
                }}))
            """),
    )
    completed = _invoke(tmp_path, probe, "--junitxml", str(junit), "-q")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    receipt = json.loads(proof.read_text())
    assert receipt["runtime"] == sys.prefix
    assert receipt["child"]
    assert not Path(receipt["home"]).exists()
    assert not Path(receipt["cwd"]).exists()
    assert junit.is_file()


@pytest.mark.parametrize("helper_name", ["security", "osascript", "keychain_backend.py"])
def test_native_credential_helpers_are_rejected_before_execution(tmp_path, helper_name):
    marker = tmp_path / "helper-ran"
    helper = _write(
        tmp_path / helper_name,
        f"#!{sys.executable}\nfrom pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
    )
    helper.chmod(0o700)
    command = [sys.executable, str(helper)] if helper_name.endswith(".py") else [str(helper)]
    probe = _write(
        tmp_path / "test_helper.py",
        textwrap.dedent(f"""\
            import subprocess
            import sys
            import pytest

            def test_helper_rejected():
                command = {command!r}
                with pytest.raises(PermissionError, match="credential|Keychain"):
                    subprocess.run(command, check=False)
            """),
    )
    completed = _invoke(tmp_path, probe, "-q")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not marker.exists()


def test_guards_cover_sqlite_and_network_without_contacting_a_remote_service(tmp_path):
    ambient = tmp_path / "ambient-home"
    probe = _write(
        tmp_path / "test_guards.py",
        textwrap.dedent(f"""\
            import ctypes
            import socket
            import sqlite3
            import sys
            from pathlib import Path
            import pytest

            def test_guards():
                private_database = Path({str(ambient)!r}) / ".koder/private.db"
                with pytest.raises(PermissionError):
                    sqlite3.connect(private_database)
                with pytest.raises(PermissionError):
                    sqlite3.connect(private_database.as_uri() + "?mode=ro", uri=True)
                with pytest.raises(PermissionError):
                    ctypes.CDLL("/nonexistent/Security.framework/Security")
                # Audit events exercise the guard without any remote syscall.
                with pytest.raises(PermissionError):
                    sys.audit("socket.connect", None, ("198.51.100.1", 443))
                with pytest.raises(PermissionError):
                    sys.audit("socket.getaddrinfo", "synthetic.invalid", 443, 0, 0, 0)
                sys.audit("socket.connect", None, ("127.0.0.1", 1234))
                with socket.socket() as listener, socket.socket() as client:
                    listener.bind(("127.0.0.1", 0))
                    listener.listen(1)
                    client.connect(listener.getsockname())
                    connection, _ = listener.accept()
                    connection.close()
            """),
    )
    completed = _invoke(tmp_path, probe, "-q")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert not (ambient / ".koder/private.db").exists()


def test_failing_tests_preserve_exit_status_and_requested_artifacts(tmp_path):
    proof = tmp_path / "failure-proof.json"
    junit = tmp_path / "failure.xml"
    probe = _write(
        tmp_path / "test_failure.py",
        "import json\nfrom pathlib import Path\n"
        "def test_expected_failure():\n"
        f"    Path({str(proof)!r}).write_text(json.dumps({{'home': str(Path.home())}}))\n"
        "    assert False, 'controlled assertion failure'\n",
    )
    completed = _invoke(tmp_path, probe, "--junitxml", str(junit), "-q")
    assert completed.returncode == 1, completed.stdout + completed.stderr
    assert "controlled assertion failure" in completed.stdout
    assert junit.is_file()
    assert not Path(json.loads(proof.read_text())["home"]).exists()


@pytest.mark.parametrize("leak_loop", [False, True])
def test_runner_owns_bootstrap_loop_without_hiding_leaks(tmp_path, leak_loop):
    probe = _write(
        tmp_path / "test_loop_ownership.py",
        textwrap.dedent(f"""\
            import asyncio
            import gc
            import pytest

            @pytest.mark.asyncio
            async def test_async_before():
                await asyncio.sleep(0)

            def test_sync_runner():
                asyncio.run(asyncio.sleep(0))
                if {leak_loop!r}:
                    leaked_loop = asyncio.new_event_loop()
                    del leaked_loop
                gc.collect()

            @pytest.mark.asyncio
            async def test_async_after():
                await asyncio.sleep(0)
            """),
    )
    completed = _invoke(
        tmp_path,
        probe,
        "-q",
        "-W",
        "error::ResourceWarning",
        "-W",
        "error::pytest.PytestUnraisableExceptionWarning",
    )
    assert completed.returncode == (1 if leak_loop else 0), completed.stdout + completed.stderr
    if leak_loop:
        assert "unclosed" in completed.stdout + completed.stderr
    else:
        assert "3 passed" in completed.stdout


def test_session_flow_sync_tests_do_not_close_the_runner_bootstrap_loop(tmp_path):
    target = (
        f"{REPOSITORY / 'tests/harness/test_session_flow_stdin.py'}"
        "::test_run_harness_session_flow_uses_piped_stdin_without_args"
    )
    completed = _invoke(
        tmp_path,
        target,
        "-q",
        "-W",
        "error::RuntimeWarning",
        "-W",
        "error::ResourceWarning",
        "-W",
        "error::pytest.PytestUnraisableExceptionWarning",
    )

    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "1 passed" in completed.stdout
    assert "Event loop is closed" not in completed.stderr


def test_keep_workspace_is_explicit(tmp_path):
    proof = tmp_path / "retained.json"
    probe = _write(
        tmp_path / "test_retained.py",
        "import json\nfrom pathlib import Path\n"
        "def test_retention():\n"
        f"    Path({str(proof)!r}).write_text(json.dumps({{'home': str(Path.home())}}))\n",
    )
    completed = _invoke(tmp_path, probe, "--keep-workspace", "-q")
    assert completed.returncode == 0, completed.stdout + completed.stderr
    retained_home = Path(json.loads(proof.read_text())["home"])
    assert retained_home.is_dir()
    assert "retained" in completed.stdout.lower()


@pytest.mark.parametrize("argument", ["--basetemp", "--rootdir", "--manual"])
def test_runner_does_not_forward_workspace_or_live_mode_overrides(tmp_path, argument):
    probe = _write(tmp_path / "test_unused.py", "def test_unused(): pass\n")
    completed = _invoke(tmp_path, probe, argument, str(tmp_path / "unsafe"))
    assert completed.returncode == 2
    assert "managed by the isolated runner" in completed.stderr
