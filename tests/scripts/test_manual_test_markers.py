"""The manual opt-in policy applies to marks, not parametrized command names."""

import subprocess
import sys
import textwrap
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("enable_manual", [False, True], ids=["default", "explicit_opt_in"])
def test_manual_policy_uses_explicit_and_inherited_markers(tmp_path, enable_manual):
    project = tmp_path / "synthetic-tests"
    project.mkdir()
    (project / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    (project / "conftest.py").write_text(
        (PROJECT_ROOT / "tests/integration/conftest.py").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    (project / "test_commands.py").write_text(
        textwrap.dedent("""
            import pytest

            @pytest.mark.parametrize("route", ["auto", "manual"])
            def test_local_command(route):
                assert route in {"auto", "manual"}

            @pytest.mark.manual
            def test_explicit_opt_in():
                pass

            @pytest.mark.manual
            class TestInheritedOptIn:
                def test_class_marker(self):
                    pass
            """),
        encoding="utf-8",
    )
    (project / "test_marked_module.py").write_text(
        "import pytest\npytestmark = pytest.mark.manual\ndef test_module_marker():\n    pass\n",
        encoding="utf-8",
    )
    receipt = project / "results.xml"
    command = [
        "uv",
        "run",
        "--no-project",
        "--no-sync",
        "--no-env-file",
        "--python",
        sys.executable,
        "python",
        "-m",
        "pytest",
        "-c",
        str(project / "pytest.ini"),
        "--confcutdir",
        str(project),
        "--rootdir",
        str(project),
        "--junitxml",
        str(receipt),
        "-q",
        str(project),
    ]
    if enable_manual:
        # Only synthetic pass-only tests exist in this child project. Never
        # enable this option for the repository's actual OAuth integration tests.
        command.append("--manual")
    result = subprocess.run(command, cwd=project, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stdout + result.stderr

    cases = list(ET.parse(receipt).getroot().iter("testcase"))
    assert len(cases) == 5
    skipped = {case.get("name") for case in cases if case.find("skipped") is not None}
    assert skipped == (
        set()
        if enable_manual
        else {"test_explicit_opt_in", "test_class_marker", "test_module_marker"}
    )
    for case in cases:
        if case.get("name", "").startswith("test_local_command"):
            assert case.find("skipped") is None
