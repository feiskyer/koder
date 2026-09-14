"""Sed programs are a language boundary, not just an in-place flag check."""

import os
import shlex
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from koder_agent.harness.permissions import service as service_module
from koder_agent.harness.permissions.modes import PermissionMode
from koder_agent.harness.permissions.service import PermissionService
from koder_agent.harness.permissions.shell_classifier import classify_shell_command
from koder_agent.harness.sandbox.policy import SandboxPolicy
from koder_agent.harness.sandbox.workspace import read_only_violation

APPROVAL_COMMANDS = (
    "sed -n 'w output.txt' sample.txt",
    "sed -n '1,2w output.txt' sample.txt",
    "sed -n '/old/w output.txt' sample.txt",
    "sed -n 'p;w output.txt' sample.txt",
    "sed -n '1{w output.txt;}' sample.txt",
    "sed 's/old/new/w output.txt' sample.txt",
    "sed -e p -e 'w output.txt' sample.txt",
    "sed -new\\ output.txt sample.txt",
    "sed -n -f program.sed sample.txt",
    "sed --file=program.sed sample.txt",
    "sed --in-pl -e p sample.txt",
    "sed -I.bak p sample.txt",
    "sed -e 'e touch executed.txt' sample.txt",
    "sed 's/old/touch executed.txt/e' sample.txt",
    "sed -n 'W output.txt' sample.txt",
    'sed -n "$SED_PROGRAM" sample.txt',
    "sed -n $SED_PROGRAM sample.txt",
    "sed -n p *",
    "sed p sample.txt -e 'w output.txt'",
    "sed -n 'a\\opaque text' sample.txt",
    "sed -n 's#old#new#gpw output.txt' sample.txt",
    "sed -n 's/old/new/ep' sample.txt",
    "sed -e 'p' --file=program.sed sample.txt",
    "sed -n ':loop;b loop' sample.txt",
    "sed -n '1~2p' sample.txt",
    "sed -n $'p;\\167 output.txt' sample.txt",
)

READ_COMMANDS = (
    "sed -n '1,20p' sample.txt",
    "sed -n '$p' sample.txt",
    'sed -n "s/$/suffix/p" sample.txt',
    "sed -n '/start/,/end/p' sample.txt",
    "sed -n '/write/p' sample.txt",
    "sed 's/old/new/g' sample.txt",
    "sed -E 's/(old|other)/new/g' sample.txt",
    "sed -e 's#old#new#g' -e '1p' sample.txt",
    "sed -n 's;write;execute;gp' sample.txt",
    "sed -n 's/old/text;w output/gp' sample.txt",
    "sed -n '1{p;}' sample.txt",
    "sed -e '1{' -e 'p;}' sample.txt",
    "sed -n '/${FIELD}/p' sample.txt",
    "sed --expression=p sample.txt",
    "sed -n '1p' -- -input.txt",
    "sed 'y/abc/xyz/' sample.txt",
    "sed -n '3q' sample.txt",
    "sed --help",
    "sed -n '1p' 'file*name'",
    "sed -n '1p' sample.txt # $UNUSED",
)


@pytest.fixture
def permission_environment(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        service_module, "resolve_sandbox_settings", lambda _cwd: SimpleNamespace(enabled=False)
    )
    monkeypatch.setattr(service_module, "is_excluded_command", lambda *_a, **_kw: False)
    return tmp_path


@pytest.mark.parametrize("command", APPROVAL_COMMANDS)
@pytest.mark.parametrize("mode", [PermissionMode.DEFAULT, PermissionMode.DONT_ASK])
def test_mutating_or_opaque_sed_requires_approval(permission_environment, command, mode):
    decision = classify_shell_command(command)
    assert decision.allowed
    assert not decision.read_only
    assert decision.requires_approval
    service = PermissionService.default(mode=mode, workspace_root=permission_environment)
    result = service.evaluate_tool_call("run_shell", {"command": command})
    assert not result.allowed
    assert result.requires_approval == (mode == PermissionMode.DEFAULT)
    policy = SandboxPolicy.from_config({"enabled": True, "mode": "read-only"})
    assert read_only_violation(command, policy=policy) is not None


@pytest.mark.parametrize("command", READ_COMMANDS)
def test_literal_read_only_sed_remains_available(permission_environment, command):
    decision = classify_shell_command(command)
    assert decision.allowed and decision.read_only and not decision.requires_approval
    service = PermissionService.default(
        mode=PermissionMode.DONT_ASK, workspace_root=permission_environment
    )
    assert service.evaluate_tool_call("run_shell", {"command": command}).allowed


@pytest.mark.parametrize("binary", ["sed", "gsed"])
@pytest.mark.parametrize(
    ("program", "expected"),
    [
        ("w output.txt", "old\n"),
        ("s/old/new/w output.txt", "new\n"),
    ],
)
def test_real_sed_writes_without_in_place(permission_environment, binary, program, expected):
    executable = shutil.which(binary)
    if executable is None:
        pytest.skip(f"{binary} is not installed")
    source = permission_environment / "sample.txt"
    source.write_text("old\n", encoding="utf-8")
    subprocess.run(
        [executable, "-n", program, "sample.txt"],
        cwd=permission_environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    assert (permission_environment / "output.txt").read_text(encoding="utf-8") == expected
    assert source.read_text(encoding="utf-8") == "old\n"
    command = f"sed -n {shlex.quote(program)} sample.txt"
    assert not classify_shell_command(command).read_only


def test_real_sed_expanded_program_can_write(permission_environment):
    if shutil.which("sed") is None or os.name != "posix":
        pytest.skip("requires a POSIX shell and sed")
    (permission_environment / "sample.txt").write_text("old\n", encoding="utf-8")
    command = 'sed -n "$SED_PROGRAM" sample.txt'
    subprocess.run(
        ["/bin/sh", "-c", command],
        cwd=permission_environment,
        env={**os.environ, "SED_PROGRAM": "w dynamic-output.txt"},
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    assert (permission_environment / "dynamic-output.txt").read_text(encoding="utf-8") == "old\n"
    assert not classify_shell_command(command).read_only


def test_real_gnu_sed_execution_command_requires_approval(permission_environment):
    executable = shutil.which("gsed")
    if executable is None:
        pytest.skip("GNU sed is not installed as gsed")
    (permission_environment / "sample.txt").write_text("old\n", encoding="utf-8")
    program = "e touch executed.txt"
    subprocess.run(
        [executable, "-n", program, "sample.txt"],
        cwd=permission_environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    assert (permission_environment / "executed.txt").exists()
    assert not classify_shell_command(f"sed -n {shlex.quote(program)} sample.txt").read_only


@pytest.mark.parametrize("binary", ["sed", "gsed"])
@pytest.mark.parametrize(
    "program",
    [
        "p",
        "1,2p",
        "$p",
        "/old/p",
        "s/old/new/gp",
        "s#old#new#p",
        "y/old/NEW/",
        "1{p;}",
        "s/old/text;w output.txt/gp",
        "s;old;new;gp",
    ],
)
def test_recognized_read_programs_run_without_file_mutation(
    permission_environment, binary, program
):
    executable = shutil.which(binary)
    if executable is None:
        pytest.skip(f"{binary} is not installed")
    source = permission_environment / "sample.txt"
    contents = "old\nnext\n"
    source.write_text(contents, encoding="utf-8")
    assert classify_shell_command(f"sed -n {shlex.quote(program)} sample.txt").read_only

    subprocess.run(
        [executable, "-n", program, "sample.txt"],
        cwd=permission_environment,
        capture_output=True,
        text=True,
        check=True,
        timeout=5,
    )
    assert sorted(path.name for path in permission_environment.iterdir()) == ["sample.txt"]
    assert source.read_text(encoding="utf-8") == contents


@pytest.mark.parametrize("program", ["p;" * 33000, "{" * 65 + "p;" + "}" * 65])
def test_sed_static_analysis_has_work_limits(program):
    decision = classify_shell_command(f"sed -n {shlex.quote(program)} sample.txt")
    assert not decision.read_only
    assert decision.requires_approval


@pytest.mark.parametrize("binary", ["sed", "gsed"])
def test_options_after_program_have_platform_dependent_effects(permission_environment, binary):
    if binary == "sed":
        if sys.platform != "darwin":
            pytest.skip("BSD sed ordering control requires macOS")
        executable = "/usr/bin/sed"
    else:
        executable = shutil.which("gsed")
        if executable is None:
            pytest.skip("GNU sed is not installed as gsed")
    (permission_environment / "sample.txt").write_text("old\n", encoding="utf-8")
    command = "sed 'w ordering-output.txt' -e p sample.txt"
    assert not classify_shell_command(command).read_only

    result = subprocess.run(
        [executable, "w ordering-output.txt", "-e", "p", "sample.txt"],
        cwd=permission_environment,
        capture_output=True,
        text=True,
        timeout=5,
    )
    assert result.returncode != 0  # Missing operands do not imply no side effects.
    assert (permission_environment / "ordering-output.txt").exists() == (binary == "sed")
