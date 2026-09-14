"""Synthetic, offline packaging contracts; never import the packaged application."""

import base64
import builtins
import csv
import hashlib
import io
import json
import os
import runpy
import stat
import sys
import warnings
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[2] / "scripts" / "verify_wheel.py"
PACKAGE = "demo_agent"
DIST = "demo-2.3.4.dist-info"


@pytest.fixture
def verifier():
    return runpy.run_path(str(SCRIPT))


@pytest.fixture
def project(tmp_path):
    source = tmp_path / "checkout"
    source.mkdir()
    (source / "pyproject.toml").write_text(
        '[project]\nname = "demo"\nversion = "2.3.4"\n'
        'requires-python = ">=3.10"\n'
        '[project.scripts]\ndemo = "demo_agent.cli:run"\n'
        '[build-system]\nrequires = ["hatchling"]\nbuild-backend = "hatchling.build"\n'
        '[tool.hatch.build.targets.wheel]\npackages = ["demo_agent"]\n'
    )
    files = {
        f"{PACKAGE}/__init__.py": b"",
        f"{PACKAGE}/cli.py": b"def run():\n    return 42\n",
        f"{PACKAGE}/instructions.md": b"# Instructions\n",
        f"{PACKAGE}/data/model_prices_and_context_window.json": b'{"model": {}}\n',
        f"{PACKAGE}/harness/skills/bundled_skills/verify.md": b"# Verify\n",
        f"{PACKAGE}/data/icon.png": b"\x89PNG\r\n\x1a\n\x00\xff",
    }
    for name, content in files.items():
        target = source / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    return source, files


def make_wheel(tmp_path, files, changes=None, omit=(), extra=(), record=True):
    members = {
        **files,
        f"{DIST}/METADATA": (
            b"Metadata-Version: 2.4\nName: demo\nVersion: 2.3.4\nRequires-Python: >=3.10\n\n"
        ),
        f"{DIST}/WHEEL": (
            b"Wheel-Version: 1.0\nGenerator: test\nRoot-Is-Purelib: true\nTag: py3-none-any\n\n"
        ),
        f"{DIST}/entry_points.txt": b"[console_scripts]\ndemo = demo_agent.cli:run\n",
    }
    members.update(changes or {})
    for name in omit:
        members.pop(name, None)
    if record:
        stream = io.StringIO(newline="")
        # Python 3.10 needs an explicit escape character to emit the NUL
        # canary; keep it intact so the verifier, not the fixture, rejects it.
        writer = csv.writer(stream, escapechar="\\")
        for name, content in members.items():
            digest = base64.urlsafe_b64encode(hashlib.sha256(content).digest()).decode().rstrip("=")
            writer.writerow([name, f"sha256={digest}", len(content)])
        writer.writerow([f"{DIST}/RECORD", "", ""])
        members[f"{DIST}/RECORD"] = stream.getvalue().encode()
    wheel = tmp_path / "demo-2.3.4-py3-none-any.whl"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        with zipfile.ZipFile(wheel, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, content in members.items():
                archive.writestr(name, content)
            for name, content in extra:
                archive.writestr(name, content)
    return wheel


def invoke(verifier, capsys, source, wheel, *, human=False):
    args = [str(wheel), "--source", str(source)]
    if not human:
        args.append("--json")
    status = verifier["main"](args)
    captured = capsys.readouterr()
    assert captured.err == ""
    return status, captured.out if human else json.loads(captured.out)


def assert_failure(result, match):
    status, report = result
    assert status == 1
    assert report["ok"] is False
    assert any(match.lower() in error.lower() for error in report["errors"]), report
    assert report["limitations"]


def test_complete_inventory_and_binary_bytes(verifier, project, tmp_path, capsys):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    before = {name: (source / name).read_bytes() for name in files}
    status, report = invoke(verifier, capsys, source, wheel)
    assert status == 0, report
    assert report["ok"]
    assert report["version"] == "2.3.4"
    assert report["requires_python"] == ">=3.10"
    assert report["syntax_floor"] == "3.10"
    assert report["source_files"] == report["matched_files"] == len(files)
    assert report["python_files_checked"] == 2
    assert report["other_assets_matched"] == 1
    assert report["bundled_skills"] == 1  # No frozen count of fourteen.
    assert report["sha256"] == hashlib.sha256(wheel.read_bytes()).hexdigest()
    assert report["errors"] == []
    assert before == {name: (source / name).read_bytes() for name in files}
    assert not list(source.rglob("__pycache__"))


@pytest.mark.parametrize("extension", ["py", "md", "json", "png"])
@pytest.mark.parametrize("operation", ["missing", "replaced", "extra"])
def test_bidirectional_inventory(verifier, project, tmp_path, capsys, extension, operation):
    source, files = project
    name = next(name for name in files if name.endswith(f".{extension}"))
    kwargs = {
        "missing": {"omit": [name]},
        "replaced": {"changes": {name: b"changed"}},
        "extra": {"changes": {f"{PACKAGE}/unexpected.{extension}": b"extra"}},
    }[operation]
    wheel = make_wheel(tmp_path, files, **kwargs)
    assert_failure(invoke(verifier, capsys, source, wheel), operation)


@pytest.mark.parametrize(
    "name",
    [
        "../escape.py",
        "/absolute.py",
        "C:/drive.py",
        "demo_agent\\evil.py",
        "demo_agent//alias.py",
        "demo_agent/./alias.py",
        "demo_agent/trailing. /file",
        "demo_agent/a\x00ignored",
        "demo_agent/a\nlog",
        "demo_agent/.env",
        "demo_agent/.env.production",
        "demo_agent/.koder/config.yaml",
        "demo_agent/.git/config",
        "demo_agent/__pycache__/module.pyc",
        "demo_agent/tests/test_code.py",
        "demo_agent/tokens/provider.json",
        "demo_agent/credentials.json",
        "demo_agent/profile.sqlite",
        "demo_agent/private.pem",
    ],
)
def test_unsafe_or_forbidden_members(verifier, project, tmp_path, capsys, name):
    source, files = project
    wheel = make_wheel(tmp_path, files, changes={name: b"do not read"})
    # zipfile itself truncates NUL names: the result is unexpected inventory,
    # still rejected rather than silently succeeding.
    result = invoke(verifier, capsys, source, wheel)
    assert_failure(result, "extra package member" if "\x00" in name else "archive")


def test_duplicate_member_rejected(verifier, project, tmp_path, capsys):
    source, files = project
    name = f"{PACKAGE}/cli.py"
    wheel = make_wheel(tmp_path, files, extra=[(name, files[name])])
    assert_failure(invoke(verifier, capsys, source, wheel), "duplicate")


def test_archive_symlink_rejected(verifier, project, tmp_path, capsys):
    source, files = project
    link = zipfile.ZipInfo(f"{PACKAGE}/linked.py")
    link.create_system = 3
    link.external_attr = (stat.S_IFLNK | 0o777) << 16
    wheel = make_wheel(tmp_path, files, extra=[(link, b"/outside")])
    assert_failure(invoke(verifier, capsys, source, wheel), "symlink")


def test_source_symlink_never_read(verifier, project, tmp_path, capsys):
    source, files = project
    (source / PACKAGE / "linked.py").symlink_to(tmp_path / "nonexistent-private")
    wheel = make_wheel(tmp_path, files)
    assert_failure(invoke(verifier, capsys, source, wheel), "symlink")


@pytest.mark.parametrize("kind", ["missing", "not-a-zip", "truncated"])
def test_bad_archive_has_structured_error(verifier, project, tmp_path, capsys, kind):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    if kind == "missing":
        wheel.unlink()
    elif kind == "not-a-zip":
        wheel.write_bytes(b"not a zip")
    else:
        wheel.write_bytes(wheel.read_bytes()[:40])
    assert_failure(invoke(verifier, capsys, source, wheel), "archive")


@pytest.mark.parametrize(
    ("member", "content", "message"),
    [
        ("METADATA", b"Name: wrong\nVersion: 2.3.4\nRequires-Python: >=3.10\n", "metadata"),
        ("METADATA", b"Name: demo\nVersion: 9.9.9\nRequires-Python: >=3.10\n", "metadata"),
        ("METADATA", b"Name: demo\nVersion: 2.3.4\nRequires-Python: >=3.11\n", "metadata"),
        ("METADATA", b"Name: demo\nName: demo\n", "metadata"),
        ("entry_points.txt", b"[console_scripts]\ndemo = bad:run\n", "entry"),
        ("entry_points.txt", b"not an ini", "entry"),
        ("WHEEL", b"Wheel-Version: 9.0\n", "wheel"),
    ],
)
def test_build_contract(verifier, project, tmp_path, capsys, member, content, message):
    source, files = project
    wheel = make_wheel(tmp_path, files, changes={f"{DIST}/{member}": content})
    assert_failure(invoke(verifier, capsys, source, wheel), message)


@pytest.mark.parametrize("member", ["METADATA", "entry_points.txt", "WHEEL", "RECORD"])
def test_required_build_files(verifier, project, tmp_path, capsys, member):
    source, files = project
    wheel = make_wheel(tmp_path, files, omit=[f"{DIST}/{member}"], record=member != "RECORD")
    assert_failure(invoke(verifier, capsys, source, wheel), member)


def test_record_hash_is_verified(verifier, project, tmp_path, capsys):
    source, files = project
    wheel = make_wheel(
        tmp_path,
        files,
        changes={f"{DIST}/RECORD": f"{PACKAGE}/cli.py,sha256=bad,1\n".encode()},
        record=False,
    )
    assert_failure(invoke(verifier, capsys, source, wheel), "record")


def test_syntax_floor_comes_from_source_contract(verifier, project, tmp_path, capsys):
    source, files = project
    name = f"{PACKAGE}/cli.py"
    files[name] = b"def run[T]():\n    return None\n"
    (source / name).write_bytes(files[name])
    wheel = make_wheel(tmp_path, files)
    assert_failure(invoke(verifier, capsys, source, wheel), "syntax")

    if sys.version_info < (3, 12):
        return  # The rejecting half still applies on 3.11.
    contract = source / "pyproject.toml"
    contract.write_text(contract.read_text().replace(">=3.10", ">=3.12"))
    wheel = make_wheel(
        tmp_path,
        files,
        changes={
            f"{DIST}/METADATA": (
                b"Metadata-Version: 2.4\nName: demo\nVersion: 2.3.4\nRequires-Python: >=3.12\n\n"
            )
        },
    )
    status, report = invoke(verifier, capsys, source, wheel)
    assert status == 0, report
    assert report["syntax_floor"] == "3.12"


def test_invalid_control_flow_is_not_just_parsed(verifier, project, tmp_path, capsys):
    source, files = project
    name = f"{PACKAGE}/cli.py"
    files[name] = b"return 42\n"
    (source / name).write_bytes(files[name])
    wheel = make_wheel(tmp_path, files)
    assert_failure(invoke(verifier, capsys, source, wheel), "syntax")


def test_unsupported_python_contract_fails_explicitly(verifier, project, tmp_path, capsys):
    source, files = project
    path = source / "pyproject.toml"
    path.write_text(path.read_text().replace(">=3.10", "~=3.10"))
    wheel = make_wheel(tmp_path, files)
    assert_failure(invoke(verifier, capsys, source, wheel), "requires-python")


@pytest.mark.parametrize(
    "target",
    ["data/model_prices_and_context_window.json", "harness/skills/bundled_skills/verify.md"],
)
def test_assets_missing_from_both_sides_are_not_success(
    verifier, project, tmp_path, capsys, target
):
    source, files = project
    name = f"{PACKAGE}/{target}"
    (source / name).unlink()
    del files[name]
    wheel = make_wheel(tmp_path, files)
    assert_failure(invoke(verifier, capsys, source, wheel), "required asset")


def test_local_bytecode_is_ignored_but_reported(verifier, project, tmp_path, capsys):
    source, files = project
    cache = source / PACKAGE / "__pycache__"
    cache.mkdir()
    (cache / "cli.pyc").write_bytes(b"cache")
    wheel = make_wheel(tmp_path, files)
    status, report = invoke(verifier, capsys, source, wheel)
    assert status == 0, report
    assert report["excluded_source_paths"] == [f"{PACKAGE}/__pycache__"]


def test_size_limits_fail_before_member_read(verifier, project, tmp_path, capsys, monkeypatch):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    monkeypatch.setitem(verifier["main"].__globals__, "MAX_MEMBER_BYTES", 16)
    assert_failure(invoke(verifier, capsys, source, wheel), "limit")


def test_human_summary_and_cli_exit(verifier, project, tmp_path, capsys, monkeypatch):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    status, text = invoke(verifier, capsys, source, wheel, human=True)
    assert status == 0
    assert "PASS" in text
    assert "6/6" in text
    assert "runtime" in text.lower()
    wheel.write_bytes(b"bad archive")
    monkeypatch.setattr(sys, "argv", [str(SCRIPT), str(wheel), "--source", str(source), "--json"])
    with pytest.raises(SystemExit) as error:
        runpy.run_path(str(SCRIPT), run_name="__main__")
    assert error.value.code == 1
    assert json.loads(capsys.readouterr().out)["ok"] is False


def test_corrupt_deflate_has_structured_error(verifier, project, tmp_path, capsys):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    with zipfile.ZipFile(wheel) as archive:
        info = archive.getinfo(f"{PACKAGE}/cli.py")
        start = info.header_offset + 30 + len(info.filename.encode()) + len(info.extra)
    payload = bytearray(wheel.read_bytes())
    payload[start] = 0xFF  # Deflate's reserved block type, not just a CRC mismatch.
    wheel.write_bytes(payload)
    assert_failure(invoke(verifier, capsys, source, wheel), "archive")


def test_fifo_archive_is_rejected_before_open(verifier, project, tmp_path, capsys, monkeypatch):
    if not hasattr(os, "mkfifo"):
        pytest.skip("FIFO only exists on POSIX")
    source, _ = project
    wheel = tmp_path / "fifo.whl"
    os.mkfifo(wheel)
    original = Path.open

    def no_fifo_open(path, *args, **kwargs):
        assert path != wheel, "must reject FIFO before a potentially blocking open"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_fifo_open)
    assert_failure(invoke(verifier, capsys, source, wheel), "regular")


def test_private_source_file_rejected_without_read(
    verifier, project, tmp_path, capsys, monkeypatch
):
    source, files = project
    forbidden = source / PACKAGE / ".env"
    forbidden.write_bytes(b"synthetic private contents")
    original = Path.open

    def no_private_open(path, *args, **kwargs):
        assert path != forbidden, "private file must not be read"
        return original(path, *args, **kwargs)

    monkeypatch.setattr(Path, "open", no_private_open)
    wheel = make_wheel(tmp_path, files)
    assert_failure(invoke(verifier, capsys, source, wheel), "forbidden")


def test_case_alias_duplicate_rejected(verifier, project, tmp_path, capsys):
    source, files = project
    wheel = make_wheel(tmp_path, files, changes={f"{PACKAGE}/CLI.py": files[f"{PACKAGE}/cli.py"]})
    assert_failure(invoke(verifier, capsys, source, wheel), "aliased")


def test_nonempty_directory_member_rejected(verifier, project, tmp_path, capsys):
    source, files = project
    wheel = make_wheel(tmp_path, files, extra=[(f"{PACKAGE}/hidden/", b"unexpected payload")])
    assert_failure(invoke(verifier, capsys, source, wheel), "directory")


@pytest.mark.parametrize("limit", ["MAX_ENTRIES", "MAX_TOTAL_BYTES"])
def test_count_and_total_limits(verifier, project, tmp_path, capsys, monkeypatch, limit):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    monkeypatch.setitem(verifier["main"].__globals__, limit, 1)
    assert_failure(invoke(verifier, capsys, source, wheel), "limit")


@pytest.mark.parametrize("content", ["invalid TOML [", '[project]\nname="demo"\n'])
def test_malformed_project_contract(verifier, project, tmp_path, capsys, content):
    source, files = project
    (source / "pyproject.toml").write_text(content)
    wheel = make_wheel(tmp_path, files)
    assert_failure(invoke(verifier, capsys, source, wheel), "contract")


def test_renamed_wheel_cannot_claim_other_version(verifier, project, tmp_path, capsys):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    renamed = wheel.rename(tmp_path / "demo-9.9.9-py3-none-any.whl")
    assert_failure(invoke(verifier, capsys, source, renamed), "filename")


@pytest.mark.parametrize("changed", [False, True])
def test_license_bytes_match_source(verifier, project, tmp_path, capsys, changed):
    source, files = project
    (source / "LICENSE").write_bytes(b"synthetic license\n")
    wheel = make_wheel(
        tmp_path,
        files,
        changes={f"{DIST}/licenses/LICENSE": b"different" if changed else b"synthetic license\n"},
    )
    result = invoke(verifier, capsys, source, wheel)
    if changed:
        assert_failure(result, "license")
    else:
        assert result[0] == 0, result


def test_other_dist_info_tree_is_rejected(verifier, project, tmp_path, capsys):
    source, files = project
    wheel = make_wheel(tmp_path, files, changes={"other-1.0.dist-info/METADATA": b"Name: other\n"})
    assert_failure(invoke(verifier, capsys, source, wheel), "unexpected build metadata")


def test_cli_cannot_execute_package_imports(verifier, project, tmp_path, capsys):
    source, files = project
    name = f"{PACKAGE}/__init__.py"
    files[name] = b'raise AssertionError("must not import application")\n'
    (source / name).write_bytes(files[name])
    wheel = make_wheel(tmp_path, files)
    status, report = invoke(verifier, capsys, source, wheel)
    assert status == 0, report


def test_unsupported_compression_is_explicit(verifier, project, tmp_path, capsys):
    source, files = project
    info = zipfile.ZipInfo(f"{PACKAGE}/compressed.dat")
    info.compress_type = zipfile.ZIP_BZIP2
    wheel = make_wheel(tmp_path, files, extra=[(info, b"synthetic data")])
    assert_failure(invoke(verifier, capsys, source, wheel), "compression")


def test_checkout_symlink_rejected(verifier, project, tmp_path, capsys):
    source, files = project
    link = tmp_path / "linked-checkout"
    link.symlink_to(source, target_is_directory=True)
    wheel = make_wheel(tmp_path, files)
    assert_failure(invoke(verifier, capsys, link, wheel), "symlink")


def test_too_deep_archive_path_is_rejected(verifier, project, tmp_path, capsys):
    source, files = project
    name = f"{PACKAGE}/" + "nested/" * 65 + "file.py"
    wheel = make_wheel(tmp_path, files, changes={name: b""})
    assert_failure(invoke(verifier, capsys, source, wheel), "unsafe path")


def test_human_failure_is_not_a_partial_success(verifier, project, tmp_path, capsys):
    source, files = project
    wheel = make_wheel(tmp_path, files, omit=[f"{PACKAGE}/cli.py"])
    status, text = invoke(verifier, capsys, source, wheel, human=True)
    assert status == 1
    assert "FAIL" in text
    assert "missing source member" in text
    assert "PASS" not in text


@pytest.fixture
def parser_imports(verifier, project, monkeypatch):
    """Substitute only parser imports and the verifier's reported Python version."""
    source, _ = project
    original_import = builtins.__import__
    source_text = (source / "pyproject.toml").read_text()

    def configure(available, version, *, invalid=False):
        attempts = []
        parsed = []

        def loads(text):
            assert text == source_text
            parsed.append(text)
            if invalid:
                raise ValueError("synthetic malformed TOML")
            return {
                "project": {
                    "name": "demo",
                    "version": "2.3.4",
                    "requires-python": ">=3.10",
                    "scripts": {"demo": "demo_agent.cli:run"},
                },
                "build-system": {"build-backend": "hatchling.build"},
                "tool": {"hatch": {"build": {"targets": {"wheel": {"packages": [PACKAGE]}}}}},
            }

        def controlled_import(name, *args, **kwargs):
            if name in {"tomllib", "tomli"}:
                attempts.append(name)
                if name not in available:
                    raise ModuleNotFoundError(f"No module named {name!r}", name=name)
                return SimpleNamespace(loads=loads)
            return original_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, "__import__", controlled_import)
        monkeypatch.setitem(
            verifier["main"].__globals__, "sys", SimpleNamespace(version_info=version)
        )
        return attempts, parsed

    return configure


def test_stdlib_parser_preferred(verifier, project, tmp_path, capsys, parser_imports):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    attempts, parsed = parser_imports({"tomllib", "tomli"}, (3, 11))
    status, report = invoke(verifier, capsys, source, wheel)
    assert status == 0, report
    assert attempts == ["tomllib"]
    assert len(parsed) == 1


def test_python310_parser_fallback(verifier, project, tmp_path, capsys, parser_imports):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    attempts, parsed = parser_imports({"tomli"}, (3, 10))
    status, report = invoke(verifier, capsys, source, wheel)
    assert status == 0, report
    assert attempts == ["tomllib", "tomli"]
    assert len(parsed) == 1
    assert report["syntax_floor"] == "3.10"
    assert report["matched_files"] == len(files)


@pytest.mark.parametrize("human", [False, True])
def test_missing_toml_parsers_explicit_error(
    verifier, project, tmp_path, capsys, parser_imports, human
):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    attempts, parsed = parser_imports(set(), (3, 10))
    status, report = invoke(verifier, capsys, source, wheel, human=human)
    assert status == 1
    if human:
        assert "FAIL" in report
        assert "TOML parser unavailable" in report
        assert "independently available tomli" in report
    else:
        assert_failure((status, report), "TOML parser unavailable")
        assert any("independently available tomli" in error for error in report["errors"])
    assert attempts == ["tomllib", "tomli"]
    assert parsed == []


def test_fallback_parser_errors_remain_validation_failures(
    verifier, project, tmp_path, capsys, parser_imports
):
    source, files = project
    wheel = make_wheel(tmp_path, files)
    attempts, parsed = parser_imports({"tomli"}, (3, 10), invalid=True)
    assert_failure(invoke(verifier, capsys, source, wheel), "project contract")
    assert attempts == ["tomllib", "tomli"]
    assert len(parsed) == 1
