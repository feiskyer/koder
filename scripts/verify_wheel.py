"""Read-only Koder wheel/source verification for Python 3.10+.

Usage: uv run python scripts/verify_wheel.py dist/package.whl --source . [--json]

Reads static Hatch package roots and PEP 621 metadata from pyproject.toml; never
imports the application, builds, extracts, installs, or contacts services. All
regular package files (including binary assets) must match in both directions.
Local bytecode caches are explicitly excluded; forbidden/private paths are never
read. This deliberately supports pure-Python, top-level Hatch package roots and
simple >=3.N Requires-Python contracts, failing closed on other build layouts.
Uses stdlib tomllib on supported Koder interpreters. Standalone inspection of older
artifacts on Python 3.10 can use an independently available tomli; this does not
lower the application's declared floor. No packages are installed by this script.
"""

from __future__ import annotations

import argparse
import ast
import base64
import configparser
import csv
import hashlib
import io
import json
import os
import re
import stat
import sys
import zipfile
import zlib
from email import policy
from email.parser import BytesParser
from pathlib import Path

MAX_MEMBER_BYTES = 32 * 1024 * 1024
MAX_TOTAL_BYTES = 128 * 1024 * 1024
MAX_ENTRIES = 10000
LIMITATIONS = [
    "Static source/metadata/syntax checks do not prove runtime or dependency compatibility.",
    "ast feature_version is a best-effort grammar check, not execution on the minimum Python.",
    "Source must be quiescent; this is not an atomic snapshot or build-provenance attestation.",
    "Private-path screening is not a secret-content scan of otherwise legitimate source files.",
    "Supports static metadata, top-level Hatch package roots, pure py3-none-any wheels and >=3.N only.",
    "Required asset policy: vendored model map and nonempty bundled skills; individual skill names follow source.",
]
FORBIDDEN_PARTS = {
    "tests",
    "test",
    "__pycache__",
    "node_modules",
    "tokens",
    "credentials",
    "profiles",
    "profile",
    "secrets",
    "venv",
    "dist",
    "build",
}
FORBIDDEN_FILES = {
    "config.yaml",
    "config.yml",
    "config.json",
    "settings.json",
    "settings.local.json",
    "credentials.json",
    "token.json",
    "tokens.json",
    "secrets.json",
    "credentials.yaml",
    "credentials.yml",
    "secrets.yaml",
    "secrets.yml",
    "profile.json",
    "id_rsa",
    "id_ed25519",
}


class ValidationError(Exception):
    """An unsupported contract or invalid input, not a successful partial check."""


def path_problem(name: str) -> str | None:
    parts = name.split("/")
    if (
        not name
        or "\\" in name
        or ":" in name
        or len(parts) > 64
        or any(ord(char) < 32 or ord(char) == 127 for char in name)
        or any(not part or part in {".", ".."} or part.rstrip(" .") != part for part in parts)
    ):
        return "unsafe path"
    if any(part.startswith(".") or part.casefold() in FORBIDDEN_PARTS for part in parts):
        return "forbidden private/test/cache path"
    if parts[-1].casefold() in FORBIDDEN_FILES or Path(name).suffix.lower() in {
        ".pyc",
        ".pyo",
        ".db",
        ".sqlite",
        ".sqlite3",
        ".pem",
        ".key",
        ".p12",
        ".pfx",
    }:
        return "forbidden profile/credential/cache file"
    return None


def read_regular(path: Path, limit: int = MAX_MEMBER_BYTES) -> bytes:
    """Reject links and nonregular files before reading bounded bytes."""
    if path.is_symlink():
        raise ValidationError(f"source symlink refused: {str(path)!r}")
    info = path.stat()
    if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
        raise ValidationError(f"regular-file/size limit exceeded: {str(path)!r}")
    with path.open("rb") as handle:
        info = os.fstat(handle.fileno())
        if not stat.S_ISREG(info.st_mode) or info.st_size > limit:
            raise ValidationError(f"regular-file/size limit exceeded: {str(path)!r}")
        payload = handle.read(limit + 1)
    if len(payload) > limit:
        raise ValidationError(f"size limit exceeded: {str(path)!r}")
    return payload


def load_contract(source: Path) -> tuple[dict, list[str], tuple[int, int]]:
    if source.is_symlink():
        raise ValidationError("source checkout symlink refused")
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib
        except ImportError as exc:
            raise ValidationError(
                "TOML parser unavailable: use stdlib tomllib on Python 3.11+ "
                "or an independently available tomli for standalone Python 3.10 inspection"
            ) from exc

    try:
        document = tomllib.loads(read_regular(source / "pyproject.toml").decode("utf-8"))
        project = document["project"]
        packages = document["tool"]["hatch"]["build"]["targets"]["wheel"]["packages"]
        backend = document["build-system"]["build-backend"]
        for field in ("name", "version", "requires-python"):
            if not isinstance(project[field], str) or not project[field]:
                raise ValueError(f"project.{field} must be a nonempty string")
        if project.get("dynamic"):
            raise ValueError("dynamic metadata is not supported")
        if backend != "hatchling.build":
            raise ValueError("only the declared Hatch build layout is supported")
        if (
            not isinstance(packages, list)
            or not packages
            or any(not isinstance(p, str) or not re.fullmatch(r"[A-Za-z_]\w*", p) for p in packages)
            or len(packages) != len(set(packages))
        ):
            raise ValueError("packages must be distinct top-level package names")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", project["name"]):
            raise ValueError("invalid project name")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9.!+_]*", project["version"]):
            raise ValueError("unsupported static project version")
        floor_match = re.fullmatch(r">=3\.(\d+)(?:\.0)?", project["requires-python"])
        if not floor_match:
            raise ValueError("unsupported requires-python; expected >=3.N")
        floor = (3, int(floor_match[1]))
        if floor < (3, 7) or floor > sys.version_info[:2]:
            raise ValueError("requires-python floor cannot be checked by this interpreter")
        scripts = project.get("scripts")
        if (
            not isinstance(scripts, dict)
            or not scripts
            or not all(isinstance(k, str) and isinstance(v, str) for k, v in scripts.items())
        ):
            raise ValueError("project.scripts must declare the CLI entrypoints")
    except (KeyError, TypeError, ValueError) as exc:
        raise ValidationError(f"project contract: {exc}") from exc
    return project, packages, floor


def inventory(source: Path, packages: list[str], report: dict) -> dict[str, bytes]:
    files = {}
    visited = 0
    total_bytes = 0

    def walk(directory: Path):
        nonlocal visited, total_bytes
        if directory.is_symlink():
            raise ValidationError(f"source symlink refused: {str(directory)!r}")
        with os.scandir(directory) as entries:
            for entry in entries:
                visited += 1
                if visited > MAX_ENTRIES:
                    raise ValidationError("source entry count limit exceeded")
                path = Path(entry.path)
                name = path.relative_to(source).as_posix()
                if entry.is_symlink():
                    raise ValidationError(f"source symlink refused: {name!r}")
                if entry.name == "__pycache__" or path.suffix in {".pyc", ".pyo"}:
                    report["excluded_source_paths"].append(name)
                    continue
                problem = path_problem(name)
                if problem:
                    raise ValidationError(f"source {problem}: {name!r}")
                if entry.is_dir(follow_symlinks=False):
                    walk(path)
                elif entry.is_file(follow_symlinks=False):
                    size = entry.stat(follow_symlinks=False).st_size
                    if size > MAX_MEMBER_BYTES or total_bytes + size > MAX_TOTAL_BYTES:
                        raise ValidationError("source size limit exceeded")
                    payload = read_regular(path, MAX_MEMBER_BYTES)
                    total_bytes += len(payload)
                    if total_bytes > MAX_TOTAL_BYTES:
                        raise ValidationError("source size limit exceeded")
                    files[name] = payload
                else:
                    raise ValidationError(f"nonregular source entry: {name!r}")

    for package in packages:
        walk(source / package)
        price_map = f"{package}/data/model_prices_and_context_window.json"
        skills = f"{package}/harness/skills/bundled_skills/"
        if price_map not in files or not any(
            name.startswith(skills) and name.endswith(".md") for name in files
        ):
            report["errors"].append(
                f"required asset missing in source: {price_map} or {skills}*.md"
            )
    report["excluded_source_paths"].sort()
    report["source_files"] = len(files)
    return files


def read_archive(wheel: Path) -> tuple[dict[str, bytes], str]:
    payload = read_regular(wheel, MAX_TOTAL_BYTES)
    with zipfile.ZipFile(io.BytesIO(payload)) as archive:
        entries = archive.infolist()
        if len(entries) > MAX_ENTRIES or sum(i.file_size for i in entries) > MAX_TOTAL_BYTES:
            raise ValidationError("archive count/expanded size limit exceeded")
        names = set()
        aliases = set()
        for info in entries:
            name = info.filename
            canonical = name.removesuffix("/") if info.is_dir() else name
            problem = path_problem(canonical)
            if problem:
                raise ValidationError(f"archive {problem}: {name!r}")
            if info.orig_filename != name:
                raise ValidationError(f"archive unsafe original path: {info.orig_filename!r}")
            if name in names or canonical.casefold() in aliases:
                raise ValidationError(f"archive duplicate/aliased member: {name!r}")
            names.add(name)
            aliases.add(canonical.casefold())
            kind = stat.S_IFMT(info.external_attr >> 16)
            if kind == stat.S_IFLNK:
                raise ValidationError(f"archive symlink refused: {name!r}")
            if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
                raise ValidationError(f"archive nonregular member: {name!r}")
            if (info.is_dir() and info.file_size) or (kind == stat.S_IFDIR and not info.is_dir()):
                raise ValidationError(f"archive inconsistent/nonempty directory: {name!r}")
            if info.file_size > MAX_MEMBER_BYTES or info.flag_bits & 1:
                raise ValidationError(f"archive member size limit or encryption: {name!r}")
            if info.compress_type not in {zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED}:
                raise ValidationError(f"archive unsupported compression: {name!r}")
        # Preflight every member before reading any member's contents.
        members = {i.filename: archive.read(i) for i in entries if not i.is_dir()}
        return members, hashlib.sha256(payload).hexdigest()


def check_metadata(members: dict[str, bytes], source: Path, project: dict, report: dict):
    errors = report["errors"]
    dist_name = re.sub(r"[-_.]+", "_", project["name"]).lower()
    dist = f"{dist_name}-{project['version']}.dist-info"
    required = {f"{dist}/{name}" for name in ("METADATA", "WHEEL", "entry_points.txt", "RECORD")}
    for name in sorted(required - members.keys()):
        errors.append(f"missing build metadata: {name}")
    metadata = BytesParser(policy=policy.default).parsebytes(members.get(f"{dist}/METADATA", b""))
    for header, expected in (
        ("Name", project["name"]),
        ("Version", project["version"]),
        ("Requires-Python", project["requires-python"]),
    ):
        values = metadata.get_all(header, [])
        if values != [expected]:
            errors.append(f"METADATA {header} does not match project contract")
    if len(metadata.get_all("Metadata-Version", [])) != 1 or metadata.defects:
        errors.append("METADATA missing/duplicate version or malformed headers")
    elif not re.fullmatch(r"2\.\d+", metadata["Metadata-Version"]):
        errors.append("METADATA unsupported Metadata-Version")
    build = BytesParser(policy=policy.default).parsebytes(members.get(f"{dist}/WHEEL", b""))
    for header, expected in (
        ("Wheel-Version", "1.0"),
        ("Root-Is-Purelib", "true"),
        ("Tag", "py3-none-any"),
    ):
        if build.get_all(header, []) != [expected]:
            errors.append(f"WHEEL {header} must be {expected!r} for this validator")
    if build.defects:
        errors.append("malformed WHEEL headers")
    try:
        entrypoints = configparser.ConfigParser(interpolation=None)
        entrypoints.optionxform = str
        entrypoints.read_string(members.get(f"{dist}/entry_points.txt", b"").decode("utf-8"))
        if entrypoints.defaults() or set(entrypoints.sections()) != {"console_scripts"}:
            raise ValueError("expected only console_scripts")
        if dict(entrypoints["console_scripts"]) != project["scripts"]:
            raise ValueError("console_scripts do not match project.scripts")
        for target in project["scripts"].values():
            match = re.fullmatch(r"([A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*):([A-Za-z_]\w*)", target)
            if not match or match[1].replace(".", "/") + ".py" not in members:
                raise ValueError("CLI entrypoint module absent or unsupported target")
    except (configparser.Error, UnicodeError, ValueError, KeyError) as exc:
        errors.append(f"entrypoint metadata: {exc}")
    for name in members:
        if ".dist-info/" not in name or name in required:
            continue
        prefix = f"{dist}/licenses/"
        if not name.startswith(prefix):
            errors.append(f"unexpected build metadata member: {name!r}")
        else:
            license_path = source / name[len(prefix) :]
            if any(
                (source / Path(*license_path.relative_to(source).parts[:depth])).is_symlink()
                for depth in range(len(license_path.relative_to(source).parts))
            ):
                raise ValidationError("source license parent symlink refused")
            if read_regular(license_path) != members[name]:
                errors.append(f"replaced license: {name!r}")
    check_record(members, f"{dist}/RECORD", errors)


def check_record(members: dict[str, bytes], record_name: str, errors: list[str]):
    try:
        rows = csv.reader(io.StringIO(members.get(record_name, b"").decode("utf-8")), strict=True)
        seen = set()
        for row in rows:
            if len(row) != 3 or row[0] in seen:
                raise ValueError("malformed or duplicate row")
            name, digest, size = row
            seen.add(name)
            if name not in members:
                raise ValueError("row references an absent member")
            if name == record_name:
                if digest or size:
                    raise ValueError("self row must have blank hash and size")
            else:
                expected = (
                    base64.urlsafe_b64encode(hashlib.sha256(members[name]).digest())
                    .decode()
                    .rstrip("=")
                )
                if digest != f"sha256={expected}" or size != str(len(members[name])):
                    raise ValueError(f"hash/size mismatch: {name!r}")
        if seen != members.keys():
            raise ValueError("inventory does not cover every archive file")
    except (UnicodeError, csv.Error, ValueError) as exc:
        errors.append(f"RECORD invalid: {exc}")


def validate(wheel: Path, source: Path) -> dict:
    report = {
        "ok": False,
        "wheel": str(wheel),
        "source": str(source),
        "source_files": 0,
        "matched_files": 0,
        "python_files_checked": 0,
        "other_assets_matched": 0,
        "bundled_skills": 0,
        "excluded_source_paths": [],
        "errors": [],
        "limitations": LIMITATIONS.copy(),
    }
    errors = report["errors"]
    try:
        project, packages, floor = load_contract(source)
        report.update(
            version=project["version"],
            requires_python=project["requires-python"],
            syntax_floor=".".join(map(str, floor)),
        )
        expected = inventory(source, packages, report)
        try:
            members, report["sha256"] = read_archive(wheel)
        except (
            OSError,
            ValueError,
            EOFError,
            zipfile.BadZipFile,
            zlib.error,
            RuntimeError,
            NotImplementedError,
            ValidationError,
        ) as exc:
            raise ValidationError(f"archive validation: {exc}") from exc
        filename = f"{re.sub(r'[-_.]+', '_', project['name']).lower()}-{project['version']}-py3-none-any.whl"
        if wheel.name != filename:
            errors.append("wheel filename does not match project/version/pure wheel tag")
        packaged = {name for name in members if any(name.startswith(f"{p}/") for p in packages)}
        for name in sorted(expected.keys() - packaged):
            errors.append(f"missing source member: {name!r}")
        for name in sorted(packaged - expected.keys()):
            errors.append(f"extra package member: {name!r}")
        for name in sorted(members.keys() - packaged):
            if ".dist-info/" not in name:
                errors.append(f"extra nonpackage member: {name!r}")
        for name in sorted(packaged & expected.keys()):
            if members[name] != expected[name]:
                errors.append(f"replaced source member: {name!r}")
                continue
            report["matched_files"] += 1
            if name.endswith(".py"):
                try:
                    tree = ast.parse(members[name], filename=name, feature_version=floor)
                    compile(
                        tree, name, "exec"
                    )  # Compile only; catch AST-level invalid control flow.
                    report["python_files_checked"] += 1
                except (SyntaxError, ValueError) as exc:
                    errors.append(
                        f"syntax floor {report['syntax_floor']} invalid: {name!r}: {type(exc).__name__}"
                    )
            elif not name.endswith((".md", ".json")):
                report["other_assets_matched"] += 1
            if "/harness/skills/bundled_skills/" in name and name.endswith(".md"):
                report["bundled_skills"] += 1
        check_metadata(members, source, project, report)
    except (OSError, UnicodeError, ValidationError) as exc:
        errors.append(str(exc))
    report["ok"] = not errors
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("wheel", type=Path, help="existing .whl; never built or extracted")
    parser.add_argument("--source", required=True, type=Path, help="quiescent source checkout")
    parser.add_argument(
        "--json", action="store_true", help="print structured result, including failures"
    )
    args = parser.parse_args(argv)
    report = validate(args.wheel.absolute(), args.source.absolute())
    if args.json:
        print(json.dumps(report, indent=2, ensure_ascii=True))
    else:
        print(
            f"{'PASS' if report['ok'] else 'FAIL'}: {report['matched_files']}/{report['source_files']} source files matched"
        )
        for error in report["errors"]:
            print(f"ERROR: {error}")
        print(
            f"Python syntax files checked: {report['python_files_checked']}; bundled skills: {report['bundled_skills']}"
        )
        for limitation in report["limitations"]:
            print(f"LIMITATION: {limitation}")
    return 0 if report["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
