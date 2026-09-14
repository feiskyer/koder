"""Public and runtime versions share metadata without importing provider code."""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from koder_agent import version
from koder_agent.harness import version_info


def _missing_distribution(_name):
    raise version.PackageNotFoundError("koder")


def test_installed_version_does_not_read_source(monkeypatch):
    monkeypatch.setattr(version, "package_version", lambda _name: "7.8.9")

    def unexpected_source_read():
        pytest.fail("installed metadata must not depend on a source checkout")

    monkeypatch.setattr(version, "_source_tree_version", unexpected_source_read)
    assert version.resolve_package_version_info() == ("7.8.9", "installed-package")
    assert version_info.resolve_runtime_version_info is version.resolve_package_version_info
    assert version_info.render_cli_version_banner() == "7.8.9 (Koder)"


def test_source_fallback_uses_package_anchor_not_working_directory(tmp_path, monkeypatch):
    source = tmp_path / "checkout" / "pyproject.toml"
    source.parent.mkdir()
    source.write_text('[project]\nname = "koder"\nversion = "2.3.4"\n', encoding="utf-8")
    unrelated = tmp_path / "unrelated"
    unrelated.mkdir()
    (unrelated / "pyproject.toml").write_text(
        '[project]\nname = "koder"\nversion = "9.9.9"\n', encoding="utf-8"
    )
    monkeypatch.chdir(unrelated)
    monkeypatch.setattr(version, "package_version", _missing_distribution)
    monkeypatch.setattr(version, "_SOURCE_PROJECT_FILE", source)
    assert version.resolve_package_version_info() == ("2.3.4", "source-tree")
    assert version_info.resolve_runtime_version() == "2.3.4"


@pytest.mark.parametrize(
    "content",
    [
        None,
        "invalid [",
        '[project]\nname = "unrelated"\nversion = "9.9.9"\n',
        '[project]\nname = "koder"\nversion = ""\n',
        '[project]\nname = "koder"\nversion = 123\n',
        "project = []",
    ],
)
def test_missing_or_invalid_metadata_is_truthfully_unknown(tmp_path, monkeypatch, content):
    source = tmp_path / "pyproject.toml"
    if content is not None:
        source.write_text(content, encoding="utf-8")
    monkeypatch.setattr(version, "package_version", _missing_distribution)
    monkeypatch.setattr(version, "_SOURCE_PROJECT_FILE", source)
    assert version.resolve_package_version_info() == ("unknown", "unavailable")


def test_source_fallback_uses_existing_tomli_when_stdlib_parser_is_unavailable(
    tmp_path, monkeypatch
):
    source = tmp_path / "pyproject.toml"
    source.write_text('[project]\nname = "koder"\nversion = "3.4.5"\n', encoding="utf-8")
    parsed = []

    def parse(text):
        parsed.append(text)
        return {"project": {"name": "koder", "version": "3.4.5"}}

    monkeypatch.setattr(version, "package_version", _missing_distribution)
    monkeypatch.setattr(version, "_SOURCE_PROJECT_FILE", source)
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "tomli", SimpleNamespace(loads=parse))
    assert version.resolve_package_version_info() == ("3.4.5", "source-tree")
    assert parsed == [source.read_text()]


def test_source_without_a_parser_does_not_invent_a_version(tmp_path, monkeypatch):
    source = tmp_path / "pyproject.toml"
    source.write_text('[project]\nname = "koder"\nversion = "3.4.5"\n', encoding="utf-8")
    monkeypatch.setattr(version, "package_version", _missing_distribution)
    monkeypatch.setattr(version, "_SOURCE_PROJECT_FILE", source)
    monkeypatch.setitem(sys.modules, "tomllib", None)
    monkeypatch.setitem(sys.modules, "tomli", None)
    assert version.resolve_package_version_info() == ("unknown", "unavailable")
