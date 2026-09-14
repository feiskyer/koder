"""A re-used catalog name must not relabel another repository's cache."""

import os
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest

from koder_agent.harness.plugins import marketplace
from koder_agent.harness.plugins.marketplace import MarketplaceStore


def test_rebinding_name_uses_a_distinct_repository_cache(tmp_path, monkeypatch):
    cache_root = tmp_path / "cache"
    monkeypatch.setattr(marketplace, "_marketplace_cache_dir", lambda: cache_root)

    def clone(repo, target):
        # Pulling an existing checkout preserves its original repository.
        if not target.exists():
            target.mkdir(parents=True)
            (target / "repository.txt").write_text(repo)
        return True

    monkeypatch.setattr(marketplace, "_clone_github_repo", clone)
    store = MarketplaceStore.for_test(tmp_path)
    first, _ = store.add("Alpha/community")
    assert store.remove("community")
    second, _ = store.add("Beta/community")
    assert first is not None and second is not None
    assert first.path != second.path
    assert (Path(first.path) / "repository.txt").read_text() == "Alpha/community"
    assert (Path(second.path) / "repository.txt").read_text() == "Beta/community"


@pytest.mark.parametrize("wrong_origin", [False, True])
def test_existing_cache_requires_matching_origin_and_successful_pull(
    tmp_path, monkeypatch, wrong_origin
):
    cache = tmp_path / "cache"
    cache.mkdir()
    calls = []

    def git(argv, **_kwargs):
        calls.append(argv)
        if "get-url" in argv:
            origin = (
                "https://github.com/Other/repo.git"
                if wrong_origin
                else "https://github.com/Acme/repo.git"
            )
            return SimpleNamespace(returncode=0, stdout=origin + "\n")
        return SimpleNamespace(returncode=0 if wrong_origin else 1, stdout="")

    monkeypatch.setattr(marketplace.subprocess, "run", git)
    assert not marketplace._clone_github_repo("Acme/repo", cache)
    if wrong_origin:
        assert not any("pull" in argv for argv in calls)


def test_matching_cache_can_be_updated(tmp_path, monkeypatch):
    cache = tmp_path / "cache"
    cache.mkdir()

    def git(argv, **_kwargs):
        return SimpleNamespace(
            returncode=0, stdout="https://github.com/Acme/repo.git\n" if "get-url" in argv else ""
        )

    monkeypatch.setattr(marketplace.subprocess, "run", git)
    assert marketplace._clone_github_repo("Acme/repo", cache)


def test_real_git_cache_keeps_origin_and_preserves_mismatched_checkout(tmp_path, monkeypatch):
    monkeypatch.setenv("GIT_CONFIG_GLOBAL", os.devnull)
    monkeypatch.setenv("GIT_CONFIG_SYSTEM", os.devnull)
    repositories = []
    for label in ("alpha", "beta"):
        repository = tmp_path / label
        repository.mkdir()
        subprocess.run(["git", "init", "-q", str(repository)], check=True)
        (repository / "payload.txt").write_text(label)
        subprocess.run(["git", "-C", str(repository), "add", "payload.txt"], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repository),
                "-c",
                "user.name=Synthetic Test",
                "-c",
                "user.email=synthetic@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        repositories.append(repository)
    cached = tmp_path / "cached"
    assert marketplace._clone_repository(str(repositories[0]), cached)
    assert marketplace._clone_repository(str(repositories[0]), cached)
    assert not marketplace._clone_repository(str(repositories[1]), cached)
    assert (cached / "payload.txt").read_text() == "alpha"
    replacement = tmp_path / "new-cache"
    assert marketplace._clone_repository(str(repositories[1]), replacement)
    assert (replacement / "payload.txt").read_text() == "beta"


def test_registry_atomic_write_failure_preserves_existing_sources(tmp_path, monkeypatch):
    from koder_agent.utils import atomic_file

    source = tmp_path / "Community"
    source.mkdir()
    store = MarketplaceStore.for_test(tmp_path)
    assert store.add(str(source))[0] is not None
    before = store._path.read_bytes()

    def fail_replace(*_args, **_kwargs):
        raise OSError("synthetic publication failure")

    monkeypatch.setattr(atomic_file.os, "replace", fail_replace)
    with pytest.raises(OSError, match="synthetic publication"):
        store.remove("community")
    assert store._path.read_bytes() == before
