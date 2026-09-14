"""Tests for marketplace source registration."""

import json
import subprocess
import sys
import types
from pathlib import Path

import pytest

if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

project_root = Path(__file__).resolve().parents[3]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from koder_agent.harness.plugins import marketplace as marketplace_module
from koder_agent.harness.plugins.marketplace import MarketplaceStore, MarketplaceStoreError


def test_add_accepts_mixed_case_github_marketplace_name(tmp_path, monkeypatch):
    cache_root = tmp_path / "marketplace-cache"
    clone_calls = []

    def fake_clone(repo, target):
        clone_calls.append((repo, target))
        target.mkdir(parents=True)
        return True

    monkeypatch.setattr(marketplace_module, "_marketplace_cache_dir", lambda: cache_root)
    monkeypatch.setattr(marketplace_module, "_clone_github_repo", fake_clone)

    source, message = MarketplaceStore.for_test(tmp_path).add("Acme/MixedCase-Market")

    assert source is not None
    assert source.name == "mixedcase-market"
    expected_cache = marketplace_module._repository_cache_path(
        "mixedcase-market", "Acme/MixedCase-Market"
    )
    assert expected_cache.parent == cache_root
    assert expected_cache.name.startswith("mixedcase-market-")
    assert source.path == str(expected_cache)
    assert clone_calls == [("Acme/MixedCase-Market", expected_cache)]
    assert message == "Added marketplace: mixedcase-market"


def test_add_accepts_mixed_case_local_marketplace_directory(tmp_path):
    marketplace_dir = tmp_path / "CommunityPlugins"
    marketplace_dir.mkdir()

    source, message = MarketplaceStore.for_test(tmp_path).add(str(marketplace_dir))

    assert source is not None
    assert source.name == "communityplugins"
    assert source.path == str(marketplace_dir.resolve())
    assert message == "Added marketplace: communityplugins"


def test_add_rejects_different_source_with_same_canonical_name(tmp_path):
    first = tmp_path / "first" / "Community"
    second = tmp_path / "second" / "community"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    store = MarketplaceStore.for_test(tmp_path)
    added, _message = store.add(str(first))
    assert added is not None

    rejected, message = store.add(str(second))

    assert rejected is None
    assert "different source" in message
    assert "remove it before adding a replacement" in message
    registered = store.get("community")
    assert registered is not None
    assert registered.path == str(first.resolve())


def test_add_same_source_is_idempotent(tmp_path):
    marketplace_dir = tmp_path / "Community"
    marketplace_dir.mkdir()
    store = MarketplaceStore.for_test(tmp_path)
    first, _message = store.add(str(marketplace_dir))

    second, message = store.add(str(marketplace_dir))

    assert first == second
    assert message == "Marketplace already added: community"
    assert store.list_all() == [first]


def test_load_migrates_legacy_mixed_case_name(tmp_path):
    source_dir = tmp_path / "Community"
    source_dir.mkdir()
    store_path = tmp_path / "marketplaces.json"
    store_path.write_text(
        json.dumps(
            {
                "Community": {
                    "source_type": "directory",
                    "path": str(source_dir),
                    "raw_source": str(source_dir),
                }
            }
        ),
        encoding="utf-8",
    )

    store = MarketplaceStore.for_test(tmp_path)
    source = store.get("COMMUNITY")

    assert source is not None
    assert source.name == "community"
    assert list(json.loads(store_path.read_text(encoding="utf-8"))) == ["community"]


def test_add_rejects_new_source_after_legacy_name_is_canonicalized(tmp_path):
    first = tmp_path / "first" / "Community"
    second = tmp_path / "second" / "community"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    store_path = tmp_path / "marketplaces.json"
    store_path.write_text(
        json.dumps(
            {
                "Community": {
                    "source_type": "directory",
                    "path": str(first),
                    "raw_source": str(first),
                }
            }
        ),
        encoding="utf-8",
    )
    store = MarketplaceStore.for_test(tmp_path)

    rejected, message = store.add(str(second))

    assert rejected is None
    assert "different source" in message
    persisted = json.loads(store_path.read_text(encoding="utf-8"))
    assert list(persisted) == ["community"]
    assert persisted["community"]["raw_source"] == str(first)


def test_fresh_process_migrates_legacy_name_before_collision_check(
    tmp_path, python_child_environment
):
    first = tmp_path / "first" / "Community"
    second = tmp_path / "second" / "community"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    store_path = tmp_path / "marketplaces.json"
    store_path.write_text(
        json.dumps(
            {
                "Community": {
                    "source_type": "directory",
                    "path": str(first),
                    "raw_source": str(first),
                }
            }
        ),
        encoding="utf-8",
    )
    script = """
import json
import sys
from pathlib import Path

from koder_agent.harness.plugins.marketplace import MarketplaceStore

store = MarketplaceStore(Path(sys.argv[1]))
source, message = store.add(sys.argv[2])
print(json.dumps({"accepted": source is not None, "message": message}))
"""

    result = subprocess.run(
        [sys.executable, "-c", script, str(store_path), str(second)],
        cwd=tmp_path,
        env=python_child_environment,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    response = json.loads(result.stdout)
    assert response["accepted"] is False
    assert "different source" in response["message"]
    persisted = json.loads(store_path.read_text(encoding="utf-8"))
    assert list(persisted) == ["community"]
    assert persisted["community"]["raw_source"] == str(first)


def test_load_merges_same_source_canonical_duplicates(tmp_path):
    source_dir = tmp_path / "Community"
    source_dir.mkdir()
    store_path = tmp_path / "marketplaces.json"
    entry = {
        "source_type": "directory",
        "path": str(source_dir),
        "raw_source": str(source_dir),
    }
    store_path.write_text(
        json.dumps({"Community": entry, "community": entry}),
        encoding="utf-8",
    )

    store = MarketplaceStore.for_test(tmp_path)

    assert [source.name for source in store.list_all()] == ["community"]
    assert json.loads(store_path.read_text(encoding="utf-8")) == {"community": entry}


def test_load_rejects_different_source_canonical_collision_without_rewriting(tmp_path):
    first = tmp_path / "first" / "Community"
    second = tmp_path / "second" / "community"
    first.mkdir(parents=True)
    second.mkdir(parents=True)
    store_path = tmp_path / "marketplaces.json"
    persisted = {
        "Community": {
            "source_type": "directory",
            "path": str(first),
            "raw_source": str(first),
        },
        "community": {
            "source_type": "directory",
            "path": str(second),
            "raw_source": str(second),
        },
    }
    original = json.dumps(persisted)
    store_path.write_text(original, encoding="utf-8")
    store = MarketplaceStore.for_test(tmp_path)

    with pytest.raises(MarketplaceStoreError, match="different sources"):
        store.list_all()

    rejected, message = store.add(str(second))
    assert rejected is None
    assert "different sources" in message
    assert store_path.read_text(encoding="utf-8") == original
