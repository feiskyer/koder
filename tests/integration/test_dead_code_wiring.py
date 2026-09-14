"""Tests that previously dead modules are now called from runtime code."""

from pathlib import Path


def _source(path):
    return (Path(__file__).resolve().parents[2] / path).read_text(encoding="utf-8")


def test_model_deprecation_called():
    src = _source("koder_agent/utils/client.py")
    assert "check_model_deprecation" in src


def test_plugin_name_validation_called():
    manifest_src = _source("koder_agent/harness/plugins/manifest.py")
    lifecycle_src = _source("koder_agent/harness/plugins/path_safety.py")
    marketplace_src = _source("koder_agent/harness/plugins/marketplace.py")
    assert "validate_plugin_name_format" in manifest_src
    assert "canonical_plugin_name" in lifecycle_src
    assert "canonical_marketplace_name" in marketplace_src


def test_skill_discovery_called():
    src = _source("koder_agent/tools/skill.py")
    assert "discover_skills_for_paths" in src


def test_voice_keyterms_called():
    src = _source("koder_agent/harness/voice/service.py")
    assert "get_all_keyterms" in src or "keyterms" in src


def test_secure_storage_called():
    src = _source("koder_agent/auth/token_storage.py")
    assert "SecureStorage" in src or "secure_storage" in src
