"""Pytest configuration for integration tests."""

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers",
        "manual: marks tests as requiring manual user interaction (skipped by default)",
    )


def pytest_addoption(parser):
    """Add command line options."""
    parser.addoption(
        "--manual",
        action="store_true",
        default=False,
        help="Run manual integration tests that require user interaction",
    )


def pytest_collection_modifyitems(config, items):
    """Skip manual tests unless --manual flag is provided."""
    if config.getoption("--manual"):
        # --manual given: do not skip manual tests
        return

    skip_manual = pytest.mark.skip(reason="Manual test - use --manual to run")
    for item in items:
        if "manual" in item.keywords:
            item.add_marker(skip_manual)
