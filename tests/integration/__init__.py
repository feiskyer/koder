"""Integration tests for OAuth authentication.

These tests require manual user interaction to complete OAuth flows.
They are marked with @pytest.mark.manual and skipped by default in CI.

To run these tests locally:
    pytest tests/integration/ -m manual --manual

Note: You will need to complete OAuth login in a browser when prompted.
"""
