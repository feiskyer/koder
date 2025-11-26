"""Tests for web tools functionality."""

import json
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

# Stub litellm before importing koder_agent to avoid optional dependency issues
if "litellm" not in sys.modules:
    litellm_stub = types.ModuleType("litellm")
    litellm_stub.model_cost = {}
    sys.modules["litellm"] = litellm_stub

# Ensure project root is on sys.path when running tests directly
from pathlib import Path

project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import the web tools modules
from koder_agent.tools.web import (  # noqa: E402
    SearchModel,
    WebFetchModel,
    web_fetch,
    web_search,
)


class TestSearchModel:
    """Tests for SearchModel Pydantic model."""

    def test_search_model_defaults(self):
        """Test SearchModel with default max_results."""
        model = SearchModel(query="test query")
        assert model.query == "test query"
        assert model.max_results == 3

    def test_search_model_custom_max_results(self):
        """Test SearchModel with custom max_results."""
        model = SearchModel(query="python programming", max_results=5)
        assert model.query == "python programming"
        assert model.max_results == 5

    def test_search_model_validation(self):
        """Test SearchModel validation requires query."""
        with pytest.raises(Exception):
            SearchModel()


class TestWebFetchModel:
    """Tests for WebFetchModel Pydantic model."""

    def test_web_fetch_model(self):
        """Test WebFetchModel with valid inputs."""
        model = WebFetchModel(url="https://example.com", prompt="extract title")
        assert model.url == "https://example.com"
        assert model.prompt == "extract title"

    def test_web_fetch_model_validation(self):
        """Test WebFetchModel validation requires both fields."""
        with pytest.raises(Exception):
            WebFetchModel(url="https://example.com")
        with pytest.raises(Exception):
            WebFetchModel(prompt="extract title")


class TestWebSearch:
    """Tests for web_search function."""

    pytestmark = pytest.mark.asyncio

    async def test_web_search_empty_query(self):
        """Test web_search with empty query returns error."""
        result = await web_search.on_invoke_tool(None, json.dumps({"query": ""}))
        assert "Invalid query" in result
        assert "empty" in result.lower()

    async def test_web_search_query_too_long(self):
        """Test web_search with query exceeding 400 chars returns error."""
        long_query = "a" * 401
        result = await web_search.on_invoke_tool(None, json.dumps({"query": long_query}))
        assert "Invalid query" in result
        assert "400" in result

    async def test_web_search_max_results_clamped_low(self):
        """Test web_search clamps max_results to minimum of 1."""
        with patch("koder_agent.tools.web.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = []
            mock_ddgs.return_value = mock_instance

            await web_search.on_invoke_tool(None, json.dumps({"query": "test", "max_results": -5}))

            # Verify text was called with max_results=1 (clamped from -5)
            mock_instance.text.assert_called_once()
            call_kwargs = mock_instance.text.call_args
            assert call_kwargs[1]["max_results"] == 1

    async def test_web_search_max_results_clamped_high(self):
        """Test web_search clamps max_results to maximum of 10."""
        with patch("koder_agent.tools.web.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = []
            mock_ddgs.return_value = mock_instance

            await web_search.on_invoke_tool(None, json.dumps({"query": "test", "max_results": 100}))

            # Verify text was called with max_results=10 (clamped from 100)
            mock_instance.text.assert_called_once()
            call_kwargs = mock_instance.text.call_args
            assert call_kwargs[1]["max_results"] == 10

    async def test_web_search_no_results(self):
        """Test web_search when no results are returned."""
        with patch("koder_agent.tools.web.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = []
            mock_ddgs.return_value = mock_instance

            result = await web_search.on_invoke_tool(
                None, json.dumps({"query": "obscure query xyz123"})
            )
            assert result == "No results found"

    async def test_web_search_formats_results(self):
        """Test web_search properly formats returned results."""
        mock_results = [
            {
                "title": "Test Title 1",
                "body": "Test body description 1",
                "href": "https://example1.com",
            },
            {
                "title": "Test Title 2",
                "body": "Test body description 2",
                "href": "https://example2.com",
            },
        ]

        with patch("koder_agent.tools.web.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = mock_results
            mock_ddgs.return_value = mock_instance

            result = await web_search.on_invoke_tool(None, json.dumps({"query": "test query"}))

            assert "**Test Title 1**" in result
            assert "Test body description 1" in result
            assert "https://example1.com" in result
            assert "**Test Title 2**" in result
            assert "Test body description 2" in result
            assert "https://example2.com" in result

    async def test_web_search_handles_missing_fields(self):
        """Test web_search handles results with missing fields."""
        mock_results = [
            {"title": "Only Title"},  # Missing body and href
            {"body": "Only body"},  # Missing title and href
        ]

        with patch("koder_agent.tools.web.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.return_value = mock_results
            mock_ddgs.return_value = mock_instance

            result = await web_search.on_invoke_tool(None, json.dumps({"query": "test query"}))

            # Should use defaults for missing fields
            assert "Only Title" in result
            assert "No description" in result
            assert "Only body" in result
            assert "No title" in result

    async def test_web_search_handles_ddgs_exception(self):
        """Test web_search handles DDGSException."""
        from ddgs.exceptions import DDGSException

        with patch("koder_agent.tools.web.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.side_effect = DDGSException("Rate limited")
            mock_ddgs.return_value = mock_instance

            result = await web_search.on_invoke_tool(None, json.dumps({"query": "test query"}))

            assert "Search failed" in result
            assert "Rate limited" in result

    async def test_web_search_handles_generic_exception(self):
        """Test web_search handles generic exceptions."""
        with patch("koder_agent.tools.web.DDGS") as mock_ddgs:
            mock_instance = MagicMock()
            mock_instance.text.side_effect = RuntimeError("Network error")
            mock_ddgs.return_value = mock_instance

            result = await web_search.on_invoke_tool(None, json.dumps({"query": "test query"}))

            assert "Search error" in result
            assert "Network error" in result


class TestWebFetch:
    """Tests for web_fetch function."""

    pytestmark = pytest.mark.asyncio

    async def test_web_fetch_invalid_url_format(self):
        """Test web_fetch with invalid URL format."""
        result = await web_fetch.on_invoke_tool(
            None, json.dumps({"url": "not-a-url", "prompt": "test"})
        )
        assert "Invalid URL format" in result

    async def test_web_fetch_missing_scheme(self):
        """Test web_fetch with URL missing scheme."""
        result = await web_fetch.on_invoke_tool(
            None, json.dumps({"url": "example.com/page", "prompt": "test"})
        )
        assert "Invalid URL format" in result

    async def test_web_fetch_unsupported_scheme(self):
        """Test web_fetch with unsupported URL scheme."""
        result = await web_fetch.on_invoke_tool(
            None, json.dumps({"url": "ftp://example.com/file", "prompt": "test"})
        )
        assert "Only HTTP/HTTPS URLs are supported" in result

    async def test_web_fetch_timeout(self):
        """Test web_fetch handles timeout."""
        import requests

        with patch("koder_agent.tools.web.requests.get") as mock_get:
            mock_get.side_effect = requests.Timeout("Connection timed out")

            result = await web_fetch.on_invoke_tool(
                None, json.dumps({"url": "https://example.com", "prompt": "test"})
            )

            assert "Request timed out" in result

    async def test_web_fetch_request_exception(self):
        """Test web_fetch handles request exceptions."""
        import requests

        with patch("koder_agent.tools.web.requests.get") as mock_get:
            mock_get.side_effect = requests.RequestException("Connection refused")

            result = await web_fetch.on_invoke_tool(
                None, json.dumps({"url": "https://example.com", "prompt": "test"})
            )

            assert "Request failed" in result
            assert "Connection refused" in result

    async def test_web_fetch_http_error_status(self):
        """Test web_fetch handles non-200 HTTP status."""
        with patch("koder_agent.tools.web.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 404
            mock_get.return_value = mock_response

            result = await web_fetch.on_invoke_tool(
                None, json.dumps({"url": "https://example.com/notfound", "prompt": "test"})
            )

            assert "Failed to fetch URL" in result
            assert "404" in result

    async def test_web_fetch_content_too_large(self):
        """Test web_fetch handles content exceeding 10MB."""
        with patch("koder_agent.tools.web.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = b"x" * (11 * 1024 * 1024)  # 11MB
            mock_get.return_value = mock_response

            result = await web_fetch.on_invoke_tool(
                None, json.dumps({"url": "https://example.com/largefile", "prompt": "test"})
            )

            assert "Content too large" in result
            assert "10MB" in result

    async def test_web_fetch_html_content(self):
        """Test web_fetch properly parses HTML content."""
        html_content = """
        <html>
        <head><title>Test Page</title></head>
        <body>
            <script>console.log('should be removed');</script>
            <style>.hidden { display: none; }</style>
            <h1>Hello World</h1>
            <p>This is test content.</p>
        </body>
        </html>
        """

        with patch("koder_agent.tools.web.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = html_content.encode()
            mock_response.text = html_content
            mock_response.headers = {"content-type": "text/html; charset=utf-8"}
            mock_get.return_value = mock_response

            result = await web_fetch.on_invoke_tool(
                None, json.dumps({"url": "https://example.com", "prompt": "extract text"})
            )

            # Check result contains expected content
            assert "https://example.com" in result
            assert "text/html" in result
            assert "extract text" in result
            # Script and style content should be removed
            assert "console.log" not in result
            assert ".hidden" not in result

    async def test_web_fetch_non_html_content(self):
        """Test web_fetch handles non-HTML content."""
        json_content = '{"key": "value", "number": 42}'

        with patch("koder_agent.tools.web.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = json_content.encode()
            mock_response.text = json_content
            mock_response.headers = {"content-type": "application/json"}
            mock_get.return_value = mock_response

            result = await web_fetch.on_invoke_tool(
                None, json.dumps({"url": "https://api.example.com/data", "prompt": "parse json"})
            )

            assert "application/json" in result
            assert "parse json" in result

    async def test_web_fetch_truncates_long_content(self):
        """Test web_fetch truncates content exceeding 50000 chars."""
        long_content = "x" * 60000

        with patch("koder_agent.tools.web.requests.get") as mock_get:
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.content = long_content.encode()
            mock_response.text = long_content
            mock_response.headers = {"content-type": "text/plain"}
            mock_get.return_value = mock_response

            result = await web_fetch.on_invoke_tool(
                None, json.dumps({"url": "https://example.com/longfile", "prompt": "test"})
            )

            # The preview should be truncated
            assert "truncated" in result.lower() or len(result) < 60000

    async def test_web_fetch_handles_generic_exception(self):
        """Test web_fetch handles unexpected exceptions."""
        with patch("koder_agent.tools.web.requests.get") as mock_get:
            mock_get.side_effect = RuntimeError("Unexpected error")

            result = await web_fetch.on_invoke_tool(
                None, json.dumps({"url": "https://example.com", "prompt": "test"})
            )

            assert "Error fetching content" in result
            assert "Unexpected error" in result


class TestWebSearchIntegration:
    """Integration tests for web_search (requires network)."""

    pytestmark = pytest.mark.asyncio

    @pytest.mark.skip(reason="Integration test - requires network access")
    async def test_web_search_real_query(self):
        """Test actual web search with real network call."""
        result = await web_search.on_invoke_tool(
            None, json.dumps({"query": "python programming language", "max_results": 3})
        )

        # Should return actual results
        assert "No results found" not in result
        assert "**" in result  # Title formatting
        assert "http" in result.lower()  # URLs present


class TestWebFetchIntegration:
    """Integration tests for web_fetch (requires network)."""

    pytestmark = pytest.mark.asyncio

    @pytest.mark.skip(reason="Integration test - requires network access")
    async def test_web_fetch_real_url(self):
        """Test actual web fetch with real network call."""
        result = await web_fetch.on_invoke_tool(
            None, json.dumps({"url": "https://httpbin.org/html", "prompt": "extract content"})
        )

        assert "httpbin.org" in result
        assert "text/html" in result.lower()
