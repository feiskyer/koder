"""Web-related operation tools."""

from urllib.parse import urlparse

import requests
from agents import function_tool
from bs4 import BeautifulSoup
from ddgs import DDGS
from ddgs.exceptions import DDGSException
from pydantic import BaseModel


class SearchModel(BaseModel):
    """Model for web search input validation."""

    query: str
    max_results: int = 3


class WebFetchModel(BaseModel):
    """Model for web fetch input validation."""

    url: str
    prompt: str


@function_tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo.

    Args:
        query: The search query (1-400 characters)
        max_results: Maximum number of results to return (1-10, default 3)

    Returns:
        Formatted search results or error message
    """
    try:
        # Validate query
        if not query:
            return "Invalid query: query cannot be empty"
        if len(query) > 400:
            return "Invalid query: query must be 400 characters or less"

        max_results = max(1, min(max_results, 10))  # Clamp between 1 and 10

        ddgs = DDGS()
        results_list = list(ddgs.text(query, max_results=max_results))

        if not results_list:
            return "No results found"

        results = []
        for r in results_list:
            title = r.get("title", "No title")
            body = r.get("body", "No description")
            href = r.get("href", "")
            results.append(f"**{title}**\n{body}\n{href}")

        return "\n\n".join(results)

    except DDGSException as e:
        return f"Search failed: {str(e)}"
    except Exception as e:
        return f"Search error: {str(e)}"


@function_tool
def web_fetch(url: str, prompt: str) -> str:
    """Fetch content from a URL and process with a prompt."""
    try:
        # Validate URL
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return "Invalid URL format"

        if parsed.scheme not in ["http", "https"]:
            return "Only HTTP/HTTPS URLs are supported"

        # Fetch content
        response = requests.get(
            url,
            timeout=15,
            headers={"User-Agent": "Mozilla/5.0 (compatible; KoderAgent/1.0)"},
            allow_redirects=True,
            verify=True,
        )

        if response.status_code != 200:
            return f"Failed to fetch URL: HTTP {response.status_code}"

        # Limit content size (10MB)
        if len(response.content) > 10 * 1024 * 1024:
            return "Content too large (>10MB)"

        # Parse HTML content
        content_type = response.headers.get("content-type", "").lower()
        if "html" in content_type:
            soup = BeautifulSoup(response.text, "html.parser")

            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()

            # Extract text
            text = soup.get_text(separator="\n", strip=True)

            # Limit text length
            if len(text) > 50000:
                text = text[:50000] + "\n... (truncated)"
        else:
            # For non-HTML content, use raw text
            text = response.text
            if len(text) > 50000:
                text = text[:50000] + "\n... (truncated)"

        # Simple prompt processing (in real implementation, this would use an AI model)
        result = f"URL: {url}\n\n"
        result += f"Content Type: {content_type}\n\n"
        result += f"Prompt: {prompt}\n\n"
        result += f"Content Preview:\n{text[:1000]}..."

        return result

    except requests.Timeout:
        return "Request timed out"
    except requests.RequestException as e:
        return f"Request failed: {str(e)}"
    except Exception as e:
        return f"Error fetching content: {str(e)}"
