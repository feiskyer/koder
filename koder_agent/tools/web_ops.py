"""Web-related operation tools."""

from urllib.parse import urlparse

import requests
from agents import function_tool
from bs4 import BeautifulSoup
from pydantic import BaseModel


class SearchModel(BaseModel):
    query: str
    max_results: int = 3


class WebFetchModel(BaseModel):
    url: str
    prompt: str


@function_tool
def web_search(query: str, max_results: int = 3) -> str:
    """Search the web using DuckDuckGo."""
    try:
        # Validate query
        if not query or len(query) > 200:
            return "Invalid query: must be between 1 and 200 characters"

        max_results = min(max_results, 10)  # Cap at 10 results

        response = requests.get(
            "https://duckduckgo.com/html/",
            params={"q": query},
            timeout=10,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"},
        )

        if response.status_code != 200:
            return f"Search failed with status code: {response.status_code}"

        soup = BeautifulSoup(response.text, "html.parser")

        results = []
        for result in soup.select(".result")[:max_results]:
            title_elem = result.select_one(".result__title")
            snippet_elem = result.select_one(".result__snippet")
            link_elem = result.select_one(".result__url")

            if title_elem and snippet_elem:
                title = title_elem.get_text(strip=True)
                snippet = snippet_elem.get_text(" ", strip=True)
                link = link_elem.get_text(strip=True) if link_elem else ""

                results.append(f"**{title}**\n{snippet}\n{link}")

        return "\n\n".join(results) if results else "No results found"

    except requests.Timeout:
        return "Search request timed out"
    except requests.RequestException as e:
        return f"Search request failed: {str(e)}"
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
