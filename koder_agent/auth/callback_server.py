"""OAuth callback server for handling authorization redirects.

Provides a local HTTP server that receives OAuth authorization callbacks
and extracts the authorization code for token exchange.
"""

import asyncio
import webbrowser
from dataclasses import dataclass
from typing import Optional

from aiohttp import web


@dataclass
class CallbackResult:
    """Result from OAuth callback."""

    success: bool
    code: Optional[str] = None
    state: Optional[str] = None
    error: Optional[str] = None
    error_description: Optional[str] = None


# HTML templates for callback responses
SUCCESS_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Authentication Successful</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        }
        .container {
            text-align: center;
            padding: 40px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            max-width: 400px;
        }
        .icon { font-size: 64px; margin-bottom: 20px; }
        h1 { color: #333; margin-bottom: 10px; }
        p { color: #666; }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">&#10004;</div>
        <h1>Authentication Successful!</h1>
        <p>You can close this window and return to the terminal.</p>
    </div>
</body>
</html>
"""

ERROR_HTML = """
<!DOCTYPE html>
<html>
<head>
    <title>Authentication Failed</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            display: flex;
            justify-content: center;
            align-items: center;
            height: 100vh;
            margin: 0;
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        }
        .container {
            text-align: center;
            padding: 40px;
            background: white;
            border-radius: 12px;
            box-shadow: 0 10px 40px rgba(0,0,0,0.2);
            max-width: 400px;
        }
        .icon { font-size: 64px; margin-bottom: 20px; }
        h1 { color: #333; margin-bottom: 10px; }
        p { color: #666; }
        .error { color: #e74c3c; font-family: monospace; margin-top: 15px; }
    </style>
</head>
<body>
    <div class="container">
        <div class="icon">&#10060;</div>
        <h1>Authentication Failed</h1>
        <p>An error occurred during authentication.</p>
        <p class="error">{error}</p>
    </div>
</body>
</html>
"""


class OAuthCallbackServer:
    """Local HTTP server for OAuth callbacks.

    Starts a temporary server on localhost to receive OAuth redirect
    with authorization code.
    """

    def __init__(self, port: int, callback_path: str = "/auth/callback"):
        """Initialize callback server.

        Args:
            port: Port to listen on
            callback_path: URL path to handle callbacks on
        """
        self.port = port
        self.callback_path = callback_path
        self._result: Optional[CallbackResult] = None
        self._callback_received = asyncio.Event()
        self._app: Optional[web.Application] = None
        self._runner: Optional[web.AppRunner] = None
        self._site: Optional[web.TCPSite] = None

    async def _handle_callback(self, request: web.Request) -> web.Response:
        """Handle OAuth callback request.

        Args:
            request: Incoming HTTP request

        Returns:
            HTML response
        """
        # Parse query parameters
        query = request.query

        # Check for error
        error = query.get("error")
        if error:
            self._result = CallbackResult(
                success=False,
                error=error,
                error_description=query.get("error_description"),
            )
            self._callback_received.set()
            return web.Response(
                text=ERROR_HTML.format(error=f"{error}: {query.get('error_description', '')}"),
                content_type="text/html",
            )

        # Extract code and state
        code = query.get("code")
        state = query.get("state")

        if not code:
            self._result = CallbackResult(
                success=False,
                error="missing_code",
                error_description="No authorization code in callback",
            )
            self._callback_received.set()
            return web.Response(
                text=ERROR_HTML.format(error="No authorization code received"),
                content_type="text/html",
            )

        self._result = CallbackResult(
            success=True,
            code=code,
            state=state,
        )
        self._callback_received.set()

        return web.Response(text=SUCCESS_HTML, content_type="text/html")

    async def start(self) -> None:
        """Start the callback server."""
        self._app = web.Application()
        self._app.router.add_get(self.callback_path, self._handle_callback)

        self._runner = web.AppRunner(self._app)
        await self._runner.setup()

        self._site = web.TCPSite(self._runner, "localhost", self.port)
        await self._site.start()

    async def stop(self) -> None:
        """Stop the callback server."""
        if self._site:
            await self._site.stop()
        if self._runner:
            await self._runner.cleanup()

    async def wait_for_callback(self, timeout: float = 300) -> CallbackResult:
        """Wait for OAuth callback.

        Args:
            timeout: Maximum time to wait in seconds

        Returns:
            CallbackResult with code or error

        Raises:
            asyncio.TimeoutError: If timeout expires
        """
        try:
            await asyncio.wait_for(self._callback_received.wait(), timeout=timeout)
            return self._result or CallbackResult(
                success=False, error="unknown", error_description="No result received"
            )
        except asyncio.TimeoutError:
            return CallbackResult(
                success=False,
                error="timeout",
                error_description=f"Callback not received within {timeout} seconds",
            )

    async def __aenter__(self):
        """Async context manager entry."""
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        await self.stop()


async def run_oauth_flow(
    auth_url: str,
    port: int,
    callback_path: str = "/auth/callback",
    timeout: float = 300,
    open_browser: bool = True,
) -> CallbackResult:
    """Run complete OAuth flow with callback server.

    Args:
        auth_url: Authorization URL to open
        port: Port for callback server to listen on
        callback_path: URL path to handle callbacks
        timeout: Maximum time to wait for callback
        open_browser: Whether to automatically open browser

    Returns:
        CallbackResult with authorization code or error
    """
    async with OAuthCallbackServer(port=port, callback_path=callback_path) as server:
        # Open browser
        if open_browser:
            print("\nOpening browser for authentication...")
            webbrowser.open(auth_url)
        else:
            print(f"\nPlease open this URL in your browser:\n{auth_url}")

        print(f"\nWaiting for authentication (timeout: {timeout}s)...")

        # Wait for callback
        result = await server.wait_for_callback(timeout=timeout)

        return result
