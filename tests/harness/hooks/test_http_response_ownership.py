"""Failed HTTP hook responses must release their body, too."""

import io
import urllib.error

import pytest

from koder_agent.harness.hooks import runtime


@pytest.mark.parametrize("close_error", [False, True])
def test_http_hook_closes_error_response_without_returning_its_body(monkeypatch, close_error):
    class ResponseBody(io.BytesIO):
        def close(self):
            was_open = not self.closed
            super().close()
            if close_error and was_open:
                raise OSError("synthetic-private-cleanup-detail")

    body = ResponseBody(b"synthetic-private-response-body")
    error = urllib.error.HTTPError(
        "https://synthetic.invalid/hook", 503, "Service Unavailable", {}, body
    )

    def open_response(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(runtime.urllib.request, "urlopen", open_response)
    try:
        code, output, diagnostic = runtime._run_http_hook(
            url="https://synthetic.invalid/hook", payload_text="{}", timeout=1
        )
        assert code == 500 and output == ""
        assert "503" in diagnostic
        assert "synthetic-private-response-body" not in diagnostic
        assert "synthetic-private-cleanup-detail" not in diagnostic
        assert body.closed, "failed hook response body was not closed"
    finally:
        try:
            error.close()
        except OSError:
            pass  # The synthetic close failure must not prevent fixture cleanup.
