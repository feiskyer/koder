"""Tests for _capture_usage fallback logic in AgentScheduler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


class TestCaptureUsage:
    """Tests for AgentScheduler._capture_usage method."""

    @pytest.fixture
    def mock_scheduler(self):
        """Create a mock scheduler with necessary components."""
        with patch("koder_agent.core.scheduler.EnhancedSQLiteSession"):
            with patch("koder_agent.core.scheduler.get_all_tools", return_value=[]):
                with patch("koder_agent.core.scheduler.get_display_hooks"):
                    with patch("koder_agent.core.scheduler.ApprovalHooks"):
                        from koder_agent.core.scheduler import AgentScheduler

                        scheduler = AgentScheduler(session_id="test")
                        # Set up mock session with encoder
                        scheduler.session = MagicMock()
                        scheduler.session.encoder = MagicMock()
                        scheduler.session.encoder.encode = MagicMock(return_value=[1, 2, 3, 4, 5])
                        scheduler.session._estimate_tokens = MagicMock(return_value=1000)
                        scheduler.session.get_items = AsyncMock(
                            return_value=[{"role": "user", "content": "Hello"}]
                        )
                        # Mock usage tracker
                        scheduler.usage_tracker = MagicMock()
                        scheduler.usage_tracker.record_usage = MagicMock()
                        return scheduler

    @pytest.mark.asyncio
    async def test_capture_usage_from_api_response(self, mock_scheduler):
        """Test that usage is captured from API response when available."""
        # Create mock result with usage data
        result = MagicMock()
        result.context_wrapper = MagicMock()
        result.context_wrapper.usage = MagicMock()
        result.context_wrapper.usage.input_tokens = 500
        result.context_wrapper.usage.output_tokens = 200
        result.context_wrapper.usage.request_usage_entries = None

        await mock_scheduler._capture_usage(result)

        # Should record the API-provided usage
        mock_scheduler.usage_tracker.record_usage.assert_called_once_with(
            500, 200, context_tokens=None
        )

    @pytest.mark.asyncio
    async def test_capture_usage_with_context_tokens(self, mock_scheduler):
        """Test that context tokens are extracted from request_usage_entries."""
        result = MagicMock()
        result.context_wrapper = MagicMock()
        result.context_wrapper.usage = MagicMock()
        result.context_wrapper.usage.input_tokens = 500
        result.context_wrapper.usage.output_tokens = 200
        last_req = MagicMock()
        last_req.total_tokens = 700
        result.context_wrapper.usage.request_usage_entries = [last_req]

        await mock_scheduler._capture_usage(result)

        mock_scheduler.usage_tracker.record_usage.assert_called_once_with(
            500, 200, context_tokens=700
        )

    @pytest.mark.asyncio
    async def test_capture_usage_fallback_when_api_returns_zero(self, mock_scheduler):
        """Test fallback to tiktoken estimation when API returns zero tokens."""
        # Create mock result with zero usage (simulating API not returning usage)
        result = MagicMock()
        result.context_wrapper = MagicMock()
        result.context_wrapper.usage = MagicMock()
        result.context_wrapper.usage.input_tokens = 0
        result.context_wrapper.usage.output_tokens = 0
        result.context_wrapper.usage.request_usage_entries = None
        result.final_output = "This is the response text"

        await mock_scheduler._capture_usage(result)

        # Should have used session._estimate_tokens for input tokens
        mock_scheduler.session._estimate_tokens.assert_called_once()
        # Should have encoded the output for output tokens
        mock_scheduler.session.encoder.encode.assert_called()
        # Should record the estimated usage
        mock_scheduler.usage_tracker.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_usage_fallback_estimates_output_tokens(self, mock_scheduler):
        """Test that output tokens are estimated from final_output."""
        result = MagicMock()
        result.context_wrapper = MagicMock()
        result.context_wrapper.usage = MagicMock()
        result.context_wrapper.usage.input_tokens = 0
        result.context_wrapper.usage.output_tokens = 0
        result.context_wrapper.usage.request_usage_entries = None
        result.final_output = "Response text"

        # Make encoder return a specific length
        mock_scheduler.session.encoder.encode.return_value = list(range(50))

        await mock_scheduler._capture_usage(result)

        # Verify output tokens were estimated (50 from encoder)
        call_args = mock_scheduler.usage_tracker.record_usage.call_args
        assert call_args[0][1] == 50  # output_tokens

    @pytest.mark.asyncio
    async def test_capture_usage_fallback_estimates_input_tokens(self, mock_scheduler):
        """Test that input tokens are estimated from session history."""
        result = MagicMock()
        result.context_wrapper = MagicMock()
        result.context_wrapper.usage = MagicMock()
        result.context_wrapper.usage.input_tokens = 0
        result.context_wrapper.usage.output_tokens = 0
        result.context_wrapper.usage.request_usage_entries = None
        result.final_output = "Response"

        # Set up session._estimate_tokens to return a specific value
        mock_scheduler.session._estimate_tokens.return_value = 1500

        await mock_scheduler._capture_usage(result)

        # Verify input tokens were estimated (1500 from _estimate_tokens)
        call_args = mock_scheduler.usage_tracker.record_usage.call_args
        assert call_args[0][0] == 1500  # input_tokens

    @pytest.mark.asyncio
    async def test_capture_usage_fallback_calculates_context_tokens(self, mock_scheduler):
        """Test that context tokens are calculated as input + output in fallback."""
        result = MagicMock()
        result.context_wrapper = MagicMock()
        result.context_wrapper.usage = MagicMock()
        result.context_wrapper.usage.input_tokens = 0
        result.context_wrapper.usage.output_tokens = 0
        result.context_wrapper.usage.request_usage_entries = None
        result.final_output = "Response"

        mock_scheduler.session._estimate_tokens.return_value = 1000
        mock_scheduler.session.encoder.encode.return_value = list(range(200))

        await mock_scheduler._capture_usage(result)

        # context_tokens should be input + output = 1000 + 200 = 1200
        call_args = mock_scheduler.usage_tracker.record_usage.call_args
        assert call_args[1]["context_tokens"] == 1200

    @pytest.mark.asyncio
    async def test_capture_usage_no_record_when_no_tokens(self, mock_scheduler):
        """Test that nothing is recorded when both API and fallback return zero."""
        result = MagicMock()
        result.context_wrapper = MagicMock()
        result.context_wrapper.usage = MagicMock()
        result.context_wrapper.usage.input_tokens = 0
        result.context_wrapper.usage.output_tokens = 0
        result.context_wrapper.usage.request_usage_entries = None
        result.final_output = None  # No output

        # Session returns empty
        mock_scheduler.session.get_items = AsyncMock(return_value=[])
        mock_scheduler.session._estimate_tokens.return_value = 0

        await mock_scheduler._capture_usage(result)

        # Should not record when both are zero
        mock_scheduler.usage_tracker.record_usage.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_usage_handles_missing_context_wrapper(self, mock_scheduler):
        """Test graceful handling when result lacks context_wrapper."""
        result = MagicMock(spec=[])  # No attributes
        result.final_output = None

        # Make session return empty to ensure no fallback estimation
        mock_scheduler.session.get_items = AsyncMock(return_value=[])
        mock_scheduler.session._estimate_tokens.return_value = 0

        await mock_scheduler._capture_usage(result)

        # Should not crash, and should not record anything when all tokens are 0
        mock_scheduler.usage_tracker.record_usage.assert_not_called()

    @pytest.mark.asyncio
    async def test_capture_usage_handles_session_error(self, mock_scheduler):
        """Test graceful handling when session.get_items raises an error."""
        result = MagicMock()
        result.context_wrapper = MagicMock()
        result.context_wrapper.usage = MagicMock()
        result.context_wrapper.usage.input_tokens = 0
        result.context_wrapper.usage.output_tokens = 0
        result.context_wrapper.usage.request_usage_entries = None
        result.final_output = "Response"

        # Make get_items raise an error
        mock_scheduler.session.get_items = AsyncMock(side_effect=Exception("DB error"))

        # Should not crash
        await mock_scheduler._capture_usage(result)

        # Should still record output tokens even if input estimation failed
        mock_scheduler.usage_tracker.record_usage.assert_called_once()

    @pytest.mark.asyncio
    async def test_capture_usage_handles_none_usage(self, mock_scheduler):
        """Test handling when usage attribute is None."""
        result = MagicMock()
        result.context_wrapper = MagicMock()
        result.context_wrapper.usage = None
        result.final_output = "Response"

        await mock_scheduler._capture_usage(result)

        # Should fall back to estimation
        mock_scheduler.session.encoder.encode.assert_called()
