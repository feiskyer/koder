"""Status line UI component for displaying model, session, and usage info."""

import os
from typing import TYPE_CHECKING, Optional

from prompt_toolkit.formatted_text import FormattedText
from prompt_toolkit.layout import FormattedTextControl, Window

from ..utils.model_info import get_context_window_size

if TYPE_CHECKING:
    from .usage_tracker import UsageTracker


class StatusLine:
    """
    Status line component showing model, directory, session, tokens, cost, and context usage.

    The status line is rendered below the input box and updates dynamically.
    """

    def __init__(
        self,
        usage_tracker: "UsageTracker",
        session_id: str,
    ):
        """
        Initialize the status line.

        Args:
            usage_tracker: UsageTracker instance for token/cost data
            session_id: Current session identifier
        """
        self.usage_tracker = usage_tracker
        self.session_id = session_id
        self._display_name: Optional[str] = None  # AI-generated display name

    def _format_tokens(self, n: int) -> str:
        """Format token count with k/M suffix for readability."""
        if n >= 1_000_000:
            return f"{n / 1_000_000:.1f}M"
        elif n >= 1000:
            # Remove trailing .0 for cleaner display (e.g., "200k" not "200.0k")
            val = n / 1000
            if val == int(val):
                return f"{int(val)}k"
            return f"{val:.1f}k"
        return str(n)

    def _truncate(self, s: str, max_len: int, from_start: bool = False) -> str:
        """Truncate string with ellipsis."""
        if len(s) <= max_len:
            return s
        if from_start:
            return s[: max_len - 3] + "..."
        return "..." + s[-(max_len - 3) :]

    def _get_context_style(self, percentage: float) -> str:
        """Get style class based on context usage percentage."""
        if percentage >= 90:
            return "class:status-context-critical"
        elif percentage >= 70:
            return "class:status-context-warn"
        return "class:status-context-ok"

    def get_formatted_text(self) -> FormattedText:
        """Generate the formatted status line text."""
        # Model name - use cached model from usage_tracker to avoid repeated lookups
        model = self.usage_tracker.model
        if "/" in model:
            display_model = model.replace("litellm/", "")
            # If still has provider prefix (e.g., openai/gpt-4o), show just the model
            if "/" in display_model:
                display_model = display_model.split("/")[-1]
        else:
            display_model = model

        # Working directory with ~ for home (only replace leading home path)
        cwd = os.getcwd()
        home = os.path.expanduser("~")
        if cwd.startswith(home):
            cwd = "~" + cwd[len(home) :]

        # Session: use display name if available, otherwise session ID (truncated)
        session_display = self._display_name or self.session_id
        session = self._truncate(session_display, 20, from_start=True)

        # Usage data
        usage = self.usage_tracker.session_usage
        cost_str = f"${usage.total_cost:.4f}"

        # Context window usage: current_context_tokens
        # This represents the total context that will be sent in the next turn
        # (assuming sessions automatically include previous conversation history)
        current_tokens = usage.current_context_tokens
        max_context = get_context_window_size(model)
        context_pct = (current_tokens / max_context * 100) if max_context > 0 else 0
        context_style = self._get_context_style(context_pct)

        # Format tokens - show dash before first API call for clearer UX
        if usage.request_count == 0:
            tokens_str = f"–/{self._format_tokens(max_context)}"
        else:
            tokens_str = f"{self._format_tokens(current_tokens)}/{self._format_tokens(max_context)}"

        return FormattedText(
            [
                ("class:status-label", " Model: "),
                ("class:status-value", self._truncate(display_model, 25)),
                ("class:status-separator", " | "),
                ("class:status-label", "Dir: "),
                ("class:status-value", self._truncate(cwd, 25)),
                ("class:status-separator", " | "),
                ("class:status-label", "Session: "),
                ("class:status-value", session),
                ("class:status-separator", " | "),
                ("class:status-label", "Tokens: "),
                ("class:status-value", tokens_str + " "),
                (context_style, f"({context_pct:.1f}%)"),
                ("class:status-separator", " | "),
                ("class:status-label", "Cost: "),
                ("class:status-value", cost_str),
            ]
        )

    def create_window(self) -> Window:
        """Create a prompt_toolkit Window for the status line."""
        return Window(
            content=FormattedTextControl(self.get_formatted_text),
            height=1,
            dont_extend_height=True,
        )

    def update_session(self, session_id: str) -> None:
        """Update the session ID displayed in the status line."""
        self.session_id = session_id
        self._display_name = None  # Reset display name on session change

    def update_display_name(self, display_name: str) -> None:
        """Update the display name (AI-generated title) for the status line."""
        self._display_name = display_name
