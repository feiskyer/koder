"""Status line UI component for displaying model, session, and usage info."""

import json
import os
import shutil
import subprocess
import threading
import unicodedata
from typing import TYPE_CHECKING, Optional

from prompt_toolkit.formatted_text import ANSI, FormattedText, to_formatted_text
from prompt_toolkit.layout import FormattedTextControl, Window

from ..harness.statusline_settings import resolve_statusline_config
from ..harness.version_info import resolve_runtime_version
from ..utils.client import get_configured_context_window

if TYPE_CHECKING:
    from ..harness.pr_status import PrStatusPoller
    from .usage_tracker import UsageTracker

# Review-state -> prompt_toolkit style class mapping.
_PR_STATE_STYLES: dict[str, str] = {
    "approved": "class:status-pr-approved",
    "pending": "class:status-pr-pending",
    "changes_requested": "class:status-pr-changes-requested",
    "draft": "class:status-pr-draft",
    "merged": "class:status-pr-merged",
}

_PR_STATE_LABELS: dict[str, str] = {
    "approved": "approved",
    "pending": "pending",
    "changes_requested": "changes requested",
    "draft": "draft",
    "merged": "merged",
}

# Absolute token warning threshold. This is independent of the context-window
# percentage: even on models with very large context windows, a session this
# large is worth flagging (cost/latency). Env-overridable for power users/tests.
_DEFAULT_ABSOLUTE_TOKEN_LIMIT = 200_000


def _absolute_token_limit() -> int:
    """Resolve the absolute token warning threshold (env-overridable)."""
    raw = os.environ.get("KODER_TOKEN_WARN_LIMIT")
    if raw:
        try:
            value = int(raw)
            if value > 0:
                return value
        except (TypeError, ValueError):
            pass
    return _DEFAULT_ABSOLUTE_TOKEN_LIMIT


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
        self._notice: Optional[str] = None
        self.pr_poller: Optional["PrStatusPoller"] = None
        self._project_dir = os.getcwd()
        self._cached_command_signature: str | None = None
        self._cached_command_output: str | None = None
        self._command_lock = threading.Lock()
        self._command_refresh_pending = False

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

    @staticmethod
    def _display_width(text: str) -> int:
        """Return the terminal display width of *text*.

        CJK ideographs, fullwidth forms, and most emoji occupy 2 columns in a
        terminal; all other printable characters occupy 1. Uses
        :func:`unicodedata.east_asian_width` for the classification.
        """
        width = 0
        for ch in text:
            eaw = unicodedata.east_asian_width(ch)
            width += 2 if eaw in ("W", "F") else 1
        return width

    def _truncate(self, s: str, max_len: int, from_start: bool = False) -> str:
        """Truncate string with ellipsis, respecting display width."""
        if max_len <= 0:
            return ""
        if max_len <= 3:
            return s[:max_len]
        if self._display_width(s) <= max_len:
            return s
        if from_start:
            # Truncate keeping the start, append "..."
            result: list[str] = []
            used = 3  # reserve for "..."
            for ch in s:
                cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                if used + cw > max_len:
                    break
                result.append(ch)
                used += cw
            return "".join(result) + "..."
        else:
            # Truncate keeping the end, prepend "..."
            result_rev: list[str] = []
            used = 3  # reserve for "..."
            for ch in reversed(s):
                cw = 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1
                if used + cw > max_len:
                    break
                result_rev.append(ch)
                used += cw
            return "..." + "".join(reversed(result_rev))

    def _terminal_columns(self) -> int:
        """Return the current terminal width with a conservative fallback."""
        try:
            return shutil.get_terminal_size((100, 24)).columns
        except Exception:
            return 100

    def _compact_cwd(self, cwd: str, max_len: int) -> str:
        """Render the cwd like Claude's compact status line: stable and short."""
        home = os.path.expanduser("~")
        if cwd.startswith(home):
            cwd = "~" + cwd[len(home) :]
        if self._display_width(cwd) <= max_len:
            return cwd

        basename = os.path.basename(cwd.rstrip(os.sep)) or cwd
        parent = os.path.basename(os.path.dirname(cwd.rstrip(os.sep)))
        compact = f".../{parent}/{basename}" if parent else basename
        if self._display_width(compact) <= max_len:
            return compact
        return self._truncate(cwd, max_len)

    def _append_field(
        self,
        fragments: list[tuple[str, str]],
        label: str,
        value: str,
        *,
        value_style: str = "class:status-value",
    ) -> None:
        if fragments:
            fragments.append(("class:status-separator", " | "))
        else:
            fragments.append(("", " "))
        fragments.extend(
            [
                ("class:status-label", label),
                (value_style, value),
            ]
        )

    def _get_context_style(self, percentage: float) -> str:
        """Get style class based on context usage percentage."""
        if percentage >= 90:
            return "class:status-context-critical"
        elif percentage >= 70:
            return "class:status-context-warn"
        return "class:status-context-ok"

    def absolute_token_warning(
        self, current_tokens: int, limit: Optional[int] = None
    ) -> Optional[str]:
        """Return a warning string when ``current_tokens`` exceeds an ABSOLUTE limit.

        This complements the context-percentage warning: a session can cross a
        large absolute token count (e.g. 200k) even when the percentage of a
        huge context window still looks small. Returns ``None`` when at or
        below the threshold.

        Args:
            current_tokens: The current context/token count to check.
            limit: Optional explicit threshold; defaults to the configured
                absolute limit (``KODER_TOKEN_WARN_LIMIT`` env or 200k).
        """
        threshold = limit if (limit is not None and limit > 0) else _absolute_token_limit()
        if current_tokens > threshold:
            return f"{self._format_tokens(current_tokens)} tokens (over {self._format_tokens(threshold)})"
        return None

    def _usage_summary(self):
        """Return a UsageSummary from the tracker, tolerating minimal test doubles.

        Falls back to reading ``session_usage`` attributes directly when the
        tracker does not implement :meth:`UsageTracker.summary` (e.g. a stubbed
        tracker), so the status line never crashes on partial objects.
        """
        summary_fn = getattr(self.usage_tracker, "summary", None)
        if callable(summary_fn):
            try:
                return summary_fn()
            except Exception:
                pass

        # Fallback: synthesize a minimal summary from session_usage attributes.
        from .usage_tracker import UsageSummary

        usage = self.usage_tracker.session_usage
        total_cost = float(getattr(usage, "total_cost", 0.0) or 0.0)
        return UsageSummary(
            request_count=int(getattr(usage, "request_count", 0) or 0),
            input_tokens=int(getattr(usage, "input_tokens", 0) or 0),
            output_tokens=int(getattr(usage, "output_tokens", 0) or 0),
            cache_read_tokens=int(getattr(usage, "cache_read_tokens", 0) or 0),
            cache_write_tokens=int(getattr(usage, "cache_write_tokens", 0) or 0),
            context_tokens=int(getattr(usage, "current_context_tokens", 0) or 0),
            total_cost=total_cost,
            # Without a real tracker we can't consult litellm pricing; treat a
            # positive cost as "known" and a zero cost as "unavailable".
            cost_unavailable=total_cost <= 0.0,
        )

    def _format_token_cost_segment(self) -> str:
        """Build the compact live token/cost segment.

        Example: ``▽ 45.2k tok (12k cached) · ~$0.14``. When per-token pricing
        is unknown (subscription/OAuth backends), the cost portion is shown as
        ``· $?`` instead of a misleading ``$0.00``. The ``(N cached)`` clause is
        omitted when no cache-read tokens have been recorded.
        """
        summary = self._usage_summary()
        billed = summary.input_tokens + summary.output_tokens
        parts = [f"▽ {self._format_tokens(billed)} tok"]
        if summary.cache_read_tokens > 0:
            parts.append(f"({self._format_tokens(summary.cache_read_tokens)} cached)")
        if summary.cost_unavailable:
            cost_part = "· $?"
        else:
            cost_part = f"· ~${summary.total_cost:.2f}"
        return f"{' '.join(parts)} {cost_part}"

    def _build_statusline_payload(self) -> dict[str, object]:
        usage = self.usage_tracker.session_usage
        model = self.usage_tracker.model
        current_dir = os.getcwd()
        max_context = get_configured_context_window(model)
        used_percentage = (
            (usage.current_context_tokens / max_context * 100) if max_context > 0 else 0.0
        )
        added_dirs = [
            item
            for item in str(os.environ.get("KODER_ADDITIONAL_DIRS", "")).split(os.pathsep)
            if item
        ]
        return {
            "session_id": self.session_id,
            "session_name": self._display_name,
            "cwd": current_dir,
            "transcript_path": None,
            "model": {
                "id": model,
                "display_name": model.replace("litellm/", "").split("/")[-1],
            },
            "workspace": {
                "current_dir": current_dir,
                "project_dir": self._project_dir,
                "added_dirs": added_dirs,
            },
            "version": resolve_runtime_version(),
            "output_style": {
                "name": "default",
            },
            "cost": {
                "total_cost_usd": usage.total_cost,
                "cost_unavailable": self._usage_summary().cost_unavailable,
                "total_duration_ms": 0,
                "total_api_duration_ms": 0,
                "total_lines_added": 0,
                "total_lines_removed": 0,
            },
            "context_window": {
                "total_input_tokens": usage.input_tokens,
                "total_output_tokens": usage.output_tokens,
                "total_cache_read_tokens": getattr(usage, "cache_read_tokens", 0),
                "total_cache_write_tokens": getattr(usage, "cache_write_tokens", 0),
                "context_window_size": max_context,
                "current_usage": {
                    "input_tokens": getattr(usage, "last_input_tokens", 0),
                    "output_tokens": getattr(usage, "last_output_tokens", 0),
                    "cache_creation_input_tokens": 0,
                    "cache_read_input_tokens": 0,
                },
                "used_percentage": round(used_percentage, 2),
                "remaining_percentage": round(max(0.0, 100.0 - used_percentage), 2),
            },
        }

    def _run_configured_command(self, command: str, payload: dict[str, object]) -> str:
        shell = os.environ.get("SHELL") or "/bin/sh"
        env = os.environ.copy()
        completed = subprocess.run(
            [shell, "-lc", command],
            input=json.dumps(payload, ensure_ascii=False),
            capture_output=True,
            text=True,
            timeout=2,
            env=env,
        )
        if completed.returncode != 0:
            detail = (
                completed.stderr.strip()
                or completed.stdout.strip()
                or f"exit {completed.returncode}"
            )
            return f"statusline error: {detail}"
        return completed.stdout

    def _normalize_command_output(self, output: str) -> str:
        lines = [line.strip() for line in output.splitlines() if line.strip()]
        return " | ".join(lines)

    def _refresh_command_in_background(self, command: str, payload: dict[str, object]) -> None:
        """Run the configured statusline command in a background thread."""
        try:
            output = self._run_configured_command(command, payload)
            with self._command_lock:
                self._cached_command_output = output
        except Exception:
            pass
        finally:
            with self._command_lock:
                self._command_refresh_pending = False

    def _render_custom_statusline(self) -> FormattedText | None:
        config = resolve_statusline_config(os.getcwd())
        if config is None:
            return None
        payload = self._build_statusline_payload()
        signature = json.dumps(
            {"command": config.command, "padding": config.padding, "payload": payload},
            ensure_ascii=False,
            sort_keys=True,
        )
        if signature != self._cached_command_signature:
            self._cached_command_signature = signature
            if self._cached_command_output is None:
                # First render (cold start): run synchronously so the status
                # line is populated immediately. Subsequent refreshes (signature
                # unchanged or changed again) dispatch to a background thread.
                self._refresh_command_in_background(config.command, payload)
            else:
                # Dispatch subprocess to a background thread so the render
                # callback never blocks on subsequent refreshes.
                with self._command_lock:
                    if not self._command_refresh_pending:
                        self._command_refresh_pending = True
                        t = threading.Thread(
                            target=self._refresh_command_in_background,
                            args=(config.command, payload),
                            daemon=True,
                        )
                        t.start()
        with self._command_lock:
            cached = self._cached_command_output
        normalized = self._normalize_command_output(cached or "")
        if not normalized and not self._notice:
            return None
        fragments: list[tuple[str, str]] = []
        if config.padding:
            fragments.append(("", " " * config.padding))
        if normalized:
            fragments.extend(list(to_formatted_text(ANSI(normalized))))
        if self._notice:
            if normalized:
                fragments.append(("class:status-separator", " | "))
            fragments.extend(
                [
                    ("class:status-label", "Notice: "),
                    ("class:status-value", self._notice),
                ]
            )
        return FormattedText(fragments)

    def get_formatted_text(self) -> FormattedText:
        """Generate the formatted status line text."""
        custom = self._render_custom_statusline()
        if custom is not None:
            return custom

        # Resolve the active model so /model changes are visible before the next request.
        model = self.usage_tracker.model
        if "/" in model:
            display_model = model.replace("litellm/", "")
            # If still has provider prefix (e.g., openai/gpt-4o), show just the model
            if "/" in display_model:
                display_model = display_model.split("/")[-1]
        else:
            display_model = model

        # Session: use display name if available, otherwise session ID (truncated)
        session_display = self._display_name or self.session_id

        # Usage data
        usage = self.usage_tracker.session_usage
        # Compact live token/cost segment (fresh vs cached split + running $).
        # Resilient to unknown pricing (shows "$?" rather than a blank/zero).
        token_cost_segment = self._format_token_cost_segment()

        # Context window usage: current_context_tokens
        # This represents the total context that will be sent in the next turn
        # (assuming sessions automatically include previous conversation history)
        current_tokens = usage.current_context_tokens
        max_context = get_configured_context_window(model)
        context_pct = (current_tokens / max_context * 100) if max_context > 0 else 0
        context_style = self._get_context_style(context_pct)

        # Format tokens - show dash before first API call for clearer UX
        if usage.request_count == 0:
            tokens_str = f"–/{self._format_tokens(max_context)}"
        else:
            tokens_str = f"{self._format_tokens(current_tokens)}/{self._format_tokens(max_context)}"

        # PR review status (populated by the background poller).
        pr_fragments: list[tuple[str, str]] = []
        if self.pr_poller is not None:
            pr_status = self.pr_poller.get_status()
            if pr_status is not None:
                style = _PR_STATE_STYLES.get(pr_status.review_state, "class:status-value")
                label = _PR_STATE_LABELS.get(pr_status.review_state, pr_status.review_state)
                pr_fragments = [
                    ("class:status-separator", " | "),
                    (style, f"PR #{pr_status.number} {label}"),
                ]

        columns = self._terminal_columns()
        cwd = os.getcwd()
        fragments: list[tuple[str, str]] = []

        if columns >= 140:
            self._append_field(
                fragments,
                "Model: ",
                self._truncate(display_model, 24),
            )
            self._append_field(fragments, "Dir: ", self._compact_cwd(cwd, 30))
            self._append_field(
                fragments,
                "Session: ",
                self._truncate(session_display, 18, from_start=True),
            )
            self._append_field(fragments, "Tokens: ", tokens_str + " ")
            fragments.append((context_style, f"({context_pct:.1f}%)"))
            # Compact live token/cost segment (fresh vs cached + running $).
            self._append_field(fragments, "", token_cost_segment)
        elif columns >= 100:
            self._append_field(
                fragments,
                "M: ",
                self._truncate(display_model, 22),
            )
            self._append_field(fragments, "Dir: ", self._compact_cwd(cwd, 28))
            self._append_field(fragments, "Tok: ", tokens_str + " ")
            fragments.append((context_style, f"({context_pct:.1f}%)"))
            self._append_field(fragments, "", token_cost_segment)
        elif columns >= 72:
            self._append_field(
                fragments,
                "M: ",
                self._truncate(display_model, 18),
            )
            self._append_field(fragments, "Dir: ", self._compact_cwd(cwd, 20))
            self._append_field(fragments, "Tok: ", tokens_str)
        else:
            self._append_field(
                fragments,
                "",
                self._truncate(display_model, 14),
            )
            self._append_field(fragments, "", self._compact_cwd(cwd, 14))
            self._append_field(fragments, "", f"{context_pct:.1f}%")

        rendered_len = sum(self._display_width(text) for _, text in fragments)

        if pr_fragments:
            pr_len = sum(self._display_width(text) for _, text in pr_fragments)
            if rendered_len + pr_len <= columns:
                fragments.extend(pr_fragments)
                rendered_len += pr_len

        # Absolute-threshold token warning (independent of context %). Only shown
        # on wider terminals and when no transient notice is already occupying
        # the line, and only if it fits within the remaining width.
        token_warning = self.absolute_token_warning(current_tokens)
        if token_warning and not self._notice and columns >= 100:
            warn_fragments = [
                ("class:status-separator", " | "),
                ("class:status-context-critical", f"⚠ {token_warning}"),
            ]
            warn_len = sum(self._display_width(text) for _, text in warn_fragments)
            if rendered_len + warn_len <= columns:
                fragments.extend(warn_fragments)
                rendered_len += warn_len

        if self._notice:
            if self._notice.startswith("Voice error:"):
                notice = self._notice
            else:
                notice = self._truncate(
                    self._notice,
                    max(10, columns - rendered_len - 12),
                )
            if notice:
                self._append_field(fragments, "Notice: ", notice)

        return FormattedText(fragments)

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

    def set_notice(self, notice: Optional[str]) -> None:
        """Set a transient notice shown on the status line."""
        self._notice = notice
