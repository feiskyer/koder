"""Interactive prompt with slash command completion using prompt_toolkit."""

from typing import TYPE_CHECKING, Dict, Optional

from prompt_toolkit.application import Application
from prompt_toolkit.buffer import Buffer
from prompt_toolkit.completion import CompleteEvent, Completer, Completion
from prompt_toolkit.document import Document
from prompt_toolkit.key_binding import KeyBindings
from prompt_toolkit.layout import Layout
from prompt_toolkit.layout.containers import HSplit, Window
from prompt_toolkit.layout.controls import BufferControl
from prompt_toolkit.layout.dimension import Dimension
from prompt_toolkit.layout.menus import CompletionsMenu
from prompt_toolkit.layout.processors import BeforeInput
from prompt_toolkit.shortcuts import confirm
from prompt_toolkit.widgets import Frame

from ..utils.terminal_theme import get_adaptive_console, get_adaptive_prompt_style

if TYPE_CHECKING:
    from .usage_tracker import UsageTracker

console = get_adaptive_console()


class DynamicCompletionsMenu(CompletionsMenu):
    """CompletionsMenu that adjusts height based on available completions."""

    def __init__(self, scroll_offset: int = 1):
        super().__init__(max_height=1, scroll_offset=scroll_offset)

    def _get_completions(self, app):
        """Get current completions from the buffer."""
        buffer = app.current_buffer
        if buffer.complete_state:
            return buffer.complete_state.completions
        return []

    def preferred_height(self, width, max_available_height):
        """Calculate preferred height based on number of completions."""
        from prompt_toolkit.application import get_app

        try:
            app = get_app()
            completions = self._get_completions(app)
            completion_count = len(completions)

            # Return height based on actual completion count, with minimum of 0
            if completion_count == 0:
                return Dimension.exact(0)
            else:
                # Cap at max_available_height to prevent overflow
                height = min(completion_count, max_available_height)
                return Dimension.exact(height)
        except Exception:
            # Fallback to no height if there's any error
            return Dimension.exact(0)


class SlashCommandCompleter(Completer):
    """Custom completer for slash commands with descriptions."""

    def __init__(self, commands: Dict[str, str]):
        """Initialize with command name -> description mapping."""
        self.commands = commands

    def get_completions(self, document: Document, complete_event: CompleteEvent):
        """Generate completions for slash commands."""
        text = document.text_before_cursor

        # Only show completions if we're at the start and typing a slash
        if text == "/" or (text.startswith("/") and " " not in text):
            # Remove the leading slash for matching
            search_text = text[1:] if text.startswith("/") else text

            for command, description in self.commands.items():
                if command.startswith(search_text):
                    # Create completion with command name and description
                    yield Completion(
                        text=command,
                        start_position=-len(search_text),
                        display=f"/{command}",
                        display_meta=description,
                    )


class InteractivePrompt:
    """Enhanced prompt with slash command support and status line."""

    def __init__(
        self,
        commands: Dict[str, str],
        usage_tracker: Optional["UsageTracker"] = None,
        session_id: str = "",
    ):
        """
        Initialize with available slash commands and optional status line.

        Args:
            commands: Dict of command name -> description
            usage_tracker: Optional UsageTracker for token/cost display
            session_id: Current session identifier
        """
        self.commands = commands
        self.completer = SlashCommandCompleter(commands)

        # Status line (optional)
        self.status_line = None
        if usage_tracker is not None:
            from .status_line import StatusLine

            self.status_line = StatusLine(
                usage_tracker=usage_tracker,
                session_id=session_id,
            )

    async def get_input(self) -> str:
        """Get user input with Rich panel display and prompt_toolkit completion."""
        # Create buffer
        buffer = Buffer(
            completer=self.completer,
            complete_while_typing=True,
        )

        # Create buffer control with "> " prefix
        buffer_control = BufferControl(
            buffer=buffer,
            input_processors=[BeforeInput("> ")],
        )

        # Create input window with dynamic height
        input_window = Window(
            content=buffer_control,
            height=Dimension(min=1, max=10),  # Allow 1-10 lines, auto-expand
            wrap_lines=False,  # Keep single line behavior, expand vertically instead
            dont_extend_height=True,
        )

        # Create simple frame without heavy styling
        framed_input = Frame(
            body=input_window,
            title="⚡ Koder",
        )

        # Key bindings
        kb = KeyBindings()

        @kb.add("enter")
        def accept_input(event):
            # Submit the input
            event.app.exit(result=buffer.text)

        @kb.add("c-j")  # For Ctrl+Enter in many terminals
        @kb.add("escape", "enter")  # For Alt+Enter as a more compatible option
        def insert_newline(event):
            # Insert a newline for multi-line input
            event.app.current_buffer.insert_text("\n")

        @kb.add("c-c")
        def cancel_input(event):
            event.app.exit(exception=KeyboardInterrupt())

        @kb.add("c-d")
        def clear_input(event):
            # Clear the input content
            event.app.current_buffer.text = ""

        # Add Tab key for completion navigation
        @kb.add("tab")
        def complete(event):
            # Tab to navigate through completions
            b = event.app.current_buffer
            if b.complete_state:
                b.complete_next()
            else:
                b.start_completion(select_first=True)

        @kb.add("s-tab")  # Shift+Tab
        def complete_previous(event):
            # Shift+Tab to navigate backwards through completions
            b = event.app.current_buffer
            if b.complete_state:
                b.complete_previous()

        @kb.add("backspace")
        def handle_backspace(event):
            # Handle backspace and retrigger completion if needed
            b = event.app.current_buffer
            # First perform the backspace
            b.delete_before_cursor()
            # If we're working with a slash command, retrigger completion
            if b.text.startswith("/"):
                # Cancel current completion state
                b.cancel_completion()
                # Start new completion
                b.start_completion(select_first=False)
            else:
                # Cancel completion if we're no longer in slash mode
                b.cancel_completion()

        @kb.add("delete")
        def handle_delete(event):
            # Handle delete key and retrigger completion if needed
            b = event.app.current_buffer
            # First perform the delete
            b.delete()
            # If we're working with a slash command, retrigger completion
            if b.text.startswith("/"):
                # Cancel current completion state
                b.cancel_completion()
                # Start new completion
                b.start_completion(select_first=False)
            else:
                # Cancel completion if we're no longer in slash mode
                b.cancel_completion()

        # Create layout with completion menu and optional status line
        components = [
            framed_input,
            DynamicCompletionsMenu(scroll_offset=1),
        ]
        if self.status_line:
            components.append(self.status_line.create_window())

        layout = Layout(HSplit(components))

        # Adaptive style that works with both light and dark terminals
        style = get_adaptive_prompt_style()

        # Create application
        app = Application(
            layout=layout,
            key_bindings=kb,
            style=style,
            full_screen=False,
            mouse_support=False,  # Disable mouse support to allow terminal scrolling
        )

        result = await app.run_async()
        if result is None:
            raise EOFError("Empty input received")
        return result.strip()

    def confirm_action(self, message: str) -> bool:
        """Ask for confirmation with yes/no prompt."""
        try:
            return confirm(message)
        except (EOFError, KeyboardInterrupt):
            return False

    def show_command_help(self) -> None:
        """Display available commands in a formatted way."""
        console.print("\n[bold cyan]Available Slash Commands:[/bold cyan]")
        for command, description in self.commands.items():
            console.print(f"  [cyan]/{command}[/cyan] - {description}")
        console.print()

    def update_session(self, session_id: str) -> None:
        """Update the session ID displayed in the status line."""
        if self.status_line:
            self.status_line.update_session(session_id)
