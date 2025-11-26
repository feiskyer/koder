"""Terminal theme detection and adaptive color scheme utility."""

import os
from typing import Optional

from rich.console import Console


def detect_terminal_theme() -> str:
    """
    Detect terminal color scheme preference.

    Returns:
        'light' for light background terminals, 'dark' for dark background terminals,
        'auto' if unable to determine.
    """
    # Check environment variables that indicate color scheme preference
    term_program = os.environ.get("TERM_PROGRAM", "").lower()
    colorfgbg = os.environ.get("COLORFGBG", "")

    # macOS Terminal.app and iTerm2 specific detection
    if term_program in ["apple_terminal", "iterm.app"]:
        # Check if we can detect theme from COLORFGBG
        if colorfgbg:
            parts = colorfgbg.split(";")
            if len(parts) >= 2:
                try:
                    bg_color = int(parts[-1])
                    # Background colors 0-7 are typically dark, 8-15 are typically light
                    if bg_color in [0, 1, 2, 3, 4, 5, 6, 8]:  # Dark backgrounds
                        return "dark"
                    elif bg_color in [7, 15]:  # Light backgrounds
                        return "light"
                except (ValueError, IndexError):
                    pass

    # Check for other terminal-specific environment variables
    if "ITERM_SESSION_ID" in os.environ:
        # iTerm2 - try to detect from profile
        iterm_profile = os.environ.get("ITERM_PROFILE", "")
        if "light" in iterm_profile.lower():
            return "light"
        elif "dark" in iterm_profile.lower():
            return "dark"

    # Check COLORFGBG more generally
    if colorfgbg:
        parts = colorfgbg.split(";")
        if len(parts) >= 2:
            try:
                fg_color = int(parts[0])
                bg_color = int(parts[-1])

                # If background is light (high numbers) and foreground is dark (low numbers)
                if bg_color >= 7 and fg_color <= 7:
                    return "light"
                # If background is dark (low numbers) and foreground is light (high numbers)
                elif bg_color <= 7 and fg_color >= 7:
                    return "dark"
            except (ValueError, IndexError):
                pass

    # Fallback: assume dark theme (most common for terminals)
    return "auto"


def get_adaptive_console(**kwargs) -> Console:
    """
    Create a Console instance with adaptive color scheme.

    Args:
        **kwargs: Additional arguments to pass to Console constructor

    Returns:
        Console instance configured for the detected terminal theme
    """
    theme = detect_terminal_theme()

    # Configure console with adaptive settings
    console_kwargs = {
        "color_system": "auto",  # Let Rich auto-detect color capabilities
        "force_terminal": kwargs.get("force_terminal", None),
        "width": kwargs.get("width", None),
        "height": kwargs.get("height", None),
        **kwargs,
    }

    # For light terminals, we might want to adjust certain default colors
    if theme == "light":
        # Rich will use 'default' colors which should adapt to terminal background
        # We can set a style that works well with light backgrounds
        console_kwargs["style"] = kwargs.get("style", "default")
        # Ensure we don't force any specific background colors
        console_kwargs["legacy_windows"] = kwargs.get("legacy_windows", False)

    return Console(**console_kwargs)


def get_adaptive_style_for_theme(base_style: str = "", theme: Optional[str] = None) -> str:
    """
    Get an adaptive style string that works well with the terminal theme.

    Args:
        base_style: Base style string (e.g., "bold", "italic")
        theme: Terminal theme ('light', 'dark', 'auto'), auto-detected if None

    Returns:
        Style string adapted for the terminal theme
    """
    if theme is None:
        theme = detect_terminal_theme()

    # For auto/unknown themes, use default colors
    if theme == "auto":
        return base_style if base_style else "default"

    # For light and dark themes, prefer default colors to adapt to terminal
    # This ensures text is readable regardless of terminal background
    if base_style:
        return f"{base_style} default"
    else:
        return "default"


def get_theme_aware_colors():
    """
    Get color definitions that adapt to terminal theme.

    Returns:
        Dictionary of semantic colors that work across themes
    """

    # Use semantic colors that adapt well to both light and dark themes
    colors = {
        "primary": "default",  # Adapts to terminal default
        "secondary": "dim default",  # Dimmed version of default
        "success": "green",  # Green works on both themes
        "warning": "yellow",  # Yellow works on both themes
        "error": "red",  # Red works on both themes
        "info": "blue",  # Blue works on both themes
        "accent": "cyan",  # Cyan works on both themes
        "muted": "dim",  # Dimmed text
    }

    return colors


def get_adaptive_prompt_style():
    """
    Get a prompt_toolkit Style that adapts to the terminal theme.

    Returns:
        Style object for prompt_toolkit that works with both light and dark themes
    """
    try:
        from prompt_toolkit.styles import Style
    except ImportError:
        # Fallback if prompt_toolkit is not available
        return None

    theme = detect_terminal_theme()

    if theme == "light":
        # Light theme: dark text on light background
        style_dict = {
            "frame.border": "#666666",  # Medium gray border (visible on light)
            "frame.title": "#444444",  # Dark gray title (readable on light)
            "": "#000000",  # Black text for input (readable on light)
            "completion-menu.completion": "#000000 bg:#f0f0f0",  # Dark text on light bg
            "completion-menu.completion.current": "#000000 bg:#d0d0d0",  # Dark text on selected
            "completion-menu.meta.completion": "#666666 bg:#e8e8e8",  # Gray desc on light
            "completion-menu.meta.completion.current": "#000000 bg:#c0c0c0",  # Dark desc selected
            # Status line styles
            "status-label": "#666666",  # Gray labels
            "status-value": "#333333",  # Dark values
            "status-separator": "#999999",  # Light gray separators
            "status-context-ok": "#228B22",  # Forest green for < 70%
            "status-context-warn": "#FF8C00",  # Dark orange for 70-90%
            "status-context-critical": "#DC143C",  # Crimson for > 90%
        }
    elif theme == "dark":
        # Dark theme: light text on dark background
        style_dict = {
            "frame.border": "#666666",  # Medium gray border (visible on dark)
            "frame.title": "#bbbbbb",  # Light gray title (readable on dark)
            "": "#ffffff",  # White text for input (readable on dark)
            "completion-menu.completion": "#ffffff bg:#333333",  # Light text on dark bg
            "completion-menu.completion.current": "#ffffff bg:#555555",  # Light text on selected
            "completion-menu.meta.completion": "#aaaaaa bg:#222222",  # Gray desc on dark
            "completion-menu.meta.completion.current": "#ffffff bg:#444444",  # Light desc selected
            # Status line styles
            "status-label": "#888888",  # Gray labels
            "status-value": "#cccccc",  # Light values
            "status-separator": "#666666",  # Dark gray separators
            "status-context-ok": "#32CD32",  # Lime green for < 70%
            "status-context-warn": "#FFA500",  # Orange for 70-90%
            "status-context-critical": "#FF6347",  # Tomato red for > 90%
        }
    else:
        # Auto/unknown theme: use system defaults where possible
        style_dict = {
            "frame.border": "#666666",  # Neutral border
            "frame.title": "",  # Use terminal default for title
            "": "",  # Use terminal default for input text
            "completion-menu.completion": "",  # Use terminal defaults
            "completion-menu.completion.current": "reverse",  # Just reverse colors
            "completion-menu.meta.completion": "",
            "completion-menu.meta.completion.current": "reverse",
            # Status line styles (neutral colors)
            "status-label": "#888888",
            "status-value": "",  # Use terminal default
            "status-separator": "#666666",
            "status-context-ok": "#32CD32",  # Green
            "status-context-warn": "#FFA500",  # Orange
            "status-context-critical": "#FF6347",  # Red
        }

    return Style.from_dict(style_dict)


# Export the main console creation function for easy import
create_console = get_adaptive_console
