"""Session management utilities for Koder Agent."""

from datetime import datetime
from typing import List, Optional, Tuple


def default_session_local_ms() -> str:
    """Generate a local time session id precise to milliseconds.

    Format: YYYY-MM-DDTHH:MM:SS.mmm (local time)
    """
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3]


def parse_session_dt(sid: str) -> Tuple[int, Optional[datetime]]:
    """Parse session id to datetime if possible. Return a sort key for desc order.

    Supports formats like YYYY-MM-DDTHH:MM:SS.mmm or with microseconds.
    Unparsable sids get None and are sorted to the end.
    """
    fmts = ["%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S"]
    for fmt in fmts:
        try:
            dt = datetime.strptime(sid, fmt)
            return (0, dt)
        except Exception:
            continue
    return (1, None)


def sort_sessions_desc(sids: List[str]) -> List[str]:
    """Sort session IDs in descending order by datetime, with unparsable ones at the end."""
    parsed = [(parse_session_dt(s), s) for s in sids]
    dated = [(dt, s) for (flag, dt), s in parsed if flag == 0 and dt is not None]
    others = [s for (flag_dt, s) in parsed if flag_dt[0] == 1]
    dated.sort(key=lambda x: x[0], reverse=True)
    return [s for _, s in dated] + others


def picker_arrows(sessions: List[str]) -> Optional[str]:
    """Simple numbered list picker for session selection."""
    import sys

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None

    print("\nAvailable sessions:")
    for i, session in enumerate(sessions, 1):
        print(f"{i}. {session}")

    print("\nEnter the number of the session to select (or press Enter to cancel):")

    max_attempts = 3
    for _ in range(max_attempts):
        try:
            user_input = input("> ").strip()

            # Empty input means cancel
            if not user_input:
                return None

            # Check if input is a valid number
            if not user_input.isdigit():
                print(f"Please enter a valid number between 1 and {len(sessions)}.")
                continue

            choice = int(user_input)

            # Check if choice is in valid range
            if 1 <= choice <= len(sessions):
                return sessions[choice - 1]
            else:
                print(f"Please enter a number between 1 and {len(sessions)}.")

        except (EOFError, KeyboardInterrupt):
            return None
        except Exception:
            print("Invalid input. Please try again.")

    print("Too many invalid attempts. Cancelling selection.")
    return None


def picker_arrows_with_titles(sessions: List[Tuple[str, Optional[str]]]) -> Optional[str]:
    """Session picker that displays titles when available.

    Args:
        sessions: List of (session_id, title) tuples

    Returns:
        Selected session ID or None if cancelled
    """
    import sys

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        return None

    print("\nAvailable sessions:")
    for i, (sid, title) in enumerate(sessions, 1):
        if title:
            # Parse datetime from session ID for suffix
            _, dt = parse_session_dt(sid)
            if dt:
                display = f"{title} - {dt.strftime('%Y-%m-%d %H:%M')}"
            else:
                display = title
        else:
            display = sid
        print(f"{i}. {display}")

    print("\nEnter the number of the session to select (or press Enter to cancel):")

    max_attempts = 3
    for _ in range(max_attempts):
        try:
            user_input = input("> ").strip()

            # Empty input means cancel
            if not user_input:
                return None

            # Check if input is a valid number
            if not user_input.isdigit():
                print(f"Please enter a valid number between 1 and {len(sessions)}.")
                continue

            choice = int(user_input)

            # Check if choice is in valid range
            if 1 <= choice <= len(sessions):
                return sessions[choice - 1][0]  # Return session ID, not title
            else:
                print(f"Please enter a number between 1 and {len(sessions)}.")

        except (EOFError, KeyboardInterrupt):
            return None
        except Exception:
            print("Invalid input. Please try again.")

    print("Too many invalid attempts. Cancelling selection.")
    return None
