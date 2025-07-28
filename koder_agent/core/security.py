"""Security guards for the Koder Agent."""

import re
from pathlib import Path
from typing import Optional


class SecurityGuard:
    """Security guard to validate paths and inputs."""

    UI_PATTERN = re.compile(r"^[\s\S]{1,2000}$")
    FORBIDDEN_WORDS = {"rm -rf", "shutdown", "reboot"}

    @staticmethod
    def validate_command(command: str) -> Optional[str]:
        """Validate a shell command for safety."""
        command_lower = command.lower()

        # Check for forbidden commands
        for forbidden in SecurityGuard.FORBIDDEN_WORDS:
            if forbidden in command_lower:
                return f"Forbidden command pattern detected: {forbidden}"

        # Check for dangerous patterns
        dangerous_patterns = [
            r">\s*/dev/",  # Writing to device files
            r"dd\s+if=",  # Disk destroyer
            r"mkfs",  # Format filesystem
            r":(){ :|:& };:",  # Fork bomb
        ]

        for pattern in dangerous_patterns:
            if re.search(pattern, command_lower):
                return "Dangerous command pattern detected"

        return None  # Valid command

    @staticmethod
    def sanitize_path(path: str) -> str:
        """Sanitize a path for safe usage."""
        # Remove any null bytes
        path = path.replace("\0", "")

        # Normalize path separators
        path = path.replace("\\", "/")

        # Remove redundant separators
        while "//" in path:
            path = path.replace("//", "/")

        # Remove trailing separators
        path = path.rstrip("/")

        return path

    @staticmethod
    def check_file_size(path: str, max_size_mb: int = 50) -> Optional[str]:
        """Check if file size is within limits."""
        try:
            path_obj = Path(path)
            if path_obj.is_file():
                size_mb = path_obj.stat().st_size / (1024 * 1024)
                if size_mb > max_size_mb:
                    return f"File too large: {size_mb:.2f}MB (max: {max_size_mb}MB)"
            return None
        except Exception as e:
            return f"Error checking file size: {str(e)}"
