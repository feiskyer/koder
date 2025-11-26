#!/usr/bin/env python3
# /// script
# requires-python = ">=3.10"
# dependencies = ["requests", "tomli"]
# ///
"""Upgrade all dependencies in pyproject.toml to their latest PyPI versions."""

try:
    import tomllib  # Python 3.11+
except ImportError:
    import tomli as tomllib  # Python < 3.11

import re
import sys
from pathlib import Path

import requests

PYPROJECT = Path("pyproject.toml")


def parse_dep(dep: str) -> tuple[str, str, str]:
    """
    Split dependency string into: package, extras, operator+version.
    Example: "openai-agents[litellm]>=0.1.0"
    Returns: ("openai-agents", "[litellm]", ">=0.1.0")
    """
    m = re.match(r"([A-Za-z0-9_.-]+)(\[[^\]]+\])?(.*)", dep)
    if not m:
        return dep, "", ""
    return m.group(1), m.group(2) or "", m.group(3) or ""


def extract_operator(version_spec: str) -> str:
    """Extract the version operator from a version specification."""
    m = re.match(r"([><=!~^]+)", version_spec.strip())
    return m.group(1) if m else ">="


def latest_version(pkg: str) -> str | None:
    """Fetch the latest version of a package from PyPI. Returns None on failure."""
    try:
        resp = requests.get(f"https://pypi.org/pypi/{pkg}/json", timeout=10)
        if resp.status_code != 200:
            print(f"  Warning: Could not fetch version for {pkg} (HTTP {resp.status_code})")
            return None
        return resp.json()["info"]["version"]
    except requests.RequestException as e:
        print(f"  Warning: Network error fetching {pkg}: {e}")
        return None


def update_pyproject() -> None:
    """Update dependencies in pyproject.toml to their latest versions."""
    if not PYPROJECT.exists():
        print(f"Error: {PYPROJECT} not found")
        sys.exit(1)

    text = PYPROJECT.read_text()

    with open(PYPROJECT, "rb") as f:
        data = tomllib.load(f)

    deps = data.get("project", {}).get("dependencies", [])
    if not deps:
        print("No dependencies found in [project.dependencies]")
        return

    replacements: list[tuple[str, str]] = []

    print(f"Checking {len(deps)} dependencies...")
    for dep in deps:
        pkg, extras, version_spec = parse_dep(dep)
        ver = latest_version(pkg)
        if ver is None:
            continue  # Skip packages we couldn't fetch

        operator = extract_operator(version_spec)
        new_dep = f"{pkg}{extras}{operator}{ver}"

        if dep != new_dep:
            replacements.append((dep, new_dep))
            print(f"  {pkg}: {version_spec.strip() or '(none)'} -> {operator}{ver}")

    if not replacements:
        print("All dependencies are already up to date.")
        return

    # Replace only within the dependencies array to avoid touching other sections
    # Find the dependencies section and replace within it
    for old, new in replacements:
        # Use word boundaries to avoid partial matches
        # Escape special regex characters in the old string
        escaped_old = re.escape(old)
        # Only replace if it's a complete dependency entry (surrounded by quotes or array boundaries)
        pattern = rf'(["\']{escaped_old}["\']|(?<=[\[\s,]){escaped_old}(?=[\],\s]))'
        text = re.sub(pattern, lambda m: m.group(0).replace(old, new), text)

    PYPROJECT.write_text(text)
    print(f"\nUpdated {len(replacements)} dependencies in pyproject.toml")
    print("Run 'uv lock' to update the lockfile.")


if __name__ == "__main__":
    update_pyproject()
