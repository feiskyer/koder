"""Koder Agent - An advanced AI coding assistant and interactive CLI tool."""

from .litellm_cost_map import configure_litellm_local_model_cost_map
from .version import resolve_package_version_info

configure_litellm_local_model_cost_map()

__version__ = resolve_package_version_info()[0]


def main(*args, **kwargs):
    from .cli import main as _main

    return _main(*args, **kwargs)


def run(*args, **kwargs):
    from .cli import run as _run

    return _run(*args, **kwargs)


__all__ = ["main", "run", "__version__"]
