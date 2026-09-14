"""Low-level utilities must not initialize provider configuration."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    "module",
    ["koder_agent.utils.async_tasks", "koder_agent.core.turn_cancellation"],
)
def test_low_level_imports_do_not_import_client(module, tmp_path):
    repository = Path(__file__).resolve().parents[2]
    script = """
import importlib
import pathlib
import sys
from unittest.mock import patch

sys.path.insert(0, sys.argv[1])
with patch.object(pathlib.Path, "home", return_value=pathlib.Path(sys.argv[3])):
    importlib.import_module(sys.argv[2])
    assert "koder_agent.utils.client" not in sys.modules
print("ISOLATED_IMPORT_OK")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", script, str(repository), module, str(tmp_path)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "ISOLATED_IMPORT_OK"


def test_public_utility_exports_remain_available():
    import koder_agent.utils as utilities
    from koder_agent.utils.queue import AsyncMessageQueue
    from koder_agent.utils.sessions import parse_session_dt

    assert utilities.AsyncMessageQueue is AsyncMessageQueue
    assert utilities.parse_session_dt is parse_session_dt
    with pytest.raises(AttributeError):
        getattr(utilities, "not_a_utility")
