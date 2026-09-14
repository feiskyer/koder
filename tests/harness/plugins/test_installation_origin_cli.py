"""Real CLI parser/dispatch persists installer provenance with a synthetic profile."""

import json
import os
import subprocess
from pathlib import Path


def test_cli_install_and_local_replacement_keep_distinct_origins(tmp_path):
    repository = Path(__file__).resolve().parents[3]
    home = tmp_path / "profile"
    home.mkdir()
    market = tmp_path / "Community"
    source = market / "demo"
    source.mkdir(parents=True)
    (source / "plugin.json").write_text('{"name":"demo","version":"1.0.0"}')
    (source / ".mcp.json").write_text(
        '{"mcpServers":{"raw-server":{"command":"synthetic-never-executed"}}}'
    )
    inherited = (
        "PATH",
        "SYSTEMROOT",
        "COMSPEC",
        "TMPDIR",
        "TEMP",
        "TMP",
        "UV_PROJECT_ENVIRONMENT",
        "UV_PYTHON",
        "UV_NO_SYNC",
    )
    env = {name: os.environ[name] for name in inherited if name in os.environ}
    env.update(
        HOME=str(home),
        KODER_NO_UPDATE_CHECK="1",
        LITELLM_LOCAL_MODEL_COST_MAP="True",
        OPENAI_AGENTS_DISABLE_TRACING="1",
    )

    def cli(*arguments):
        result = subprocess.run(
            [
                "uv",
                "run",
                "--project",
                str(repository),
                "--no-env-file",
                "koder",
                "plugin",
                *arguments,
            ],
            cwd=home,
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        assert result.returncode == 0, result.stdout + result.stderr
        return result.stdout

    cli("marketplace", "add", str(market))
    cli("install", "demo@community")
    state_path = home / ".koder" / "plugins" / "state.json"
    state = json.loads(state_path.read_text())["demo"]
    assert state["origin"]["marketplace"] == "community"
    assert len(state["origin"]["source_digest"]) == 64
    listing = json.loads(cli("list", "--json"))
    assert [entry["name"] for entry in listing] == ["demo"]
    cli("disable", "demo")
    cli("enable", "demo")
    assert json.loads(state_path.read_text())["demo"]["origin"] == state["origin"]
    cli("install", str(source))
    assert json.loads(state_path.read_text())["demo"]["origin"] is None
