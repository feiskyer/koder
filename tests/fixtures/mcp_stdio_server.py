"""Offline MCP wire peer used by real SDK and terminal startup regressions.

This intentionally uses only JSON-RPC/stdin/stdout, not a mocked client session.
It supports the legacy handshake used by deployed stdio servers such as Context7.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", type=Path, required=True)
    parser.add_argument("--mode", choices=("normal", "fail", "silent"), default="normal")
    options = parser.parse_args()

    def record(**event):
        with options.trace.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event) + "\n")

    def send(message):
        print(json.dumps({"jsonrpc": "2.0", **message}), flush=True)

    def result(request, value):
        send({"id": request["id"], "result": value})

    record(pid=os.getpid())
    pending_call = None
    try:
        for line in sys.stdin:
            request = json.loads(line)
            method = request.get("method")
            record(method=method, **({"result": request["result"]} if "result" in request else {}))
            if method == "initialize":
                record(capabilities=request["params"].get("capabilities", {}))
                if options.mode == "silent":
                    continue
                if options.mode == "fail":
                    send(
                        {
                            "id": request["id"],
                            "error": {"code": -32603, "message": "fixture failure"},
                        }
                    )
                    continue
                result(
                    request,
                    {
                        "protocolVersion": "2025-11-25",
                        "capabilities": {"tools": {}, "experimental": {"claude/channel": {}}},
                        "serverInfo": {"name": "koder-startup-fixture", "version": "1.0"},
                    },
                )
            elif method == "tools/list":
                result(
                    request,
                    {
                        "tools": [
                            {
                                "name": "startup_echo",
                                "description": "Offline startup test tool",
                                "inputSchema": {"type": "object", "properties": {}},
                            }
                        ]
                    },
                )
            elif method == "tools/call":
                if request["params"].get("arguments", {}).get("elicit"):
                    pending_call = request
                    send(
                        {
                            "id": "fixture-elicit",
                            "method": "elicitation/create",
                            "params": {
                                "mode": "form",
                                "message": "Offline fixture question",
                                "requestedSchema": {
                                    "type": "object",
                                    "properties": {"answer": {"type": "string"}},
                                },
                            },
                        }
                    )
                else:
                    if request["params"].get("arguments", {}).get("channel"):
                        send(
                            {
                                "method": "notifications/claude/channel",
                                "params": {"content": "channel handshake verified"},
                            }
                        )
                    result(request, {"content": [{"type": "text", "text": "MCP_HANDSHAKE_OK"}]})
            elif request.get("id") == "fixture-elicit" and pending_call is not None:
                result(pending_call, {"content": [{"type": "text", "text": "ELICITATION_OK"}]})
                pending_call = None
            elif method in ("prompts/list", "resources/list", "resources/templates/list"):
                field = {
                    "prompts/list": "prompts",
                    "resources/list": "resources",
                    "resources/templates/list": "resourceTemplates",
                }[method]
                result(request, {field: []})
            elif "id" in request:
                send(
                    {"id": request["id"], "error": {"code": -32601, "message": "Method not found"}}
                )
    finally:
        record(closed=True)


if __name__ == "__main__":
    main()
