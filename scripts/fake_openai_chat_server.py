#!/usr/bin/env python3
"""Serve deterministic OpenAI-compatible Chat/Responses for tmux scenarios."""

from __future__ import annotations

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

_TOOL_SCENARIOS = (
    "streaming_subagent_tool",
    "streaming_tool_queue",
    "streaming_tool_error",
    "sandbox_shell_tool",
    "git_query_mutation",
    "sed_query_mutation",
)
_SCENARIOS = ("single", *_TOOL_SCENARIOS)
_SUBAGENT_CHILD_MARKERS = {
    "alpha": "KODER_SUBAGENT_ALPHA",
    "beta": "KODER_SUBAGENT_BETA",
}


class _ReusableHTTPServer(ThreadingHTTPServer):
    allow_reuse_address = True


class _Handler(BaseHTTPRequestHandler):
    response_text: str = "fixture response"
    log_file: Path | None = None
    scenario: str = "single"
    subagent_delay: float = 0.0
    stream_delay: float = 0.0
    stream_lines: int = 2
    log_lock = threading.Lock()

    def do_POST(self) -> None:  # noqa: N802 - stdlib handler API
        length = int(self.headers.get("content-length") or 0)
        raw_body = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            body: dict[str, Any] = json.loads(raw_body)
        except json.JSONDecodeError:
            body = {"raw_body": raw_body}

        self._write_log(body)

        if self.path.rstrip("/").endswith("/responses") and self.scenario == "single":
            self._send_response(body)
            return

        if self.scenario in _TOOL_SCENARIOS:
            if not body.get("tools"):
                self._send_json(self._chat_completion(body, self.response_text))
                return
            if self.scenario == "streaming_subagent_tool":
                self._handle_subagent_tool_scenario(body)
                return
            if self.scenario == "streaming_tool_error" and self._request_has_tool_output(body):
                self._send_error(self.response_text)
                return
            if body.get("stream"):
                if self._request_has_tool_output(body):
                    self._send_text_stream(body)
                else:
                    self._send_tool_call_stream(body)
                return
            if self._request_has_tool_output(body):
                self._send_json(self._chat_completion(body, self.response_text))
            else:
                self._send_json(self._tool_call_completion(body))
            return

        if body.get("stream"):
            self._send_text_stream(body)
        else:
            self._send_json(self._chat_completion(body, self.response_text))

    def _write_log(self, body: dict[str, Any]) -> None:
        if self.log_file is None:
            return
        record: dict[str, Any] = {
            "path": self.path,
            "body": body,
        }
        marker = self._subagent_log_marker(body)
        if marker is not None:
            record["marker"] = marker
        if self.scenario == "git_query_mutation":
            record.update(self._git_permission_denials(body))
        elif self.scenario == "sed_query_mutation":
            record.update(self._sed_permission_outcomes(body))
        self.log_file.parent.mkdir(parents=True, exist_ok=True)
        with self.log_lock, self.log_file.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

    @staticmethod
    def _tool_result_contains(body: dict[str, Any], call_id: str, text: str) -> bool:
        messages = body.get("messages")
        messages = messages if isinstance(messages, list) else []
        return any(
            isinstance(message, dict)
            and message.get("role") == "tool"
            and message.get("tool_call_id") == call_id
            and text in str(message.get("content"))
            for message in messages
        )

    @classmethod
    def _git_permission_denials(cls, body: dict[str, Any]) -> dict[str, bool]:
        return {
            f"git_{name}_denied": cls._tool_result_contains(
                body, f"call_koder_git_{name}", "dontAsk mode: approval auto-denied"
            )
            for name in ("reflog", "fsck")
        }

    @classmethod
    def _sed_permission_outcomes(cls, body: dict[str, Any]) -> dict[str, bool]:
        return {
            "sed_write_denied": cls._tool_result_contains(
                body, "call_koder_sed_write", "dontAsk mode: approval auto-denied"
            ),
            "sed_read_succeeded": cls._tool_result_contains(body, "call_koder_sed_read", "initial"),
        }

    def _subagent_log_marker(self, body: dict[str, Any]) -> str | None:
        if self.scenario != "streaming_subagent_tool":
            return None
        child = self._subagent_child(body)
        if child is None:
            return None
        if not self._request_has_tool_output(body):
            if body.get("tools") and self._has_tool(body, "read_file"):
                return f"subagent-{child}-read_file-request"
            return None
        if self._request_has_tool_call(body, "read_file"):
            return f"subagent-{child}-read_file-result"
        return None

    def _request_has_tool_output(self, body: dict[str, Any]) -> bool:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return False

        last_user_index = -1
        for index, message in enumerate(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                last_user_index = index

        for message in messages[last_user_index + 1 :]:
            if not isinstance(message, dict):
                continue
            if message.get("role") == "tool":
                return True
            if "Queued user input" in json.dumps(message, ensure_ascii=False):
                return True
        return False

    @staticmethod
    def _request_has_tool_call(body: dict[str, Any], name: str) -> bool:
        messages = body.get("messages")
        if not isinstance(messages, list):
            return False
        for message in messages:
            if not isinstance(message, dict):
                continue
            for tool_call in message.get("tool_calls") or []:
                if not isinstance(tool_call, dict):
                    continue
                function = tool_call.get("function")
                if isinstance(function, dict) and function.get("name") == name:
                    return True
        return False

    @staticmethod
    def _subagent_child(body: dict[str, Any]) -> str | None:
        messages = body.get("messages")
        serialized = json.dumps(messages, ensure_ascii=False) if isinstance(messages, list) else ""
        matches = [
            child for child, marker in _SUBAGENT_CHILD_MARKERS.items() if marker in serialized
        ]
        return matches[0] if len(matches) == 1 else None

    def _chat_completion(self, body: dict[str, Any], content: str) -> dict[str, Any]:
        payload = {
            "id": "chatcmpl-koder-fixture",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": body.get("model") or "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": self.response_text},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        }
        payload["choices"][0]["message"]["content"] = content
        return payload

    def _tool_call_completion(self, body: dict[str, Any]) -> dict[str, Any]:
        payload = self._chat_completion(body, "")
        payload["choices"][0]["message"] = {
            "role": "assistant",
            "content": None,
            "tool_calls": [self._tool_call_payload(body)],
        }
        payload["choices"][0]["finish_reason"] = "tool_calls"
        return payload

    @staticmethod
    def _function_tool_call(call_id: str, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": call_id,
            "type": "function",
            "function": {
                "name": name,
                "arguments": json.dumps(arguments),
            },
        }

    def _tool_call_payload(self, body: dict[str, Any] | None = None) -> dict[str, Any]:
        request_body = body or {}
        child = self._subagent_child(request_body)
        if (
            self.scenario == "streaming_subagent_tool"
            and child is None
            and self._has_tool(request_body, "task_delegate")
        ):
            return self._function_tool_call(
                "call_koder_subagent_fixture",
                "task_delegate",
                {
                    "tasks": [
                        {
                            "description": "Inspect sample alpha",
                            "prompt": (
                                "Read sample.txt and report alpha. Marker: KODER_SUBAGENT_ALPHA."
                            ),
                        },
                        {
                            "description": "Inspect sample beta",
                            "prompt": (
                                "Read sample.txt and report beta. Marker: KODER_SUBAGENT_BETA."
                            ),
                        },
                    ]
                },
            )
        if self.scenario == "streaming_subagent_tool" and child is not None:
            return self._function_tool_call(
                f"call_koder_subagent_{child}_read_file",
                "read_file",
                {"path": "sample.txt"},
            )
        if self.scenario == "sandbox_shell_tool":
            return self._function_tool_call(
                "call_koder_sandbox_fixture",
                "run_shell",
                {"command": "touch model-tool-created.txt"},
            )
        if self.scenario in {"git_query_mutation", "sed_query_mutation"}:
            latest_user = next(
                (
                    message
                    for message in reversed(request_body.get("messages") or [])
                    if isinstance(message, dict) and message.get("role") == "user"
                ),
                {},
            )
            if self.scenario == "sed_query_mutation":
                read = "KODER_SED_READ" in json.dumps(latest_user, ensure_ascii=False)
                return self._function_tool_call(
                    f"call_koder_sed_{'read' if read else 'write'}",
                    "run_shell",
                    {
                        "command": (
                            "sed -n '1p' sample.txt"
                            if read
                            else "sed -n 'w sed-write-proof.txt' sample.txt"
                        )
                    },
                )
            fsck = "KODER_GIT_FSCK" in json.dumps(latest_user, ensure_ascii=False)
            return self._function_tool_call(
                f"call_koder_git_{'fsck' if fsck else 'reflog'}",
                "run_shell",
                {
                    "command": (
                        "git fsck --lost" if fsck else "git reflog expire --expire=all --all"
                    )
                },
            )
        return self._function_tool_call(
            "call_koder_queue_fixture",
            "read_file",
            {"path": "sample.txt"},
        )

    def _stream_chunk(self, body: dict[str, Any], delta: dict[str, Any], finish_reason=None):
        return {
            "id": "chatcmpl-koder-fixture",
            "object": "chat.completion.chunk",
            "created": int(time.time()),
            "model": body.get("model") or "gpt-4.1",
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                }
            ],
        }

    def _send_sse(self, chunks: list[dict[str, Any]]) -> None:
        self.send_response(200)
        self.send_header("content-type", "text/event-stream")
        self.send_header("cache-control", "no-cache")
        self.end_headers()
        for chunk in chunks:
            if str(chunk.get("type", "")).startswith("response."):
                self.wfile.write(f"event: {chunk['type']}\n".encode())
            encoded = f"data: {json.dumps(chunk)}\n\n".encode("utf-8")
            self.wfile.write(encoded)
            self.wfile.flush()
            if self.stream_delay:
                time.sleep(self.stream_delay)
        self.wfile.write(b"data: [DONE]\n\n")
        self.wfile.flush()

    def _send_tool_call_stream(self, body: dict[str, Any]) -> None:
        tool_payload = self._tool_call_payload(body)
        function_payload = tool_payload["function"]
        argument_json = function_payload["arguments"]
        if self.stream_lines <= 2:
            content_chunks = [
                self._stream_chunk(body, {"content": "streaming fixture line 1\n"}),
                self._stream_chunk(body, {"content": "streaming fixture line 2\n"}),
            ]
        else:
            content_chunks = [
                self._stream_chunk(
                    body,
                    {"content": f"streaming fixture long line {line_number:03d}\n"},
                )
                for line_number in range(1, self.stream_lines + 1)
            ]
        chunks = [
            self._stream_chunk(body, {"role": "assistant"}),
            *content_chunks,
            self._stream_chunk(
                body,
                {
                    "tool_calls": [
                        {
                            "index": 0,
                            "id": tool_payload["id"],
                            "type": "function",
                            "function": {
                                "name": function_payload["name"],
                                "arguments": "",
                            },
                        }
                    ]
                },
            ),
            self._stream_chunk(
                body,
                {"tool_calls": [{"index": 0, "function": {"arguments": argument_json}}]},
            ),
            self._stream_chunk(body, {}, "tool_calls"),
        ]
        self._send_sse(chunks)

    def _handle_subagent_tool_scenario(self, body: dict[str, Any]) -> None:
        child = self._subagent_child(body)
        is_parent = child is None
        has_tool_output = self._request_has_tool_output(body)
        if has_tool_output:
            if not is_parent and self.subagent_delay:
                time.sleep(self.subagent_delay)
            content = self.response_text if is_parent else f"subagent {child} fixture result"
            if body.get("stream"):
                self._send_text_stream(body, content=content)
            else:
                self._send_json(self._chat_completion(body, content))
            return

        if body.get("stream"):
            self._send_tool_call_stream(body)
        else:
            self._send_json(self._tool_call_completion(body))

    @staticmethod
    def _has_tool(body: dict[str, Any], name: str) -> bool:
        for tool in body.get("tools") or []:
            if not isinstance(tool, dict):
                continue
            function = tool.get("function")
            if isinstance(function, dict) and function.get("name") == name:
                return True
            if tool.get("name") == name:
                return True
        return False

    def _send_text_stream(self, body: dict[str, Any], *, content: str | None = None) -> None:
        chunks = [
            self._stream_chunk(body, {"role": "assistant"}),
            self._stream_chunk(body, {"content": content or self.response_text}),
            self._stream_chunk(body, {}, "stop"),
        ]
        self._send_sse(chunks)

    def _send_response(self, body: dict[str, Any]) -> None:
        part = {"type": "output_text", "text": self.response_text, "annotations": []}
        message = {
            "type": "message",
            "id": "msg_koder_fixture",
            "role": "assistant",
            "status": "completed",
            "content": [part],
        }
        response = {
            "id": "resp_koder_fixture",
            "object": "response",
            "created_at": int(time.time()),
            "model": body.get("model", "fixture"),
            "status": "completed",
            "output": [message],
            "parallel_tool_calls": True,
            "tool_choice": "auto",
            "tools": [],
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 0},
            },
        }
        if not body.get("stream"):
            self._send_json(response)
            return
        empty_part = {**part, "text": ""}
        location = {"output_index": 0, "item_id": message["id"], "content_index": 0}
        events = [
            {
                "type": "response.created",
                "response": {**response, "status": "in_progress", "output": []},
            },
            {
                "type": "response.output_item.added",
                "output_index": 0,
                "item": {**message, "status": "in_progress", "content": []},
            },
            {"type": "response.content_part.added", **location, "part": empty_part},
            {"type": "response.output_text.delta", **location, "delta": self.response_text},
            {"type": "response.output_text.done", **location, "text": self.response_text},
            {"type": "response.content_part.done", **location, "part": part},
            {"type": "response.output_item.done", "output_index": 0, "item": message},
            {"type": "response.completed", "response": response},
        ]
        self._send_sse([{**event, "sequence_number": index} for index, event in enumerate(events)])

    def _send_json(self, payload: dict[str, Any]) -> None:
        encoded = json.dumps(payload).encode("utf-8")
        self.send_response(200)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def _send_error(self, message: str) -> None:
        encoded = json.dumps(
            {
                "error": {
                    "message": message,
                    "type": "invalid_request_error",
                    "param": "input",
                    "code": None,
                }
            }
        ).encode("utf-8")
        self.send_response(400)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(encoded)))
        self.end_headers()
        self.wfile.write(encoded)

    def log_message(self, _format: str, *_args: Any) -> None:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, required=True)
    parser.add_argument("--response", required=True)
    parser.add_argument("--log-file", type=Path, required=True)
    parser.add_argument("--ready-file", type=Path, required=True)
    parser.add_argument(
        "--scenario",
        default="single",
        choices=_SCENARIOS,
    )
    parser.add_argument("--stream-delay", type=float, default=0.0)
    parser.add_argument("--subagent-delay", type=float, default=0.0)
    parser.add_argument("--stream-lines", type=int, default=2)
    args = parser.parse_args()

    _Handler.response_text = args.response
    _Handler.log_file = args.log_file
    _Handler.scenario = args.scenario
    _Handler.subagent_delay = max(0.0, args.subagent_delay)
    _Handler.stream_delay = max(0.0, args.stream_delay)
    _Handler.stream_lines = max(1, args.stream_lines)

    server = _ReusableHTTPServer(("127.0.0.1", args.port), _Handler)
    bound_port = int(server.server_address[1])
    args.ready_file.parent.mkdir(parents=True, exist_ok=True)
    args.ready_file.write_text(f"ready http://127.0.0.1:{bound_port}/v1\n", encoding="utf-8")
    try:
        server.serve_forever()
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
