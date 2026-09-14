"""Standalone pipe protocol for existing file-based macOS Keychain items.

Importing this module never loads a framework. Native calls run only in the
helper process, with temporary noninteractive policy and owned native buffers.
The deprecated SecKeychain APIs preserve the existing store and item ACLs.
"""

from __future__ import annotations

import base64
import ctypes
import json
import platform
import sys
from dataclasses import dataclass

PROTOCOL_VERSION = 1
MAX_SECRET_BYTES = 1024 * 1024
MAX_IDENTIFIER_BYTES = 64 * 1024
MAX_PROTOCOL_BYTES = 2 * 1024 * 1024
ERR_SUCCESS = 0
ERR_ITEM_NOT_FOUND = -25300
ERR_DUPLICATE_ITEM = -25299
SECURITY_FRAMEWORK = "/System/Library/Frameworks/Security.framework/Security"
FOUNDATION_FRAMEWORK = "/System/Library/Frameworks/CoreFoundation.framework/CoreFoundation"


@dataclass(frozen=True)
class NativeResult:
    status: int
    data: bytes | None = None


def _identifier(value) -> bytes:
    if not isinstance(value, str) or not value or "\x00" in value:
        raise ValueError("Invalid identifier")
    encoded = value.encode("utf-8")
    if len(encoded) > MAX_IDENTIFIER_BYTES:
        raise ValueError("Identifier too large")
    return encoded


def parse_request(payload: object) -> tuple[str, bytes, bytes, bytes | None]:
    if not isinstance(payload, dict) or type(payload.get("version")) is not int:
        raise ValueError("Invalid request")
    if payload["version"] != PROTOCOL_VERSION:
        raise ValueError("Unsupported protocol")
    operation = payload.get("operation")
    if operation not in {"store", "retrieve", "delete"}:
        raise ValueError("Invalid operation")
    expected = {"version", "operation", "service", "account"}
    if operation == "store":
        expected.add("data_b64")
    if set(payload) != expected:
        raise ValueError("Invalid request fields")
    service, account = _identifier(payload["service"]), _identifier(payload["account"])
    data = None
    if operation == "store":
        encoded = payload["data_b64"]
        if not isinstance(encoded, str) or len(encoded) > MAX_PROTOCOL_BYTES:
            raise ValueError("Invalid data")
        data = base64.b64decode(encoded, validate=True)
        if len(data) > MAX_SECRET_BYTES:
            raise ValueError("Data too large")
    return operation, service, account, data


class NativeKeychain:
    """Byte-preserving operations against the original file-based search list."""

    def __init__(self, *, security=None, foundation=None):
        if (security is None) != (foundation is None):
            raise ValueError("Supply both framework adapters")
        if security is None:
            if platform.system() != "Darwin":
                raise OSError("Unsupported platform")
            security = ctypes.CDLL(SECURITY_FRAMEWORK)
            foundation = ctypes.CDLL(FOUNDATION_FRAMEWORK)
        self.security = security
        self.foundation = foundation
        pointer, length, status = ctypes.c_void_p, ctypes.c_uint32, ctypes.c_int32
        self._bind(
            security,
            "SecKeychainFindGenericPassword",
            [
                pointer,
                length,
                pointer,
                length,
                pointer,
                ctypes.POINTER(length),
                ctypes.POINTER(pointer),
                ctypes.POINTER(pointer),
            ],
            status,
        )
        self._bind(
            security,
            "SecKeychainAddGenericPassword",
            [pointer, length, pointer, length, pointer, length, pointer, ctypes.POINTER(pointer)],
            status,
        )
        self._bind(
            security,
            "SecKeychainItemModifyAttributesAndData",
            [pointer, pointer, length, pointer],
            status,
        )
        self._bind(security, "SecKeychainItemDelete", [pointer], status)
        self._bind(security, "SecKeychainItemFreeContent", [pointer, pointer], status)
        self._bind(
            security,
            "SecKeychainGetUserInteractionAllowed",
            [ctypes.POINTER(ctypes.c_ubyte)],
            status,
        )
        self._bind(security, "SecKeychainSetUserInteractionAllowed", [ctypes.c_ubyte], status)
        self._bind(foundation, "CFRelease", [pointer], None)

    @staticmethod
    def _bind(library, name, arguments, result):
        function = getattr(library, name)
        function.argtypes, function.restype = arguments, result

    def perform(
        self, operation: str, service: bytes, account: bytes, data: bytes | None
    ) -> NativeResult:
        # Zero-length attributes are omitted by SecKeychainFindGenericPassword;
        # reject them rather than broadening a lookup/update/delete.
        for value in (service, account):
            if (
                not isinstance(value, bytes)
                or not value
                or b"\x00" in value
                or len(value) > MAX_IDENTIFIER_BYTES
            ):
                raise ValueError("Invalid native identifier")
        if operation == "store" and (not isinstance(data, bytes) or len(data) > MAX_SECRET_BYTES):
            raise ValueError("Invalid native store data")
        previous = ctypes.c_ubyte()
        status = self.security.SecKeychainGetUserInteractionAllowed(ctypes.byref(previous))
        if status != ERR_SUCCESS:
            return NativeResult(status)
        try:
            status = self.security.SecKeychainSetUserInteractionAllowed(0)
            if status != ERR_SUCCESS:
                return NativeResult(status)
            if operation == "retrieve":
                return self._retrieve(service, account)
            if operation == "store":
                if data is None:
                    raise ValueError("Missing store data")
                return NativeResult(self._store(service, account, data))
            if operation == "delete":
                return NativeResult(self._delete(service, account))
            raise ValueError("Invalid operation")
        finally:
            if self.security.SecKeychainSetUserInteractionAllowed(previous.value) != ERR_SUCCESS:
                raise RuntimeError("Interaction policy restoration failed")

    def _find(self, service: bytes, account: bytes, *, password: bool):
        service_buffer = ctypes.create_string_buffer(service)
        account_buffer = ctypes.create_string_buffer(account)
        length, data, item = ctypes.c_uint32(), ctypes.c_void_p(), ctypes.c_void_p()
        status = self.security.SecKeychainFindGenericPassword(
            None,
            len(service),
            ctypes.cast(service_buffer, ctypes.c_void_p),
            len(account),
            ctypes.cast(account_buffer, ctypes.c_void_p),
            ctypes.byref(length) if password else None,
            ctypes.byref(data) if password else None,
            None if password else ctypes.byref(item),
        )
        return status, length, data, item

    def _retrieve(self, service: bytes, account: bytes) -> NativeResult:
        status, length, data, _item = self._find(service, account, password=True)
        try:
            if status != ERR_SUCCESS:
                return NativeResult(status)
            if length.value > MAX_SECRET_BYTES or (length.value and not data.value):
                raise ValueError("Invalid native result")
            value = ctypes.string_at(data.value, length.value) if length.value else b""
            return NativeResult(status, value)
        finally:
            if data.value:
                if self.security.SecKeychainItemFreeContent(None, data) != ERR_SUCCESS:
                    raise RuntimeError("Native buffer release failed")

    def _store(self, service: bytes, account: bytes, data: bytes) -> int:
        status, _length, _data, item = self._find(service, account, password=False)
        try:
            if status == ERR_ITEM_NOT_FOUND:
                service_buffer = ctypes.create_string_buffer(service)
                account_buffer = ctypes.create_string_buffer(account)
                value_buffer = ctypes.create_string_buffer(data)
                status = self.security.SecKeychainAddGenericPassword(
                    None,
                    len(service),
                    ctypes.cast(service_buffer, ctypes.c_void_p),
                    len(account),
                    ctypes.cast(account_buffer, ctypes.c_void_p),
                    len(data),
                    ctypes.cast(value_buffer, ctypes.c_void_p),
                    None,
                )
                if status != ERR_DUPLICATE_ITEM:
                    return status
                # Retry a concurrent insertion once, without deleting its ACL.
                if item.value:
                    self.foundation.CFRelease(item)
                    item = ctypes.c_void_p()
                status, _length, _data, item = self._find(service, account, password=False)
            if status != ERR_SUCCESS:
                return status
            if not item.value:
                raise ValueError("Missing native item reference")
            # A non-NULL buffer with zero length really writes an empty value.
            value_buffer = ctypes.create_string_buffer(data)
            return self.security.SecKeychainItemModifyAttributesAndData(
                item, None, len(data), ctypes.cast(value_buffer, ctypes.c_void_p)
            )
        finally:
            if item.value:
                self.foundation.CFRelease(item)

    def _delete(self, service: bytes, account: bytes) -> int:
        status, _length, _data, item = self._find(service, account, password=False)
        try:
            if status != ERR_SUCCESS:
                return status
            if not item.value:
                raise ValueError("Missing native item reference")
            return self.security.SecKeychainItemDelete(item)
        finally:
            if item.value:
                self.foundation.CFRelease(item)


def handle_request(payload: object, *, backend_factory=None) -> dict:
    """Validate before constructing the backend; never include exception text."""
    operation = payload.get("operation") if isinstance(payload, dict) else None
    if not isinstance(operation, str) or operation not in {"store", "retrieve", "delete"}:
        operation = None
    reply = {"version": PROTOCOL_VERSION, "operation": operation}
    try:
        operation, service, account, data = parse_request(payload)
    except Exception:
        return {**reply, "error": "invalid_request"}
    try:
        backend = (NativeKeychain if backend_factory is None else backend_factory)()
    except Exception:
        return {**reply, "error": "unavailable"}
    try:
        result = backend.perform(operation, service, account, data)
        if type(result.status) is not int or not -(2**31) <= result.status < 2**31:
            raise ValueError("Invalid native status")
        response = {**reply, "status": result.status}
        if operation == "retrieve" and result.status == ERR_SUCCESS:
            if not isinstance(result.data, bytes) or len(result.data) > MAX_SECRET_BYTES:
                raise ValueError("Invalid native data")
            response["data_b64"] = base64.b64encode(result.data).decode("ascii")
        return response
    except Exception:
        return {**reply, "error": "operation_failed"}


def main() -> int:
    try:
        raw = sys.stdin.buffer.read(MAX_PROTOCOL_BYTES + 1)
        if len(raw) > MAX_PROTOCOL_BYTES:
            raise ValueError("Request too large")
        payload = json.loads(raw)
    except Exception:
        payload = None
    response = handle_request(payload)
    sys.stdout.buffer.write(json.dumps(response, ensure_ascii=True).encode("utf-8") + b"\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
