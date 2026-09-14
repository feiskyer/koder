"""Checked, byte-preserving macOS Keychain access through a private pipe."""

from __future__ import annotations

import base64
import json
import os
import platform
import subprocess
import sys
from pathlib import Path
from typing import NoReturn, Optional

from .keychain_backend import (
    ERR_ITEM_NOT_FOUND,
    ERR_SUCCESS,
    MAX_PROTOCOL_BYTES,
    MAX_SECRET_BYTES,
    PROTOCOL_VERSION,
    parse_request,
)

KEYCHAIN_TIMEOUT_SECONDS = 5
_HELPER = Path(__file__).with_name("keychain_backend.py")
_ENVIRONMENT_KEYS = {"PATH", "HOME", "LANG", "LC_ALL", "LC_CTYPE", "TMPDIR", "TEMP", "TMP"}


class KeychainUnavailableError(OSError):
    pass


class KeychainMutationUncertainError(OSError):
    """The helper may have mutated storage, but no authoritative result arrived."""


class SecureStorage:
    """File-based Keychain compatibility without secret argv or CLI output parsing."""

    def is_available(self) -> bool:
        """Check platform/helper availability without accessing any keychain."""
        return platform.system() == "Darwin" and _HELPER.is_file()

    @staticmethod
    def _failed(operation: str, *, uncertain: bool = True) -> NoReturn:
        if uncertain and operation in {"store", "delete"}:
            raise KeychainMutationUncertainError(
                f"Keychain {operation} result is uncertain"
            ) from None
        raise OSError(f"Unable to {operation} credentials with Keychain") from None

    def _invoke(self, operation: str, service: str, account: str, data: str | None = None) -> dict:
        if not self.is_available():
            raise KeychainUnavailableError("Keychain storage unavailable")
        request = {
            "version": PROTOCOL_VERSION,
            "operation": operation,
            "service": service,
            "account": account,
        }
        if operation == "store":
            if not isinstance(data, str):
                raise ValueError("Credential data must be a string")
            if len(data) > MAX_SECRET_BYTES:
                raise ValueError("Credential data too large")
            raw_data = data.encode("utf-8")
            if len(raw_data) > MAX_SECRET_BYTES:
                raise ValueError("Credential data too large")
            request["data_b64"] = base64.b64encode(raw_data).decode("ascii")
        parse_request(request)
        encoded = json.dumps(request, ensure_ascii=True).encode("utf-8")
        if len(encoded) > MAX_PROTOCOL_BYTES:
            raise ValueError("Keychain request too large")
        environment = {key: os.environ[key] for key in _ENVIRONMENT_KEYS if key in os.environ}
        try:
            result = subprocess.run(
                [sys.executable, "-I", str(_HELPER)],
                input=encoded,
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                check=False,
                timeout=KEYCHAIN_TIMEOUT_SECONDS,
                env=environment,
                close_fds=True,
            )
        except (FileNotFoundError, PermissionError):
            raise KeychainUnavailableError("Keychain helper unavailable") from None
        except Exception:
            self._failed(operation)
        if type(result.returncode) is not int or result.returncode != 0:
            self._failed(operation)
        try:
            if not isinstance(result.stdout, bytes) or len(result.stdout) > MAX_PROTOCOL_BYTES:
                raise ValueError("Invalid helper output")
            response = json.loads(result.stdout)
            if (
                not isinstance(response, dict)
                or type(response.get("version")) is not int
                or response["version"] != PROTOCOL_VERSION
                or response.get("operation") != operation
            ):
                raise ValueError("Invalid helper response")
            if "error" in response:
                if set(response) != {"version", "operation", "error"}:
                    raise ValueError("Invalid error response")
                if response["error"] == "unavailable":
                    raise KeychainUnavailableError("Keychain backend unavailable")
                raise ValueError("Helper operation failed")
            status = response["status"]
            if type(status) is not int or not -(2**31) <= status < 2**31:
                raise ValueError("Invalid native status")
            expected = {"version", "operation", "status"}
            if operation == "retrieve" and status == ERR_SUCCESS:
                expected.add("data_b64")
            if set(response) != expected:
                raise ValueError("Invalid result fields")
            if operation == "retrieve" and status == ERR_SUCCESS:
                value = base64.b64decode(response["data_b64"], validate=True)
                if len(value) > MAX_SECRET_BYTES:
                    raise ValueError("Invalid native value")
                response["data"] = value.decode("utf-8")
        except KeychainUnavailableError:
            raise
        except Exception:
            self._failed(operation)
        return response

    def store_checked(self, service: str, account: str, data: str) -> bool:
        """Return a known native outcome; propagate uncertain writes for callers."""
        try:
            return self._invoke("store", service, account, data)["status"] == ERR_SUCCESS
        except KeychainUnavailableError:
            return False

    def store(self, service: str, account: str, data: str) -> bool:
        """Legacy boolean wrapper; durable token writers use store_checked."""
        try:
            return self.store_checked(service, account, data)
        except (OSError, ValueError):
            return False

    def retrieve_checked(self, service: str, account: str) -> Optional[str]:
        """Only native errSecItemNotFound is absence; transport errors are not."""
        result = self._invoke("retrieve", service, account)
        if result["status"] == ERR_SUCCESS:
            return result["data"]
        if result["status"] == ERR_ITEM_NOT_FOUND:
            return None
        self._failed("retrieve", uncertain=False)

    def retrieve(self, service: str, account: str) -> Optional[str]:
        try:
            return self.retrieve_checked(service, account)
        except (OSError, ValueError):
            return None

    def delete_checked(self, service: str, account: str) -> bool:
        result = self._invoke("delete", service, account)
        if result["status"] == ERR_SUCCESS:
            return True
        if result["status"] == ERR_ITEM_NOT_FOUND:
            return False
        self._failed("delete", uncertain=False)

    def delete(self, service: str, account: str) -> bool:
        try:
            return self.delete_checked(service, account)
        except (OSError, ValueError):
            return False


def get_storage() -> Optional[SecureStorage]:
    storage = SecureStorage()
    return storage if storage.is_available() else None
