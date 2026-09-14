"""Exercise ctypes marshalling and ownership against Python function adapters."""

import ctypes

import pytest

from koder_agent.auth import keychain_backend as backend


class Function:
    def __init__(self, implementation):
        self.implementation = implementation

    def __call__(self, *args):
        return self.implementation(*args)


def set_pointer(pointer, value, kind=ctypes.c_void_p):
    ctypes.cast(pointer, ctypes.POINTER(kind)).contents.value = value


class NativeFixture:
    """No CDLL, system calls, native keychain or user data."""

    def __init__(self):
        self.entries = {}
        self.acls = {}
        self.references = {}
        self.buffers = {}
        self.next_reference = 100
        self.ui_allowed = True
        self.ui_calls = []
        self.calls = []
        self.find_status = None
        self.length_override = None
        self.duplicate_once = False
        self.disable_status = 0
        self.restore_status = 0
        self.free_count = 0
        self.release_count = 0
        self.nil_empty_data = False
        for name, implementation in (
            ("SecKeychainFindGenericPassword", self.find),
            ("SecKeychainAddGenericPassword", self.add),
            ("SecKeychainItemModifyAttributesAndData", self.modify),
            ("SecKeychainItemDelete", self.delete),
            ("SecKeychainItemFreeContent", self.free),
            ("SecKeychainGetUserInteractionAllowed", self.get_ui),
            ("SecKeychainSetUserInteractionAllowed", self.set_ui),
            ("CFRelease", self.release),
        ):
            setattr(self, name, Function(implementation))

    def get_ui(self, output):
        set_pointer(output, int(self.ui_allowed), ctypes.c_ubyte)
        return 0

    def set_ui(self, value):
        self.ui_calls.append(value)
        self.ui_allowed = bool(value)
        return self.restore_status if value else self.disable_status

    @staticmethod
    def key(service_length, service, account_length, account):
        return ctypes.string_at(service, service_length), ctypes.string_at(account, account_length)

    def find(self, keychain, sn, service, an, account, length, data, item):
        assert keychain is None
        assert self.ui_allowed is False
        key = self.key(sn, service, an, account)
        self.calls.append(("find", key))
        if self.find_status is not None:
            return self.find_status
        # Match the native API's optional-attribute behavior: zero-length
        # identifiers omit that predicate rather than matching an empty value.
        candidates = [
            candidate
            for candidate in self.entries
            if (not sn or candidate[0] == key[0]) and (not an or candidate[1] == key[1])
        ]
        if not candidates:
            return backend.ERR_ITEM_NOT_FOUND
        key = candidates[0]
        if data is not None:
            value = self.entries[key]
            set_pointer(
                length,
                self.length_override if self.length_override is not None else len(value),
                ctypes.c_uint32,
            )
            if value or not self.nil_empty_data:
                buffer = ctypes.create_string_buffer(value)
                address = ctypes.addressof(buffer)
                self.buffers[address] = buffer
                set_pointer(data, address)
        else:
            assert length is None
        if item is not None:
            self.next_reference += 1
            self.references[self.next_reference] = key
            set_pointer(item, self.next_reference)
        return 0

    def add(self, keychain, sn, service, an, account, length, data, item):
        assert keychain is None and item is None
        key = self.key(sn, service, an, account)
        self.calls.append(("add", key))
        if self.duplicate_once:
            self.duplicate_once = False
            self.entries[key] = b"other writer"
            self.acls[key] = "existing ACL"
            return backend.ERR_DUPLICATE_ITEM
        if key in self.entries:
            return backend.ERR_DUPLICATE_ITEM
        assert data.value is not None
        self.entries[key] = ctypes.string_at(data, length)
        self.acls[key] = "default creator ACL"
        return 0

    def modify(self, item, attributes, length, data):
        assert attributes is None and data.value is not None
        key = self.references[item.value]
        self.calls.append(("modify", key))
        self.entries[key] = ctypes.string_at(data, length)
        return 0

    def delete(self, item):
        key = self.references[item.value]
        self.calls.append(("delete", key))
        self.entries.pop(key)
        self.acls.pop(key)
        return 0

    def free(self, attributes, data):
        assert attributes is None
        self.buffers.pop(data.value)
        self.free_count += 1
        return 0

    def release(self, item):
        self.references.pop(item.value)
        self.release_count += 1


@pytest.fixture
def native(monkeypatch):
    monkeypatch.setattr(
        backend.ctypes,
        "CDLL",
        lambda *_args, **_kwargs: pytest.fail("Never load a native framework in tests"),
    )
    functions = NativeFixture()
    adapter = backend.NativeKeychain(security=functions, foundation=functions)
    return adapter, functions


@pytest.mark.parametrize("value", [b"", b"  value  ", b"a\x00b\n", "合成 λ".encode()])
def test_native_data_is_length_delimited_and_owned(native, value):
    adapter, functions = native
    service, account = "服务".encode(), "账户suffix".encode()
    assert adapter.perform("store", service, account, value).status == 0
    assert functions.entries[(service, account)] == value
    result = adapter.perform("retrieve", service, account, None)
    assert result.status == 0 and result.data == value
    assert functions.buffers == {}
    assert functions.free_count == 1
    assert functions.ui_calls == [0, 1, 0, 1]
    assert functions.ui_allowed


def test_native_update_preserves_item_acl_and_empty_value(native):
    adapter, functions = native
    key = (b"service", b"account")
    functions.entries[key] = b"before"
    functions.acls[key] = "user ACL"
    assert adapter.perform("store", *key, b"").status == 0
    assert functions.entries[key] == b""
    assert functions.acls[key] == "user ACL"
    assert [name for name, _ in functions.calls] == ["find", "modify"]
    assert functions.references == {}
    assert functions.release_count == 1


def test_duplicate_insert_race_retries_lookup_once_without_delete(native):
    adapter, functions = native
    functions.duplicate_once = True
    key = (b"service", b"account")
    assert adapter.perform("store", *key, b"replacement").status == 0
    assert functions.entries[key] == b"replacement"
    assert functions.acls[key] == "existing ACL"
    assert [name for name, _ in functions.calls] == ["find", "add", "find", "modify"]
    assert functions.references == {}


@pytest.mark.parametrize("operation", ["store", "retrieve", "delete"])
@pytest.mark.parametrize("status", [-25300, -25308, -25293, -25556])
def test_lookup_status_is_not_folded_or_used_as_permission_to_mutate(native, operation, status):
    adapter, functions = native
    functions.find_status = status
    result = adapter.perform(operation, b"service", b"account", b"data")
    if operation == "store" and status == backend.ERR_ITEM_NOT_FOUND:
        assert result.status == 0
        assert [name for name, _ in functions.calls] == ["find", "add"]
    else:
        assert result.status == status
        assert [name for name, _ in functions.calls] == ["find"]
    assert functions.ui_allowed


def test_delete_uses_only_the_found_reference_and_releases_it(native):
    adapter, functions = native
    target, other = (b"service", b"account"), (b"other", b"account")
    functions.entries.update({target: b"target", other: b"keep"})
    functions.acls.update({target: "target ACL", other: "keep ACL"})
    assert adapter.perform("delete", *target, None).status == 0
    assert target not in functions.entries and functions.entries[other] == b"keep"
    assert functions.references == {} and functions.release_count == 1


def test_oversized_native_buffer_is_released_without_copying(native):
    adapter, functions = native
    functions.entries[(b"service", b"account")] = b"small backing allocation"
    functions.length_override = backend.MAX_SECRET_BYTES + 1
    with pytest.raises(ValueError, match="native result"):
        adapter.perform("retrieve", b"service", b"account", None)
    assert functions.buffers == {} and functions.free_count == 1
    assert functions.ui_allowed


def test_zero_length_null_result_is_a_real_empty_secret(native):
    adapter, functions = native
    functions.entries[(b"service", b"account")] = b""
    functions.nil_empty_data = True
    assert adapter.perform("retrieve", b"service", b"account", None).data == b""
    assert functions.free_count == 0


def test_failed_ui_suppression_performs_no_lookup_and_restores_previous_policy(native):
    adapter, functions = native
    functions.disable_status = -25308
    assert adapter.perform("delete", b"service", b"account", None).status == -25308
    assert functions.calls == []
    assert functions.ui_calls == [0, 1]
    assert functions.ui_allowed


def test_ui_policy_is_restored_when_an_operation_raises(native):
    adapter, functions = native
    functions.SecKeychainFindGenericPassword.implementation = lambda *_: (_ for _ in ()).throw(
        RuntimeError("synthetic native failure")
    )
    with pytest.raises(RuntimeError):
        adapter.perform("retrieve", b"service", b"account", None)
    assert functions.ui_allowed and functions.ui_calls == [0, 1]


def test_native_abi_uses_32_bit_status_and_lengths(native):
    adapter, functions = native
    assert functions.SecKeychainFindGenericPassword.restype is ctypes.c_int32
    assert functions.SecKeychainFindGenericPassword.argtypes[1] is ctypes.c_uint32
    assert functions.SecKeychainItemModifyAttributesAndData.argtypes[2] is ctypes.c_uint32
    assert functions.CFRelease.restype is None


def test_preexisting_noninteractive_policy_is_preserved(native):
    adapter, functions = native
    functions.ui_allowed = False
    assert (
        adapter.perform("retrieve", b"service", b"account", None).status
        == backend.ERR_ITEM_NOT_FOUND
    )
    assert functions.ui_calls == [0, 0]
    assert functions.ui_allowed is False


@pytest.mark.parametrize("service,account", [(b"", b"account"), (b"service", b"")])
def test_empty_identifier_cannot_widen_native_delete(native, service, account):
    adapter, functions = native
    target = (b"service", b"account")
    functions.entries[target] = b"must remain"
    functions.acls[target] = "must remain"
    with pytest.raises(ValueError):
        adapter.perform("delete", service, account, None)
    assert functions.entries[target] == b"must remain"
    assert functions.calls == []
