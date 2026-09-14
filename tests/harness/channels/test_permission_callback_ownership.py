"""A pending relay callback belongs to one registration, not a reused ID."""

import pytest

from koder_agent.harness.channels.permissions import ChannelPermissionCallbacks


def test_duplicate_pending_registration_is_not_silently_replaced():
    callbacks = ChannelPermissionCallbacks()
    first, second = [], []
    callbacks.on_response("ABCDE", first.append)
    with pytest.raises(ValueError):
        callbacks.on_response("abcde", second.append)
    assert callbacks.resolve("abcde", "allow", "synthetic-server")
    assert len(first) == 1 and second == []


def test_retired_unsubscribe_cannot_remove_a_new_registration_of_the_same_handler():
    callbacks = ChannelPermissionCallbacks()
    received = []
    handler = received.append
    retired = callbacks.on_response("abcde", handler)
    assert callbacks.resolve("abcde", "deny", "synthetic-server")
    current = callbacks.on_response("abcde", handler)
    retired()
    assert callbacks.pending_count == 1
    assert callbacks.resolve("abcde", "allow", "synthetic-server")
    assert [response.behavior for response in received] == ["deny", "allow"]
    current()
    assert callbacks.pending_count == 0


@pytest.mark.parametrize("behavior", ["yes", "", None])
def test_invalid_verdict_keeps_the_pending_request(behavior):
    callbacks = ChannelPermissionCallbacks()
    received = []
    callbacks.on_response("abcde", received.append)
    assert not callbacks.resolve("abcde", behavior, "synthetic-server")
    assert callbacks.pending_count == 1 and not received
    assert callbacks.resolve("abcde", "deny", "synthetic-server")
