"""A successful no-follow write must consume all bytes, including short writes."""

import os

import pytest

from koder_agent.tools import file as file_tools


@pytest.mark.parametrize("append", [False, True])
def test_no_follow_write_handles_short_system_writes(tmp_path, monkeypatch, append):
    target = tmp_path / "sample.txt"
    target.write_bytes(b"before\n")
    content = "multibyte: 测试\n".encode("utf-8")
    real_write = os.write

    def short_write(descriptor, data):
        return real_write(descriptor, data[:3])

    monkeypatch.setattr(file_tools.os, "write", short_write)
    file_tools._write_bytes_no_follow(str(target), content, append=append)
    assert target.read_bytes() == (b"before\n" if append else b"") + content


def test_no_follow_write_rejects_zero_progress(tmp_path, monkeypatch):
    monkeypatch.setattr(file_tools.os, "write", lambda _descriptor, _data: 0)
    with pytest.raises(OSError, match="no progress"):
        file_tools._write_bytes_no_follow(str(tmp_path / "sample.txt"), b"content", append=False)
