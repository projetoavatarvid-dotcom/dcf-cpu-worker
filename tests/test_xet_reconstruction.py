from types import SimpleNamespace
import pytest
from dcf_xet_reconstruction import reconstruct_range, ReconstructionError


def _chunk(data):
    return bytes([0]) + len(data).to_bytes(3, "little") + b"\x00" + len(data).to_bytes(3, "little") + data


def _metadata(length):
    return {
        "offset_into_first_range": 2,
        "terms": [{"hash": "a", "range": {"start": 0, "end": 2}, "unpacked_length": 8}],
        "xorbs": {"a": [{"url": "https://example.invalid/xorb", "ranges": [
            {"bytes": {"start": 10, "end": 10 + length - 1}, "chunks": {"start": 0, "end": 2}}
        ]}]}
    }


def test_reconstruct_offset_and_truncation():
    body = _chunk(b"abcd") + _chunk(b"efgh")
    def fetch(url, byte_range):
        assert byte_range == f"bytes=10-{9 + len(body)}"
        return SimpleNamespace(status_code=206, headers={"Content-Range": f"bytes 10-{9 + len(body)}/100"}, content=body)
    assert reconstruct_range(_metadata(len(body)), 5, fetch) == b"cdefg"


def test_reject_ignored_range():
    body = _chunk(b"abcd") + _chunk(b"efgh")
    def fetch(url, byte_range):
        return SimpleNamespace(status_code=200, headers={}, content=body)
    with pytest.raises(ReconstructionError, match="range request failed"):
        reconstruct_range(_metadata(len(body)), 5, fetch)


def test_reject_corrupt_term_size():
    body = _chunk(b"abcd") + _chunk(b"efgh")
    meta = _metadata(len(body))
    meta["terms"][0]["unpacked_length"] = 7
    def fetch(url, byte_range):
        return SimpleNamespace(status_code=206, headers={"Content-Range": f"bytes 10-{9 + len(body)}/100"}, content=body)
    with pytest.raises(ReconstructionError, match="term size mismatch"):
        reconstruct_range(meta, 5, fetch)
