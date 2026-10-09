import pytest

from dcf_xet_xorb import XorbDecodeError, _undo_byte_grouping, decode_xorb_chunks


def _chunk(data, scheme=0, packed=None):
    packed = data if packed is None else packed
    return bytes([0]) + len(packed).to_bytes(3, "little") + bytes([scheme]) + len(data).to_bytes(3, "little") + packed


def test_uncompressed_multiple_chunks():
    assert decode_xorb_chunks(_chunk(b"abc") + _chunk(b"defg")) == [b"abc", b"defg"]


@pytest.mark.parametrize("length", range(1, 25))
def test_undo_byte_grouping(length):
    original = bytes(range(length))
    grouped = b"".join(original[i::4] for i in range(4))
    assert _undo_byte_grouping(grouped) == original


@pytest.mark.parametrize("payload", [
    b"short", bytes([1, 1, 0, 0, 0, 1, 0, 0]) + b"x",
    bytes([0, 2, 0, 0, 0, 2, 0, 0]) + b"x",
    bytes([0, 1, 0, 0, 3, 1, 0, 0]) + b"x",
    _chunk(b"x")[:-1],
])
def test_invalid_chunks_rejected(payload):
    with pytest.raises(XorbDecodeError):
        decode_xorb_chunks(payload)


def test_limit_enforced():
    with pytest.raises(XorbDecodeError):
        decode_xorb_chunks(_chunk(b"abcdef"), max_unpacked=5)


def test_no_data_accepted_as_empty_range():
    assert decode_xorb_chunks(b"") == []
