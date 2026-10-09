"""Bounded Xet xorb chunk decoder (protocol v0).

This module is intentionally not wired to production transfers yet.
See https://huggingface.co/docs/xet/xorb for the wire format.
"""
from __future__ import annotations

from io import BytesIO


class XorbDecodeError(ValueError):
    pass


def _undo_byte_grouping(data: bytes) -> bytes:
    size = len(data)
    groups = []
    offset = 0
    for lane in range(4):
        length = (size + 3 - lane) // 4 if size > lane else 0
        groups.append(data[offset:offset + length])
        offset += length
    output = bytearray(size)
    for lane, group in enumerate(groups):
        output[lane::4] = group
    return bytes(output)


def decode_xorb_chunks(payload: bytes, *, max_unpacked: int = 128 * 1024 * 1024) -> list[bytes]:
    """Decode one serialized xorb byte range; enforce strict bounds."""
    if len(payload) > 64 * 1024 * 1024:
        raise XorbDecodeError("xorb range exceeds 64 MiB")
    reader = BytesIO(payload)
    result = []
    total = 0
    while reader.tell() < len(payload):
        header = reader.read(8)
        if len(header) != 8:
            raise XorbDecodeError("truncated chunk header")
        version = header[0]
        compressed = int.from_bytes(header[1:4], "little")
        scheme = header[4]
        unpacked = int.from_bytes(header[5:8], "little")
        if version != 0 or scheme not in (0, 1, 2):
            raise XorbDecodeError("unsupported xorb chunk format")
        if not 0 < unpacked <= 128 * 1024 or not 0 < compressed <= 128 * 1024:
            raise XorbDecodeError("invalid xorb chunk length")
        total += unpacked
        if total > max_unpacked:
            raise XorbDecodeError("unpacked xorb limit exceeded")
        body = reader.read(compressed)
        if len(body) != compressed:
            raise XorbDecodeError("truncated chunk data")
        if scheme == 0:
            data = body
        else:
            try:
                import lz4.block
                data = lz4.block.decompress(body, uncompressed_size=unpacked)
            except Exception as error:
                raise XorbDecodeError("LZ4 chunk decompression failed") from error
            if scheme == 2:
                data = _undo_byte_grouping(data)
        if len(data) != unpacked:
            raise XorbDecodeError("unpacked chunk size mismatch")
        result.append(data)
    return result
