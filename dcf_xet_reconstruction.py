"""Bounded reconstruction of Xet CAS byte ranges."""
from collections import defaultdict
from dcf_xet_xorb import decode_xorb_chunks


class ReconstructionError(ValueError):
    pass


def reconstruct_range(metadata, expected_length, fetch):
    """Fetch signed xorb ranges and reconstruct one bounded file slice."""
    if not 0 < expected_length <= 64 * 1024 * 1024:
        raise ReconstructionError("invalid slice length")
    chunks = {}
    required = defaultdict(set)
    for term in metadata["terms"]:
        a, b = term["range"]["start"], term["range"]["end"]
        if a < 0 or b <= a:
            raise ReconstructionError("invalid chunk range")
        required[term["hash"]].update(range(a, b))
    for xorb_hash, entries in metadata["xorbs"].items():
        for entry in entries:
            descriptors = entry["ranges"]
            if len(descriptors) != 1:
                raise ReconstructionError("multipart xorb not yet supported")
            desc = descriptors[0]
            a, b = desc["bytes"]["start"], desc["bytes"]["end"]
            if a < 0 or b < a or b - a + 1 > 64 * 1024 * 1024:
                raise ReconstructionError("invalid xorb byte range")
            response = fetch(entry["url"], f"bytes={a}-{b}")
            if response.status_code != 206:
                raise ReconstructionError("xorb range request failed")
            if not response.headers.get("Content-Range", "").startswith(f"bytes {a}-{b}/"):
                raise ReconstructionError("xorb range mismatch")
            if len(response.content) != b - a + 1:
                raise ReconstructionError("truncated xorb response")
            decoded = decode_xorb_chunks(response.content)
            start, end = desc["chunks"]["start"], desc["chunks"]["end"]
            if len(decoded) != end - start:
                raise ReconstructionError("chunk count mismatch")
            for index, data in enumerate(decoded, start):
                if index in required[xorb_hash]:
                    chunks[xorb_hash, index] = data
    result = bytearray()
    for term in metadata["terms"]:
        start, end = term["range"]["start"], term["range"]["end"]
        try:
            data = b"".join(chunks[term["hash"], i] for i in range(start, end))
        except KeyError as error:
            raise ReconstructionError("missing chunk") from error
        if len(data) != term["unpacked_length"]:
            raise ReconstructionError("term size mismatch")
        result.extend(data)
        if len(result) > expected_length + 128 * 1024:
            raise ReconstructionError("reconstructed slice too large")
    skip = metadata["offset_into_first_range"]
    if not 0 <= skip <= len(result):
        raise ReconstructionError("invalid offset")
    data = bytes(result[skip:skip + expected_length])
    if len(data) != expected_length:
        raise ReconstructionError("incomplete reconstructed slice")
    return data
