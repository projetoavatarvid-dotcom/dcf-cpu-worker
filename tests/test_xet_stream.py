import hashlib
import sys
from types import ModuleType, SimpleNamespace

import pytest

import dcf_xet_stream


class R2:
    def __init__(self):
        self.uploads = []
        self.aborted = False
        self.completed = False
        self.completed_parts = None

    def create_multipart_upload(self, **kw):
        return {"UploadId": "u"}

    def upload_part(self, PartNumber, Body, **kw):
        body = bytes(Body)
        self.uploads.append((PartNumber, body))
        return {"ETag": str(PartNumber)}

    def complete_multipart_upload(self, **kw):
        self.completed = True
        self.completed_parts = kw["MultipartUpload"]["Parts"]

    def abort_multipart_upload(self, **kw):
        self.aborted = True


def _install(monkeypatch, data, *, corrupt_part=None):
    calls = []

    class Stream:
        def __init__(self, payload):
            self.payload = payload
            self.cancelled = False

        def __iter__(self):
            # Split each requested range to ensure the implementation
            # does not assume one Xet chunk == one multipart part.
            midpoint = max(1, len(self.payload) // 2)

            yield self.payload[:midpoint]

            if midpoint < len(self.payload):
                yield self.payload[midpoint:]

        def cancel(self):
            self.cancelled = True

    class Group:
        def download_stream(
            self,
            info,
            start=None,
            end=None,
        ):
            start = 0 if start is None else start
            end = len(data) if end is None else end

            calls.append((start, end))

            payload = data[start:end]

            if (
                corrupt_part is not None
                and start == corrupt_part
            ):
                payload = payload[:-1]

            return Stream(payload)

    class Session:
        def new_download_stream_group(self, **kw):
            return Group()

    hx = ModuleType("hf_xet")
    hx.XetFileInfo = lambda h, s: (h, s)
    monkeypatch.setitem(sys.modules, "hf_xet", hx)

    fd = ModuleType("huggingface_hub.file_download")
    fd.hf_hub_url = lambda *a, **k: "url"
    fd.get_hf_file_metadata = (
        lambda *a, **k: SimpleNamespace(
            size=len(data),
            xet_file_data=SimpleNamespace(
                file_hash="hash",
                refresh_route="refresh",
            ),
        )
    )
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.file_download",
        fd,
    )

    hd = ModuleType("huggingface_hub.utils._headers")
    hd.build_hf_headers = lambda **k: {}
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.utils._headers",
        hd,
    )

    xu = ModuleType("huggingface_hub.utils._xet")
    xu.get_xet_session = lambda: Session()
    xu.refresh_xet_connection_info = (
        lambda **k: SimpleNamespace(
            endpoint="e",
            access_token="t",
            expiration_unix_epoch=1,
        )
    )
    xu.xet_headers_without_auth = lambda h: {}
    monkeypatch.setitem(
        sys.modules,
        "huggingface_hub.utils._xet",
        xu,
    )

    return calls


def _artifact(data):
    return {
        "repository": "r",
        "repository_path": "f",
        "revision": "rev",
        "r2_key": "k",
        "expected_size_bytes": len(data),
        "expected_sha256":
            hashlib.sha256(data).hexdigest(),
    }


def test_ranged_xet_to_multipart(monkeypatch):
    data = b"abcdefghij"

    calls = _install(
        monkeypatch,
        data,
    )

    r2 = R2()

    size, digest = (
        dcf_xet_stream.transfer_xet_stream_to_r2(
            _artifact(data),
            r2,
            "b",
            None,
            part_size=4,
            upload_workers=2,
        )
    )

    assert size == len(data)

    assert digest == hashlib.sha256(
        data
    ).hexdigest()

    # Ranges must cover the file exactly:
    # [0,4), [4,8), [8,10)
    assert sorted(calls) == [
        (0, 4),
        (4, 8),
        (8, 10),
    ]

    assert sorted(r2.uploads) == [
        (1, b"abcd"),
        (2, b"efgh"),
        (3, b"ij"),
    ]

    assert r2.completed
    assert not r2.aborted

    assert r2.completed_parts == [
        {
            "PartNumber": 1,
            "ETag": "1",
        },
        {
            "PartNumber": 2,
            "ETag": "2",
        },
        {
            "PartNumber": 3,
            "ETag": "3",
        },
    ]


def test_corrupt_range_aborts(monkeypatch):
    data = b"abcdefghij"

    # Corrupt the range beginning at offset 4.
    _install(
        monkeypatch,
        data,
        corrupt_part=4,
    )

    r2 = R2()

    with pytest.raises(
        dcf_xet_stream.XetStreamError
    ):
        dcf_xet_stream.transfer_xet_stream_to_r2(
            _artifact(data),
            r2,
            "b",
            None,
            part_size=4,
            upload_workers=2,
        )

    assert r2.aborted
    assert not r2.completed


def test_ranged_single_worker(monkeypatch):
    data = b"abcdefghijklmnopq"

    calls = _install(
        monkeypatch,
        data,
    )

    r2 = R2()

    size, digest = (
        dcf_xet_stream.transfer_xet_stream_to_r2(
            _artifact(data),
            r2,
            "b",
            None,
            part_size=5,
            upload_workers=1,
        )
    )

    assert size == len(data)

    assert digest == hashlib.sha256(
        data
    ).hexdigest()

    assert calls == [
        (0, 5),
        (5, 10),
        (10, 15),
        (15, 17),
    ]

    assert r2.completed
    assert not r2.aborted


def test_progress_reports_only_completed_r2_parts(
    monkeypatch,
):
    data = b"abcdefghij"

    _install(
        monkeypatch,
        data,
    )

    r2 = R2()
    progress = []

    size, digest = (
        dcf_xet_stream.transfer_xet_stream_to_r2(
            _artifact(data),
            r2,
            "b",
            None,
            part_size=4,
            upload_workers=2,
            progress_callback=lambda done, total:
                progress.append((done, total)),
        )
    )

    assert size == len(data)

    assert digest == hashlib.sha256(
        data
    ).hexdigest()

    assert progress

    # Parallel completion order is intentionally not assumed.
    done_values = [
        done
        for done, total in progress
    ]

    assert all(
        total == len(data)
        for done, total in progress
    )

    assert sorted(done_values) == [
        4,
        8,
        10,
    ]

    assert max(done_values) == len(data)

    assert r2.completed
    assert not r2.aborted
