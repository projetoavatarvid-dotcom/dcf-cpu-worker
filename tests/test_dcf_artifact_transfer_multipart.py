import hashlib
import sys
from types import SimpleNamespace

import pytest

import dcf_artifact_transfer as transfer


class FakeResponse:
    def __init__(self, chunks, status_code=200):
        self._chunks = chunks
        self.status_code = status_code

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def iter_content(self, chunk_size):
        yield from self._chunks


class FakeR2:
    def __init__(self):
        self.parts = []
        self.completed = False
        self.aborted = False

    def create_multipart_upload(self, **kwargs):
        return {"UploadId": "upload-1"}

    def upload_part(
        self,
        *,
        Bucket,
        Key,
        UploadId,
        PartNumber,
        Body,
    ):
        self.parts.append(bytes(Body))
        return {"ETag": f'"etag-{PartNumber}"'}

    def complete_multipart_upload(self, **kwargs):
        self.completed = True

    def abort_multipart_upload(self, **kwargs):
        self.aborted = True


def _artifact(payload, expected_size=None):
    return {
        "artifact_type": "MODEL_ARTIFACT",
        "repository": "owner/repo",
        "revision": "main",
        "repository_path": "models/model.safetensors",
        "r2_key": "models/model.safetensors",
        "expected_size_bytes": (
            len(payload)
            if expected_size is None
            else expected_size
        ),
        "expected_sha256": hashlib.sha256(payload).hexdigest(),
        "gated": False,
    }


def test_model_streams_directly_to_r2_multipart(monkeypatch):
    payload = b"abcdefghijklmnop"

    fake_requests = SimpleNamespace(
        get=lambda *args, **kwargs: FakeResponse(
            [payload[:5], payload[5:11], payload[11:]]
        )
    )

    monkeypatch.setitem(
        sys.modules,
        "requests",
        fake_requests,
    )

    client = FakeR2()

    size, digest = (
        transfer._stream_huggingface_to_r2_multipart(
            _artifact(payload),
            client,
            "bucket",
            None,
            part_size=6,
            chunk_size=4,
        )
    )

    assert size == len(payload)
    assert digest == hashlib.sha256(payload).hexdigest()
    assert b"".join(client.parts) == payload
    assert len(client.parts) == 3
    assert client.completed is True
    assert client.aborted is False


def test_multipart_is_aborted_on_size_mismatch(monkeypatch):
    payload = b"abcdef"

    fake_requests = SimpleNamespace(
        get=lambda *args, **kwargs: FakeResponse([payload])
    )

    monkeypatch.setitem(
        sys.modules,
        "requests",
        fake_requests,
    )

    client = FakeR2()

    with pytest.raises(
        transfer.TransferError,
        match="download size mismatch",
    ):
        transfer._stream_huggingface_to_r2_multipart(
            _artifact(
                payload,
                expected_size=len(payload) + 1,
            ),
            client,
            "bucket",
            None,
            part_size=5,
            chunk_size=3,
        )

    assert client.completed is False
    assert client.aborted is True



def test_multipart_is_aborted_on_sha256_mismatch(monkeypatch):
    payload = b"abcdef"

    fake_requests = SimpleNamespace(
        get=lambda *args, **kwargs: FakeResponse([payload])
    )

    monkeypatch.setitem(
        sys.modules,
        "requests",
        fake_requests,
    )

    artifact = _artifact(payload)
    artifact["expected_sha256"] = "0" * 64

    client = FakeR2()

    with pytest.raises(
        transfer.TransferError,
        match="sha256 mismatch",
    ):
        transfer._stream_huggingface_to_r2_multipart(
            artifact,
            client,
            "bucket",
            None,
            part_size=5,
            chunk_size=3,
        )

    assert client.completed is False
    assert client.aborted is True


def test_execute_request_routes_model_to_streaming_without_staging(
    monkeypatch,
    tmp_path,
):
    payload = b"abcdef"
    artifact = _artifact(payload)

    request = {
        "schema_version": "1.0",
        "mode": "SEQUENTIAL",
        "artifacts": [artifact],
        "transfer": {
            "object_workers": 1,
            "part_concurrency": 1,
        },
        "result_channel": None,
    }

    client = object()
    head_calls = {"count": 0}

    monkeypatch.setenv("DCF_R2_ENDPOINT", "https://example.invalid")
    monkeypatch.setenv("DCF_R2_BUCKET", "bucket")
    monkeypatch.setenv("DCF_R2_ACCESS_KEY", "access")
    monkeypatch.setenv("DCF_R2_SECRET_KEY", "secret")

    monkeypatch.setattr(
        transfer,
        "_build_r2_client",
        lambda *args, **kwargs: client,
    )

    def fake_head(client_arg, bucket, key):
        head_calls["count"] += 1

        if head_calls["count"] == 1:
            return None

        return {
            "ContentLength": len(payload),
            "ETag": '"etag-final"',
        }

    monkeypatch.setattr(
        transfer,
        "_head_or_none",
        fake_head,
    )

    monkeypatch.setattr(
        transfer,
        "_stream_huggingface_to_r2_multipart",
        lambda *args, **kwargs: (
            len(payload),
            hashlib.sha256(payload).hexdigest(),
        ),
    )

    def forbidden(*args, **kwargs):
        raise AssertionError(
            "MODEL_ARTIFACT must not use local staging/s5cmd"
        )

    monkeypatch.setattr(
        transfer,
        "_download",
        forbidden,
    )

    monkeypatch.setattr(
        transfer,
        "_upload_with_s5cmd",
        forbidden,
    )

    result = transfer.execute_request(
        request,
        staging_root=tmp_path,
    )

    assert result["status"] == "COMPLETED"
    assert len(result["artifacts"]) == 1

    evidence = result["artifacts"][0]

    assert evidence["status"] == "TRANSFERRED_AND_VERIFIED"
    assert evidence["transfer_method"] == "R2_MULTIPART_STREAMING"
    assert evidence["size_bytes"] == len(payload)
    assert evidence["r2_size_bytes"] == len(payload)

    assert list(tmp_path.iterdir()) == []



def test_default_multipart_part_size_is_64_mib_for_normal_models():
    assert transfer._select_multipart_part_size(
        20 * 1024 * 1024 * 1024
    ) == 64 * 1024 * 1024


def test_multipart_part_size_grows_for_very_large_models():
    size = (
        transfer.DEFAULT_MULTIPART_PART_SIZE
        * transfer.MAX_MULTIPART_PARTS
        + 1
    )

    part_size = transfer._select_multipart_part_size(size)

    assert part_size > transfer.DEFAULT_MULTIPART_PART_SIZE

    part_count = (
        size + part_size - 1
    ) // part_size

    assert part_count <= transfer.MAX_MULTIPART_PARTS
