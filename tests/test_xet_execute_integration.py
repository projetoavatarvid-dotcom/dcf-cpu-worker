import os
from pathlib import Path
from unittest.mock import patch

import dcf_artifact_transfer as transfer
import dcf_xet_stream


def test_large_model_routes_to_native_xet_stream(monkeypatch, tmp_path):
    request = transfer.validate_request({
        "schema_version": "1.0",
        "artifacts": [{
            "repository": "org/repo",
            "revision": "abc123",
            "repository_path": "model.safetensors",
            "expected_size_bytes": transfer.DEFAULT_MULTIPART_PART_SIZE + 1,
            "expected_sha256": "a" * 64,
            "r2_key": "models/model.safetensors",
            "gated": True,
        }],
    })
    fake_client = object()
    called = {}

    def fake_xet(
        artifact,
        client,
        bucket,
        token,
        *,
        progress_callback=None,
    ):
        called.update(
            artifact=artifact,
            client=client,
            bucket=bucket,
            token=token,
            progress_callback=progress_callback,
        )
        return (
            artifact["expected_size_bytes"],
            artifact["expected_sha256"],
        )

    monkeypatch.setattr(dcf_xet_stream, "transfer_xet_stream_to_r2", fake_xet)
    env = {
        "DCF_R2_ENDPOINT": "https://example.invalid",
        "DCF_R2_BUCKET": "bucket",
        "DCF_R2_ACCESS_KEY": "access",
        "DCF_R2_SECRET_KEY": "secret",
        "HF_TOKEN": "token",
    }
    with patch.dict(os.environ, env, clear=True),          patch.object(transfer, "_build_r2_client", return_value=fake_client),          patch.object(transfer, "_head_or_none", side_effect=[None, {"ContentLength": request["artifacts"][0]["expected_size_bytes"]}]):
        result = transfer.execute_request(request, tmp_path)

    assert called["client"] is fake_client
    assert called["bucket"] == "bucket"
    assert called["token"] == "token"
    assert result["artifacts"][0]["transfer_method"] == "HF_XET_STREAM_R2_MULTIPART"


def test_large_xet_retries_and_reports_attempt_number(
    monkeypatch,
    tmp_path,
):
    request = transfer.validate_request({
        "schema_version": "1.0",
        "artifacts": [{
            "repository": "org/repo",
            "revision": "abc123",
            "repository_path": "model.safetensors",
            "expected_size_bytes":
                transfer.DEFAULT_MULTIPART_PART_SIZE + 1,
            "expected_sha256": "a" * 64,
            "r2_key": "models/model.safetensors",
            "gated": True,
        }],
        "result_channel": {
            "transfer_id": "b" * 64,
            "result_key":
                "workflow-preparation/transfer-results/" + ("b" * 64) + "/result.json",
        },
    })

    class FakeClient:
        def __init__(self):
            self.puts = []
            self.objects = {}

        def put_object(self, **kwargs):
            self.puts.append(kwargs)
            self.objects[kwargs["Key"]] = kwargs["Body"]
            return {"ETag": '"fake-etag"'}

    fake_client = FakeClient()
    calls = []
    progress = []

    def fake_xet(
        artifact,
        client,
        bucket,
        token,
        *,
        progress_callback=None,
    ):
        attempt = len(calls) + 1
        calls.append(attempt)

        if progress_callback is not None:
            progress_callback(
                artifact["expected_size_bytes"],
                artifact["expected_size_bytes"],
            )

        if attempt == 1:
            raise RuntimeError("forced first attempt failure")

        return (
            artifact["expected_size_bytes"],
            artifact["expected_sha256"],
        )

    def fake_progress(
        client,
        bucket,
        result_channel,
        payload,
    ):
        progress.append(dict(payload))

    monkeypatch.setattr(
        dcf_xet_stream,
        "transfer_xet_stream_to_r2",
        fake_xet,
    )

    monkeypatch.setattr(
        transfer,
        "_publish_progress_json",
        fake_progress,
    )

    monkeypatch.setenv(
        "DCF_R2_ENDPOINT",
        "https://example.invalid",
    )
    monkeypatch.setenv("DCF_R2_BUCKET", "bucket")
    monkeypatch.setenv("DCF_R2_ACCESS_KEY", "access")
    monkeypatch.setenv("DCF_R2_SECRET_KEY", "secret")
    monkeypatch.setenv("HF_TOKEN", "token")

    monkeypatch.setattr(
        transfer,
        "_build_r2_client",
        lambda *args, **kwargs: fake_client,
    )

    head_calls = {"model": 0}

    def fake_head(client, bucket, key):
        if key == "models/model.safetensors":
            head_calls["model"] += 1

            if head_calls["model"] == 1:
                return None

            return {
                "ContentLength":
                    request["artifacts"][0][
                        "expected_size_bytes"
                    ],
                "ETag": '"model-etag"',
            }

        if key in fake_client.objects:
            return {
                "ContentLength":
                    len(fake_client.objects[key]),
                "ETag": '"result-etag"',
            }

        return None

    monkeypatch.setattr(
        transfer,
        "_head_or_none",
        fake_head,
    )

    result = transfer.execute_request(
        request,
        tmp_path,
    )

    assert calls == [1, 2]
    assert [
        (item["attempt"], item["attempts"])
        for item in progress
    ] == [(1, 3), (2, 3)]

    assert result["artifacts"][0]["transfer_method"] == (
        "HF_XET_STREAM_R2_MULTIPART"
    )
