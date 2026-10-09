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

    def fake_xet(artifact, client, bucket, token):
        called.update(artifact=artifact, client=client, bucket=bucket, token=token)
        return artifact["expected_size_bytes"], artifact["expected_sha256"]

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
