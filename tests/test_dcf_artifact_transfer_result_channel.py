from __future__ import annotations

import json
import unittest
from unittest.mock import patch

import dcf_artifact_transfer as transfer


TRANSFER_ID = "a" * 64


def result_channel():
    return {
        "transfer_id": TRANSFER_ID,
        "result_key": (
            "workflow-preparation/transfer-results/"
            + TRANSFER_ID
            + "/result.json"
        ),
        "content_type": "application/json",
    }


def request():
    return {
        "schema_version": "1.0",
        "artifacts": [{
            "repository": "Lightricks/LTX-2.5",
            "revision": "main",
            "repository_path": "vae/example.safetensors",
            "expected_size_bytes": 123,
            "r2_key": "models/vae/example.safetensors",
            "gated": True,
        }],
        "result_channel": result_channel(),
    }


class ResultChannelValidationTests(unittest.TestCase):
    def test_accepts_canonical_result_channel(self):
        validated = transfer.validate_request(request())

        self.assertEqual(
            validated["result_channel"],
            result_channel(),
        )

    def test_backward_compatible_without_result_channel(self):
        payload = request()
        del payload["result_channel"]

        validated = transfer.validate_request(payload)

        self.assertIsNone(
            validated["result_channel"]
        )

    def test_rejects_invalid_transfer_id(self):
        payload = request()
        payload["result_channel"]["transfer_id"] = "abc"

        with self.assertRaises(transfer.TransferError):
            transfer.validate_request(payload)

    def test_rejects_key_not_matching_transfer_id(self):
        payload = request()
        payload["result_channel"]["result_key"] = (
            "workflow-preparation/transfer-results/"
            + ("b" * 64)
            + "/result.json"
        )

        with self.assertRaises(transfer.TransferError):
            transfer.validate_request(payload)

    def test_rejects_non_json_content_type(self):
        payload = request()
        payload["result_channel"]["content_type"] = "text/plain"

        with self.assertRaises(transfer.TransferError):
            transfer.validate_request(payload)


class ResultPublicationTests(unittest.TestCase):
    def test_refuses_result_overwrite(self):
        with patch.object(
            transfer,
            "_head_or_none",
            return_value={"ContentLength": 10},
        ):
            with self.assertRaisesRegex(
                transfer.TransferError,
                "refusing overwrite",
            ):
                transfer._publish_result_json(
                    object(),
                    "bucket",
                    result_channel(),
                    {"status": "COMPLETED"},
                )

    def test_writes_exact_result_json_and_verifies_size(self):
        stored = {}

        class Client:
            def put_object(self, **kwargs):
                stored.update(kwargs)
                return {}

        heads = [
            None,
            "REMOTE",
        ]

        def fake_head(client, bucket, key):
            current = heads.pop(0)

            if current is None:
                return None

            return {
                "ContentLength": len(stored["Body"]),
                "ETag": '"informational-only"',
            }

        result = {
            "schema_version": "1.0",
            "status": "COMPLETED",
            "mode": "SEQUENTIAL",
            "gpu_required": False,
            "network_volume_required": False,
            "catalog_write_performed": False,
            "artifacts": [{
                "status": "TRANSFERRED_AND_VERIFIED",
                "repository": "Lightricks/LTX-2.5",
                "revision": "main",
                "repository_path": "vae/example.safetensors",
                "r2_key": "models/vae/example.safetensors",
                "size_bytes": 123,
                "sha256": "c" * 64,
                "sha256_source": "calculated_on_cpu_staging",
                "r2_size_bytes": 123,
                "r2_etag_informational": "artifact-etag",
                "etag_is_sha256": False,
                "local_staging_removed": True,
            }],
        }

        with patch.object(
            transfer,
            "_head_or_none",
            side_effect=fake_head,
        ):
            transfer._publish_result_json(
                Client(),
                "bucket",
                result_channel(),
                result,
            )

        self.assertEqual(
            stored["Bucket"],
            "bucket",
        )
        self.assertEqual(
            stored["Key"],
            result_channel()["result_key"],
        )
        self.assertEqual(
            stored["ContentType"],
            "application/json",
        )

        decoded = json.loads(
            stored["Body"].decode("utf-8")
        )

        self.assertEqual(
            decoded,
            result,
        )
        self.assertEqual(
            decoded["artifacts"][0]["sha256"],
            "c" * 64,
        )
        self.assertFalse(
            decoded["artifacts"][0]["etag_is_sha256"]
        )

    def test_rejects_remote_result_size_mismatch(self):
        class Client:
            def put_object(self, **kwargs):
                return {}

        heads = [
            None,
            {"ContentLength": 1},
        ]

        with patch.object(
            transfer,
            "_head_or_none",
            side_effect=heads,
        ):
            with self.assertRaisesRegex(
                transfer.TransferError,
                "size mismatch",
            ):
                transfer._publish_result_json(
                    Client(),
                    "bucket",
                    result_channel(),
                    {"status": "COMPLETED"},
                )


if __name__ == "__main__":
    unittest.main()
