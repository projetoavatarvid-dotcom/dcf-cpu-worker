import os
import unittest
from pathlib import Path
from unittest.mock import patch

import dcf_artifact_transfer as transfer


class ArtifactTransferValidationTests(unittest.TestCase):
    def request(self):
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
        }

    def normalized(self):
        return transfer.validate_request(self.request())

    def test_valid_request(self):
        result = self.normalized()
        self.assertEqual(result["mode"], "SEQUENTIAL")
        self.assertEqual(result["transfer"]["object_workers"], 32)
        self.assertEqual(result["transfer"]["part_concurrency"], 80)

    def test_rejects_non_model_r2_key(self):
        request = self.request()
        request["artifacts"][0]["r2_key"] = "other/example.safetensors"
        with self.assertRaises(transfer.TransferError):
            transfer.validate_request(request)

    def test_rejects_unknown_size(self):
        request = self.request()
        request["artifacts"][0]["expected_size_bytes"] = None
        with self.assertRaises(transfer.TransferError):
            transfer.validate_request(request)

    def test_rejects_duplicate_destination(self):
        request = self.request()
        request["artifacts"].append(dict(request["artifacts"][0]))
        with self.assertRaises(transfer.TransferError):
            transfer.validate_request(request)

    def test_rejects_bad_sha256(self):
        request = self.request()
        request["artifacts"][0]["expected_sha256"] = "abc"
        with self.assertRaises(transfer.TransferError):
            transfer.validate_request(request)

    @patch.dict(os.environ, {
        "DCF_R2_ENDPOINT": "https://example.invalid",
        "DCF_R2_BUCKET": "bucket",
        "DCF_R2_ACCESS_KEY": "access",
        "DCF_R2_SECRET_KEY": "secret",
    }, clear=True)
    def test_gated_artifact_requires_hf_token_before_network_client(self):
        with patch.object(transfer, "_build_r2_client") as build_client:
            with self.assertRaisesRegex(transfer.TransferError, "HF_TOKEN is required"):
                transfer.execute_request(self.normalized(), Path("unused"))
            build_client.assert_not_called()

    @patch.dict(os.environ, {
        "DCF_R2_ENDPOINT": "https://example.invalid",
        "DCF_R2_BUCKET": "bucket",
        "DCF_R2_ACCESS_KEY": "access",
        "DCF_R2_SECRET_KEY": "secret",
        "HF_TOKEN": "super-secret-token",
    }, clear=True)
    def test_existing_r2_object_refuses_overwrite_before_download(self):
        fake_client = object()
        with patch.object(transfer, "_build_r2_client", return_value=fake_client), \
             patch.object(transfer, "_head_or_none", return_value={"ContentLength": 123}), \
             patch.object(transfer, "_download") as download, \
             patch.object(transfer, "_upload_with_s5cmd") as upload:
            with self.assertRaisesRegex(transfer.TransferError, "refusing overwrite"):
                transfer.execute_request(self.normalized(), Path("unused"))
            download.assert_not_called()
            upload.assert_not_called()

    def test_hf_url_contains_no_token(self):
        url = transfer._hf_url(
            "Lightricks/LTX-2.5",
            "main",
            "vae/example.safetensors",
        )
        self.assertEqual(
            url,
            "https://huggingface.co/Lightricks/LTX-2.5/resolve/main/vae/example.safetensors",
        )
        self.assertNotIn("token", url.lower())


    def test_download_error_does_not_expose_redirect_url_or_token(self):
        class FakeRequestException(Exception):
            pass

        request_exception = FakeRequestException(
            "failure at https://cdn-lfs.example.invalid/file?X-Amz-Signature=SECRET"
        )

        artifact = self.normalized()["artifacts"][0]
        destination = Path("unused.part")

        class FakeRequests:
            RequestException = FakeRequestException

            @staticmethod
            def get(*args, **kwargs):
                raise request_exception

        import sys
        with patch.dict(sys.modules, {"requests": FakeRequests}):
            with self.assertRaises(transfer.TransferError) as caught:
                transfer._download(artifact, destination, "super-secret-token")

        message = str(caught.exception)
        self.assertIn("FakeRequestException", message)
        self.assertNotIn("cdn-lfs", message)
        self.assertNotIn("X-Amz-Signature", message)
        self.assertNotIn("SECRET", message)
        self.assertNotIn("super-secret-token", message)
if __name__ == "__main__":
    unittest.main()