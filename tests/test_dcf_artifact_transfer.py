import os
import unittest
import pytest
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
def test_model_artifact_keeps_legacy_default():
    request = ArtifactTransferValidationTests().request()
    result = transfer.validate_request(request)

    artifact = result["artifacts"][0]

    assert artifact["artifact_type"] == "MODEL_ARTIFACT"
    assert artifact["r2_key"].startswith("models/")
    assert artifact["repository_path"] == "vae/example.safetensors"


def test_accepts_custom_node_package_with_exact_commit():
    request = {
        "schema_version": "1.0",
        "artifacts": [{
            "artifact_type": "CUSTOM_NODE_PACKAGE",
            "repository": "example-owner/example-node",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "r2_key": (
                "custom-nodes/example-owner/example-node/"
                "0123456789abcdef0123456789abcdef01234567/source.zip"
            ),
            "gated": False,
        }],
    }

    result = transfer.validate_request(request)
    artifact = result["artifacts"][0]

    assert artifact["artifact_type"] == "CUSTOM_NODE_PACKAGE"
    assert artifact["repository"] == "example-owner/example-node"
    assert artifact["revision"] == "0123456789abcdef0123456789abcdef01234567"
    assert "repository_path" not in artifact


def test_custom_node_package_requires_exact_commit():
    request = {
        "schema_version": "1.0",
        "artifacts": [{
            "artifact_type": "CUSTOM_NODE_PACKAGE",
            "repository": "example-owner/example-node",
            "revision": "main",
            "r2_key": "custom-nodes/example-owner/example-node/main/source.zip",
        }],
    }

    with pytest.raises(transfer.TransferError, match="exact 40-char Git commit"):
        transfer.validate_request(request)


def test_custom_node_package_requires_custom_nodes_namespace():
    request = {
        "schema_version": "1.0",
        "artifacts": [{
            "artifact_type": "CUSTOM_NODE_PACKAGE",
            "repository": "example-owner/example-node",
            "revision": "0123456789abcdef0123456789abcdef01234567",
            "r2_key": "models/not-allowed.zip",
        }],
    }

    with pytest.raises(transfer.TransferError, match="custom-nodes/"):
        transfer.validate_request(request)

def test_github_archive_url_uses_exact_commit():
    commit = "0123456789abcdef0123456789abcdef01234567"

    url = transfer._github_archive_url(
        "example-owner/example-node",
        commit,
    )

    assert url == (
        "https://codeload.github.com/"
        "example-owner/example-node/zip/"
        + commit
    )


def test_custom_node_download_dispatches_to_github():
    artifact = {
        "artifact_type": "CUSTOM_NODE_PACKAGE",
        "repository": "example-owner/example-node",
        "revision": "0123456789abcdef0123456789abcdef01234567",
        "r2_key": (
            "custom-nodes/example-owner/example-node/"
            "0123456789abcdef0123456789abcdef01234567/source.zip"
        ),
        "gated": False,
    }

    with patch.object(
        transfer,
        "_download_github_package",
        return_value=(123, "a" * 64),
    ) as github_download, patch.object(
        transfer,
        "_download_huggingface",
    ) as hf_download:

        result = transfer._download(
            artifact,
            Path("unused.part"),
            None,
        )

    assert result == (123, "a" * 64)
    github_download.assert_called_once()
    hf_download.assert_not_called()


def test_model_download_still_dispatches_to_huggingface():
    request = ArtifactTransferValidationTests().request()
    artifact = transfer.validate_request(request)["artifacts"][0]

    with patch.object(
        transfer,
        "_download_huggingface",
        return_value=(123, "b" * 64),
    ) as hf_download, patch.object(
        transfer,
        "_download_github_package",
    ) as github_download:

        result = transfer._download(
            artifact,
            Path("unused.part"),
            "token",
        )

    assert result == (123, "b" * 64)
    hf_download.assert_called_once()
    github_download.assert_not_called()

def test_inspects_custom_node_dependency_manifests(tmp_path):
    archive_path = tmp_path / "node.zip"

    import zipfile

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "example-node-abc123/requirements.txt",
            "requests\n",
        )
        archive.writestr(
            "example-node-abc123/requirements/extra.txt",
            "numpy\n",
        )
        archive.writestr(
            "example-node-abc123/pyproject.toml",
            "[build-system]\n",
        )
        archive.writestr(
            "example-node-abc123/nodes.py",
            "print('not executed')\n",
        )

    result = transfer._inspect_custom_node_requirements(
        archive_path
    )

    assert result == [
        "pyproject.toml",
        "requirements.txt",
        "requirements/extra.txt",
    ]


def test_custom_node_requirement_inspection_does_not_execute_code(tmp_path):
    archive_path = tmp_path / "node.zip"

    import zipfile

    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr(
            "node/setup.py",
            "raise RuntimeError('MUST NOT RUN')\n",
        )

    result = transfer._inspect_custom_node_requirements(
        archive_path
    )

    assert result == ["setup.py"]


def test_invalid_custom_node_zip_is_rejected(tmp_path):
    archive_path = tmp_path / "invalid.zip"
    archive_path.write_bytes(b"not-a-zip")

    with pytest.raises(
        transfer.TransferError,
        match="not a valid ZIP archive",
    ):
        transfer._inspect_custom_node_requirements(
            archive_path
        )
