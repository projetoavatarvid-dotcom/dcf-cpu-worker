#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any
from urllib.parse import quote

DEFAULT_STAGING_ROOT = Path("/tmp/dcf-artifact-transfer")
DEFAULT_OBJECT_WORKERS = 32
DEFAULT_PART_CONCURRENCY = 80


class TransferError(RuntimeError):
    pass


def _require_text(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise TransferError(f"{name} is required")
    return value.strip()


def _require_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise TransferError(f"{name} must be a positive integer")
    return value


def validate_request(payload: dict[str, Any]) -> dict[str, Any]:
    if payload.get("schema_version") != "1.0":
        raise TransferError("schema_version must be '1.0'")

    artifacts = payload.get("artifacts")
    if not isinstance(artifacts, list) or not artifacts:
        raise TransferError("artifacts must be a non-empty list")

    normalized = []
    seen_r2_keys = set()

    for index, raw in enumerate(artifacts):
        if not isinstance(raw, dict):
            raise TransferError(f"artifacts[{index}] must be an object")

        artifact_type = raw.get("artifact_type", "MODEL_ARTIFACT")
        if artifact_type not in {"MODEL_ARTIFACT", "CUSTOM_NODE_PACKAGE"}:
            raise TransferError(
                f"artifacts[{index}].artifact_type must be "
                "MODEL_ARTIFACT or CUSTOM_NODE_PACKAGE"
            )

        repository = _require_text(raw.get("repository"), f"artifacts[{index}].repository")
        revision = _require_text(raw.get("revision", "main"), f"artifacts[{index}].revision")
        r2_key = _require_text(raw.get("r2_key"), f"artifacts[{index}].r2_key")

        if artifact_type == "MODEL_ARTIFACT":
            repository_path = _require_text(
                raw.get("repository_path"),
                f"artifacts[{index}].repository_path",
            )
            expected_size_bytes = _require_positive_int(
                raw.get("expected_size_bytes"),
                f"artifacts[{index}].expected_size_bytes",
            )
            if not r2_key.startswith("models/"):
                raise TransferError(
                    f"artifacts[{index}].r2_key must start with 'models/'"
                )
        else:
            repository_path = raw.get("repository_path")
            expected_size_bytes = raw.get("expected_size_bytes")

            if not re.fullmatch(r"[0-9a-fA-F]{40}", revision):
                raise TransferError(
                    f"artifacts[{index}].revision must be an exact 40-char "
                    "Git commit for CUSTOM_NODE_PACKAGE"
                )

            if not r2_key.startswith("custom-nodes/"):
                raise TransferError(
                    f"artifacts[{index}].r2_key must start with 'custom-nodes/' "
                    "for CUSTOM_NODE_PACKAGE"
                )

            if repository_path not in (None, ""):
                raise TransferError(
                    f"artifacts[{index}].repository_path must be omitted "
                    "for CUSTOM_NODE_PACKAGE"
                )

            if expected_size_bytes is not None:
                expected_size_bytes = _require_positive_int(
                    expected_size_bytes,
                    f"artifacts[{index}].expected_size_bytes",
                )
        if r2_key in seen_r2_keys:
            raise TransferError(f"duplicate r2_key: {r2_key}")
        seen_r2_keys.add(r2_key)

        expected_sha256 = raw.get("expected_sha256")
        if expected_sha256 is not None:
            expected_sha256 = _require_text(expected_sha256, f"artifacts[{index}].expected_sha256").lower()
            if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
                raise TransferError(f"artifacts[{index}].expected_sha256 must be 64 hex chars")

        normalized_artifact = {
            "artifact_type": artifact_type,
            "repository": repository,
            "revision": revision,
            "expected_sha256": expected_sha256,
            "r2_key": r2_key,
            "gated": bool(raw.get("gated", False)),
        }

        if repository_path not in (None, ""):
            normalized_artifact["repository_path"] = repository_path
        if expected_size_bytes is not None:
            normalized_artifact["expected_size_bytes"] = expected_size_bytes

        normalized.append(normalized_artifact)

    result_channel = payload.get("result_channel")
    normalized_result_channel = None

    if result_channel is not None:
        if not isinstance(result_channel, dict):
            raise TransferError("result_channel must be an object")

        transfer_id = _require_text(
            result_channel.get("transfer_id"),
            "result_channel.transfer_id",
        ).lower()

        if not re.fullmatch(r"[0-9a-f]{64}", transfer_id):
            raise TransferError(
                "result_channel.transfer_id must be 64 lowercase hex chars"
            )

        result_key = _require_text(
            result_channel.get("result_key"),
            "result_channel.result_key",
        )

        expected_result_key = (
            "workflow-preparation/transfer-results/"
            f"{transfer_id}/result.json"
        )

        if result_key != expected_result_key:
            raise TransferError(
                "result_channel.result_key does not match transfer_id"
            )

        content_type = _require_text(
            result_channel.get(
                "content_type",
                "application/json",
            ),
            "result_channel.content_type",
        )

        if content_type != "application/json":
            raise TransferError(
                "result_channel.content_type must be application/json"
            )

        normalized_result_channel = {
            "transfer_id": transfer_id,
            "result_key": result_key,
            "content_type": content_type,
        }

    transfer = payload.get("transfer") or {}
    if not isinstance(transfer, dict):
        raise TransferError("transfer must be an object")

    object_workers = transfer.get("object_workers", DEFAULT_OBJECT_WORKERS)
    part_concurrency = transfer.get("part_concurrency", DEFAULT_PART_CONCURRENCY)
    _require_positive_int(object_workers, "transfer.object_workers")
    _require_positive_int(part_concurrency, "transfer.part_concurrency")

    return {
        "schema_version": "1.0",
        "mode": "SEQUENTIAL",
        "artifacts": normalized,
        "transfer": {
            "object_workers": object_workers,
            "part_concurrency": part_concurrency,
        },
        "result_channel": normalized_result_channel,
    }


def _hf_url(repository: str, revision: str, repository_path: str) -> str:
    return (
        f"https://huggingface.co/{repository}/resolve/"
        f"{quote(revision, safe='')}/{quote(repository_path, safe='/')}"
    )


def _github_archive_url(repository: str, revision: str) -> str:
    return (
        f"https://codeload.github.com/{repository}/zip/{revision}"
    )


def _download_huggingface(
    artifact: dict[str, Any],
    destination: Path,
    hf_token: str | None,
) -> tuple[int, str]:
    import requests

    headers = {}
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    sha256 = hashlib.sha256()
    size = 0

    try:
        with requests.get(
            _hf_url(
                artifact["repository"],
                artifact["revision"],
                artifact["repository_path"],
            ),
            headers=headers,
            stream=True,
            timeout=(30, 300),
        ) as response:
            if response.status_code >= 400:
                raise TransferError(
                    f"Hugging Face download failed for "
                    f"{artifact['repository_path']}: "
                    f"HTTP {response.status_code}"
                )

            with destination.open("wb") as handle:
                for chunk in response.iter_content(
                    chunk_size=8 * 1024 * 1024
                ):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    size += len(chunk)
                    sha256.update(chunk)

    except TransferError:
        raise
    except requests.RequestException as exc:
        raise TransferError(
            f"Hugging Face download failed for "
            f"{artifact['repository_path']}: "
            f"{type(exc).__name__}"
        ) from None

    if size != artifact["expected_size_bytes"]:
        raise TransferError(
            f"download size mismatch: "
            f"expected={artifact['expected_size_bytes']} actual={size}"
        )

    digest = sha256.hexdigest()
    expected = artifact.get("expected_sha256")

    if expected and digest.lower() != expected.lower():
        raise TransferError(
            f"sha256 mismatch: expected={expected} actual={digest}"
        )

    return size, digest


def _download_github_package(
    artifact: dict[str, Any],
    destination: Path,
) -> tuple[int, str]:
    import requests

    sha256 = hashlib.sha256()
    size = 0

    try:
        with requests.get(
            _github_archive_url(
                artifact["repository"],
                artifact["revision"],
            ),
            stream=True,
            timeout=(30, 300),
        ) as response:
            if response.status_code >= 400:
                raise TransferError(
                    "GitHub custom-node package download failed: "
                    f"HTTP {response.status_code}"
                )

            with destination.open("wb") as handle:
                for chunk in response.iter_content(
                    chunk_size=8 * 1024 * 1024
                ):
                    if not chunk:
                        continue
                    handle.write(chunk)
                    size += len(chunk)
                    sha256.update(chunk)

    except TransferError:
        raise
    except requests.RequestException as exc:
        raise TransferError(
            "GitHub custom-node package download failed: "
            f"{type(exc).__name__}"
        ) from None

    if size <= 0:
        raise TransferError(
            "GitHub custom-node package download produced an empty artifact"
        )

    expected_size = artifact.get("expected_size_bytes")
    if expected_size is not None and size != expected_size:
        raise TransferError(
            f"download size mismatch: "
            f"expected={expected_size} actual={size}"
        )

    digest = sha256.hexdigest()
    expected = artifact.get("expected_sha256")

    if expected and digest.lower() != expected.lower():
        raise TransferError(
            f"sha256 mismatch: expected={expected} actual={digest}"
        )

    return size, digest


def _download(
    artifact: dict[str, Any],
    destination: Path,
    hf_token: str | None,
) -> tuple[int, str]:
    if artifact.get("artifact_type") == "CUSTOM_NODE_PACKAGE":
        return _download_github_package(
            artifact,
            destination,
        )

    return _download_huggingface(
        artifact,
        destination,
        hf_token,
    )

def _inspect_custom_node_requirements(
    archive_path: Path,
) -> list[str]:
    candidates = []

    try:
        with zipfile.ZipFile(archive_path, "r") as archive:
            for name in archive.namelist():
                normalized = name.replace("\\", "/")
                relative = normalized.split("/", 1)[-1] if "/" in normalized else normalized
                lower = relative.lower()

                is_requirement = (
                    lower == "requirements.txt"
                    or (
                        lower.startswith("requirements/")
                        and lower.endswith(".txt")
                    )
                    or lower == "pyproject.toml"
                    or lower == "setup.py"
                    or lower == "setup.cfg"
                )

                if is_requirement:
                    candidates.append(relative)

    except zipfile.BadZipFile:
        raise TransferError(
            "CUSTOM_NODE_PACKAGE is not a valid ZIP archive"
        ) from None

    return sorted(set(candidates))


def _build_r2_client(endpoint: str, access_key: str, secret_key: str):
    import boto3
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
        region_name="auto",
    )


def _head_or_none(client: Any, bucket: str, key: str):
    try:
        return client.head_object(Bucket=bucket, Key=key)
    except Exception as exc:
        response = getattr(exc, "response", None)
        code = None
        if isinstance(response, dict):
            code = (response.get("Error") or {}).get("Code")
        if code in {"404", "NoSuchKey", "NotFound"}:
            return None
        raise


def _upload_with_s5cmd(
    local_path: Path,
    bucket: str,
    r2_key: str,
    endpoint: str,
    access_key: str,
    secret_key: str,
    object_workers: int,
    part_concurrency: int,
) -> None:
    env = os.environ.copy()
    env["AWS_ACCESS_KEY_ID"] = access_key
    env["AWS_SECRET_ACCESS_KEY"] = secret_key
    env["AWS_REGION"] = "auto"
    env["AWS_DEFAULT_REGION"] = "auto"

    subprocess.run(
        [
            "s5cmd",
            "--endpoint-url", endpoint,
            "--numworkers", str(object_workers),
            "cp",
            "--concurrency", str(part_concurrency),
            str(local_path),
            f"s3://{bucket}/{r2_key}",
        ],
        check=True,
        env=env,
        stdout=sys.stderr,
        stderr=sys.stderr,
    )


def _publish_result_json(
    client: Any,
    bucket: str,
    result_channel: dict[str, Any],
    result: dict[str, Any],
) -> None:
    result_key = result_channel["result_key"]

    if _head_or_none(client, bucket, result_key) is not None:
        raise TransferError(
            "result destination already exists; refusing overwrite: "
            f"{result_key}"
        )

    body = json.dumps(
        result,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    ).encode("utf-8")

    try:
        client.put_object(
            Bucket=bucket,
            Key=result_key,
            Body=body,
            ContentType=result_channel["content_type"],
        )
    except Exception as exc:
        raise TransferError(
            f"result publication failed: {type(exc).__name__}"
        ) from None

    remote = _head_or_none(
        client,
        bucket,
        result_key,
    )

    if remote is None:
        raise TransferError(
            f"result publication verification failed: {result_key}"
        )

    remote_size = int(
        remote.get("ContentLength", -1)
    )

    if remote_size != len(body):
        raise TransferError(
            "result publication size mismatch: "
            f"local={len(body)} remote={remote_size}"
        )


def execute_request(request: dict[str, Any], staging_root: Path = DEFAULT_STAGING_ROOT) -> dict[str, Any]:
    endpoint = _require_text(os.getenv("DCF_R2_ENDPOINT"), "DCF_R2_ENDPOINT")
    bucket = _require_text(os.getenv("DCF_R2_BUCKET"), "DCF_R2_BUCKET")
    access_key = _require_text(os.getenv("DCF_R2_ACCESS_KEY"), "DCF_R2_ACCESS_KEY")
    secret_key = _require_text(os.getenv("DCF_R2_SECRET_KEY"), "DCF_R2_SECRET_KEY")
    hf_token = os.getenv("HF_TOKEN")

    if any(a["gated"] for a in request["artifacts"]) and not hf_token:
        raise TransferError("HF_TOKEN is required because at least one artifact is gated")

    client = _build_r2_client(endpoint, access_key, secret_key)
    staging_root.mkdir(parents=True, exist_ok=True)
    evidence = []

    for artifact in request["artifacts"]:
        if _head_or_none(client, bucket, artifact["r2_key"]) is not None:
            raise TransferError(f"destination already exists; refusing overwrite: {artifact['r2_key']}")

        if artifact["artifact_type"] == "CUSTOM_NODE_PACKAGE":
            repository_name = artifact["repository"].rsplit("/", 1)[-1]
            safe_name = re.sub(
                r"[^A-Za-z0-9._-]+",
                "_",
                f"{repository_name}-{artifact['revision']}.zip",
            )
        else:
            safe_name = re.sub(
                r"[^A-Za-z0-9._-]+",
                "_",
                Path(artifact["repository_path"]).name,
            )

        local_path = staging_root / f"{safe_name}.part"
        if local_path.exists():
            local_path.unlink()

        verified = False
        try:
            size, digest = _download(artifact, local_path, hf_token)

            dependency_manifests = []
            if artifact["artifact_type"] == "CUSTOM_NODE_PACKAGE":
                dependency_manifests = _inspect_custom_node_requirements(
                    local_path
                )

            _upload_with_s5cmd(
                local_path,
                bucket,
                artifact["r2_key"],
                endpoint,
                access_key,
                secret_key,
                request["transfer"]["object_workers"],
                request["transfer"]["part_concurrency"],
            )

            remote = _head_or_none(client, bucket, artifact["r2_key"])
            if remote is None:
                raise TransferError(f"R2 verification failed: {artifact['r2_key']}")

            remote_size = int(remote.get("ContentLength", -1))
            if remote_size != size:
                raise TransferError(
                    f"R2 size mismatch: local={size} remote={remote_size}"
                )

            verified = True
            evidence.append(
                {
                    "status": "TRANSFERRED_AND_VERIFIED",
                    "artifact_type": artifact["artifact_type"],
                    "repository": artifact["repository"],
                    "revision": artifact["revision"],
                    "repository_path": artifact.get("repository_path"),
                    "r2_key": artifact["r2_key"],
                    "size_bytes": size,
                    "sha256": digest,
                    "sha256_source": (
                        "expected_and_calculated"
                        if artifact.get("expected_sha256")
                        else "calculated_on_cpu_staging"
                    ),
                    "r2_size_bytes": remote_size,
                    "r2_etag_informational": str(remote.get("ETag", "")).strip('"') or None,
                    "etag_is_sha256": False,
                    "dependency_manifests": dependency_manifests,
                    "requirements_detected": bool(dependency_manifests),
                    "local_staging_removed": True,
                }
            )
        finally:
            if verified and local_path.exists():
                local_path.unlink()

    result = {
        "schema_version": "1.0",
        "status": "COMPLETED",
        "mode": "SEQUENTIAL",
        "gpu_required": False,
        "network_volume_required": False,
        "catalog_write_performed": False,
        "artifacts": evidence,
    }

    result_channel = request.get("result_channel")

    if result_channel is not None:
        _publish_result_json(
            client,
            bucket,
            result_channel,
            result,
        )

    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--staging-root", default=str(DEFAULT_STAGING_ROOT))
    args = parser.parse_args()

    payload = json.loads(Path(args.request).read_text(encoding="utf-8"))
    request = validate_request(payload)

    if not args.execute:
        print(json.dumps({
            "schema_version": "1.0",
            "status": "VALID",
            "execution_performed": False,
            "gpu_required": False,
            "network_volume_required": False,
            "artifact_count": len(request["artifacts"]),
            "request": request,
        }, indent=2, sort_keys=True))
        return 0

    print(json.dumps(execute_request(request, Path(args.staging_root)), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except TransferError as exc:
        print(json.dumps({"schema_version": "1.0", "status": "ERROR", "error": str(exc)}), file=sys.stderr)
        raise SystemExit(2)