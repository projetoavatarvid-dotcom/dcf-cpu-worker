"""Native hf_xet ordered stream -> R2 multipart, with bounded memory."""
from __future__ import annotations
import hashlib
from concurrent.futures import ThreadPoolExecutor


class XetStreamError(RuntimeError):
    pass


def transfer_xet_stream_to_r2(artifact, client, bucket, hf_token, *, part_size=64*1024*1024, upload_workers=4):
    from hf_xet import XetFileInfo
    from huggingface_hub.file_download import get_hf_file_metadata, hf_hub_url
    from huggingface_hub.utils._headers import build_hf_headers
    from huggingface_hub.utils._xet import get_xet_session, refresh_xet_connection_info, xet_headers_without_auth

    expected = artifact["expected_size_bytes"]
    url = hf_hub_url(artifact["repository"], artifact["repository_path"], revision=artifact["revision"])
    headers = build_hf_headers(token=hf_token)
    metadata = get_hf_file_metadata(url, token=hf_token, headers=headers, timeout=30, retry_on_errors=True)
    if metadata.size != expected:
        raise XetStreamError(f"Xet metadata size mismatch: expected={expected} actual={metadata.size}")
    if metadata.xet_file_data is None:
        raise XetStreamError("artifact is not backed by Xet storage")
    xet_data = metadata.xet_file_data
    connection = refresh_xet_connection_info(file_data=xet_data, headers=headers)
    session = get_xet_session()
    group = session.new_download_stream_group(
        endpoint=connection.endpoint,
        token=connection.access_token,
        token_expiry_unix_secs=connection.expiration_unix_epoch,
        token_refresh_url=xet_data.refresh_route,
        token_refresh_headers=headers,
        custom_headers=xet_headers_without_auth(headers),
    )
    stream = group.download_stream(XetFileInfo(xet_data.file_hash, expected))
    key = artifact["r2_key"]
    upload_id = client.create_multipart_upload(Bucket=bucket, Key=key)["UploadId"]
    sha = hashlib.sha256()
    total = 0
    part_number = 1
    buffer = bytearray()
    parts = []
    pending = []

    def upload(number, body):
        result = client.upload_part(Bucket=bucket, Key=key, UploadId=upload_id, PartNumber=number, Body=body)
        return {"PartNumber": number, "ETag": result["ETag"]}

    try:
        with ThreadPoolExecutor(max_workers=upload_workers) as pool:
            for chunk in stream:
                if not chunk:
                    continue
                total += len(chunk)
                if total > expected:
                    raise XetStreamError("Xet stream exceeded expected size")
                sha.update(chunk)
                buffer.extend(chunk)
                while len(buffer) >= part_size:
                    body = bytes(buffer[:part_size])
                    del buffer[:part_size]
                    pending.append(pool.submit(upload, part_number, body))
                    part_number += 1
                    if len(pending) >= upload_workers:
                        parts.append(pending.pop(0).result())
            if buffer:
                pending.append(pool.submit(upload, part_number, bytes(buffer)))
            parts.extend(f.result() for f in pending)
        if total != expected:
            raise XetStreamError(f"Xet stream size mismatch: expected={expected} actual={total}")
        digest = sha.hexdigest()
        expected_sha = artifact.get("expected_sha256")
        if expected_sha and digest.lower() != expected_sha.lower():
            raise XetStreamError("Xet stream sha256 mismatch")
        client.complete_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id,
                                         MultipartUpload={"Parts": sorted(parts, key=lambda p:p["PartNumber"])})
        return total, digest
    except BaseException:
        try:
            stream.cancel()
        except Exception:
            pass
        try:
            client.abort_multipart_upload(Bucket=bucket, Key=key, UploadId=upload_id)
        except Exception:
            pass
        raise
