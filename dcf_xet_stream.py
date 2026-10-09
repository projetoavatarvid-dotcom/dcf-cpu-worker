"""Native hf_xet ranged download -> R2 multipart with bounded memory."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor


class XetStreamError(RuntimeError):
    pass


def transfer_xet_stream_to_r2(
    artifact,
    client,
    bucket,
    hf_token,
    *,
    part_size=32 * 1024 * 1024,
    upload_workers=2,
    progress_callback=None,
):
    from hf_xet import XetFileInfo
    from huggingface_hub.file_download import (
        get_hf_file_metadata,
        hf_hub_url,
    )
    from huggingface_hub.utils._headers import build_hf_headers
    from huggingface_hub.utils._xet import (
        get_xet_session,
        refresh_xet_connection_info,
        xet_headers_without_auth,
    )

    expected = artifact["expected_size_bytes"]

    url = hf_hub_url(
        artifact["repository"],
        artifact["repository_path"],
        revision=artifact["revision"],
    )

    headers = build_hf_headers(token=hf_token)

    metadata = get_hf_file_metadata(
        url,
        token=hf_token,
        headers=headers,
        timeout=30,
        retry_on_errors=True,
    )

    if metadata.size != expected:
        raise XetStreamError(
            f"Xet metadata size mismatch: "
            f"expected={expected} actual={metadata.size}"
        )

    if metadata.xet_file_data is None:
        raise XetStreamError(
            "artifact is not backed by Xet storage"
        )

    xet_data = metadata.xet_file_data

    connection = refresh_xet_connection_info(
        file_data=xet_data,
        headers=headers,
    )

    session = get_xet_session()

    group = session.new_download_stream_group(
        endpoint=connection.endpoint,
        token=connection.access_token,
        token_expiry_unix_secs=connection.expiration_unix_epoch,
        token_refresh_url=xet_data.refresh_route,
        token_refresh_headers=headers,
        custom_headers=xet_headers_without_auth(headers),
    )

    file_info = XetFileInfo(
        xet_data.file_hash,
        expected,
    )

    key = artifact["r2_key"]

    upload_id = client.create_multipart_upload(
        Bucket=bucket,
        Key=key,
    )["UploadId"]

    def transfer_part(part_number, start, end):
        stream = group.download_stream(
            file_info,
            start=start,
            end=end,
        )

        body = bytearray()
        wanted = end - start

        try:
            for chunk in stream:
                if not chunk:
                    continue

                body.extend(chunk)

                if len(body) > wanted:
                    raise XetStreamError(
                        f"Xet range exceeded requested size: "
                        f"part={part_number}"
                    )

        except BaseException:
            try:
                stream.cancel()
            except Exception:
                pass
            raise

        if len(body) != wanted:
            raise XetStreamError(
                f"Xet range size mismatch: "
                f"part={part_number} "
                f"expected={wanted} actual={len(body)}"
            )

        payload = bytes(body)

        result = client.upload_part(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            PartNumber=part_number,
            Body=payload,
        )

        return {
            "PartNumber": part_number,
            "ETag": result["ETag"],
            "start": start,
            "size": len(payload),
            "digest_payload": payload,
        }

    def ranges():
        part_number = 1
        start = 0

        while start < expected:
            end = min(
                start + part_size,
                expected,
            )

            yield (
                part_number,
                start,
                end,
            )

            part_number += 1
            start = end

    parts = []
    sha = hashlib.sha256()
    total = 0
    range_iterator = iter(ranges())

    try:
        with ThreadPoolExecutor(
            max_workers=upload_workers
        ) as pool:

            while True:
                batch = []

                for _ in range(upload_workers):
                    try:
                        item = next(range_iterator)
                    except StopIteration:
                        break

                    batch.append(item)

                if not batch:
                    break

                futures = [
                    pool.submit(
                        transfer_part,
                        part_number,
                        range_start,
                        range_end,
                    )
                    for (
                        part_number,
                        range_start,
                        range_end,
                    ) in batch
                ]

                results = [
                    future.result()
                    for future in futures
                ]

                results.sort(
                    key=lambda item:
                        item["PartNumber"]
                )

                for result in results:
                    parts.append(
                        {
                            "PartNumber":
                                result["PartNumber"],
                            "ETag":
                                result["ETag"],
                        }
                    )

                    payload = result[
                        "digest_payload"
                    ]

                    sha.update(payload)
                    total += len(payload)

                    if progress_callback is not None:
                        progress_callback(
                            total,
                            expected,
                        )

                    # Drop the payload reference as soon
                    # as this part has been hashed.
                    result["digest_payload"] = None

                del results
                del futures

        if total != expected:
            raise XetStreamError(
                f"Xet transfer size mismatch: "
                f"expected={expected} actual={total}"
            )

        digest = sha.hexdigest()

        expected_sha = artifact.get(
            "expected_sha256"
        )

        if (
            expected_sha
            and digest.lower()
            != expected_sha.lower()
        ):
            raise XetStreamError(
                "Xet stream sha256 mismatch"
            )

        client.complete_multipart_upload(
            Bucket=bucket,
            Key=key,
            UploadId=upload_id,
            MultipartUpload={
                "Parts": sorted(
                    parts,
                    key=lambda part:
                        part["PartNumber"],
                )
            },
        )

        return total, digest

    except BaseException:
        try:
            client.abort_multipart_upload(
                Bucket=bucket,
                Key=key,
                UploadId=upload_id,
            )
        except Exception:
            pass

        raise
