#!/usr/bin/env python3
# Copyright 2025 Individual Contributor: Fengyuan Miao
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Download STARK images referenced by an OPD-MM record store.

The public STARK image corpus has two different storage layouts. URL-backed
images are resolved through the Hugging Face Dataset Viewer, while generated
and retrieved images live inside large WebDataset tar shards. The latter are
streamed one shard at a time and only requested members are retained locally.
"""

from __future__ import annotations

import argparse
import io
import json
import os
import socket
import tarfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import requests
from PIL import Image

DATASET_SERVER_FILTER = "https://datasets-server.huggingface.co/filter"
IMAGE_REPOSITORY = "passing2961/stark-image"
URL_REPOSITORY = "passing2961/stark-image-url"
IMAGE_EXTENSIONS = {".bmp", ".gif", ".jpeg", ".jpg", ".png", ".webp"}
FORMAT_EXTENSIONS = {
    "BMP": ".bmp",
    "GIF": ".gif",
    "JPEG": ".jpg",
    "PNG": ".png",
    "WEBP": ".webp",
}
USER_AGENT = "Mozilla/5.0 (compatible; STARK dataset downloader/1.0)"


def configure_direct_dns(nameserver: str) -> None:
    """Resolve archive hosts without relying on a broken local proxy or system DNS."""

    import dns.resolver

    resolver = dns.resolver.Resolver(configure=False)
    resolver.nameservers = [nameserver]
    resolver.lifetime = 15
    original_getaddrinfo = socket.getaddrinfo
    cache: dict[str, list[str]] = {}
    lock = threading.Lock()

    def direct_getaddrinfo(
        host: str,
        port: int | str,
        family: int = 0,
        type: int = 0,
        proto: int = 0,
        flags: int = 0,
    ) -> list[tuple[Any, ...]]:
        if host == "huggingface.co" or host.endswith(".hf.co"):
            with lock:
                addresses = cache.get(host)
                if addresses is None:
                    addresses = [str(answer) for answer in resolver.resolve(host, "A")]
                    cache[host] = addresses
            results: list[tuple[Any, ...]] = []
            for address in addresses:
                results.extend(
                    original_getaddrinfo(
                        address,
                        port,
                        socket.AF_INET,
                        type,
                        proto,
                        flags | socket.AI_NUMERICHOST,
                    )
                )
            return results
        return original_getaddrinfo(host, port, family, type, proto, flags)

    socket.getaddrinfo = direct_getaddrinfo
    for variable in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
        os.environ.pop(variable, None)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _local_image_keys(image_root: Path) -> set[str]:
    return {
        path.stem
        for path in image_root.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _missing_keys(records_path: Path, image_root: Path) -> tuple[set[str], set[str]]:
    local_keys = _local_image_keys(image_root)
    url_keys: set[str] = set()
    archive_keys: set[str] = set()
    for record in _read_jsonl(records_path):
        if record.get("modality") != "image":
            continue
        image_key = str((record.get("metadata") or {}).get("image_key") or "")
        lookup_key = image_key.removeprefix("url:").removeprefix("face:")
        if not lookup_key or lookup_key in local_keys:
            continue
        if image_key.startswith("url:"):
            url_keys.add(lookup_key)
        else:
            archive_keys.add(lookup_key)
    return url_keys, archive_keys


def _chunks(values: list[str], size: int) -> list[list[str]]:
    return [values[start : start + size] for start in range(0, len(values), size)]


def _lookup_url_batch(keys: list[str], timeout: float) -> dict[str, dict[str, Any]]:
    escaped_keys = [key.replace("'", "''") for key in keys]
    where = " OR ".join(f'"__key__"=\'{key}\'' for key in escaped_keys)
    response = requests.get(
        DATASET_SERVER_FILTER,
        params={
            "dataset": URL_REPOSITORY,
            "config": "default",
            "split": "train",
            "where": where,
            "length": len(keys),
        },
        timeout=timeout,
    )
    response.raise_for_status()
    return {
        str(item["row"]["__key__"]): item["row"]["json"]
        for item in response.json().get("rows", [])
    }


def lookup_image_urls(
    keys: set[str], *, workers: int, timeout: float
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    rows: dict[str, dict[str, Any]] = {}
    errors: dict[str, str] = {}
    batches = _chunks(sorted(keys), 20)
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(batches)))) as executor:
        futures = {executor.submit(_lookup_url_batch, batch, timeout): batch for batch in batches}
        for future in as_completed(futures):
            batch = futures[future]
            try:
                rows.update(future.result())
            except Exception as exc:  # Dataset server failures are recorded for retry.
                message = f"{type(exc).__name__}: {exc}"
                errors.update({key: message for key in batch})
    return rows, errors


def _validated_extension(payload: bytes) -> str:
    with Image.open(io.BytesIO(payload)) as image:
        image.verify()
        image_format = str(image.format or "").upper()
    if image_format not in FORMAT_EXTENSIONS:
        raise ValueError(f"unsupported image format: {image_format or 'unknown'}")
    return FORMAT_EXTENSIONS[image_format]


def _fetch_image(url: str, *, timeout: float, max_bytes: int) -> tuple[bytes, str]:
    candidates = [url]
    if url.startswith("http://"):
        candidates.insert(0, "https://" + url.removeprefix("http://"))
    errors: list[str] = []
    for candidate in candidates:
        try:
            with requests.get(
                candidate,
                headers={"User-Agent": USER_AGENT},
                stream=True,
                timeout=(15, timeout),
            ) as response:
                response.raise_for_status()
                payload = bytearray()
                for chunk in response.iter_content(256 * 1024):
                    payload.extend(chunk)
                    if len(payload) > max_bytes:
                        raise ValueError(f"image exceeds {max_bytes} bytes")
            data = bytes(payload)
            return data, _validated_extension(data)
        except Exception as exc:
            errors.append(f"{candidate}: {type(exc).__name__}: {exc}")
    raise RuntimeError("; ".join(errors))


def _download_url_image(
    key: str,
    metadata: dict[str, Any],
    *,
    image_root: Path,
    timeout: float,
    max_bytes: int,
) -> tuple[str, str]:
    image_url = str(metadata.get("image_url") or "")
    if not image_url.startswith(("http://", "https://")):
        raise ValueError(f"invalid source URL: {image_url!r}")
    payload, extension = _fetch_image(image_url, timeout=timeout, max_bytes=max_bytes)
    output_path = image_root / f"{key}{extension}"
    output_path.write_bytes(payload)
    metadata_path = image_root / f"{key}.json"
    metadata_path.write_text(
        json.dumps(
            {
                "index": key,
                "image_url": image_url,
                "image_source": metadata.get("image_source", ""),
            },
            ensure_ascii=True,
        ),
        encoding="utf-8",
    )
    return key, str(output_path)


def download_url_images(
    rows: dict[str, dict[str, Any]],
    *,
    image_root: Path,
    workers: int,
    timeout: float,
    max_bytes: int,
) -> tuple[dict[str, str], dict[str, str]]:
    downloaded: dict[str, str] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(rows)))) as executor:
        futures = {
            executor.submit(
                _download_url_image,
                key,
                metadata,
                image_root=image_root,
                timeout=timeout,
                max_bytes=max_bytes,
            ): key
            for key, metadata in rows.items()
        }
        for future in as_completed(futures):
            key = futures[future]
            try:
                _, path = future.result()
                downloaded[key] = path
            except Exception as exc:
                errors[key] = f"{type(exc).__name__}: {exc}"
    return downloaded, errors


def _load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"completed_archive_shards": [], "archive_errors": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_state(path: Path, state: dict[str, Any], lock: threading.Lock) -> None:
    with lock:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        temporary.replace(path)


class _SignedArchiveURL:
    def __init__(self, resolve_url: str, timeout: float) -> None:
        self.resolve_url = resolve_url
        self.timeout = timeout
        self._lock = threading.Lock()
        self.url = ""
        self.size = 0

    def refresh(self) -> tuple[str, int]:
        with self._lock:
            errors: list[str] = []
            for attempt in range(10):
                try:
                    response = requests.head(
                        self.resolve_url,
                        headers={"User-Agent": USER_AGENT},
                        allow_redirects=False,
                        timeout=(15, self.timeout),
                    )
                    try:
                        response.raise_for_status()
                        location = str(response.headers.get("Location") or "")
                        linked_size = str(response.headers.get("X-Linked-Size") or "")
                        if (
                            response.status_code not in {301, 302, 303, 307, 308}
                            or not location
                            or not linked_size
                        ):
                            raise RuntimeError(
                                f"archive probe failed: {response.status_code} "
                                f"location={bool(location)} size={linked_size!r}"
                            )
                        self.url = location
                        self.size = int(linked_size)
                        return self.url, self.size
                    finally:
                        response.close()
                except Exception as exc:
                    errors.append(f"{type(exc).__name__}: {exc}")
                    time.sleep(min(8.0, 0.5 * (attempt + 1)))
            raise RuntimeError(f"archive URL probe failed: {'; '.join(errors[-3:])}")

    def current(self) -> tuple[str, int]:
        with self._lock:
            if self.url and self.size:
                return self.url, self.size
        return self.refresh()


def _download_range(
    provider: _SignedArchiveURL,
    file_descriptor: int,
    start: int,
    end: int,
    *,
    timeout: float,
    attempts: int = 12,
) -> None:
    expected_size = end - start + 1
    errors: list[str] = []
    for attempt in range(attempts):
        url, _ = provider.current()
        try:
            response = requests.get(
                url,
                headers={"Range": f"bytes={start}-{end}", "User-Agent": USER_AGENT},
                timeout=(15, timeout),
            )
            try:
                response.raise_for_status()
                content_range = str(response.headers.get("Content-Range") or "")
                if response.status_code != 206 or not content_range.startswith(f"bytes {start}-{end}/"):
                    raise RuntimeError(f"unexpected range response: {response.status_code} {content_range!r}")
                if len(response.content) != expected_size:
                    raise RuntimeError(f"short range: {len(response.content)} != {expected_size}")
                os.pwrite(file_descriptor, response.content, start)
                return
            finally:
                response.close()
        except Exception as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
            if attempt in {3, 7}:
                provider.refresh()
            time.sleep(min(8.0, 0.5 * (attempt + 1)))
    raise RuntimeError(f"range {start}-{end} failed: {'; '.join(errors[-3:])}")


def _download_archive_file(
    shard_name: str,
    *,
    archive_cache: Path,
    timeout: float,
    range_workers: int,
    chunk_size: int,
    archive_endpoint: str,
) -> tuple[Path, Path]:
    archive_cache.mkdir(parents=True, exist_ok=True)
    archive_path = archive_cache / shard_name
    range_state_path = archive_cache / f"{shard_name}.ranges.json"
    resolve_url = (
        f"{archive_endpoint.rstrip('/')}/datasets/{IMAGE_REPOSITORY}/resolve/main/{shard_name}"
    )
    provider = _SignedArchiveURL(resolve_url, timeout)
    _, archive_size = provider.current()

    completed: set[int] = set()
    if range_state_path.exists():
        range_state = json.loads(range_state_path.read_text(encoding="utf-8"))
        if range_state.get("size") == archive_size and range_state.get("chunk_size") == chunk_size:
            completed = {int(index) for index in range_state.get("completed", [])}

    descriptor = os.open(archive_path, os.O_CREAT | os.O_RDWR, 0o644)
    os.ftruncate(descriptor, archive_size)
    state_lock = threading.Lock()
    ranges = [
        (index, start, min(archive_size - 1, start + chunk_size - 1))
        for index, start in enumerate(range(0, archive_size, chunk_size))
        if index not in completed
    ]
    try:
        with ThreadPoolExecutor(max_workers=min(range_workers, max(1, len(ranges)))) as executor:
            futures = {
                executor.submit(
                    _download_range,
                    provider,
                    descriptor,
                    start,
                    end,
                    timeout=timeout,
                ): index
                for index, start, end in ranges
            }
            for future in as_completed(futures):
                index = futures[future]
                future.result()
                with state_lock:
                    completed.add(index)
                    temporary = range_state_path.with_suffix(range_state_path.suffix + ".tmp")
                    temporary.write_text(
                        json.dumps(
                            {
                                "size": archive_size,
                                "chunk_size": chunk_size,
                                "completed": sorted(completed),
                            }
                        ),
                        encoding="utf-8",
                    )
                    temporary.replace(range_state_path)
    finally:
        os.close(descriptor)
    return archive_path, range_state_path


def _scan_archive_shard(
    shard_index: int,
    *,
    requested_keys: set[str],
    image_root: Path,
    archive_cache: Path,
    timeout: float,
    range_workers: int,
    chunk_size: int,
    archive_endpoint: str,
) -> set[str]:
    shard_name = f"stark-train-{shard_index:06d}-of-000034.tar"
    archive_path, range_state_path = _download_archive_file(
        shard_name,
        archive_cache=archive_cache,
        timeout=timeout,
        range_workers=range_workers,
        chunk_size=chunk_size,
        archive_endpoint=archive_endpoint,
    )
    found: set[str] = set()
    try:
        with tarfile.open(archive_path, mode="r:") as archive:
            for member in archive:
                if not member.isfile():
                    continue
                member_path = Path(member.name)
                key = member_path.stem
                if key not in requested_keys:
                    continue
                suffix = member_path.suffix.lower()
                if suffix not in IMAGE_EXTENSIONS and suffix != ".json":
                    continue
                extracted = archive.extractfile(member)
                if extracted is None:
                    continue
                payload = extracted.read()
                if suffix in IMAGE_EXTENSIONS:
                    suffix = _validated_extension(payload)
                    found.add(key)
                (image_root / f"{key}{suffix}").write_bytes(payload)
    finally:
        archive_path.unlink(missing_ok=True)
        range_state_path.unlink(missing_ok=True)
    return found


def download_archive_images(
    requested_keys: set[str],
    *,
    image_root: Path,
    archive_cache: Path,
    state_path: Path,
    workers: int,
    range_workers: int,
    chunk_size: int,
    archive_endpoint: str,
    timeout: float,
    first_shard: int,
    last_shard: int,
) -> tuple[set[str], dict[str, str]]:
    state = _load_state(state_path)
    completed = {int(index) for index in state.get("completed_archive_shards", [])}
    archive_errors = dict(state.get("archive_errors", {}))
    existing = _local_image_keys(image_root)
    remaining = requested_keys - existing
    found = requested_keys & existing
    lock = threading.Lock()
    shards = [index for index in range(first_shard, last_shard + 1) if index not in completed]
    with ThreadPoolExecutor(max_workers=min(workers, max(1, len(shards)))) as executor:
        futures = {
            executor.submit(
                _scan_archive_shard,
                index,
                requested_keys=remaining,
                image_root=image_root,
                archive_cache=archive_cache,
                timeout=timeout,
                range_workers=range_workers,
                chunk_size=chunk_size,
                archive_endpoint=archive_endpoint,
            ): index
            for index in shards
        }
        for future in as_completed(futures):
            index = futures[future]
            try:
                found.update(future.result())
                completed.add(index)
                archive_errors.pop(str(index), None)
            except Exception as exc:
                archive_errors[str(index)] = f"{type(exc).__name__}: {exc}"
            state["completed_archive_shards"] = sorted(completed)
            state["archive_errors"] = archive_errors
            state["archive_found"] = len(found)
            state["archive_requested"] = len(requested_keys)
            _write_state(state_path, state, lock)
    return found, archive_errors


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        type=Path,
        default=Path("dataset/Stark/opd_mm_store_rounds_3000/records.jsonl"),
    )
    parser.add_argument("--image-root", type=Path, default=Path("dataset/Stark/images_sample"))
    parser.add_argument("--mode", choices=("all", "url", "archive"), default="all")
    parser.add_argument("--url-workers", type=int, default=16)
    parser.add_argument("--archive-workers", type=int, default=2)
    parser.add_argument("--archive-cache", type=Path, default=None)
    parser.add_argument("--range-workers", type=int, default=24)
    parser.add_argument("--range-chunk-mib", type=int, default=8)
    parser.add_argument("--archive-endpoint", default="https://huggingface.co")
    parser.add_argument(
        "--direct-dns-server",
        help="Bypass proxy variables and resolve Hugging Face archive hosts with this DNS server.",
    )
    parser.add_argument("--timeout", type=float, default=120.0)
    parser.add_argument("--max-image-bytes", type=int, default=50 * 1024 * 1024)
    parser.add_argument("--first-shard", type=int, default=0)
    parser.add_argument("--last-shard", type=int, default=33)
    parser.add_argument("--state", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.direct_dns_server:
        configure_direct_dns(args.direct_dns_server)
    args.image_root.mkdir(parents=True, exist_ok=True)
    state_path = args.state or args.image_root / "download_missing_manifest.json"
    url_keys, archive_keys = _missing_keys(args.records, args.image_root)
    summary: dict[str, Any] = {
        "records": str(args.records.resolve()),
        "url_requested": len(url_keys),
        "archive_requested": len(archive_keys),
    }

    if args.mode in {"all", "url"} and url_keys:
        rows, lookup_errors = lookup_image_urls(
            url_keys, workers=args.url_workers, timeout=args.timeout
        )
        downloaded, download_errors = download_url_images(
            rows,
            image_root=args.image_root,
            workers=args.url_workers,
            timeout=args.timeout,
            max_bytes=args.max_image_bytes,
        )
        summary.update(
            {
                "url_resolved": len(rows),
                "url_downloaded": len(downloaded),
                "url_lookup_errors": lookup_errors,
                "url_download_errors": download_errors,
            }
        )

    if args.mode in {"all", "archive"} and archive_keys:
        found, archive_errors = download_archive_images(
            archive_keys,
            image_root=args.image_root,
            archive_cache=args.archive_cache or args.image_root / ".stark_tar_cache",
            state_path=state_path,
            workers=args.archive_workers,
            range_workers=args.range_workers,
            chunk_size=args.range_chunk_mib * 1024 * 1024,
            archive_endpoint=args.archive_endpoint,
            timeout=args.timeout,
            first_shard=args.first_shard,
            last_shard=args.last_shard,
        )
        summary.update(
            {
                "archive_found": len(found),
                "archive_errors": archive_errors,
            }
        )

    previous_state = _load_state(state_path)
    previous_state.update(summary)
    state_path.write_text(json.dumps(previous_state, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({key: value for key, value in summary.items() if not isinstance(value, dict)}, indent=2))
    for key in ("url_lookup_errors", "url_download_errors", "archive_errors"):
        if summary.get(key):
            print(f"{key}: {len(summary[key])}")


if __name__ == "__main__":
    main()
