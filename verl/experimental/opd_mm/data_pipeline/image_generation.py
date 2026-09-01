"""GPT Image generation and normalization for MMem construction artifacts."""

from __future__ import annotations

import base64
import copy
import io
import json
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib import error, request

from PIL import Image
from openai import OpenAI


DEFAULT_IMAGE_BASE_URL = "https://xiaohondou.com/v1"
DEFAULT_IMAGE_MODEL = "gpt-image-2"
DEFAULT_IMAGE_REQUEST_SIZE = "1024x1024"
DEFAULT_IMAGE_OUTPUT_WIDTH = 512
DEFAULT_IMAGE_OUTPUT_HEIGHT = 512


@dataclass(frozen=True)
class GeneratedImage:
    image_id: str
    path: Path


def resize_image_bytes(
    image_bytes: bytes,
    *,
    width: int = DEFAULT_IMAGE_OUTPUT_WIDTH,
    height: int = DEFAULT_IMAGE_OUTPUT_HEIGHT,
) -> bytes:
    """Decode an image and return a normalized RGB PNG of the requested size."""

    if width < 1 or height < 1:
        raise ValueError("image output dimensions must be positive")
    with Image.open(io.BytesIO(image_bytes)) as source:
        source.load()
        image = source.convert("RGB")
        if image.size != (width, height):
            image = image.resize((width, height), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=True)
    return output.getvalue()


class GPTImageClient:
    """Client for the fixed OpenAI-compatible GPT Image endpoint."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_IMAGE_BASE_URL,
        api_key: str | None = None,
        timeout: float = 300.0,
        retries: int = 2,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get("MMEM_IMAGE_API_KEY", "")
        self.timeout = timeout
        self.retries = retries
        self._opener = request.build_opener()

    def generate(self, prompt: str) -> bytes:
        if not self.api_key:
            raise RuntimeError("MMEM_IMAGE_API_KEY is required for GPT Image generation")
        payload = {
            "model": DEFAULT_IMAGE_MODEL,
            "prompt": prompt,
            "n": 1,
            "size": DEFAULT_IMAGE_REQUEST_SIZE,
            "quality": "medium",
        }
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "User-Agent": "MMem-Data-Pipeline/1.0",
        }
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            req = request.Request(
                f"{self.base_url}/images/generations",
                data=body,
                headers=headers,
                method="POST",
            )
            try:
                with self._opener.open(req, timeout=self.timeout) as response:
                    value = json.loads(response.read().decode("utf-8"))
                data = value.get("data") if isinstance(value, dict) else None
                item = data[0] if isinstance(data, list) and data and isinstance(data[0], dict) else None
                if item is None:
                    raise RuntimeError(f"GPT Image response has no image item: {value}")
                if item.get("b64_json"):
                    return base64.b64decode(item["b64_json"], validate=True)
                if item.get("url"):
                    image_req = request.Request(str(item["url"]), headers={"User-Agent": headers["User-Agent"]})
                    with self._opener.open(image_req, timeout=self.timeout) as response:
                        return response.read()
                raise RuntimeError(f"GPT Image response has no image data: {value}")
            except error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:1000]
                last_error = RuntimeError(f"HTTP {exc.code}: {detail}")
            except (error.URLError, TimeoutError, ValueError, RuntimeError) as exc:
                last_error = exc
            if attempt < self.retries:
                time.sleep(2**attempt)
        assert last_error is not None
        raise RuntimeError(
            f"GPT Image generation failed after {self.retries + 1} attempts: {last_error}"
        ) from last_error

    def edit(self, prompt: str, *, reference_paths: list[str | Path]) -> bytes:
        """Edit one or more reference images while preserving their visual identity."""

        if not self.api_key:
            raise RuntimeError("MMEM_IMAGE_API_KEY is required for GPT Image editing")
        paths = [Path(item) for item in reference_paths]
        if not paths or any(not item.is_file() for item in paths):
            raise ValueError("GPT Image editing requires existing reference image paths")
        last_error: Exception | None = None
        for attempt in range(self.retries + 1):
            files = [item.open("rb") for item in paths]
            client = OpenAI(
                base_url=self.base_url,
                api_key=self.api_key,
                timeout=self.timeout,
                max_retries=0,
            )
            try:
                response = client.images.edit(
                    model=DEFAULT_IMAGE_MODEL,
                    image=files[0] if len(files) == 1 else files,
                    prompt=prompt,
                    n=1,
                    size=DEFAULT_IMAGE_REQUEST_SIZE,
                    quality="medium",
                )
                item = response.data[0] if response.data else None
                if item is None:
                    raise RuntimeError(f"GPT Image edit response has no image item: {response}")
                if item.b64_json:
                    return base64.b64decode(item.b64_json, validate=True)
                if item.url:
                    image_req = request.Request(str(item.url), headers={"User-Agent": "MMem-Data-Pipeline/1.0"})
                    with self._opener.open(image_req, timeout=self.timeout) as result:
                        return result.read()
                raise RuntimeError(f"GPT Image edit response has no image data: {response}")
            except Exception as exc:
                last_error = exc
            finally:
                for file in files:
                    file.close()
                client.close()
            if attempt < self.retries:
                time.sleep(2**attempt)
        assert last_error is not None
        raise RuntimeError(
            f"GPT Image editing failed after {self.retries + 1} attempts: {last_error}"
        ) from last_error


def materialize_image_requests(
    artifact: dict[str, Any],
    *,
    output_dir: str | Path,
    client: GPTImageClient,
    workers: int = 1,
    overwrite: bool = False,
) -> tuple[dict[str, Any], list[GeneratedImage]]:
    """Generate private ``image_requests`` and expose only normalized images."""

    result = copy.deepcopy(artifact)
    requests = result.pop("image_requests", [])
    if not requests:
        return result, []
    if not isinstance(requests, list):
        raise ValueError("image_requests must be a list")
    images = result.setdefault("images", [])
    if not isinstance(images, list):
        raise ValueError("images must be a list")
    existing = {str(item.get("image_id")) for item in images if isinstance(item, dict)}
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)

    def generate_one(index_and_spec: tuple[int, Any]) -> tuple[int, dict[str, Any], GeneratedImage | None]:
        index, spec = index_and_spec
        if not isinstance(spec, dict):
            raise ValueError(f"image_requests[{index}] must be an object")
        image_id = str(spec.get("image_id") or "").strip()
        prompt = str(spec.get("prompt") or "").strip()
        if not image_id or not prompt:
            raise ValueError(f"image_requests[{index}] requires image_id and prompt")
        if image_id in existing:
            raise ValueError(f"image_requests[{index}] duplicates existing image_id {image_id!r}")
        safe_name = "".join(character if character.isalnum() or character in "-_" else "_" for character in image_id)
        path = destination / f"{safe_name}.png"
        generated: GeneratedImage | None = None
        if overwrite or not path.exists():
            image_bytes = resize_image_bytes(client.generate(prompt))
            temporary = path.with_suffix(".png.tmp")
            temporary.write_bytes(image_bytes)
            temporary.replace(path)
            generated = GeneratedImage(image_id=image_id, path=path)
        image = {
            "image_id": image_id,
            "path": str(path.resolve()),
            "role": str(spec.get("role") or "memory"),
            "public_retrieval_description": str(spec.get("public_retrieval_description") or ""),
            "private_verified_visual_facts": list(spec.get("private_verified_visual_facts") or []),
        }
        return index, image, generated

    rows: list[tuple[int, dict[str, Any], GeneratedImage | None]] = []
    if workers <= 1:
        rows = [generate_one(item) for item in enumerate(requests)]
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(generate_one, item) for item in enumerate(requests)]
            for future in as_completed(futures):
                rows.append(future.result())
    generated_images: list[GeneratedImage] = []
    for _, image, generated in sorted(rows, key=lambda item: item[0]):
        images.append(image)
        if generated is not None:
            generated_images.append(generated)
    return result, generated_images
