"""OpenAI-compatible multimodal generation client."""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import httpx
from openai import OpenAI


DEFAULT_MULTIMODAL_BASE_URL = "https://1pkapi.com/v1"
DEFAULT_MULTIMODAL_MODEL = "gpt-5.6-terra"
MULTIMODAL_API_KEY_ENV = "MMEM_MULTIMODAL_API_KEY"


def _extract_json_object(text: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    candidates: list[tuple[int, dict[str, Any]]] = []
    for position, character in enumerate(text):
        if character != "{":
            continue
        try:
            value, end = decoder.raw_decode(text[position:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            candidates.append((end, value))
    if not candidates:
        raise ValueError("model output does not contain a JSON object")
    return max(candidates, key=lambda item: item[0])[1]


def _image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    if not image_path.is_file():
        raise FileNotFoundError(image_path)
    mime_type = mimetypes.guess_type(image_path.name)[0] or "image/jpeg"
    if not mime_type.startswith("image/"):
        raise ValueError(f"unsupported image type for {image_path}: {mime_type}")
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


@dataclass(frozen=True)
class MultimodalResult:
    content: str
    model: str
    response_id: str
    usage: dict[str, Any]
    response: dict[str, Any]


class MultimodalResponsesClient:
    """OpenAI Responses client used by generative and visual pipeline stages."""

    def __init__(
        self,
        *,
        base_url: str = DEFAULT_MULTIMODAL_BASE_URL,
        model: str = DEFAULT_MULTIMODAL_MODEL,
        api_key: str | None = None,
        timeout: float = 300.0,
        max_retries: int = 2,
        use_env_proxy: bool = True,
    ) -> None:
        resolved_key = api_key or os.environ.get(MULTIMODAL_API_KEY_ENV) or os.environ.get("OPENAI_API_KEY")
        if not resolved_key:
            raise ValueError(
                f"multimodal API key is missing; set {MULTIMODAL_API_KEY_ENV} or pass api_key explicitly"
            )
        self.base_url = base_url.rstrip("/")
        self.model = model
        self._http_client = httpx.Client(timeout=timeout, trust_env=use_env_proxy)
        self._client = OpenAI(
            base_url=self.base_url,
            api_key=resolved_key,
            timeout=timeout,
            max_retries=max_retries,
            http_client=self._http_client,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MultimodalResponsesClient":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    @staticmethod
    def instructions(task_contract: str) -> str:
        contract = str(task_contract or "").strip()
        if not contract:
            raise ValueError("task_contract must be non-empty")
        return contract

    def generate(
        self,
        *,
        task_contract: str,
        prompt: str,
        image_paths: Iterable[str | Path] = (),
        max_output_tokens: int = 2048,
        model: str | None = None,
    ) -> MultimodalResult:
        content: list[dict[str, Any]] = [{"type": "input_text", "text": str(prompt)}]
        content.extend(
            {"type": "input_image", "image_url": _image_data_url(path)}
            for path in image_paths
        )
        response = self._client.responses.create(
            model=model or self.model,
            instructions=self.instructions(task_contract),
            input=[{"role": "user", "content": content}],
            max_output_tokens=max_output_tokens,
        )
        response_dict = response.model_dump(mode="json")
        output_text = str(getattr(response, "output_text", "") or "").strip()
        if not output_text:
            texts = []
            for item in response_dict.get("output", []):
                if not isinstance(item, dict):
                    continue
                for part in item.get("content", []):
                    if isinstance(part, dict) and part.get("text"):
                        texts.append(str(part["text"]))
            output_text = "\n".join(texts).strip()
        if not output_text:
            raise RuntimeError(f"multimodal model returned no output text: {response_dict}")
        usage = response_dict.get("usage")
        return MultimodalResult(
            content=output_text,
            model=str(response_dict.get("model") or model or self.model),
            response_id=str(response_dict.get("id") or ""),
            usage=dict(usage) if isinstance(usage, dict) else {},
            response=response_dict,
        )

    def generate_json(
        self,
        *,
        task_contract: str,
        prompt: str,
        image_paths: Iterable[str | Path] = (),
        max_output_tokens: int = 2048,
        model: str | None = None,
    ) -> tuple[dict[str, Any], MultimodalResult]:
        json_contract = (
            f"{task_contract.strip()}\nReturn exactly one valid JSON object. Do not add Markdown or commentary."
        )
        result = self.generate(
            task_contract=json_contract,
            prompt=prompt,
            image_paths=image_paths,
            max_output_tokens=max_output_tokens,
            model=model,
        )
        return _extract_json_object(result.content), result
