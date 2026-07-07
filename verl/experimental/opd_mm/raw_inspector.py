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

"""Raw image/media inspectors for OPD-MM.

The agent-loop tool observations are text-only, so INSPECT_RAW converts raw
images into a concise visual observation by calling a remote OpenAI-compatible
vLLM vision-language service.
"""

from __future__ import annotations

import base64
import json
import mimetypes
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any
from urllib import request as urllib_request
from urllib.error import URLError


DEFAULT_RAW_INSPECTOR_URL = "http://192.168.1.113:31208"


def _endpoint(base_url: str, suffix: str) -> str:
    base = str(base_url or "").rstrip("/")
    if base.endswith(suffix):
        return base
    if base.endswith("/v1"):
        return f"{base}{suffix.removeprefix('/v1')}"
    return f"{base}{suffix}"


def _image_to_data_url(path: str) -> str:
    image_path = Path(path)
    mime_type = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    payload = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:{mime_type};base64,{payload}"


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        parts = []
        for item in value:
            if isinstance(item, dict) and item.get("type") == "text":
                parts.append(str(item.get("text") or ""))
            elif isinstance(item, str):
                parts.append(item)
        return "\n".join(part.strip() for part in parts if part and part.strip())
    return str(value or "").strip()


@lru_cache(maxsize=16)
def _discover_model(base_url: str, api_key: str, timeout: float) -> str:
    inspector = RemoteVLLMRawInspector(
        base_url=base_url,
        model="__discover__",
        api_key=api_key or None,
        timeout=min(float(timeout), 5.0),
    )
    response = inspector._get_json(_endpoint(base_url, "/v1/models"))
    data = response.get("data") if isinstance(response, dict) else None
    if isinstance(data, list) and data:
        model_id = data[0].get("id") if isinstance(data[0], dict) else None
        if model_id:
            return str(model_id)
    return ""


@dataclass(frozen=True)
class RemoteVLLMRawInspector:
    """Call a remote OpenAI-compatible vLLM VLM endpoint for raw image details."""

    base_url: str = DEFAULT_RAW_INSPECTOR_URL
    model: str | None = None
    api_key: str | None = None
    timeout: float = 60.0
    max_tokens: int = 256
    temperature: float = 0.0

    def inspect(
        self,
        image_path: str,
        query: str,
        question_image: str | None = None,
        text_context: str | None = None,
    ) -> str:
        """Return a concise textual observation of ``image_path``.

        Failures are returned as text observations instead of exceptions. That
        keeps a transient remote-inspector issue from aborting the whole
        multi-turn rollout.
        """
        try:
            image = Path(str(image_path))
            if not image.exists():
                return f"RAW_INSPECT_ERROR: image path does not exist: {image_path}"

            content: list[dict[str, Any]] = [
                {
                    "type": "text",
                    "text": self._prompt(query=query, text_context=text_context),
                },
                {"type": "image_url", "image_url": {"url": _image_to_data_url(str(image))}},
            ]
            if question_image:
                question_path = Path(str(question_image))
                if question_path.exists():
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {"url": _image_to_data_url(str(question_path))},
                        }
                    )

            payload = {
                "model": self._resolve_model(),
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an OPD-MM raw visual inspector. "
                            "Inspect the provided image directly and return only concise, "
                            "query-relevant visual facts. Do not invent hidden memory IDs."
                        ),
                    },
                    {"role": "user", "content": content},
                ],
                "max_tokens": int(self.max_tokens),
                "temperature": float(self.temperature),
            }
            response = self._post_json(_endpoint(self.base_url, "/v1/chat/completions"), payload)
            choices = response.get("choices") if isinstance(response, dict) else None
            if not choices:
                return f"RAW_INSPECT_ERROR: empty response from {self.base_url}"
            message = choices[0].get("message", {}) if isinstance(choices[0], dict) else {}
            observation = _content_text(message.get("content"))
            return observation or f"RAW_INSPECT_ERROR: empty message content from {self.base_url}"
        except Exception as exc:
            return f"RAW_INSPECT_ERROR: {type(exc).__name__}: {exc}"

    @staticmethod
    def _prompt(query: str, text_context: str | None = None) -> str:
        context = (text_context or "").strip()
        return (
            "Inspect the raw memory image for the OPD-MM retrieval trajectory.\n"
            f"User question: {query}\n"
            f"Linked text context from the same turn: {context or '(none)'}\n\n"
            "Return a compact observation that helps answer the user question. "
            "Mention visible objects, text, people, clothing, scene, layout, colors, "
            "or identity cues only when they are visible or supported by the linked context."
        )

    def _resolve_model(self) -> str:
        configured = self.model or os.getenv("OPD_MM_RAW_INSPECTOR_MODEL")
        if configured:
            return configured
        try:
            discovered = _discover_model(
                self.base_url,
                self.api_key or os.getenv("OPD_MM_RAW_INSPECTOR_API_KEY") or "",
                float(self.timeout),
            )
            if discovered:
                return discovered
        except Exception:
            pass
        return "default"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        api_key = self.api_key or os.getenv("OPD_MM_RAW_INSPECTOR_API_KEY")
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        return headers

    def _post_json(self, url: str, payload: dict[str, Any]) -> dict[str, Any]:
        data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        req = urllib_request.Request(url, data=data, headers=self._headers(), method="POST")
        return self._read_json(req)

    def _get_json(self, url: str) -> dict[str, Any]:
        req = urllib_request.Request(url, headers=self._headers(), method="GET")
        return self._read_json(req)

    def _read_json(self, req: urllib_request.Request) -> dict[str, Any]:
        try:
            with urllib_request.urlopen(req, timeout=float(self.timeout)) as response:
                raw = response.read().decode("utf-8")
        except URLError as exc:
            raise RuntimeError(f"request to {req.full_url} failed: {exc}") from exc
        parsed = json.loads(raw)
        if isinstance(parsed, dict) and parsed.get("error"):
            raise RuntimeError(f"vLLM error from {req.full_url}: {parsed['error']}")
        if not isinstance(parsed, dict):
            raise RuntimeError(f"unexpected JSON response from {req.full_url}: {type(parsed).__name__}")
        return parsed
