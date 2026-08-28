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

"""Model-based semantic screening between retrieval and public evidence."""

from __future__ import annotations

import asyncio
import base64
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import aiohttp


def build_semantic_selector_prompt(
    query: str,
    candidates: list[dict[str, Any]],
    action: dict[str, Any] | None = None,
    *,
    has_question_image: bool = False,
) -> str:
    """Build the shared high-recall screening instruction for local teachers."""
    return (
        "Select every memory candidate that may help answer the question. Favor recall: when relevance is uncertain, "
        "retain the candidate. Keep direct support, necessary context, temporal or relational links, and images that "
        "may match the attached question image even when captions are incomplete. Use the current action as retrieval "
        "intent, but judge usefulness against the original question. For conflict questions, retain evidence about "
        "the same entity/event that establishes a compatible or incompatible alternative even if it omits a disputed "
        "claim word. A shared topic or capability is not enough when the candidate concerns a different entity, "
        "object type, or event. Exclude clearly unrelated candidates or exactly redundant copies. Return an empty "
        "list only when every candidate is clearly irrelevant. "
        "Use only the supplied candidate_id values. Return exactly one JSON object with key "
        "selected_candidate_ids and no other text.\n\n"
        + json.dumps(
            {
                "question": str(query),
                "question_image_attached": bool(has_question_image),
                "current_action": action or {},
                "candidates": candidates,
            },
            ensure_ascii=False,
            separators=(",", ":"),
        )
    )


def _image_data_url(path: str | Path) -> str:
    image_path = Path(path)
    subtype = {".png": "png", ".webp": "webp", ".gif": "gif"}.get(
        image_path.suffix.casefold(), "jpeg"
    )
    encoded = base64.b64encode(image_path.read_bytes()).decode("ascii")
    return f"data:image/{subtype};base64,{encoded}"


def parse_semantic_selection(text: str, allowed_ids: set[str]) -> list[str]:
    """Parse and validate request-local candidate IDs from model output."""
    decoder = json.JSONDecoder()
    parsed = None
    for index, character in enumerate(str(text or "")):
        if character != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            parsed = value
            break
    if parsed is None:
        raise ValueError("selector response contains no JSON object")
    selected = parsed.get("selected_candidate_ids")
    if not isinstance(selected, list):
        raise ValueError("selected_candidate_ids is not a list")
    normalized = list(dict.fromkeys(str(value).strip() for value in selected if str(value).strip()))
    unknown = [value for value in normalized if value not in allowed_ids]
    if unknown:
        raise ValueError(f"unknown candidate IDs: {unknown[:8]}")
    return normalized


@dataclass(frozen=True)
class SemanticSelection:
    """A validated semantic-selection result using request-local candidate IDs."""

    selected_candidate_ids: list[str]
    status: str = "ok"
    error: str = ""


class RemoteSemanticEvidenceSelector:
    """Select potentially answer-relevant candidates with an OpenAI API model.

    Candidate IDs are temporary (``C1``, ``C2``, ...), so this private helper
    never exposes hidden memory IDs to either the selector prompt or the policy.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        timeout: float = 120.0,
        max_tokens: int = 512,
        retries: int = 2,
    ) -> None:
        self.base_url = str(base_url).rstrip("/")
        self.model = str(model).strip()
        self.api_key = str(api_key or "EMPTY")
        self.timeout = max(1.0, float(timeout))
        self.max_tokens = max(64, int(max_tokens))
        self.retries = max(0, int(retries))

    async def select(
        self,
        *,
        query: str,
        candidates: list[dict[str, Any]],
        question_image: str | None = None,
        action: dict[str, Any] | None = None,
    ) -> SemanticSelection:
        if not candidates:
            return SemanticSelection([])
        allowed_ids = {
            str(candidate.get("candidate_id") or "").strip()
            for candidate in candidates
            if candidate.get("candidate_id")
        }
        prompt = build_semantic_selector_prompt(
            query,
            candidates,
            action,
            has_question_image=bool(question_image),
        )
        user_content: str | list[dict[str, Any]] = prompt
        if question_image:
            try:
                user_content = [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(question_image)},
                    },
                ]
            except OSError:
                user_content = prompt
        selection_schema = {
            "type": "object",
            "properties": {
                "selected_candidate_ids": {
                    "type": "array",
                    "items": {"type": "string", "enum": sorted(allowed_ids)},
                }
            },
            "required": ["selected_candidate_ids"],
            "additionalProperties": False,
        }
        payload = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are an OPD-MM semantic evidence selector. Follow the user instruction exactly and "
                        "return no prose."
                    ),
                },
                {
                    "role": "user",
                    "content": user_content,
                },
            ],
            "temperature": 0.0,
            "max_tokens": self.max_tokens,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": "semantic_selection",
                    "strict": True,
                    "schema": selection_schema,
                },
            },
            "chat_template_kwargs": {"enable_thinking": False},
        }
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        endpoint = self.base_url
        if not endpoint.endswith("/chat/completions"):
            endpoint = (
                f"{endpoint}/chat/completions"
                if endpoint.endswith("/v1")
                else f"{endpoint}/v1/chat/completions"
            )

        last_error = ""
        for attempt in range(self.retries + 1):
            try:
                timeout = aiohttp.ClientTimeout(total=self.timeout)
                # trust_env=False deliberately bypasses HTTP(S)_PROXY for the
                # private vLLM endpoint used by training.
                async with aiohttp.ClientSession(timeout=timeout, trust_env=False) as session:
                    async with session.post(endpoint, json=payload, headers=headers) as response:
                        body = await response.text()
                        if response.status >= 400:
                            raise RuntimeError(f"HTTP {response.status}: {body[:400]}")
                response_json = json.loads(body)
                content = response_json["choices"][0]["message"].get("content") or ""
                normalized = parse_semantic_selection(str(content), allowed_ids)
                return SemanticSelection(normalized)
            except Exception as exc:
                last_error = str(exc)
                if attempt < self.retries:
                    await asyncio.sleep(min(2**attempt, 4))
        return SemanticSelection(
            [str(candidate["candidate_id"]) for candidate in candidates],
            status="fallback_all_error",
            error=last_error,
        )


__all__ = [
    "RemoteSemanticEvidenceSelector",
    "SemanticSelection",
    "build_semantic_selector_prompt",
    "parse_semantic_selection",
]
