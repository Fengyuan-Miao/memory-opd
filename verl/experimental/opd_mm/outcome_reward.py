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

"""Terminal answer-correctness reward for OPD-MM GRPO.

The retrieval policy never sees the gold answer.  Once a trajectory terminates,
a fixed answer model consumes the public evidence and a separate judge compares
that generated answer with the private gold answer.  Both calls use an
OpenAI-compatible endpoint so the reward model can run outside the actor's Ray
resource pool.
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import mimetypes
import os
import re
from pathlib import Path
from typing import Any

import aiohttp


DEFAULT_OUTCOME_BASE_URL = "http://127.0.0.1:8011"
DEFAULT_OUTCOME_MODEL = "opd-mm-outcome"
OPD_MM_DATA_SOURCE_PREFIX = "opd_mm"
logger = logging.getLogger(__name__)


def _image_data_url(path: str | None) -> str | None:
    image_path = Path(str(path or ""))
    if not image_path.is_file():
        return None
    data = image_path.read_bytes()
    if data.startswith(b"\x89PNG\r\n\x1a\n"):
        mime = "image/png"
    elif data.startswith(b"\xff\xd8\xff"):
        mime = "image/jpeg"
    elif data.startswith((b"GIF87a", b"GIF89a")):
        mime = "image/gif"
    elif data.startswith(b"RIFF") and data[8:12] == b"WEBP":
        mime = "image/webp"
    else:
        mime = mimetypes.guess_type(image_path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}"


def _is_cd_question(query: str, point: str | None = None) -> bool:
    """Return whether the sample uses the conflict-detection answer contract."""
    if str(point or "").strip().upper() == "CD":
        return True
    query_lower = str(query or "").lower()
    return "conflict" in query_lower or "contradict" in query_lower


def _cd_question_mode(query: str) -> str:
    """Distinguish asking for a conflict from asking whether a claim is true."""
    query_lower = str(query or "").lower()
    return "conflict_presence" if "conflict" in query_lower or "contradict" in query_lower else "claim_truth"


def _cd_comparison_scope(query: str) -> str:
    """Separate a tested claim from contradictions internal to the memory."""
    query_lower = " ".join(str(query or "").lower().split())
    internal_markers = (
        "conversation contain conflicting",
        "conversation contains conflicting",
        "conversation contain any contradictory",
        "conversation contains any contradictory",
        "contradictory statements",
        "contradictory definitions",
        "contradict himself",
        "contradict herself",
        "contradict themselves",
        "contradicting earlier",
        "after describing",
        "contradiction between",
        "contradiction in the conversation",
        "contradiction in the dialogue",
    )
    if any(marker in query_lower for marker in internal_markers):
        return "memory_internal"
    if "contradiction in" in query_lower and any(
        marker in query_lower for marker in ("description", " wanting ", " saying ")
    ):
        return "memory_internal"
    if "instance where" in query_lower and "contradictory" in query_lower:
        return "memory_internal"
    if "initial reference" in query_lower and "image" in query_lower:
        return "memory_internal"
    return "claim_vs_memory"


def _cd_answer_contract(query: str) -> str:
    """Build the structured CD decision contract used by answer models."""
    if _cd_comparison_scope(query) == "memory_internal":
        return (
            "This is a memory-internal conflict-detection (CD) sample. Decide only whether public evidence contains "
            "a pair of statements or linked items that cannot both be correct in the entity, event, and temporal scope "
            "asked about. A topic introduced by whether in the question is not itself a statement in memory. Different "
            "contexts, perspectives, degrees, or non-exclusive effects are compatible. For a linked name and image, "
            "compare their atomic identities rather than summarizing the relationship. A later correction does not "
            "erase an earlier mismatch when the question asks about the initial statement or originally supplied image. "
            "Set conflicting_pair_found to false and comparison to NONE when no evidence-backed opposing pair exists. "
            "When it is true, comparison must be one of EXPLICIT_NEGATION, INCOMPATIBLE_VALUE, "
            "INCOMPATIBLE_IDENTITY, INCOMPATIBLE_ATTRIBUTE, or INCOMPATIBLE_ACTION. Return only one JSON object: "
            '{"statement_a":"first evidence-backed atomic fact","statement_b":"second evidence-backed atomic fact '
            'or empty string","comparison":"NONE","conflicting_pair_found":false,"evidence_ids":["E1"]}. '
            "Use only public evidence IDs. Do not answer Yes/No."
        )
    return (
        "This is a conflict-detection (CD) sample. Do not answer Yes/No and do not choose a conflict verdict. "
        "This question tests a claim against memory. Put the concrete claim from the question in "
        "question_proposition and the corresponding public-evidence fact in memory_proposition, then classify only "
        "their semantic difference. "
        "question_proposition must contain the concrete identity, value, attribute, or action attributed by the "
        "question, without the words conflict, contradiction, compatible, or a conclusion about their relationship. "
        "memory_proposition must contain the corresponding evidence-backed fact, also without a relationship verdict. "
        "When the question compares two items from memory, place one factual item in each proposition field. "
        "Use NONE whenever both propositions can be true at the same time. Different contexts, perspectives, degrees, "
        "or non-exclusive effects are not conflicts. Use EXPLICIT_NEGATION for positive-versus-negative statements, "
        "INCOMPATIBLE_VALUE for different exact values, INCOMPATIBLE_IDENTITY for different names or identities in "
        "the same role/event, INCOMPATIBLE_ATTRIBUTE for incompatible properties or rankings, INCOMPATIBLE_ACTION "
        "for incompatible actions or occurrence histories, and MISSING_COMPARISON when evidence does not address the "
        "exact proposition. A shared broad category does not erase a different exact identity or value. Respect the "
        "entity, event, and temporal scope named by the question: a later correction or explanation does not erase an "
        "earlier mismatch when the question asks about the initial statement or the originally supplied image. When "
        "the question explicitly presents two conditions and asks whether they contradict, compare whether those "
        "conditions can both be true in the supplied memory context; use NONE when they are compatible. Do not use "
        "MISSING_COMPARISON merely because one of those stated conditions is not repeated verbatim in evidence; use it "
        "only when evidence does not identify the relevant entity, event, or proposition well enough to compare. "
        "Return only one JSON object with exactly this shape: "
        '{"question_proposition":"brief proposition","memory_proposition":"brief evidence-backed proposition",'
        '"comparison":"INCOMPATIBLE_IDENTITY","evidence_ids":["E1"]}. '
        "comparison must be exactly one of: NONE, EXPLICIT_NEGATION, INCOMPATIBLE_VALUE, "
        "INCOMPATIBLE_IDENTITY, INCOMPATIBLE_ATTRIBUTE, INCOMPATIBLE_ACTION, MISSING_COMPARISON. "
        "Use only public evidence IDs in evidence_ids."
    )


def _parse_cd_decision(query: str, text: str) -> tuple[str, str, dict[str, Any]]:
    """Parse a semantic comparison type and deterministically map it to Yes/No."""
    value = _json_object(text)
    raw_comparison = value.get("comparison")
    if not isinstance(raw_comparison, str):
        raise ValueError("CD answer JSON field 'comparison' must be a string")
    comparison = raw_comparison.strip().upper()
    incompatible = {
        "EXPLICIT_NEGATION",
        "INCOMPATIBLE_VALUE",
        "INCOMPATIBLE_IDENTITY",
        "INCOMPATIBLE_ATTRIBUTE",
        "INCOMPATIBLE_ACTION",
    }
    if _cd_comparison_scope(query) == "memory_internal":
        conflict_found = value.get("conflicting_pair_found")
        if not isinstance(conflict_found, bool):
            raise ValueError("memory-internal CD JSON field 'conflicting_pair_found' must be a boolean")
        if conflict_found and comparison not in incompatible:
            raise ValueError("memory-internal CD conflict requires an incompatible comparison type")
        if not conflict_found and comparison != "NONE":
            raise ValueError("memory-internal CD without a conflict must use comparison NONE")
        return comparison, "Yes" if conflict_found else "No", value
    if comparison == "MISSING_COMPARISON":
        return comparison, "INSUFFICIENT_EVIDENCE", value
    if comparison not in incompatible | {"NONE"}:
        raise ValueError(f"invalid CD comparison type {raw_comparison!r}")
    conflict_present = comparison in incompatible
    if _cd_question_mode(query) == "conflict_presence":
        verdict = "Yes" if conflict_present else "No"
    else:
        verdict = "No" if conflict_present else "Yes"
    return comparison, verdict, value


def _cd_verdict_matches_gold(candidate_answer: str, gold_answer: str) -> bool | None:
    """Compare a normalized CD verdict without asking another model to reinterpret it."""
    candidate = re.sub(r"[^a-z]+", " ", str(candidate_answer or "").lower()).strip()
    gold = re.sub(r"[^a-z]+", " ", str(gold_answer or "").lower()).strip()
    if candidate not in {"yes", "no"} or gold not in {"yes", "no"}:
        return None
    return candidate == gold


def _plain(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _as_dict(value: Any) -> dict[str, Any]:
    value = _plain(value)
    return dict(value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    value = _plain(value)
    if isinstance(value, list | tuple):
        return list(value)
    if hasattr(value, "tolist"):
        converted = value.tolist()
        return converted if isinstance(converted, list) else []
    return []


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


def _endpoint(base_url: str) -> str:
    base = str(base_url or "").rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _json_object(text: str) -> dict[str, Any]:
    stripped = str(text or "").strip()
    candidates = [stripped]
    if "```" in stripped:
        for part in stripped.split("```"):
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].lstrip()
            if part:
                candidates.append(part)
    decoder = json.JSONDecoder()
    for candidate in candidates:
        try:
            value = json.loads(candidate)
            if isinstance(value, dict):
                return value
        except json.JSONDecodeError:
            pass
        for index, char in enumerate(candidate):
            if char != "{":
                continue
            try:
                value, _ = decoder.raw_decode(candidate[index:])
            except json.JSONDecodeError:
                continue
            if isinstance(value, dict):
                return value
    raise ValueError(f"judge did not return a JSON object: {stripped[:240]}")


def _parse_correct(text: str) -> tuple[bool, str]:
    correct, reason, _ = _parse_correct_with_recovery(text)
    return correct, reason


def _parse_correct_with_recovery(text: str) -> tuple[bool, str, bool]:
    """Parse the judge verdict, recovering only an unambiguous boolean.

    The judge sometimes truncates the final closing brace while still emitting
    a complete ``correct`` field. First try strict JSON, then accept a single,
    unambiguous JSON-style boolean literal. Never coerce quoted strings such as
    ``"TRUE"`` because that can hide prompt-following failures.
    """
    try:
        value = _json_object(text)
        correct = value.get("correct")
        if not isinstance(correct, bool):
            raise ValueError("judge JSON field 'correct' must be a boolean")
        return correct, str(value.get("reason") or ""), False
    except ValueError as strict_error:
        matches = re.findall(r'["\']correct["\']\s*:\s*(true|false)\b', str(text or ""), flags=re.IGNORECASE)
        verdicts = {match.lower() == "true" for match in matches}
        if len(verdicts) != 1:
            raise strict_error
        return verdicts.pop(), "", True


def _gold_means_absence(gold_answer: str) -> bool:
    normalized = re.sub(r"[^a-z]+", " ", str(gold_answer or "").lower()).strip()
    return normalized in {
        "not mentioned",
        "not specified",
        "not provided",
        "unknown",
        "cannot be determined",
        "insufficient information",
    }


def _parse_answerable_with_recovery(
    text: str,
    *,
    query: str | None = None,
    gold_answer: str | None = None,
    search_progress: dict[str, Any] | None = None,
    question_image_attached: bool = False,
) -> tuple[bool, str, bool]:
    """Parse a structured evidence-support class, with legacy boolean support."""
    try:
        value = _json_object(text)
        support = value.get("support")
        if isinstance(support, str):
            normalized_support = support.strip().upper()
            allowed = {"DIRECT", "ABSENCE_WITH_COVERAGE", "INSUFFICIENT", "CONTRADICTED"}
            if normalized_support not in allowed:
                raise ValueError(f"invalid evidence support class {support!r}")
            reason = str(value.get("reason") or "")
            answerable = normalized_support in {"DIRECT", "ABSENCE_WITH_COVERAGE"}
            if normalized_support == "ABSENCE_WITH_COVERAGE" and gold_answer is not None:
                absence_valid = _gold_means_absence(gold_answer) and _search_progress_allows_empty_answer(
                    str(query or ""),
                    search_progress or {},
                    question_image_attached=question_image_attached,
                )
                answerable = bool(absence_valid)
                if not absence_valid:
                    reason = (reason + " [ABSENCE_WITH_COVERAGE rejected by deterministic coverage rules]").strip()
            return answerable, reason, False
        answerable = value.get("answerable")
        if not isinstance(answerable, bool):
            raise ValueError("evidence judge JSON must contain a valid 'support' class")
        if answerable and gold_answer is not None and _gold_means_absence(gold_answer):
            answerable = _search_progress_allows_empty_answer(
                str(query or ""),
                search_progress or {},
                question_image_attached=question_image_attached,
            )
        return answerable, str(value.get("reason") or ""), False
    except ValueError as strict_error:
        support_matches = re.findall(
            r'["\']support["\']\s*:\s*["\'](DIRECT|ABSENCE_WITH_COVERAGE|INSUFFICIENT|CONTRADICTED)["\']',
            str(text or ""),
            flags=re.IGNORECASE,
        )
        supports = {match.upper() for match in support_matches}
        if len(supports) == 1:
            recovered_support = supports.pop()
            answerable = recovered_support in {"DIRECT", "ABSENCE_WITH_COVERAGE"}
            if recovered_support == "ABSENCE_WITH_COVERAGE" and gold_answer is not None:
                answerable = _gold_means_absence(gold_answer) and _search_progress_allows_empty_answer(
                    str(query or ""),
                    search_progress or {},
                    question_image_attached=question_image_attached,
                )
            return bool(answerable), "", True
        matches = re.findall(
            r'["\']answerable["\']\s*:\s*(true|false)\b',
            str(text or ""),
            flags=re.IGNORECASE,
        )
        verdicts = {match.lower() == "true" for match in matches}
        if len(verdicts) != 1:
            raise strict_error
        answerable = verdicts.pop()
        if answerable and gold_answer is not None and _gold_means_absence(gold_answer):
            answerable = _search_progress_allows_empty_answer(
                str(query or ""),
                search_progress or {},
                question_image_attached=question_image_attached,
            )
        return bool(answerable), "", True


def _answer_messages(
    query: str,
    evidence: list[Any],
    search_progress: dict[str, Any] | None = None,
    point: str | None = None,
    question_image: str | None = None,
) -> list[dict[str, Any]]:
    evidence_json = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), default=str)
    progress_json = json.dumps(search_progress or {}, ensure_ascii=False, separators=(",", ":"), default=str)
    is_conflict_query = _is_cd_question(query, point)
    if is_conflict_query:
        messages: list[dict[str, Any]] = [
            {
                "role": "system",
                "content": (
                    "Compare the question with only the supplied public evidence. Do not use outside knowledge or "
                    "hidden memory. " + _cd_answer_contract(query)
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Question:\n{query}\n\nPublic evidence:\n{evidence_json}\n\n"
                    f"Public search progress:\n{progress_json}"
                ),
            },
        ]
        image_url = _image_data_url(question_image)
        if image_url:
            messages[1]["content"] = [
                {"type": "text", "text": messages[1]["content"]},
                {"type": "image_url", "image_url": {"url": image_url}},
            ]
        return messages
    stalled_absence_contract = ""
    if not is_conflict_query and str((search_progress or {}).get("state") or "") == "stalled":
        stalled_absence_contract = (
            "Search is stalled. If the exact requested proposition is neither explicitly affirmed nor explicitly "
            "negated in evidence, return exactly Not mentioned. Related concepts, a different object type, and a "
            "future intention do not establish the requested proposition or its negation. "
        )
    messages = [
        {
            "role": "system",
            "content": (
                "Answer the memory question using only the supplied public evidence. Do not use outside knowledge or "
                "the hidden memory store. If the evidence cannot support an answer, return exactly "
                "INSUFFICIENT_EVIDENCE. Never guess an unstated exact name, value, date, cause, or capability from a "
                "related fact; a nearby action is not a cause unless evidence explicitly links them. Not mentioned and "
                "No are distinct: No requires an explicit negative statement, and missing information never supports "
                "No. When asked whether a person, conversation, dialogue, or record mentioned, stated, or provided "
                "information, return exactly Not mentioned only when search_progress is stalled after multiple "
                "complementary searches that cover the requested modality and the detail remains absent. For other "
                "unsupported yes/no questions, "
                "use INSUFFICIENT_EVIDENCE. Ignore unrelated evidence and return only the final answer. "
                + stalled_absence_contract
                + "An explicit incompatible alternative for the same entity/event is a conflict rather than missing "
                "information; absence alone is not a conflict. "
                "Do not substitute a nearby entity, object type, event, or capability for the one asked about, and "
                "do not turn a plausible implication into a stated fact. A future intention does not prove that a "
                "past action did not occur. "
                "For questions requesting multiple people, items, or events, include every supported member. "
                "Resolve relative dates from session_date and explicit event wording, not from an unrelated timestamp. "
                "Return an image_id only when the question asks which image or asks for an image ID; an image attached "
                "to the question or present in evidence does not by itself make an image ID the answer. "
                "For ordinary factual questions, answer with the requested fact rather than an image_id."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\nPublic evidence:\n{evidence_json}\n\n"
                f"Public search progress:\n{progress_json}"
            ),
        },
    ]
    image_url = _image_data_url(question_image)
    if image_url:
        messages[1]["content"] = [
            {"type": "text", "text": messages[1]["content"]},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    return messages


def _normalize_answer_output(query: str, answer: str, point: str | None = None) -> str:
    """Normalize answer output, including deterministic structured CD relations."""
    normalized = str(answer or "").strip()
    if _is_cd_question(query, point):
        _, verdict, _ = _parse_cd_decision(query, normalized)
        return verdict
    try:
        wrapped = _json_object(normalized)
        if isinstance(wrapped.get("answer"), str):
            normalized = wrapped["answer"].strip()
    except ValueError:
        pass
    return normalized


def _judge_messages(
    query: str,
    gold_answer: str,
    evidence: list[Any],
    candidate_answer: str,
) -> list[dict[str, str]]:
    evidence_json = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), default=str)
    return [
        {
            "role": "system",
            "content": (
                "Judge answer correctness for a memory-QA benchmark. Use the gold answer as the sole correctness "
                "reference: identify the minimal core proposition or requested set needed to answer the question, then "
                "mark correct only if the candidate conveys that answer at the requested granularity. Exact wording is "
                "unnecessary, and a concise answer need not repeat illustrative examples or explanations from a verbose "
                "gold answer. The gold is not automatically an exhaustive list: omission of a detail from gold is not "
                "a contradiction. Public evidence is provided only to validate relevant extra details in the candidate; "
                "it cannot replace a missing required answer or excuse a wrong core answer. "
                "Do not mark an answer incorrect merely because it is more detailed than the gold or embeds the core "
                "answer in an explanation, provided every material extra claim is relevant, evidence-supported, and "
                "does not contradict the gold. Unsupported or contradictory material additions are incorrect. "
                "A refusal, unknown, "
                "or INSUFFICIENT_EVIDENCE is incorrect when the gold answer provides a substantive answer. When the "
                "gold means unknown or not mentioned, a candidate that explicitly says the detail is unstated or "
                "cannot be determined from the record is equivalent. A bare No is equivalent only when the question "
                "itself asks whether the conversation or record mentioned, stated, or provided information; it is not "
                "equivalent when the question asks whether an event occurred. If the candidate explicitly says the "
                "record does not mention the detail, do not reinterpret an introductory No as a factual denial. For "
                "a conflict or contradiction question, Yes means the candidate claim conflicts with memory and No "
                "means it is compatible; a verdict whose explanation states the opposite is incorrect. "
                "Return only JSON: {\"correct\":true|false,\"reason\":\"short reason\"}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\nGold answer:\n{gold_answer}\n\n"
                f"Candidate answer:\n{candidate_answer}\n\nPublic evidence (for validating additions only):\n"
                f"{evidence_json}"
            ),
        },
    ]


def _evidence_answerable_messages(
    query: str,
    gold_answer: str,
    evidence: list[Any],
    search_progress: dict[str, Any] | None = None,
    question_image: str | None = None,
) -> list[dict[str, Any]]:
    """Build the private verifier prompt for public-evidence sufficiency."""
    evidence_json = json.dumps(evidence, ensure_ascii=False, separators=(",", ":"), default=str)
    progress_json = json.dumps(search_progress or {}, ensure_ascii=False, separators=(",", ":"), default=str)
    messages = [
        {
            "role": "system",
            "content": (
                "Judge whether the supplied public evidence alone contains enough information to derive the gold "
                "answer to the question. Use the gold only as a private reference; do not require exact wording and "
                "do not use outside knowledge. For a question asking which image, the evidence must expose the "
                "corresponding image_id. When the gold means not mentioned, direct factual evidence is not required, "
                "but search_progress must be stalled after at least two distinct complementary searches covering the "
                "question-visible subject and modality. Search progress never supports a positive factual answer. "
                "When those absence-search conditions hold and the requested detail remains absent, use "
                "ABSENCE_WITH_COVERAGE; do not require an explicit sentence saying that the detail is absent. "
                "ABSENCE_WITH_COVERAGE is valid only when the gold answer itself means missing or not mentioned; "
                "never use it for a substantive gold answer. Relative dates and counts that can be deterministically "
                "derived from explicit evidence and timestamps are DIRECT support and need not appear verbatim. Basic "
                "arithmetic such as subtracting 'three days ago' from an explicit session date is permitted reasoning, "
                "not outside knowledge; when that calculation uniquely yields the gold answer, classify DIRECT. "
                "For a conflict or contradiction question, an explicit incompatible value, identity, attribute, or "
                "action for the same entity/event is sufficient evidence for conflict; absence alone is not. "
                "Classify support as DIRECT when evidence directly derives the gold, ABSENCE_WITH_COVERAGE only for "
                "a justified not-mentioned answer, CONTRADICTED when evidence explicitly supports a conflicting "
                "answer, or INSUFFICIENT otherwise. Unrelated or partial evidence is INSUFFICIENT. Return only JSON: "
                '{"support":"DIRECT|ABSENCE_WITH_COVERAGE|INSUFFICIENT|CONTRADICTED","reason":"short reason"}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{query}\n\nGold answer:\n{gold_answer}\n\n"
                f"Public evidence:\n{evidence_json}\n\nPublic search progress:\n{progress_json}"
            ),
        },
    ]
    image_url = _image_data_url(question_image)
    if image_url:
        messages[1]["content"] = [
            {"type": "text", "text": messages[1]["content"]},
            {"type": "image_url", "image_url": {"url": image_url}},
        ]
    return messages


def _search_progress_allows_empty_answer(
    query: str,
    progress: dict[str, Any],
    *,
    question_image_attached: bool = False,
) -> bool:
    """Return whether an empty-evidence absence answer is worth evaluating."""
    if str(progress.get("state") or "") != "stalled":
        return False
    if int(progress.get("distinct_discovery_count") or 0) < 2:
        return False
    if int(progress.get("consecutive_no_gain_count") or 0) < 2:
        return False
    methods = {str(value) for value in _as_list(progress.get("retrieval_methods_tried"))}
    metadata_fields = {str(value) for value in _as_list(progress.get("metadata_fields_tried"))}
    neighbor_windows = {int(value) for value in _as_list(progress.get("neighbor_windows_tried"))}
    complementary = (
        len(methods) >= 2
        or int(progress.get("rewritten_query_count") or 0) >= 1
        or len(metadata_fields) >= 2
        or (bool(metadata_fields) and bool(methods or neighbor_windows))
        or (bool(neighbor_windows) and bool(methods))
    )
    if not complementary:
        return False
    modalities = {str(value) for value in _as_list(progress.get("modalities_searched"))}
    query_text = str(query or "").lower()
    requires_image = bool(question_image_attached) or any(
        marker in query_text
        for marker in ("image", "photo", "picture", "visual", "图像", "图片", "照片")
    )
    return "image" in modalities if requires_image else "text" in modalities


async def _chat_completion(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    api_key: str,
    timeout: float,
    max_tokens: int,
    retries: int,
    json_mode: bool = False,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": int(max_tokens),
        "chat_template_kwargs": {"enable_thinking": False},
    }
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    client_timeout = aiohttp.ClientTimeout(total=float(timeout))
    last_error: Exception | None = None

    for attempt in range(max(1, int(retries))):
        try:
            async with aiohttp.ClientSession(timeout=client_timeout) as session:
                async with session.post(_endpoint(base_url), json=payload, headers=headers) as response:
                    body = await response.text()
                    if response.status >= 400:
                        error = RuntimeError(f"outcome service HTTP {response.status}: {body[:400]}")
                        if response.status < 500:
                            raise error
                        last_error = error
                    else:
                        parsed = json.loads(body)
                        choices = parsed.get("choices") if isinstance(parsed, dict) else None
                        if not choices or not isinstance(choices[0], dict):
                            raise RuntimeError(f"outcome service returned no choices: {body[:400]}")
                        message = choices[0].get("message") or {}
                        content = _content_text(message.get("content"))
                        if not content:
                            raise RuntimeError("outcome service returned empty content")
                        return content
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if isinstance(exc, RuntimeError) and "HTTP 4" in str(exc):
                break
        if attempt + 1 < max(1, int(retries)):
            await asyncio.sleep(min(2**attempt, 4))

    raise RuntimeError(f"outcome model request failed after {max(1, int(retries))} attempts: {last_error}")


def _repeat_count(trace: list[Any]) -> int:
    seen: set[str] = set()
    repeats = 0
    for item in trace:
        if not isinstance(item, dict):
            continue
        signature = json.dumps(item, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)
        if signature in seen:
            repeats += 1
        else:
            seen.add(signature)
    return repeats


async def _dump_result(payload: dict[str, Any]) -> None:
    dump_dir = str(os.getenv("OPD_MM_OUTCOME_REWARD_DUMP_DIR") or "").strip()
    if not dump_dir:
        return
    path = Path(dump_dir) / f"outcome_reward_{os.getpid()}.jsonl"
    line = json.dumps(payload, ensure_ascii=False, default=str) + "\n"

    def write() -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(line)

    try:
        await asyncio.to_thread(write)
    except OSError as exc:
        # Reward dumps are diagnostic only. A full or temporarily unavailable
        # filesystem must not discard the computed reward or stop training.
        logger.warning("Failed to write OPD-MM outcome reward dump %s: %s", path, exc)


async def compute_outcome_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    *,
    answer_base_url: str | None = None,
    answer_model: str | None = None,
    judge_base_url: str | None = None,
    judge_model: str | None = None,
    api_key: str | None = None,
    timeout: float = 180.0,
    answer_max_tokens: int = 256,
    judge_max_tokens: int = 192,
    retries: int = 3,
    repeat_penalty: float = 0.02,
    max_action_penalty: float = 0.1,
    error_penalty: float = 0.1,
    non_stop_penalty: float = 0.1,
    empty_evidence_penalty: float = 0.1,
    evidence_answerable_weight: float = 0.2,
    efficiency_action_free: int = 3,
    efficiency_action_penalty: float = 0.01,
    efficiency_evidence_free: int = 16,
    efficiency_evidence_penalty: float = 0.005,
    **kwargs: Any,
) -> dict[str, float]:
    """Generate and judge a final answer from an OPD-MM terminal state."""
    del solution_str, kwargs
    # Validation splits use distinct labels so verl reports their metrics
    # separately (for example, ``opd_mm_mmem_val`` and
    # ``opd_mm_memgallery_val``). They share the same reward semantics as the
    # training source; unrelated data sources must still fail closed.
    if data_source != OPD_MM_DATA_SOURCE_PREFIX and not data_source.startswith(
        f"{OPD_MM_DATA_SOURCE_PREFIX}_"
    ):
        raise NotImplementedError(f"Outcome reward is not implemented for {data_source=}")

    info = _as_dict(extra_info)
    state = _as_dict(info.get("opd_mm"))
    evidence = _as_list(state.get("evidence"))
    trace = _as_list(state.get("trace"))
    search_progress = _as_dict(state.get("search_progress"))
    query = str(state.get("query") or "").strip()
    tools_kwargs = _as_dict(info.get("tools_kwargs"))
    opd_tool_kwargs = _as_dict(tools_kwargs.get("opd_mm"))
    if not query:
        query = str(opd_tool_kwargs.get("query") or "").strip()
    question_image = str(opd_tool_kwargs.get("question_image") or info.get("question_image") or "").strip()
    gold_answer = str(_plain(info.get("gold_answer", ground_truth)) or "").strip()
    point = str(_plain(info.get("point", state.get("point"))) or "").strip()
    is_cd_question = _is_cd_question(query, point)

    terminated = bool(state.get("terminated", state.get("stopped")))
    policy_stopped = bool(state.get("policy_stopped", state.get("stopped")))
    termination_reason = str(
        state.get("termination_reason") or ("policy_stop" if policy_stopped else "")
    )
    trajectory_error = bool(state.get("error"))
    max_actions_reached = bool(state.get("max_actions_reached"))
    repeats = _repeat_count(trace)
    correct = False
    evidence_answerable = False
    evaluated = False
    candidate_answer = ""
    judge_raw = ""
    judge_reason = ""
    evidence_judge_raw = ""
    evidence_judge_reason = ""
    outcome_error = ""
    answer_request_failed = False
    answer_parse_failed = False
    judge_request_failed = False
    judge_parse_recovered = False
    judge_parse_failed = False
    evidence_judge_request_failed = False
    evidence_judge_parse_recovered = False
    evidence_judge_parse_failed = False

    # ``answer_correct=False`` is not sufficient to route a rollout: an
    # unavailable answer/judge service must not become a synthetic negative
    # example. Keep infrastructure failures separate from genuine policy
    # failures while retaining the existing numeric metrics.
    outcome_infrastructure_failure = False

    can_answer_without_evidence = _search_progress_allows_empty_answer(
        query,
        search_progress,
        question_image_attached=bool(question_image or state.get("question_image_attached")),
    )
    if terminated and (evidence or can_answer_without_evidence) and query and gold_answer and not trajectory_error:
        outcome_url = answer_base_url or os.getenv("OPD_MM_OUTCOME_BASE_URL") or DEFAULT_OUTCOME_BASE_URL
        outcome_model = answer_model or os.getenv("OPD_MM_OUTCOME_MODEL") or DEFAULT_OUTCOME_MODEL
        judge_url = judge_base_url or os.getenv("OPD_MM_JUDGE_BASE_URL") or outcome_url
        judge_model_name = judge_model or os.getenv("OPD_MM_JUDGE_MODEL") or outcome_model
        outcome_api_key = api_key or os.getenv("OPD_MM_OUTCOME_API_KEY") or ""

        answer_task = asyncio.create_task(
            _chat_completion(
                base_url=outcome_url,
                model=outcome_model,
                messages=_answer_messages(query, evidence, search_progress, point, question_image),
                api_key=outcome_api_key,
                timeout=float(timeout),
                max_tokens=int(answer_max_tokens),
                retries=int(retries),
                json_mode=is_cd_question,
            )
        )
        evidence_task = asyncio.create_task(
            _chat_completion(
                base_url=judge_url,
                model=judge_model_name,
                messages=_evidence_answerable_messages(
                    query,
                    gold_answer,
                    evidence,
                    search_progress,
                    question_image,
                ),
                api_key=outcome_api_key,
                timeout=float(timeout),
                max_tokens=int(judge_max_tokens),
                retries=int(retries),
                json_mode=True,
            )
        )
        answer_result, evidence_result = await asyncio.gather(
            answer_task,
            evidence_task,
            return_exceptions=True,
        )
        if isinstance(answer_result, Exception):
            answer_request_failed = True
            outcome_infrastructure_failure = True
            outcome_error = f"answer_request_failed:{type(answer_result).__name__}:{answer_result}"[:500]
        else:
            try:
                candidate_answer = _normalize_answer_output(query, str(answer_result or ""), point)
            except ValueError as exc:
                answer_parse_failed = True
                outcome_infrastructure_failure = True
                outcome_error = f"answer_parse_failed:{type(exc).__name__}:{exc}"[:500]
            if not candidate_answer and not answer_parse_failed:
                answer_request_failed = True
                outcome_infrastructure_failure = True
                outcome_error = "answer_request_failed:empty_response"

        if isinstance(evidence_result, Exception):
            evidence_judge_request_failed = True
            outcome_infrastructure_failure = True
            if not outcome_error:
                outcome_error = (
                    f"evidence_judge_request_failed:{type(evidence_result).__name__}:{evidence_result}"
                )[:500]
        else:
            evidence_judge_raw = evidence_result
            try:
                evidence_answerable, evidence_judge_reason, recovered = _parse_answerable_with_recovery(
                    evidence_judge_raw,
                    query=query,
                    gold_answer=gold_answer,
                    search_progress=search_progress,
                    question_image_attached=bool(question_image or state.get("question_image_attached")),
                )
                evidence_judge_parse_recovered = recovered
            except ValueError as exc:
                evidence_judge_parse_failed = True
                outcome_infrastructure_failure = True
                if not outcome_error:
                    outcome_error = f"evidence_judge_parse_failed:{type(exc).__name__}:{exc}"[:500]

        deterministic_cd_correct = (
            _cd_verdict_matches_gold(candidate_answer, gold_answer) if is_cd_question else None
        )
        if deterministic_cd_correct is not None:
            correct = deterministic_cd_correct
            judge_reason = "deterministic structured CD relation comparison"
            evaluated = True
        elif candidate_answer:
            parse_error: Exception | None = None
            for judge_attempt in range(max(1, int(retries))):
                try:
                    judge_raw = await _chat_completion(
                        base_url=judge_url,
                        model=judge_model_name,
                        messages=_judge_messages(query, gold_answer, evidence, candidate_answer),
                        api_key=outcome_api_key,
                        timeout=float(timeout),
                        max_tokens=int(judge_max_tokens),
                        retries=int(retries),
                        json_mode=True,
                    )
                    correct, judge_reason, recovered = _parse_correct_with_recovery(judge_raw)
                    judge_parse_recovered = judge_parse_recovered or recovered
                    parse_error = None
                    evaluated = True
                    break
                except ValueError as exc:
                    parse_error = exc
                except Exception as exc:
                    judge_request_failed = True
                    outcome_infrastructure_failure = True
                    parse_error = exc
                if judge_attempt + 1 < max(1, int(retries)):
                    await asyncio.sleep(min(2**judge_attempt, 4))
            if parse_error is not None:
                judge_parse_failed = not judge_request_failed
                outcome_infrastructure_failure = True
                outcome_error = f"judge_failed:{type(parse_error).__name__}:{parse_error}"[:500]

    action_count = len(trace)
    action_over_budget = max(0, action_count - max(0, int(efficiency_action_free)))
    evidence_over_budget = max(0, len(evidence) - max(0, int(efficiency_evidence_free)))
    efficiency_eligible = bool(correct or evidence_answerable)
    efficiency_penalty = 0.0
    if efficiency_eligible:
        efficiency_penalty = (
            float(efficiency_action_penalty) * action_over_budget
            + float(efficiency_evidence_penalty) * evidence_over_budget
        )

    score = float(correct) + float(evidence_answerable_weight) * float(evidence_answerable)
    score -= efficiency_penalty
    score -= float(repeat_penalty) * repeats
    score -= float(max_action_penalty) if max_actions_reached else 0.0
    score -= float(error_penalty) if trajectory_error else 0.0
    score -= float(non_stop_penalty) if not policy_stopped else 0.0
    score -= float(empty_evidence_penalty) if policy_stopped and not evidence and not can_answer_without_evidence else 0.0
    score = max(-1.0, min(max(1.0, 1.0 + float(evidence_answerable_weight)), score))

    await _dump_result(
        {
            "query": query,
            "gold_answer": gold_answer,
            "evidence": evidence,
            "trace": trace,
            "candidate_answer": candidate_answer,
            "judge_raw": judge_raw,
            "judge_reason": judge_reason,
            "evidence_judge_raw": evidence_judge_raw,
            "evidence_judge_reason": evidence_judge_reason,
            "outcome_error": outcome_error,
            "correct": correct,
            "evidence_answerable": evidence_answerable,
            "score": score,
            "terminated": terminated,
            "termination_reason": termination_reason,
            "policy_stopped": policy_stopped,
            "stopped": policy_stopped,
            "max_actions_reached": max_actions_reached,
            "trajectory_error": state.get("error") or "",
            "repeat_count": repeats,
            "action_count": action_count,
            "action_over_budget": action_over_budget,
            "evidence_over_budget": evidence_over_budget,
            "efficiency_eligible": efficiency_eligible,
            "efficiency_penalty": efficiency_penalty,
            "drop_calls": int(state.get("drop_calls") or 0),
            "dropped_evidence_count": int(state.get("dropped_evidence_count") or 0),
            "answer_request_failed": answer_request_failed,
            "answer_parse_failed": answer_parse_failed,
            "judge_request_failed": judge_request_failed,
            "judge_parse_recovered": judge_parse_recovered,
            "judge_parse_failed": judge_parse_failed,
            "evidence_judge_request_failed": evidence_judge_request_failed,
            "evidence_judge_parse_recovered": evidence_judge_parse_recovered,
            "evidence_judge_parse_failed": evidence_judge_parse_failed,
            "outcome_infrastructure_failure": outcome_infrastructure_failure,
        }
    )

    return {
        "score": score,
        "opd_mm/answer_correct": float(correct),
        "opd_mm/evidence_answerable": float(evidence_answerable),
        "opd_mm/outcome_evaluated": float(evaluated),
        "opd_mm/outcome_infrastructure_failure": float(outcome_infrastructure_failure),
        "opd_mm/terminal_stopped": float(policy_stopped),
        "opd_mm/terminated": float(terminated),
        "opd_mm/policy_stop_rate": float(policy_stopped),
        "opd_mm/budget_exhausted_rate": float(termination_reason == "budget_exhausted"),
        "opd_mm/evidence_count": float(len(evidence)),
        "opd_mm/evidence_event_count": float(state.get("evidence_event_count", len(evidence)) or 0),
        "opd_mm/evidence_record_count": float(state.get("evidence_record_count", len(evidence)) or 0),
        "opd_mm/action_count": float(action_count),
        "opd_mm/action_over_budget": float(action_over_budget),
        "opd_mm/evidence_over_budget": float(evidence_over_budget),
        "opd_mm/efficiency_eligible": float(efficiency_eligible),
        "opd_mm/efficiency_penalty": float(efficiency_penalty),
        "opd_mm/empty_evidence": float(not evidence),
        "opd_mm/repeated_actions": float(repeats),
        "opd_mm/blocked_action_count": float(state.get("blocked_action_count") or 0),
        "opd_mm/search_stalled_rate": float(search_progress.get("state") == "stalled"),
        "opd_mm/drop_calls": float(state.get("drop_calls") or 0),
        "opd_mm/dropped_evidence_count": float(state.get("dropped_evidence_count") or 0),
        "opd_mm/max_actions_reached": float(max_actions_reached),
        "opd_mm/trajectory_error": float(trajectory_error),
        "opd_mm/answer_request_failed": float(answer_request_failed),
        "opd_mm/answer_parse_failed": float(answer_parse_failed),
        "opd_mm/judge_request_failed": float(judge_request_failed),
        "opd_mm/judge_parse_recovered": float(judge_parse_recovered),
        "opd_mm/judge_parse_failed": float(judge_parse_failed),
        "opd_mm/evidence_judge_request_failed": float(evidence_judge_request_failed),
        "opd_mm/evidence_judge_parse_recovered": float(evidence_judge_parse_recovered),
        "opd_mm/evidence_judge_parse_failed": float(evidence_judge_parse_failed),
    }


async def compute_opsd_validation_score(
    data_source: str,
    solution_str: str,
    ground_truth: Any,
    extra_info: dict[str, Any] | None = None,
    **kwargs: Any,
) -> dict[str, float]:
    """Use zero task reward for OPSD training and answer correctness for validation.

    Pure OPSD optimizes only teacher/student KL. Validation sources use either
    ``opd_mm_eval`` or a split-specific ``opd_mm_*_val`` label, so only
    validation rollouts invoke the answer and judge service.
    """
    if data_source == "opd_mm_eval":
        return await compute_outcome_score(
            data_source="opd_mm",
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
    if data_source.startswith("opd_mm_") and data_source.endswith("_val"):
        return await compute_outcome_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )
    if data_source != "opd_mm":
        raise NotImplementedError(f"OPSD reward is not implemented for {data_source=}")
    return {
        "score": 0.0,
        "opd_mm/answer_correct": 0.0,
        "opd_mm/evidence_answerable": 0.0,
        "opd_mm/outcome_evaluated": 0.0,
        "opd_mm/outcome_infrastructure_failure": 0.0,
        "opd_mm/terminal_stopped": 0.0,
        "opd_mm/terminated": 0.0,
        "opd_mm/policy_stop_rate": 0.0,
        "opd_mm/budget_exhausted_rate": 0.0,
        "opd_mm/evidence_count": 0.0,
        "opd_mm/evidence_event_count": 0.0,
        "opd_mm/evidence_record_count": 0.0,
        "opd_mm/action_count": 0.0,
        "opd_mm/action_over_budget": 0.0,
        "opd_mm/evidence_over_budget": 0.0,
        "opd_mm/efficiency_eligible": 0.0,
        "opd_mm/efficiency_penalty": 0.0,
        "opd_mm/empty_evidence": 0.0,
        "opd_mm/repeated_actions": 0.0,
        "opd_mm/blocked_action_count": 0.0,
        "opd_mm/search_stalled_rate": 0.0,
        "opd_mm/drop_calls": 0.0,
        "opd_mm/dropped_evidence_count": 0.0,
        "opd_mm/max_actions_reached": 0.0,
        "opd_mm/trajectory_error": 0.0,
        "opd_mm/answer_request_failed": 0.0,
        "opd_mm/judge_request_failed": 0.0,
        "opd_mm/judge_parse_recovered": 0.0,
        "opd_mm/judge_parse_failed": 0.0,
        "opd_mm/evidence_judge_request_failed": 0.0,
        "opd_mm/evidence_judge_parse_recovered": 0.0,
        "opd_mm/evidence_judge_parse_failed": 0.0,
    }


__all__ = ["compute_opsd_validation_score", "compute_outcome_score"]
