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

"""Remove STARK QAs that cannot be answered from their annotated support.

For every non-AR sample, an answer model first sees only the question and the
gold support records (including raw images when available). A separate request
then asks a judge to compare that generated answer with the gold answer. The
source data is never modified; audit rows and cleaned QA/RLHF files are written
under a new output directory. Request failures are retained for retry rather
than treated as bad training examples.
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import io
import json
import mimetypes
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import aiohttp
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.outcome_reward import _parse_correct_with_recovery
from verl.experimental.opd_mm.stark_expansion import finalize_expansion_dataset


SPLITS = ("train", "validation", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _endpoint(base_url: str) -> str:
    base = base_url.rstrip("/")
    if base.endswith("/v1/chat/completions"):
        return base
    if base.endswith("/v1"):
        return f"{base}/chat/completions"
    return f"{base}/v1/chat/completions"


def _content_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, list):
        return "\n".join(
            str(item.get("text") or "").strip()
            for item in value
            if isinstance(item, dict) and item.get("type") == "text"
        ).strip()
    return str(value or "").strip()


def _image_data_url(path: str | Path, max_side: int = 768) -> str:
    image_path = Path(path)
    mime = mimetypes.guess_type(str(image_path))[0] or "image/jpeg"
    with Image.open(image_path) as image:
        if max(image.size) <= max_side:
            payload = image_path.read_bytes()
        else:
            image.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            if image.mode not in {"RGB", "L"}:
                image = image.convert("RGB")
            buffer = io.BytesIO()
            output_format = "PNG" if image.mode == "L" else "JPEG"
            image.save(buffer, format=output_format, quality=92)
            payload = buffer.getvalue()
            mime = "image/png" if output_format == "PNG" else "image/jpeg"
    encoded = base64.b64encode(payload).decode("ascii")
    return f"data:{mime};base64,{encoded}"


def _public_record(record: dict[str, Any]) -> dict[str, Any]:
    metadata = record.get("metadata") or {}
    content = str(record.get("content") or record.get("summary") or "")
    profile = metadata.get("character_profile") or {}
    speaker_name = str(profile.get("name") or "").strip() if isinstance(profile, dict) else ""
    if content and speaker_name and str(record.get("modality") or "").lower() == "text":
        content = re.sub(r"(?m)^User:", f"{speaker_name}:", content)
    value = {
        "content": content,
        "timestamp": record.get("timestamp"),
        "session_date": metadata.get("session_date"),
        "modality": record.get("modality"),
        "image_id": metadata.get("image_id"),
    }
    return {key: item for key, item in value.items() if item not in (None, "")}


def _answer_messages(
    qa: dict[str, Any],
    evidence_records: list[dict[str, Any]],
    *,
    evidence_description: str = "Gold support records follow in their original annotated order.",
    record_label: str = "Support",
    system_instruction: str | None = None,
    output_rule_override: str | None = None,
    max_image_side: int = 768,
) -> list[dict[str, Any]]:
    point = str(qa.get("point") or "")
    output_rule = output_rule_override or {
        "VS": (
            "For this image-selection question, return only every qualifying public image_id value, "
            "comma-separated. Never return support numbers."
        ),
        "CD": (
            "For this conflict question, return exactly Yes., No., or "
            "INSUFFICIENT_EVIDENCE."
        ),
    }.get(point, "Return only the shortest final answer without explanation.")
    content: list[dict[str, Any]] = [
        {
            "type": "text",
            "text": (
                f"Question:\n{qa['question']}\n\n"
                f"{evidence_description}"
            ),
        }
    ]
    for index, record in enumerate(evidence_records, start=1):
        public = _public_record(record)
        content.append(
            {
                "type": "text",
                "text": (
                    f"{record_label} {index}:\n"
                    f"{json.dumps(public, ensure_ascii=False, separators=(',', ':'))}"
                ),
            }
        )
        raw_pointer = record.get("raw_pointer")
        if raw_pointer and Path(str(raw_pointer)).is_file():
            content.append(
                {
                    "type": "image_url",
                    "image_url": {"url": _image_data_url(raw_pointer, max_side=max_image_side)},
                }
            )

    question_image = qa.get("question_image") or qa.get("question_image_relative")
    if question_image and Path(str(question_image)).is_file():
        content.append({"type": "text", "text": "Question image:"})
        content.append(
            {
                "type": "image_url",
                "image_url": {"url": _image_data_url(question_image, max_side=max_image_side)},
            }
        )

    if system_instruction is None:
        system_instruction = (
            "Answer the memory benchmark question using only the supplied gold support records and images. "
            "Treat timestamps, session dates, and public image IDs as part of the evidence. Do not use outside "
            "knowledge. If the supplied support does not determine the answer, return exactly "
            f"INSUFFICIENT_EVIDENCE. {output_rule}"
        )
    else:
        system_instruction = f"{system_instruction.strip()} {output_rule}"

    return [
        {
            "role": "system",
            "content": system_instruction,
        },
        {"role": "user", "content": content},
    ]


def _judge_messages(qa: dict[str, Any], candidate: str) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Judge answer correctness for a memory-QA dataset. Compare the candidate only against the gold "
                "answer in the context of the question. Correct means fully answering every requested part and "
                "preserving every material entity, item, count, date, location, and temporal qualifier in the gold "
                "answer. Partial overlap is incorrect; never waive a missing part merely because the candidate "
                "captures the main idea. Allow only harmless differences in wording, punctuation, date formatting, "
                "or ordering of unordered items. Extra detail is acceptable only when it does not replace, omit, or "
                "contradict required information. A refusal or INSUFFICIENT_EVIDENCE is incorrect for a substantive "
                "gold answer. Return only JSON: "
                "{\"correct\":true|false,\"reason\":\"short reason\"}."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{qa['question']}\n\nGold answer:\n{qa['gold_answer']}\n\n"
                f"Candidate answer:\n{candidate}"
            ),
        },
    ]


def _cd_answer_from_relation(question: str, relation: str) -> str:
    if relation == "INSUFFICIENT_EVIDENCE":
        return relation
    asks_conflict = bool(re.search(r"\b(?:conflict|contradict)\w*\b", question.casefold()))
    if asks_conflict:
        return "Yes." if relation == "CONTRADICTED" else "No."
    return "Yes." if relation == "CONSISTENT" else "No."


def _parse_cd_support_response(text: str) -> tuple[str, list[int], str]:
    value = text.strip()
    if value.startswith("```"):
        value = value.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
        value = value.removesuffix("```").strip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        try:
            if start < 0 or end <= start:
                raise json.JSONDecodeError("no JSON object", value, 0)
            payload = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            relation_match = re.search(r'"relation"\s*:\s*"([^"]+)', value)
            numbers_match = re.search(r'"evidence_numbers"\s*:\s*\[([^\]]*)', value)
            if relation_match is None or numbers_match is None:
                raise ValueError(f"CD support response is not recoverable JSON: {text[:200]}")
            payload = {
                "relation": relation_match.group(1),
                "evidence_numbers": re.findall(r"\d+", numbers_match.group(1)),
                "reason": "",
            }
    if not isinstance(payload, dict):
        raise ValueError("CD support response must be a JSON object")

    relation = str(payload.get("relation") or "").strip().upper()
    if relation not in {"CONSISTENT", "CONTRADICTED", "INSUFFICIENT_EVIDENCE"}:
        raise ValueError(f"invalid CD relation: {payload.get('relation')!r}")
    raw_numbers = payload.get("evidence_numbers") or []
    if not isinstance(raw_numbers, list):
        raise ValueError("CD evidence_numbers must be a list")
    evidence_numbers: list[int] = []
    for number in raw_numbers:
        if isinstance(number, bool):
            continue
        try:
            evidence_numbers.append(int(number))
        except (TypeError, ValueError):
            continue
    return relation, list(dict.fromkeys(evidence_numbers)), str(payload.get("reason") or "").strip()


def _cd_support_verifier_messages(
    qa: dict[str, Any],
    relation: str,
    evidence_numbers: list[int],
    support_records: list[dict[str, Any]],
    answer_reason: str,
) -> list[dict[str, str]]:
    citations = [
        {
            "support_number": number,
            "record": _public_record(support_records[number - 1]),
        }
        for number in evidence_numbers
        if 1 <= number <= len(support_records)
    ]
    return [
        {
            "role": "system",
            "content": (
                "Verify whether the cited gold support records explicitly establish the proposed semantic relation "
                "between the claim and memory. CONSISTENT means the claim agrees with memory; CONTRADICTED means "
                "memory explicitly opposes it. Record timestamps and session dates explicitly establish event order; "
                "a different stated value for the same event or attribute establishes contradiction. Reject reliance "
                "on absence, plausibility, or unstated implication. A question inside a support record is not "
                "confirmation. Every material part must be established. "
                "Return only JSON: "
                '{"supported":true|false,"reason":"brief reason"}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{qa['question']}\n\nProposed relation:\n{relation}\n\n"
                f"Proposed reason:\n{answer_reason}\n\nCited gold supports:\n"
                f"{json.dumps(citations, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]


def _parse_cd_support_verdict(text: str) -> tuple[bool, str]:
    value = text.strip()
    if value.startswith("```"):
        value = value.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
        value = value.removesuffix("```").strip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        try:
            if start < 0 or end <= start:
                raise json.JSONDecodeError("no JSON object", value, 0)
            payload = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            supported_match = re.search(r'"supported"\s*:\s*(true|false)', value, re.IGNORECASE)
            if supported_match is None:
                raise ValueError(f"CD support verdict is not recoverable JSON: {text[:200]}")
            payload = {
                "supported": supported_match.group(1).casefold() == "true",
                "reason": "",
            }
    if not isinstance(payload, dict) or not isinstance(payload.get("supported"), bool):
        raise ValueError("CD support verdict must contain a boolean supported field")
    return bool(payload["supported"]), str(payload.get("reason") or "").strip()


def _cd_gold_answer_verifier_messages(
    qa: dict[str, Any],
    support_records: list[dict[str, Any]],
) -> list[dict[str, str]]:
    numbered_supports = [
        {
            "support_number": index,
            "record": _public_record(record),
        }
        for index, record in enumerate(support_records, start=1)
    ]
    return [
        {
            "role": "system",
            "content": (
                "Determine whether the numbered gold support records make the supplied gold answer correct for the "
                "literal question. Do not reinterpret an ordinary yes/no question as a conflict question merely "
                "because of a dataset category. For a question asking whether a statement conflicts, Yes is "
                "supported only when memory explicitly contradicts the statement, and No only when memory agrees. "
                "Treat the YYYY-MM-DD prefix of record timestamps, and session dates, as event dates. A reverse "
                "event order or a different stated value for the same event or attribute is explicit contradiction. "
                "Absence is not contradiction, and a question inside a record is not confirmation. Every material "
                "part must be established. Return only JSON: "
                '{"supported":true|false,"evidence_numbers":[1,2],"reason":"brief reason"}. '
                "Use an empty evidence_numbers list when unsupported."
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{qa['question']}\n\nGold answer:\n"
                f"{qa.get('gold_answer') or qa.get('answer')}\n\nNumbered gold supports:\n"
                f"{json.dumps(numbered_supports, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]


def _parse_cd_gold_answer_verdict(text: str, support_count: int) -> tuple[bool, list[int], str]:
    value = text.strip()
    if value.startswith("```"):
        value = value.removeprefix("```json").removeprefix("```JSON").removeprefix("```")
        value = value.removesuffix("```").strip()
    try:
        payload = json.loads(value)
    except json.JSONDecodeError:
        start = value.find("{")
        end = value.rfind("}")
        try:
            if start < 0 or end <= start:
                raise json.JSONDecodeError("no JSON object", value, 0)
            payload = json.loads(value[start : end + 1])
        except json.JSONDecodeError:
            supported_match = re.search(r'"supported"\s*:\s*(true|false)', value, re.IGNORECASE)
            numbers_match = re.search(r'"evidence_numbers"\s*:\s*\[([^\]]*)', value)
            if supported_match is None:
                raise ValueError(f"CD gold-answer verdict is not recoverable JSON: {text[:200]}")
            payload = {
                "supported": supported_match.group(1).casefold() == "true",
                "evidence_numbers": (
                    re.findall(r"\d+", numbers_match.group(1)) if numbers_match is not None else []
                ),
                "reason": "",
            }
    if not isinstance(payload, dict) or not isinstance(payload.get("supported"), bool):
        raise ValueError("CD gold-answer verdict must contain a boolean supported field")
    raw_numbers = payload.get("evidence_numbers") or []
    if not isinstance(raw_numbers, list):
        raise ValueError("CD gold-answer evidence_numbers must be a list")
    evidence_numbers: list[int] = []
    for number in raw_numbers:
        if isinstance(number, bool):
            continue
        try:
            value_number = int(number)
        except (TypeError, ValueError):
            continue
        if 1 <= value_number <= support_count:
            evidence_numbers.append(value_number)
    supported = bool(payload["supported"])
    evidence_numbers = list(dict.fromkeys(evidence_numbers))
    if supported and not evidence_numbers:
        supported = False
    return supported, evidence_numbers, str(payload.get("reason") or "").strip()


async def _chat(
    session: aiohttp.ClientSession,
    *,
    url: str,
    model: str,
    messages: list[dict[str, Any]],
    max_tokens: int,
    retries: int,
) -> str:
    payload = {
        "model": model,
        "messages": messages,
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
        "chat_template_kwargs": {"enable_thinking": False},
    }
    last_error: Exception | None = None
    for attempt in range(max(1, retries)):
        try:
            async with session.post(url, json=payload) as response:
                body = await response.text()
                if response.status >= 400:
                    raise RuntimeError(f"HTTP {response.status}: {body[:400]}")
                value = json.loads(body)
                choices = value.get("choices") if isinstance(value, dict) else None
                if not choices:
                    raise RuntimeError(f"response has no choices: {body[:400]}")
                text = _content_text((choices[0].get("message") or {}).get("content"))
                if not text:
                    raise RuntimeError("response has empty content")
                return text
        except (aiohttp.ClientError, asyncio.TimeoutError, json.JSONDecodeError, RuntimeError) as exc:
            last_error = exc
            if attempt + 1 < max(1, retries):
                await asyncio.sleep(min(2**attempt, 4))
    raise RuntimeError(f"request failed after {max(1, retries)} attempts: {last_error}")


async def _evaluate_one(
    qa: dict[str, Any],
    records_by_id: dict[str, dict[str, Any]],
    session: aiohttp.ClientSession,
    semaphore: asyncio.Semaphore,
    args: argparse.Namespace,
) -> dict[str, Any]:
    sample_id = str(qa["sample_id"])
    support_ids = [str(value) for value in qa.get("support_memory_ids") or []]
    if not support_ids:
        return {
            "sample_id": sample_id,
            "point": qa.get("point"),
            "status": "exempt_no_support",
            "keep": True,
            "candidate_answer": "",
            "judge_correct": None,
            "judge_reason": "AR/no-support sample is outside support-answerability cleaning",
        }
    missing = [memory_id for memory_id in support_ids if memory_id not in records_by_id]
    if missing:
        return {
            "sample_id": sample_id,
            "point": qa.get("point"),
            "status": "error",
            "keep": None,
            "error": f"missing support records: {missing[:3]}",
        }

    async with semaphore:
        try:
            support_records = [records_by_id[memory_id] for memory_id in support_ids]
            is_cd = str(qa.get("point") or "") == "CD"
            system_instruction = None
            output_rule_override = None
            prompt_version = "gold_support_answerability_v1"
            if is_cd:
                system_instruction = (
                    "Classify the candidate claim against the supplied numbered gold support records. Choose "
                    "CONSISTENT when every material part agrees with memory, CONTRADICTED when any material part is "
                    "explicitly incompatible, or INSUFFICIENT_EVIDENCE when the records do not determine the "
                    "relation. The relation label always describes the candidate claim itself, not the surrounding "
                    "question: if your reason says the claim is contradicted, relation must be CONTRADICTED. Treat "
                    "the YYYY-MM-DD prefix of each record timestamp, and any session date, as the event date for "
                    "before/after claims. A reverse date order or a different value for the same event or attribute "
                    "is a contradiction, not missing evidence. Combine records when the claim compares multiple "
                    "events, and never infer a contradiction merely from absence."
                )
                output_rule_override = (
                    "Return only one JSON object with this schema: "
                    '{"relation":"CONSISTENT|CONTRADICTED|INSUFFICIENT_EVIDENCE",'
                    '"evidence_numbers":[1,2],"reason":"brief evidence-based reason"}. '
                    "For CONSISTENT or CONTRADICTED, cite every numbered Support needed. For "
                    "INSUFFICIENT_EVIDENCE, use an empty list."
                )
                prompt_version = "cd_gold_support_relation_citation_v4"
            candidate_raw = await _chat(
                session,
                url=_endpoint(args.base_url),
                model=args.model,
                messages=_answer_messages(
                    qa,
                    support_records,
                    system_instruction=system_instruction,
                    output_rule_override=output_rule_override,
                ),
                max_tokens=args.answer_max_tokens,
                retries=args.retries,
            )
            evidence_numbers: list[int] = []
            answer_reason = ""
            evidence_backed = True
            support_verified = True
            support_verifier_raw = ""
            support_verifier_reason = ""
            gold_support_verified = True
            gold_support_evidence_numbers: list[int] = []
            gold_support_verifier_raw = ""
            gold_support_verifier_reason = ""
            evidence_relation = ""
            if is_cd:
                evidence_relation, evidence_numbers, answer_reason = _parse_cd_support_response(candidate_raw)
                candidate = _cd_answer_from_relation(str(qa.get("question") or ""), evidence_relation)
                evidence_backed = (
                    evidence_relation in {"CONSISTENT", "CONTRADICTED"}
                    and bool(evidence_numbers)
                    and all(1 <= number <= len(support_records) for number in evidence_numbers)
                )
                support_verified = False
                if evidence_backed:
                    support_verifier_raw = await _chat(
                        session,
                        url=_endpoint(args.base_url),
                        model=args.model,
                        messages=_cd_support_verifier_messages(
                            qa,
                            evidence_relation,
                            evidence_numbers,
                            support_records,
                            answer_reason,
                        ),
                        max_tokens=args.judge_max_tokens,
                        retries=args.retries,
                    )
                    support_verified, support_verifier_reason = _parse_cd_support_verdict(
                        support_verifier_raw
                    )
                gold_support_verifier_raw = await _chat(
                    session,
                    url=_endpoint(args.base_url),
                    model=args.model,
                    messages=_cd_gold_answer_verifier_messages(qa, support_records),
                    max_tokens=args.judge_max_tokens,
                    retries=args.retries,
                )
                (
                    gold_support_verified,
                    gold_support_evidence_numbers,
                    gold_support_verifier_reason,
                ) = _parse_cd_gold_answer_verdict(
                    gold_support_verifier_raw,
                    len(support_records),
                )
            else:
                candidate = candidate_raw
            judge_raw = await _chat(
                session,
                url=_endpoint(args.base_url),
                model=args.model,
                messages=_judge_messages(qa, candidate),
                max_tokens=args.judge_max_tokens,
                retries=args.retries,
            )
            correct, reason, recovered = _parse_correct_with_recovery(judge_raw)
            # CD generation can explain the right temporal/attribute conflict
            # while emitting the opposite relation label, and the literal gold
            # adjudicator can make the symmetric label error. Keep a CD when
            # either independently grounded route establishes the gold answer.
            keep = (
                (correct and evidence_backed) or gold_support_verified
                if is_cd
                else correct
            )
            return {
                "sample_id": sample_id,
                "point": qa.get("point"),
                "status": "evaluated",
                "keep": keep,
                "candidate_answer": candidate,
                "candidate_raw": candidate_raw,
                "evidence_relation": evidence_relation,
                "answer_reason": answer_reason,
                "evidence_numbers": evidence_numbers,
                "evidence_backed": evidence_backed if is_cd else None,
                "support_verified": support_verified if is_cd else None,
                "support_verifier_reason": support_verifier_reason,
                "support_verifier_raw": support_verifier_raw,
                "gold_support_verified": gold_support_verified if is_cd else None,
                "gold_support_evidence_numbers": gold_support_evidence_numbers,
                "gold_support_verifier_reason": gold_support_verifier_reason,
                "gold_support_verifier_raw": gold_support_verifier_raw,
                "judge_correct": correct,
                "judge_reason": reason,
                "judge_parse_recovered": recovered,
                "judge_raw": judge_raw,
                "prompt_version": "cd_gold_support_dual_adjudication_v6" if is_cd else prompt_version,
            }
        except Exception as exc:
            return {
                "sample_id": sample_id,
                "point": qa.get("point"),
                "status": "error",
                "keep": None,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
            }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    records = _read_jsonl(Path(args.records))
    records_by_id = {str(record["memory_id"]): record for record in records}
    input_dir = Path(args.input_qa_dir)
    output_dir = Path(args.output_dir)
    audit_dir = output_dir / "audit"
    qa_dir = output_dir / "qa"
    audit_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=float(args.timeout))
    connector = aiohttp.TCPConnector(limit=max(args.workers * 2, 8))
    semaphore = asyncio.Semaphore(args.workers)
    summary: dict[str, Any] = {"model": args.model, "base_url": args.base_url, "splits": {}}

    async with aiohttp.ClientSession(timeout=timeout, connector=connector) as session:
        for split in args.splits:
            qas = _read_jsonl(input_dir / f"{split}_qa.jsonl")
            audit_path = audit_dir / f"{split}.jsonl"
            previous = {str(row["sample_id"]): row for row in _read_jsonl(audit_path)} if args.resume else {}
            finished = {
                sample_id
                for sample_id, row in previous.items()
                if row.get("status") in {"evaluated", "exempt_no_support"}
            }
            pending = [qa for qa in qas if str(qa["sample_id"]) not in finished]
            print(f"[{split}] total={len(qas)} resumed={len(finished)} pending={len(pending)}", flush=True)

            mode = "a" if args.resume and audit_path.exists() else "w"
            completed = len(finished)
            with audit_path.open(mode, encoding="utf-8") as handle:
                for offset in range(0, len(pending), args.batch_size):
                    batch = pending[offset : offset + args.batch_size]
                    results = await asyncio.gather(
                        *(
                            _evaluate_one(qa, records_by_id, session, semaphore, args)
                            for qa in batch
                        )
                    )
                    for row in results:
                        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        previous[str(row["sample_id"])] = row
                    handle.flush()
                    completed += len(results)
                    counts = Counter(str(row.get("status")) for row in previous.values())
                    kept = sum(row.get("keep") is True for row in previous.values())
                    print(
                        f"[{split}] {completed}/{len(qas)} kept={kept} "
                        f"evaluated={counts['evaluated']} errors={counts['error']}",
                        flush=True,
                    )

            # Deduplicate append-only retries by sample ID and leave errors out
            # of the cleaned data until a later resume successfully evaluates them.
            audit_rows = list(previous.values())
            error_rows = [row for row in audit_rows if row.get("status") == "error"]
            keep_ids = {str(row["sample_id"]) for row in audit_rows if row.get("keep") is True}
            cleaned = [qa for qa in qas if str(qa["sample_id"]) in keep_ids]
            with (qa_dir / f"{split}_qa.jsonl").open("w", encoding="utf-8") as handle:
                for qa in cleaned:
                    handle.write(json.dumps(qa, ensure_ascii=False) + "\n")

            by_point: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "kept": 0, "dropped": 0})
            verdicts = {str(row["sample_id"]): row for row in audit_rows}
            for qa in qas:
                point = str(qa.get("point") or "")
                by_point[point]["input"] += 1
                verdict = verdicts.get(str(qa["sample_id"])) or {}
                if verdict.get("keep") is True:
                    by_point[point]["kept"] += 1
                elif verdict.get("keep") is False:
                    by_point[point]["dropped"] += 1
            summary["splits"][split] = {
                "input": len(qas),
                "kept": len(cleaned),
                "dropped": sum(row.get("keep") is False for row in audit_rows),
                "errors": len(error_rows),
                "by_point": dict(sorted(by_point.items())),
            }

    total_errors = sum(value["errors"] for value in summary["splits"].values())
    summary["complete"] = total_errors == 0
    summary_path = output_dir / "cleaning_summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    all_qa_splits_exist = all((qa_dir / f"{split}_qa.jsonl").exists() for split in SPLITS)
    if total_errors == 0 and all_qa_splits_exist:
        manifest = finalize_expansion_dataset(
            expansion_dir=args.expansion_dir,
            records_path=args.records,
            qa_dir=qa_dir,
            output_dir=output_dir,
        )
        summary["final_manifest"] = manifest
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-qa-dir",
        default="dataset/Stark/opd_mm_store_direct_3000/expansion_v3/direct_api_3000/qa",
    )
    parser.add_argument(
        "--expansion-dir",
        default="dataset/Stark/opd_mm_store_direct_3000/expansion_v3",
    )
    parser.add_argument("--records", default="dataset/Stark/opd_mm_store_direct_3000/records.jsonl")
    parser.add_argument(
        "--output-dir",
        default="dataset/Stark/opd_mm_store_direct_3000/expansion_v3/direct_api_3000_clean_qwen35_9b",
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-9b")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=180.0)
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--answer-max-tokens", type=int, default=256)
    parser.add_argument("--judge-max-tokens", type=int, default=128)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.set_defaults(resume=True)
    args = parser.parse_args()
    if args.workers <= 0 or args.batch_size <= 0:
        parser.error("--workers and --batch-size must be positive")
    summary = asyncio.run(_run(args))
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
