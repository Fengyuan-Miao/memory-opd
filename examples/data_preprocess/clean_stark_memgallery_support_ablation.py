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

"""Keep STARK QAs that become unanswerable after removing their support.

Each non-AR question is answered from the complete memory of its episode after
all records belonging to annotated support turns are removed. Other memories
remain as distractors. A separate judge compares the generated answer with the
gold answer: an incorrect answer is retained because it indicates that the
removed support was necessary, while a correct answer is filtered as support
leakage or a redundant question. Source QA and memory files are never changed.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import aiohttp

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_ROOT = Path(__file__).resolve().parent
for path in (REPO_ROOT, SCRIPT_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from clean_stark_memgallery_qas import (  # noqa: E402
    SPLITS,
    _answer_messages,
    _cd_answer_from_relation,
    _chat,
    _endpoint,
    _judge_messages,
    _public_record,
    _read_jsonl,
)
from verl.experimental.opd_mm.outcome_reward import _parse_correct_with_recovery  # noqa: E402
from verl.experimental.opd_mm.stark_expansion import finalize_expansion_dataset  # noqa: E402


def _episode_sort_key(record: dict[str, Any]) -> tuple[Any, ...]:
    metadata = record.get("metadata") or {}
    return (
        str(metadata.get("session_date") or ""),
        str(metadata.get("session_id") or ""),
        int(metadata.get("turn_index") or 0),
        0 if str(record.get("modality") or "").lower() == "text" else 1,
        str(record.get("memory_id") or ""),
    )


def _coalesce_episode_records(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Represent a dialogue round once while retaining every linked image."""

    by_turn: dict[str, list[dict[str, Any]]] = defaultdict(list)
    order: list[str] = []
    for record in sorted(records, key=_episode_sort_key):
        turn_id = str(record.get("turn_id") or record.get("memory_id") or "")
        if turn_id not in by_turn:
            order.append(turn_id)
        by_turn[turn_id].append(record)

    output: list[dict[str, Any]] = []
    for turn_id in order:
        turn_records = by_turn[turn_id]
        text_record = next(
            (
                record
                for record in turn_records
                if str(record.get("modality") or "").lower() == "text"
            ),
            None,
        )
        if text_record is not None:
            output.append(text_record)
        image_records = [
            record
            for record in turn_records
            if str(record.get("modality") or "").lower() == "image"
        ]
        for image_index, record in enumerate(image_records):
            public_image_record = dict(record)
            if text_record is not None or image_index > 0:
                # The paired text memory already exposes the round dialogue.
                # Keep only the image payload and its public image ID here.
                public_image_record["content"] = ""
                public_image_record["summary"] = ""
            output.append(public_image_record)
    return output


def _request_image_side(qa: dict[str, Any], records: list[dict[str, Any]]) -> int:
    image_count = sum(
        bool(record.get("raw_pointer") and Path(str(record["raw_pointer"])).is_file())
        for record in records
    )
    question_image = qa.get("question_image") or qa.get("question_image_relative")
    image_count += bool(question_image and Path(str(question_image)).is_file())
    if image_count <= 1:
        return 672
    # Keep total visual pixels roughly bounded without removing any image.
    side = int(math.sqrt(1_800_000 / image_count))
    return max(224, min(560, side // 28 * 28))


def _ablation_system_instruction(qa: dict[str, Any]) -> tuple[str, str]:
    if str(qa.get("point") or "") == "CD":
        return (
            (
                "Classify the candidate claim against only the supplied episode memory. Choose CONSISTENT when "
                "every material part agrees with memory, CONTRADICTED when any material part is explicitly "
                "incompatible, or INSUFFICIENT_EVIDENCE when memory does not determine the relation. Treat record "
                "labels as describing the candidate claim itself: if the reason says the claim is contradicted, "
                "relation must be CONTRADICTED. Treat the YYYY-MM-DD prefix of each record timestamp, and any "
                "session date, as the event date for before/after claims. A reverse date order or a different value "
                "for the same event or attribute is a contradiction, not missing evidence. Combine records when "
                "the claim compares multiple events. Absence alone is not contradiction; do not guess."
            ),
            "cd_relation_citation_dual_verified_v7",
        )
    return (
        (
            "Answer the memory benchmark question from the supplied episode memory and question "
            "input. Treat timestamps, session dates, public image IDs, and images as evidence. "
            "If you cannot determine the answer, return exactly INSUFFICIENT_EVIDENCE."
        ),
        "general_support_ablation_v1",
    )


def _parse_cd_ablation_response(text: str) -> tuple[str, list[int], str]:
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
                raise ValueError(f"CD response is not recoverable JSON: {text[:200]}")
            payload = {
                "relation": relation_match.group(1),
                "evidence_numbers": re.findall(r"\d+", numbers_match.group(1)),
                "reason": "",
            }
    if not isinstance(payload, dict):
        raise ValueError("CD response must be a JSON object")

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


def _cd_citation_verifier_messages(
    qa: dict[str, Any],
    relation: str,
    evidence_numbers: list[int],
    remaining: list[dict[str, Any]],
    answer_reason: str,
) -> list[dict[str, str]]:
    citations = [
        {
            "memory_number": number,
            "record": _public_record(remaining[number - 1]),
        }
        for number in evidence_numbers
        if 1 <= number <= len(remaining)
    ]
    return [
        {
            "role": "system",
            "content": (
                "Verify whether the cited memory records explicitly establish the proposed semantic relation. "
                "CONSISTENT means the claim agrees with memory; CONTRADICTED means memory explicitly opposes it. "
                "Be strict. Reject support based only on absence, plausibility, or an unstated implication. A "
                "question inside a memory is not confirmation. Doing an activity does not prove a preference. A "
                "temporal relation requires explicit events and order or dates. Every material part must be "
                "established. Return only JSON: "
                '{"supported":true|false,"reason":"brief reason"}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{qa['question']}\n\nProposed relation:\n{relation}\n\n"
                f"Proposed reason:\n{answer_reason}\n\nCited records:\n"
                f"{json.dumps(citations, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]


def _parse_cd_citation_verdict(text: str) -> tuple[bool, str]:
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
                raise ValueError(f"CD citation verdict is not recoverable JSON: {text[:200]}")
            payload = {
                "supported": supported_match.group(1).casefold() == "true",
                "reason": "",
            }
    if not isinstance(payload, dict) or not isinstance(payload.get("supported"), bool):
        raise ValueError("CD citation verdict must contain a boolean supported field")
    return bool(payload["supported"]), str(payload.get("reason") or "").strip()


def _cd_ablation_gold_verifier_messages(
    qa: dict[str, Any],
    evidence_numbers: list[int],
    remaining: list[dict[str, Any]],
) -> list[dict[str, str]]:
    citations = [
        {
            "memory_number": number,
            "record": _public_record(remaining[number - 1]),
        }
        for number in evidence_numbers
        if 1 <= number <= len(remaining)
    ]
    return [
        {
            "role": "system",
            "content": (
                "Decide whether the cited non-support memory records alone explicitly make the gold answer correct "
                "for the literal question. Be conservative: every material entity, event, attribute, qualifier, and "
                "comparison must refer to the same proposition. A related fact is insufficient; an event's location "
                "does not establish an item's origin, overall sentiment does not establish a specific sub-event, and "
                "two different but compatible activities do not exclude each other. Absence is not contradiction. "
                "A question or suggestion inside memory is not a factual answer. Timestamps may explicitly establish "
                "event order. Return only JSON: "
                '{"supported":true|false,"reason":"brief reason"}.'
            ),
        },
        {
            "role": "user",
            "content": (
                f"Question:\n{qa['question']}\n\nGold answer:\n"
                f"{qa.get('gold_answer') or qa.get('answer')}\n\nCited non-support records:\n"
                f"{json.dumps(citations, ensure_ascii=False, separators=(',', ':'))}"
            ),
        },
    ]


def _ablation_records(
    qa: dict[str, Any],
    episode_records: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    records_by_id = {str(record["memory_id"]): record for record in episode_records}
    support_ids = [str(value) for value in qa.get("support_memory_ids") or []]
    missing = [memory_id for memory_id in support_ids if memory_id not in records_by_id]
    if missing:
        raise ValueError(f"missing support records: {missing[:3]}")

    # Linked text/image memories share public round content. Removing the whole
    # turn prevents a paired image record from leaking an ablated text support
    # (and vice versa).
    support_turn_ids = {str(records_by_id[memory_id].get("turn_id") or "") for memory_id in support_ids}
    removed_ids = [
        str(record["memory_id"])
        for record in episode_records
        if str(record.get("turn_id") or "") in support_turn_ids
    ]
    remaining = [
        record
        for record in episode_records
        if str(record.get("turn_id") or "") not in support_turn_ids
    ]
    return _coalesce_episode_records(remaining), support_ids, removed_ids


async def _evaluate_one(
    qa: dict[str, Any],
    records_by_episode: dict[str, list[dict[str, Any]]],
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
            "answer_correct_without_support": None,
            "candidate_answer": "",
            "judge_reason": "AR/no-support sample is outside support-ablation cleaning",
        }

    scenario = str(qa.get("scenario") or "")
    episode_records = records_by_episode.get(scenario)
    if not episode_records:
        return {
            "sample_id": sample_id,
            "point": qa.get("point"),
            "status": "error",
            "keep": None,
            "error": f"unknown episode: {scenario}",
        }
    try:
        remaining, annotated_support_ids, removed_ids = _ablation_records(qa, episode_records)
    except ValueError as exc:
        return {
            "sample_id": sample_id,
            "point": qa.get("point"),
            "status": "error",
            "keep": None,
            "error": str(exc),
        }

    max_image_side = _request_image_side(qa, remaining)
    system_instruction, prompt_version = _ablation_system_instruction(qa)
    is_cd = str(qa.get("point") or "") == "CD"
    output_rule_override = None
    if is_cd:
        output_rule_override = (
            "Return only one JSON object with this schema: "
            '{"relation":"CONSISTENT|CONTRADICTED|INSUFFICIENT_EVIDENCE",'
            '"evidence_numbers":[1,2],"reason":"brief evidence-based reason"}. '
            "For CONSISTENT or CONTRADICTED, evidence_numbers must cite every numbered Memory record needed. "
            "For INSUFFICIENT_EVIDENCE, use an empty list."
        )
    distractor_image_count = sum(
        bool(record.get("raw_pointer") and Path(str(record["raw_pointer"])).is_file())
        for record in remaining
    )
    async with semaphore:
        try:
            candidate_raw = await _chat(
                session,
                url=_endpoint(args.base_url),
                model=args.model,
                messages=_answer_messages(
                    qa,
                    remaining,
                    evidence_description="Episode memory records follow in chronological order.",
                    record_label="Memory",
                    system_instruction=system_instruction,
                    output_rule_override=output_rule_override,
                    max_image_side=max_image_side,
                ),
                max_tokens=args.answer_max_tokens,
                retries=args.retries,
            )
            evidence_numbers: list[int] = []
            answer_reason = ""
            evidence_backed = True
            evidence_relation = ""
            if is_cd:
                evidence_relation, evidence_numbers, answer_reason = _parse_cd_ablation_response(candidate_raw)
                candidate = _cd_answer_from_relation(str(qa.get("question") or ""), evidence_relation)
                evidence_backed = (
                    evidence_relation in {"CONSISTENT", "CONTRADICTED"}
                    and bool(evidence_numbers)
                    and all(1 <= number <= len(remaining) for number in evidence_numbers)
                )
            else:
                candidate = candidate_raw
            citation_supported = True
            citation_judge_raw = ""
            citation_judge_reason = ""
            if is_cd:
                citation_supported = False
                if evidence_backed:
                    citation_judge_raw = await _chat(
                        session,
                        url=_endpoint(args.base_url),
                        model=args.model,
                        messages=_cd_citation_verifier_messages(
                            qa,
                            evidence_relation,
                            evidence_numbers,
                            remaining,
                            answer_reason,
                        ),
                        max_tokens=args.judge_max_tokens,
                        retries=args.retries,
                    )
                    citation_supported, citation_judge_reason = _parse_cd_citation_verdict(
                        citation_judge_raw
                    )
            judge_raw = await _chat(
                session,
                url=_endpoint(args.base_url),
                model=args.model,
                messages=_judge_messages(qa, candidate),
                max_tokens=args.judge_max_tokens,
                retries=args.retries,
            )
            correct, reason, recovered = _parse_correct_with_recovery(judge_raw)
            preliminary_answerable = correct and (
                not is_cd or (evidence_backed and citation_supported)
            )
            ablation_gold_supported = True
            ablation_gold_verifier_raw = ""
            ablation_gold_verifier_reason = ""
            if is_cd and preliminary_answerable:
                ablation_gold_verifier_raw = await _chat(
                    session,
                    url=_endpoint(args.base_url),
                    model=args.model,
                    messages=_cd_ablation_gold_verifier_messages(
                        qa,
                        evidence_numbers,
                        remaining,
                    ),
                    max_tokens=args.judge_max_tokens,
                    retries=args.retries,
                )
                (
                    ablation_gold_supported,
                    ablation_gold_verifier_reason,
                ) = _parse_cd_citation_verdict(ablation_gold_verifier_raw)
            answerable_without_support = preliminary_answerable and (
                not is_cd or ablation_gold_supported
            )
            return {
                "sample_id": sample_id,
                "point": qa.get("point"),
                "status": "evaluated",
                # The ablation cleaner intentionally inverts the judge result.
                "keep": not answerable_without_support,
                "answer_correct_without_support": answerable_without_support,
                "candidate_answer": candidate,
                "candidate_raw": candidate_raw,
                "evidence_relation": evidence_relation,
                "answer_reason": answer_reason,
                "evidence_numbers": evidence_numbers,
                "evidence_backed": evidence_backed if is_cd else None,
                "citation_supported": citation_supported if is_cd else None,
                "citation_judge_reason": citation_judge_reason,
                "citation_judge_raw": citation_judge_raw,
                "ablation_gold_supported": ablation_gold_supported if is_cd else None,
                "ablation_gold_verifier_reason": ablation_gold_verifier_reason,
                "ablation_gold_verifier_raw": ablation_gold_verifier_raw,
                "judge_reason": reason,
                "judge_parse_recovered": recovered,
                "judge_raw": judge_raw,
                "prompt_version": prompt_version,
                "annotated_support_count": len(annotated_support_ids),
                "removed_memory_count": len(removed_ids),
                "removed_memory_ids": removed_ids,
                "remaining_memory_count": len(remaining),
                "remaining_image_count": distractor_image_count,
                "request_image_max_side": max_image_side,
            }
        except Exception as exc:
            return {
                "sample_id": sample_id,
                "point": qa.get("point"),
                "status": "error",
                "keep": None,
                "error": f"{type(exc).__name__}: {exc}"[:1000],
                "prompt_version": prompt_version,
                "annotated_support_count": len(annotated_support_ids),
                "removed_memory_count": len(removed_ids),
                "remaining_memory_count": len(remaining),
                "remaining_image_count": distractor_image_count,
                "request_image_max_side": max_image_side,
            }


async def _run(args: argparse.Namespace) -> dict[str, Any]:
    records = _read_jsonl(Path(args.records))
    records_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        scenario = str((record.get("metadata") or {}).get("scenario") or "")
        records_by_episode[scenario].append(record)

    input_dir = Path(args.input_qa_dir)
    output_dir = Path(args.output_dir)
    audit_dir = output_dir / "audit"
    qa_dir = output_dir / "qa"
    audit_dir.mkdir(parents=True, exist_ok=True)
    qa_dir.mkdir(parents=True, exist_ok=True)

    timeout = aiohttp.ClientTimeout(total=float(args.timeout))
    connector = aiohttp.TCPConnector(limit=max(args.workers * 2, 8))
    semaphore = asyncio.Semaphore(args.workers)
    summary: dict[str, Any] = {
        "cleaning": "support_turn_ablation_with_full_episode_distractors",
        "model": args.model,
        "base_url": args.base_url,
        "splits": {},
    }

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
                            _evaluate_one(qa, records_by_episode, session, semaphore, args)
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
                    leaked = sum(row.get("answer_correct_without_support") is True for row in previous.values())
                    print(
                        f"[{split}] {completed}/{len(qas)} kept={kept} answerable_without_support={leaked} "
                        f"evaluated={counts['evaluated']} errors={counts['error']}",
                        flush=True,
                    )

            audit_rows = list(previous.values())
            error_rows = [row for row in audit_rows if row.get("status") == "error"]
            keep_ids = {str(row["sample_id"]) for row in audit_rows if row.get("keep") is True}
            cleaned = [qa for qa in qas if str(qa["sample_id"]) in keep_ids]
            with (qa_dir / f"{split}_qa.jsonl").open("w", encoding="utf-8") as handle:
                for qa in cleaned:
                    handle.write(json.dumps(qa, ensure_ascii=False) + "\n")

            by_point: dict[str, dict[str, int]] = defaultdict(
                lambda: {
                    "input": 0,
                    "kept_support_dependent": 0,
                    "exempt_no_support": 0,
                    "dropped_answerable_without_support": 0,
                    "errors": 0,
                }
            )
            verdicts = {str(row["sample_id"]): row for row in audit_rows}
            for qa in qas:
                point = str(qa.get("point") or "")
                by_point[point]["input"] += 1
                verdict = verdicts.get(str(qa["sample_id"])) or {}
                if verdict.get("status") == "exempt_no_support":
                    by_point[point]["exempt_no_support"] += 1
                elif verdict.get("keep") is True:
                    by_point[point]["kept_support_dependent"] += 1
                elif verdict.get("keep") is False:
                    by_point[point]["dropped_answerable_without_support"] += 1
                elif verdict.get("status") == "error":
                    by_point[point]["errors"] += 1
            exempt_count = sum(row.get("status") == "exempt_no_support" for row in audit_rows)
            summary["splits"][split] = {
                "input": len(qas),
                "kept_total": len(cleaned),
                "kept_support_dependent": sum(
                    row.get("status") == "evaluated" and row.get("keep") is True
                    for row in audit_rows
                ),
                "exempt_no_support": exempt_count,
                "dropped_answerable_without_support": sum(
                    row.get("keep") is False for row in audit_rows
                ),
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
        default=(
            "dataset/Stark/opd_mm_store_rounds_3000/expansion_v3/"
            "direct_api_3000_rounds_clean_qwen35_9b/qa"
        ),
    )
    parser.add_argument(
        "--expansion-dir",
        default="dataset/Stark/opd_mm_store_rounds_3000/expansion_v3",
    )
    parser.add_argument("--records", default="dataset/Stark/opd_mm_store_rounds_3000/records.jsonl")
    parser.add_argument(
        "--output-dir",
        default=(
            "dataset/Stark/opd_mm_store_rounds_3000/expansion_v3/"
            "direct_api_3000_rounds_clean_qwen35_9b_support_ablation"
        ),
    )
    parser.add_argument("--base-url", default="http://127.0.0.1:8000/v1")
    parser.add_argument("--model", default="qwen35-9b")
    parser.add_argument("--splits", nargs="+", choices=SPLITS, default=list(SPLITS))
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--timeout", type=float, default=240.0)
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
