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

"""Map STARK message-level QA support to dialogue-round memory records."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.stark_expansion import finalize_expansion_dataset


SPLITS = ("train", "validation", "test")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _append_unique(values: list[str], *items: str | None) -> None:
    for item in items:
        if item and item not in values:
            values.append(item)


def remap_qa_splits(
    *,
    input_qa_dir: str | Path,
    round_records_path: str | Path,
    output_qa_dir: str | Path,
) -> dict[str, Any]:
    records = _read_jsonl(Path(round_records_path))
    records_by_id = {str(record["memory_id"]): record for record in records}
    source_to_text: dict[str, str] = {}
    source_to_image: dict[str, str] = {}
    for record in records:
        memory_id = str(record["memory_id"])
        metadata = record.get("metadata") or {}
        if record.get("modality") == "text":
            for source_memory_id in metadata.get("source_memory_ids") or []:
                source_to_text[str(source_memory_id)] = memory_id
        elif metadata.get("source_memory_id"):
            source_to_image[str(metadata["source_memory_id"])] = memory_id

    input_root = Path(input_qa_dir)
    output_root = Path(output_qa_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"splits": {}}
    for split in SPLITS:
        source_qas = _read_jsonl(input_root / f"{split}_qa.jsonl")
        remapped_qas: list[dict[str, Any]] = []
        image_support_rows = 0
        support_before = 0
        support_after = 0
        for source_qa in source_qas:
            qa = dict(source_qa)
            remapped_support: list[str] = []
            for source_memory_id in source_qa.get("support_memory_ids") or []:
                source_memory_id = str(source_memory_id)
                text_memory_id = source_to_text.get(source_memory_id)
                image_memory_id = source_to_image.get(source_memory_id)
                if text_memory_id is None:
                    raise ValueError(
                        f"{qa.get('sample_id')} support has no round mapping: {source_memory_id}"
                    )
                _append_unique(remapped_support, text_memory_id, image_memory_id)
                image_support_rows += int(image_memory_id is not None)

            question_image_source_id = str(
                (source_qa.get("raw_qa") or {}).get("question_image_memory_id") or ""
            )
            question_image_memory_id = (
                source_to_image.get(question_image_source_id) if question_image_source_id else None
            )
            if question_image_source_id and question_image_memory_id is None:
                raise ValueError(
                    f"{qa.get('sample_id')} question image has no round mapping: "
                    f"{question_image_source_id}"
                )

            support_records = [records_by_id[memory_id] for memory_id in remapped_support]
            qa["support_memory_ids"] = remapped_support
            qa["support_turn_ids"] = list(
                dict.fromkeys(str(record["turn_id"]) for record in support_records)
            )
            qa["session_id"] = list(
                dict.fromkeys(
                    str((record.get("metadata") or {}).get("session_id") or "")
                    for record in support_records
                )
            )
            qa["clue"] = list(
                dict.fromkeys(
                    str((record.get("metadata") or {}).get("local_turn_id") or "")
                    for record in support_records
                )
            )
            raw_qa = dict(source_qa.get("raw_qa") or {})
            raw_qa["source_support_memory_ids"] = list(
                source_qa.get("support_memory_ids") or []
            )
            raw_qa["support_memory_ids"] = remapped_support
            if question_image_source_id:
                raw_qa["source_question_image_memory_id"] = question_image_source_id
                raw_qa["question_image_memory_id"] = question_image_memory_id
            qa["raw_qa"] = raw_qa
            support_before += len(source_qa.get("support_memory_ids") or [])
            support_after += len(remapped_support)
            remapped_qas.append(qa)

        output_path = output_root / f"{split}_qa.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for qa in remapped_qas:
                handle.write(json.dumps(qa, ensure_ascii=False) + "\n")
        summary["splits"][split] = {
            "qa_count": len(remapped_qas),
            "support_count_before": support_before,
            "support_count_after": support_after,
            "mapped_image_support_count": image_support_rows,
            "qa_path": str(output_path.resolve()),
        }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-qa-dir",
        default="dataset/Stark/opd_mm_store_direct_3000/expansion_v3/direct_api_3000/qa",
    )
    parser.add_argument(
        "--round-records",
        default="dataset/Stark/opd_mm_store_rounds_3000/records.jsonl",
    )
    parser.add_argument(
        "--expansion-dir",
        default="dataset/Stark/opd_mm_store_rounds_3000/expansion_v3",
    )
    parser.add_argument(
        "--output-dir",
        default="dataset/Stark/opd_mm_store_rounds_3000/expansion_v3/direct_api_3000_rounds",
    )
    args = parser.parse_args()

    output = Path(args.output_dir)
    summary = remap_qa_splits(
        input_qa_dir=args.input_qa_dir,
        round_records_path=args.round_records,
        output_qa_dir=output / "qa",
    )
    manifest = finalize_expansion_dataset(
        expansion_dir=args.expansion_dir,
        records_path=args.round_records,
        qa_dir=output / "qa",
        output_dir=output,
    )
    summary["manifest"] = manifest
    (output / "remap_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
