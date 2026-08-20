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

"""Drop STARK QAs that reference records removed from the memory store."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def filter_qa_directory(
    *,
    records_path: Path,
    qa_dir: Path,
    write: bool,
) -> dict[str, Any]:
    record_ids = {str(record["memory_id"]) for record in _read_jsonl(records_path)}
    totals: Counter[str] = Counter()
    splits: dict[str, Any] = {}
    for qa_path in sorted(qa_dir.glob("*_qa.jsonl")):
        rows = _read_jsonl(qa_path)
        kept: list[dict[str, Any]] = []
        counts: Counter[str] = Counter()
        for qa in rows:
            counts["input"] += 1
            support_ids = [str(value) for value in qa.get("support_memory_ids") or []]
            question_image_id = str(
                (qa.get("raw_qa") or {}).get("question_image_memory_id") or ""
            )
            missing_support = [value for value in support_ids if value not in record_ids]
            if missing_support:
                counts["dropped_missing_support"] += 1
                continue
            if question_image_id and question_image_id not in record_ids:
                counts["dropped_missing_question_image"] += 1
                continue
            kept.append(qa)
            counts["kept"] += 1
        if write:
            temporary = qa_path.with_suffix(qa_path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as handle:
                for qa in kept:
                    handle.write(json.dumps(qa, ensure_ascii=False) + "\n")
            temporary.replace(qa_path)
        splits[qa_path.stem.removesuffix("_qa")] = dict(counts)
        totals.update(counts)
    return {
        "records_path": str(records_path.resolve()),
        "qa_dir": str(qa_dir.resolve()),
        "write": write,
        "splits": splits,
        "totals": dict(totals),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--qa-dir", type=Path, required=True)
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--summary", type=Path)
    args = parser.parse_args()
    summary = filter_qa_directory(
        records_path=args.records,
        qa_dir=args.qa_dir,
        write=args.write,
    )
    if args.summary is not None:
        args.summary.parent.mkdir(parents=True, exist_ok=True)
        args.summary.write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
