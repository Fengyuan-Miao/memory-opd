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

"""Merge STARK QA directories and reassign rows to an episode-level split."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


SPLITS = ("train", "validation", "test")
AVAILABLE_WORD = re.compile(r"\bavailable\b\s*", re.IGNORECASE)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def merge_qa_directories(
    *,
    expansion_dir: str | Path,
    qa_dirs: list[str | Path],
    output_dir: str | Path,
) -> dict[str, Any]:
    expansion = Path(expansion_dir)
    output = Path(output_dir)
    qa_output = output / "qa"
    qa_output.mkdir(parents=True, exist_ok=True)
    split_by_episode: dict[str, str] = {}
    for split in SPLITS:
        ids = (expansion / "splits" / f"{split}_episode_ids.txt").read_text(
            encoding="utf-8"
        ).splitlines()
        for conversation_id in ids:
            if conversation_id in split_by_episode:
                raise ValueError(f"episode appears in multiple splits: {conversation_id}")
            split_by_episode[conversation_id] = split

    rows_by_split: dict[str, list[dict[str, Any]]] = {split: [] for split in SPLITS}
    seen_sample_ids: set[str] = set()
    seen_episode_questions: set[tuple[str, str]] = set()
    source_counts: Counter[str] = Counter()
    point_counts: Counter[str] = Counter()
    for qa_dir_value in qa_dirs:
        qa_dir = Path(qa_dir_value)
        for source_split in SPLITS:
            for row in _read_jsonl(qa_dir / f"{source_split}_qa.jsonl"):
                sample_id = str(row.get("sample_id") or "")
                conversation_id = str(row.get("scenario") or "")
                destination_split = split_by_episode.get(conversation_id)
                if destination_split is None:
                    raise ValueError(f"QA references an episode outside the expansion: {sample_id}")
                if not sample_id or sample_id in seen_sample_ids:
                    raise ValueError(f"duplicate or empty sample ID: {sample_id}")
                copied = dict(row)
                if copied.get("point") in {"VS", "VR"}:
                    copied["question"] = " ".join(
                        AVAILABLE_WORD.sub("", str(copied.get("question") or "")).split()
                    )
                question_key = re.sub(
                    r"[^a-z0-9]+", " ", str(copied.get("question") or "").casefold()
                ).strip()
                episode_question = (conversation_id, question_key)
                if not question_key or episode_question in seen_episode_questions:
                    raise ValueError(f"duplicate or empty episode question: {sample_id}")
                seen_sample_ids.add(sample_id)
                seen_episode_questions.add(episode_question)
                rows_by_split[destination_split].append(copied)
                source_counts[str(qa_dir.resolve())] += 1
                point_counts[str(copied.get("point") or "")] += 1

    split_counts = {}
    for split, rows in rows_by_split.items():
        rows.sort(key=lambda row: str(row["sample_id"]))
        path = qa_output / f"{split}_qa.jsonl"
        path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        split_counts[split] = len(rows)
    summary = {
        "qa_count": sum(split_counts.values()),
        "split_counts": split_counts,
        "point_counts": dict(sorted(point_counts.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "qa_dir": str(qa_output.resolve()),
    }
    (output / "merge_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-dir", required=True)
    parser.add_argument("--qa-dir", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    summary = merge_qa_directories(
        expansion_dir=args.expansion_dir,
        qa_dirs=args.qa_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
