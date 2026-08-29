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

"""Build a Mem-Gallery training split from teacher-successful rollouts.

The teacher result files must cover every candidate QA exactly once. A sample
is retained only when the terminal answer judge reports ``judge_correct=true``.
Fixed held-out IDs are excluded before coverage is checked, so evaluation
examples cannot silently leak into the generated training split.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from examples.data_preprocess.build_mem_gallery_opd_mm_train_subset import _samples_for_qas
from verl.experimental.opd_mm.dataset import write_opd_rlhf_jsonl, write_opd_rlhf_parquet
from verl.experimental.opd_mm.mem_gallery import load_mem_gallery_qas, qas_to_jsonl


def _read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def _read_ids(path: str | Path | None) -> set[str]:
    if not path:
        return set()
    return {
        line.strip()
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }


def _counts_by_point(rows: Iterable[dict[str, Any]]) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get("point") or "") for row in rows).items()))


def build_teacher_success_subset(
    *,
    dataset_root: str | Path,
    teacher_results: list[str | Path],
    output_dir: str | Path,
    heldout_sample_ids: str | Path | None = None,
    data_source: str = "opd_mm",
    agent_name: str = "tool_agent",
    dataset_name: str = "mem_gallery",
) -> dict[str, Any]:
    qas = load_mem_gallery_qas(dataset_root, dataset_name=dataset_name)
    heldout_ids = _read_ids(heldout_sample_ids)
    candidates = [qa for qa in qas if str(qa["sample_id"]) not in heldout_ids]
    candidate_by_id = {str(qa["sample_id"]): qa for qa in candidates}

    result_by_id: dict[str, dict[str, Any]] = {}
    duplicate_ids: set[str] = set()
    for result_path in teacher_results:
        for row in _read_jsonl(result_path):
            sample_id = str(row.get("sample_id") or "")
            if not sample_id:
                raise ValueError(f"teacher result without sample_id in {result_path}")
            if sample_id in result_by_id:
                duplicate_ids.add(sample_id)
            result_by_id[sample_id] = row
    if duplicate_ids:
        raise ValueError(f"duplicate teacher results: {sorted(duplicate_ids)[:5]}")

    unexpected_ids = sorted(set(result_by_id) - set(candidate_by_id) - heldout_ids)
    if unexpected_ids:
        raise ValueError(f"teacher results contain unknown sample IDs: {unexpected_ids[:5]}")
    leaked_ids = sorted(set(result_by_id) & heldout_ids)
    if leaked_ids:
        raise ValueError(f"teacher results contain held-out sample IDs: {leaked_ids[:5]}")
    missing_ids = sorted(set(candidate_by_id) - set(result_by_id))
    if missing_ids:
        raise ValueError(
            f"teacher results cover {len(result_by_id)}/{len(candidate_by_id)} candidates; "
            f"missing examples include {missing_ids[:5]}"
        )

    successful_ids = {
        sample_id
        for sample_id, row in result_by_id.items()
        if row.get("judge_correct") is True
        and not row.get("rollout_error")
        and not row.get("answer_error")
        and not row.get("judge_error")
    }
    successful_qas = [qa for qa in candidates if str(qa["sample_id"]) in successful_ids]
    rejected_qas = [qa for qa in candidates if str(qa["sample_id"]) not in successful_ids]

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    qas_to_jsonl(successful_qas, output / "train_qas.jsonl")
    (output / "train_sample_ids.txt").write_text(
        "\n".join(str(qa["sample_id"]) for qa in successful_qas) + "\n",
        encoding="utf-8",
    )
    (output / "rejected_sample_ids.txt").write_text(
        "\n".join(str(qa["sample_id"]) for qa in rejected_qas) + "\n",
        encoding="utf-8",
    )

    samples = _samples_for_qas(
        successful_qas,
        dataset_root=dataset_root,
        data_source=data_source,
        agent_name=agent_name,
        dataset_name=dataset_name,
    )
    write_opd_rlhf_jsonl(samples, output / "train.jsonl", data_source=data_source, agent_name=agent_name)
    write_opd_rlhf_parquet(samples, output / "train.parquet", data_source=data_source, agent_name=agent_name)

    candidate_counts = _counts_by_point(candidates)
    successful_counts = _counts_by_point(successful_qas)
    manifest = {
        "dataset": dataset_name,
        "selection_policy": "teacher_terminal_judge_correct",
        "dataset_root": str(Path(dataset_root).resolve()),
        "heldout_sample_ids": str(Path(heldout_sample_ids).resolve()) if heldout_sample_ids else None,
        "teacher_results": [str(Path(path).resolve()) for path in teacher_results],
        "candidate_count": len(candidates),
        "heldout_count": len(heldout_ids),
        "teacher_success_count": len(successful_qas),
        "teacher_failure_count": len(rejected_qas),
        "teacher_success_rate": len(successful_qas) / len(candidates) if candidates else 0.0,
        "candidate_counts_by_point": candidate_counts,
        "teacher_success_counts_by_point": successful_counts,
        "teacher_success_rates_by_point": {
            point: successful_counts.get(point, 0) / count
            for point, count in candidate_counts.items()
        },
        "files": {
            "train_qas_jsonl": str((output / "train_qas.jsonl").resolve()),
            "train_sample_ids": str((output / "train_sample_ids.txt").resolve()),
            "rejected_sample_ids": str((output / "rejected_sample_ids.txt").resolve()),
            "train_jsonl": str((output / "train.jsonl").resolve()),
            "train_parquet": str((output / "train.parquet").resolve()),
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset/mem_gallery")
    parser.add_argument("--teacher-results", nargs="+", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--heldout-sample-ids")
    parser.add_argument("--data-source", default="opd_mm")
    parser.add_argument("--agent-name", default="tool_agent")
    parser.add_argument("--dataset-name", default="mem_gallery")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    manifest = build_teacher_success_subset(
        dataset_root=args.dataset_root,
        teacher_results=args.teacher_results,
        output_dir=args.output_dir,
        heldout_sample_ids=args.heldout_sample_ids,
        data_source=args.data_source,
        agent_name=args.agent_name,
        dataset_name=args.dataset_name,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
