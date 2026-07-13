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

"""Build a balanced Mem-Gallery OPD-MM training subset.

The full Mem-Gallery QA split is uneven across scenarios and QA categories.
This script builds a small, reproducible training subset by sampling up to
``per_cell_cap`` QAs from every non-empty ``(scenario, point)`` cell. The output
contains both an inspectable QA JSONL and verl-ready RLHF rows with hidden
scenario memories in ``extra_info.tools_kwargs``.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.dataset import (
    DEFAULT_AGENT_NAME,
    DEFAULT_DATA_SOURCE,
    write_opd_rlhf_jsonl,
    write_opd_rlhf_parquet,
)
from verl.experimental.opd_mm.mem_gallery import (
    load_mem_gallery_qas,
    load_mem_gallery_records,
    qas_to_jsonl,
)
from verl.experimental.opd_mm.models import OPDSample
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore


DEFAULT_OUTPUT_DIR = "dataset/mem_gallery/opd_mm_store/subsets/balanced_train_cap2"


def _group_qas(qas: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for qa in qas:
        groups[(str(qa.get("scenario") or ""), str(qa.get("point") or ""))].append(qa)
    return groups


def _sample_group(
    rows: list[dict[str, Any]],
    *,
    cap: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    """Sample one cell, preferring image/text diversity when available."""
    if len(rows) <= cap:
        return sorted(rows, key=lambda qa: qa["sample_id"])

    candidates = sorted(rows, key=lambda qa: qa["sample_id"])
    rng.shuffle(candidates)
    selected: list[dict[str, Any]] = []
    selected_ids: set[str] = set()

    def take_first(predicate) -> None:
        if len(selected) >= cap:
            return
        for qa in candidates:
            sample_id = str(qa["sample_id"])
            if sample_id not in selected_ids and predicate(qa):
                selected.append(qa)
                selected_ids.add(sample_id)
                return

    take_first(lambda qa: bool(qa.get("question_image")))
    take_first(lambda qa: not bool(qa.get("question_image")))
    for qa in candidates:
        if len(selected) >= cap:
            break
        sample_id = str(qa["sample_id"])
        if sample_id not in selected_ids:
            selected.append(qa)
            selected_ids.add(sample_id)

    return sorted(selected, key=lambda qa: qa["sample_id"])


def stratified_mem_gallery_subset(
    qas: list[dict[str, Any]],
    *,
    per_cell_cap: int = 2,
    seed: int = 20260705,
) -> list[dict[str, Any]]:
    """Return a scenario/category balanced Mem-Gallery subset."""
    if per_cell_cap <= 0:
        raise ValueError("per_cell_cap must be positive")
    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for key, rows in sorted(_group_qas(qas).items()):
        del key
        selected.extend(_sample_group(rows, cap=per_cell_cap, rng=rng))
    return sorted(selected, key=lambda qa: (qa["scenario"], qa["point"], qa["sample_id"]))


def stratified_holdout_subset(
    qas: list[dict[str, Any]],
    *,
    max_samples: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Match the fixed scenario/category-balanced OPD-MM evaluation sampling."""
    if max_samples <= 0:
        return []
    if max_samples >= len(qas):
        return sorted(qas, key=lambda qa: (qa["scenario"], qa["point"], qa["sample_id"]))

    rng = random.Random(seed)
    groups = _group_qas(qas)
    selected: list[dict[str, Any]] = []
    keys = sorted(groups)
    rng.shuffle(keys)
    while len(selected) < max_samples and keys:
        next_keys = []
        for key in keys:
            rows = groups[key]
            if not rows:
                continue
            rows.sort(key=lambda qa: str(qa["sample_id"]))
            selected.append(rows.pop(rng.randrange(len(rows))))
            if rows:
                next_keys.append(key)
            if len(selected) >= max_samples:
                break
        keys = next_keys
    return sorted(selected, key=lambda qa: (qa["scenario"], qa["point"], qa["sample_id"]))


def extend_stratified_mem_gallery_subset(
    qas: list[dict[str, Any]],
    *,
    base_sample_ids: set[str],
    excluded_sample_ids: set[str],
    per_cell_cap: int,
    seed: int,
) -> list[dict[str, Any]]:
    """Retain a base subset and fill each cell without sampling excluded QAs."""
    if per_cell_cap <= 0:
        raise ValueError("per_cell_cap must be positive")
    available_ids = {str(qa["sample_id"]) for qa in qas}
    missing_base = sorted(base_sample_ids - available_ids)
    if missing_base:
        raise ValueError(f"base sample IDs are missing from Mem-Gallery: {missing_base[:3]}")
    overlap = base_sample_ids & excluded_sample_ids
    if overlap:
        raise ValueError(f"base and excluded sample IDs overlap: {sorted(overlap)[:3]}")

    rng = random.Random(seed)
    selected: list[dict[str, Any]] = []
    for _, rows in sorted(_group_qas(qas).items()):
        base_rows = [qa for qa in rows if str(qa["sample_id"]) in base_sample_ids]
        candidates = [
            qa
            for qa in rows
            if str(qa["sample_id"]) not in base_sample_ids
            and str(qa["sample_id"]) not in excluded_sample_ids
        ]
        selected.extend(base_rows)
        remaining = max(0, per_cell_cap - len(base_rows))
        if remaining:
            selected.extend(_sample_group(candidates, cap=remaining, rng=rng))
    return sorted(selected, key=lambda qa: (qa["scenario"], qa["point"], qa["sample_id"]))


def _counter_by(rows: Iterable[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row.get(key) or "") for row in rows).items()))


def _scenario_point_counts(rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    counter = Counter((str(row.get("scenario") or ""), str(row.get("point") or "")) for row in rows)
    return [
        {"scenario": scenario, "point": point, "count": count}
        for (scenario, point), count in sorted(counter.items())
    ]


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    support_lengths = [len(row.get("support_turn_ids") or []) for row in rows]
    return {
        "count": len(rows),
        "scenario_count": len({row.get("scenario") for row in rows}),
        "point_count": len({row.get("point") for row in rows}),
        "question_image_count": sum(bool(row.get("question_image")) for row in rows),
        "no_support_count": sum(not row.get("support_turn_ids") for row in rows),
        "avg_support_turns": (
            round(sum(support_lengths) / len(support_lengths), 4)
            if support_lengths
            else 0.0
        ),
        "counts_by_point": _counter_by(rows, "point"),
        "counts_by_scenario": _counter_by(rows, "scenario"),
        "counts_by_scenario_point": _scenario_point_counts(rows),
    }


def _write_json(path: str | Path, value: dict[str, Any]) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return output


def _samples_for_qas(
    qas: list[dict[str, Any]],
    *,
    dataset_root: str | Path,
    data_source: str,
    agent_name: str,
) -> list[OPDSample]:
    records = load_mem_gallery_records(dataset_root)
    records_by_scenario: dict[str, list[Any]] = defaultdict(list)
    for record in records:
        records_by_scenario[str(record.metadata.get("scenario") or "")].append(record)

    samples = []
    for index, qa in enumerate(qas):
        scenario = str(qa.get("scenario") or "")
        metadata = {
            "index": index,
            "data_source": data_source,
            "agent_name": agent_name,
            "opd_mm_online_self_distill": True,
            "scenario": scenario,
            "point": qa.get("point"),
            "qa_index": qa.get("qa_index"),
            "question_image": qa.get("question_image"),
            "question_image_relative": qa.get("question_image_relative"),
            "extra_info": {
                "mem_gallery_sample_id": qa.get("sample_id"),
                "scenario": scenario,
                "point": qa.get("point"),
                "qa_index": qa.get("qa_index"),
                "support_turn_ids": qa.get("support_turn_ids") or [],
                "clue": qa.get("clue") or [],
                "question_image": qa.get("question_image"),
                "question_image_relative": qa.get("question_image_relative"),
            },
        }
        samples.append(
            OPDSample(
                sample_id=str(qa["sample_id"]),
                query=str(qa.get("question") or ""),
                gold_answer=str(qa.get("gold_answer") or qa.get("answer") or ""),
                memory_store=HiddenMemoryStore(records_by_scenario[scenario]),
                metadata=metadata,
            )
        )
    return samples


def build_subset(
    *,
    dataset_root: str | Path,
    output_dir: str | Path,
    per_cell_cap: int,
    seed: int,
    data_source: str,
    agent_name: str,
    write_rlhf: bool = True,
    base_sample_ids: Iterable[str] | None = None,
    excluded_sample_ids: Iterable[str] | None = None,
    reserve_eval_samples: int = 0,
    reserve_eval_seed: int = 20260705,
) -> dict[str, Any]:
    qas = load_mem_gallery_qas(dataset_root)
    base_ids = {str(sample_id).strip() for sample_id in (base_sample_ids or []) if str(sample_id).strip()}
    excluded_ids = {
        str(sample_id).strip() for sample_id in (excluded_sample_ids or []) if str(sample_id).strip()
    }
    if base_ids & excluded_ids:
        raise ValueError("base_sample_ids and excluded_sample_ids must be disjoint")

    reserve_candidates = [
        qa
        for qa in qas
        if str(qa["sample_id"]) not in base_ids and str(qa["sample_id"]) not in excluded_ids
    ]
    reserved_eval = stratified_holdout_subset(
        reserve_candidates,
        max_samples=reserve_eval_samples,
        seed=reserve_eval_seed,
    )
    excluded_ids.update(str(qa["sample_id"]) for qa in reserved_eval)
    heldout = [qa for qa in qas if str(qa["sample_id"]) in excluded_ids]

    if base_ids or excluded_ids:
        selected = extend_stratified_mem_gallery_subset(
            qas,
            base_sample_ids=base_ids,
            excluded_sample_ids=excluded_ids,
            per_cell_cap=per_cell_cap,
            seed=seed,
        )
    else:
        selected = stratified_mem_gallery_subset(
            qas,
            per_cell_cap=per_cell_cap,
            seed=seed,
        )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    qas_to_jsonl(selected, output / "train_qas.jsonl")
    (output / "train_sample_ids.txt").write_text(
        "\n".join(str(qa["sample_id"]) for qa in selected) + "\n",
        encoding="utf-8",
    )

    files = {
        "train_qas_jsonl": str((output / "train_qas.jsonl").resolve()),
        "train_sample_ids": str((output / "train_sample_ids.txt").resolve()),
    }
    if base_ids:
        (output / "base_sample_ids.txt").write_text("\n".join(sorted(base_ids)) + "\n", encoding="utf-8")
        files["base_sample_ids"] = str((output / "base_sample_ids.txt").resolve())
    if heldout:
        qas_to_jsonl(heldout, output / "heldout_qas.jsonl")
        (output / "heldout_sample_ids.txt").write_text(
            "\n".join(sorted(str(qa["sample_id"]) for qa in heldout)) + "\n",
            encoding="utf-8",
        )
        files["heldout_qas_jsonl"] = str((output / "heldout_qas.jsonl").resolve())
        files["heldout_sample_ids"] = str((output / "heldout_sample_ids.txt").resolve())
    if write_rlhf:
        samples = _samples_for_qas(
            selected,
            dataset_root=dataset_root,
            data_source=data_source,
            agent_name=agent_name,
        )
        write_opd_rlhf_jsonl(samples, output / "train.jsonl", data_source=data_source, agent_name=agent_name)
        write_opd_rlhf_parquet(samples, output / "train.parquet", data_source=data_source, agent_name=agent_name)
        files.update(
            {
                "train_jsonl": str((output / "train.jsonl").resolve()),
                "train_parquet": str((output / "train.parquet").resolve()),
            }
        )

    manifest = {
        "dataset": "mem_gallery",
        "dataset_root": str(Path(dataset_root).resolve()),
        "output_dir": str(output.resolve()),
        "selection_policy": {
            "type": "scenario_point_stratified_extension" if base_ids else "scenario_point_stratified",
            "per_cell_cap": per_cell_cap,
            "seed": seed,
            "non_empty_cells": len(_group_qas(qas)),
            "image_text_diversity_preference": True,
            "base_sample_count": len(base_ids),
            "excluded_sample_count": len(excluded_ids),
            "reserve_eval_samples": reserve_eval_samples,
            "reserve_eval_seed": reserve_eval_seed,
        },
        "full": _summary(qas),
        "train": _summary(selected),
        "heldout": _summary(heldout) if heldout else None,
        "files": files,
    }
    _write_json(output / "manifest.json", manifest)
    return manifest


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="dataset/mem_gallery")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--per-cell-cap", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260705)
    parser.add_argument("--data-source", default=DEFAULT_DATA_SOURCE)
    parser.add_argument("--agent-name", default=DEFAULT_AGENT_NAME)
    parser.add_argument("--base-sample-ids", default="")
    parser.add_argument("--exclude-sample-ids", default="")
    parser.add_argument("--reserve-eval-samples", type=int, default=0)
    parser.add_argument("--reserve-eval-seed", type=int, default=20260705)
    parser.add_argument("--skip-rlhf", action="store_true")
    return parser.parse_args()


def _read_sample_ids(path: str) -> set[str]:
    if not path:
        return set()
    return {line.strip() for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()}


def main() -> None:
    args = _parse_args()
    manifest = build_subset(
        dataset_root=args.dataset_root,
        output_dir=args.output_dir,
        per_cell_cap=args.per_cell_cap,
        seed=args.seed,
        data_source=args.data_source,
        agent_name=args.agent_name,
        write_rlhf=not args.skip_rlhf,
        base_sample_ids=_read_sample_ids(args.base_sample_ids),
        excluded_sample_ids=_read_sample_ids(args.exclude_sample_ids),
        reserve_eval_samples=args.reserve_eval_samples,
        reserve_eval_seed=args.reserve_eval_seed,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
