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

from __future__ import annotations

from types import SimpleNamespace

import pandas as pd

from examples.data_preprocess.build_mem_gallery_opd_mm_train_subset import (
    extend_stratified_mem_gallery_subset,
    stratified_holdout_subset,
)
from examples.opd_mm_baseline.evaluate_opd_mm_llm_judge import load_eval_qas


def _qa(sample_id: str, scenario: str, point: str, image: bool = False) -> dict:
    return {
        "sample_id": sample_id,
        "scenario": scenario,
        "point": point,
        "question_image": "image.png" if image else None,
    }


def test_stratified_extension_retains_base_and_excludes_holdout() -> None:
    qas = [
        _qa("a", "s1", "p1"),
        _qa("b", "s1", "p1", image=True),
        _qa("c", "s1", "p1"),
        _qa("d", "s1", "p1", image=True),
        _qa("e", "s2", "p1"),
        _qa("f", "s2", "p1"),
    ]

    selected = extend_stratified_mem_gallery_subset(
        qas,
        base_sample_ids={"a", "e"},
        excluded_sample_ids={"b"},
        per_cell_cap=3,
        seed=7,
    )
    selected_ids = {qa["sample_id"] for qa in selected}

    assert {"a", "e"} <= selected_ids
    assert "b" not in selected_ids
    assert len([qa for qa in selected if qa["scenario"] == "s1"]) == 3
    assert len([qa for qa in selected if qa["scenario"] == "s2"]) == 2


def test_stratified_holdout_is_deterministic_and_unique() -> None:
    qas = [
        _qa(f"{scenario}-{point}-{index}", scenario, point)
        for scenario in ("s1", "s2")
        for point in ("p1", "p2")
        for index in range(4)
    ]

    first = stratified_holdout_subset(qas, max_samples=7, seed=11)
    second = stratified_holdout_subset(qas, max_samples=7, seed=11)

    assert [qa["sample_id"] for qa in first] == [qa["sample_id"] for qa in second]
    assert len({qa["sample_id"] for qa in first}) == 7


def test_eval_loader_accepts_explicit_fixed_sample_ids(tmp_path) -> None:
    qas_path = tmp_path / "qas.parquet"
    ids_path = tmp_path / "eval_ids.txt"
    pd.DataFrame(
        [
            _qa("train", "s1", "p1"),
            _qa("eval-a", "s1", "p1"),
            _qa("eval-b", "s2", "p2"),
        ]
    ).to_parquet(qas_path, index=False)
    ids_path.write_text("eval-a\neval-b\n", encoding="utf-8")
    args = SimpleNamespace(
        qas_path=str(qas_path),
        eval_sample_ids=str(ids_path),
        train_sample_ids="unused",
        max_samples=100,
        seed=3,
    )

    selected = load_eval_qas(args)

    assert {qa["sample_id"] for qa in selected} == {"eval-a", "eval-b"}
