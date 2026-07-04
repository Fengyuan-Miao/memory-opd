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

"""Reward bridge for OPD-MM rollouts."""

from __future__ import annotations

from typing import Any

import torch

from verl.protocol import DataProto
from verl.workers.reward_manager.abstract import AbstractRewardManager, RawRewardFn


def opd_mm_score(row: dict[str, Any]) -> tuple[float, dict[str, Any]]:
    """Compute a small dense reward from OPD-MM metadata."""
    if row.get("correct") is True:
        return 1.0, {"opd_mm/correct": 1.0}
    if row.get("correct") is False:
        return 0.0, {"opd_mm/correct": 0.0}

    support_recall = row.get("support_recall")
    if support_recall is not None:
        value = float(max(0.0, min(1.0, support_recall)))
        return value, {"opd_mm/support_recall": value}

    evidence_count = row.get("evidence_count")
    if evidence_count is None:
        evidence = row.get("evidence") or row.get("observed_evidence") or []
        evidence_count = len(evidence) if isinstance(evidence, list) else 0
    value = 1.0 if int(evidence_count or 0) > 0 else 0.0
    return value, {"opd_mm/evidence_present": value}


class OPDMMRewardManager(AbstractRewardManager):
    """A verl reward manager for OPD-MM metadata-bearing batches."""

    def __init__(
        self,
        tokenizer: Any,
        num_examine: int,
        compute_score: RawRewardFn | None,
        reward_fn_key: str = "data_source",
        **kwargs: Any,
    ):
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.compute_score = compute_score or opd_mm_score
        self.reward_fn_key = reward_fn_key

    def __call__(self, data: DataProto, return_dict: bool = False) -> torch.Tensor | dict[str, Any]:
        batch_size = len(data)
        response_mask = data.batch.get("response_mask") if data.batch is not None else None
        if response_mask is None:
            reward_tensor = torch.zeros(batch_size, 1, dtype=torch.float32)
        else:
            reward_tensor = torch.zeros_like(response_mask, dtype=torch.float32)

        extra_infos: list[dict[str, Any]] = []
        for idx in range(batch_size):
            row = self._row_from_data(data, idx)
            score, info = self.compute_score(row)
            if response_mask is None:
                reward_tensor[idx, 0] = float(score)
            else:
                valid_positions = torch.nonzero(response_mask[idx] > 0, as_tuple=False).flatten()
                target_idx = int(valid_positions[-1].item()) if len(valid_positions) else reward_tensor.shape[1] - 1
                reward_tensor[idx, target_idx] = float(score)
            extra_infos.append(info)

        if not return_dict:
            return reward_tensor
        keys = sorted({key for info in extra_infos for key in info})
        reward_extra_info = {key: [info.get(key) for info in extra_infos] for key in keys}
        return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}

    @staticmethod
    def _row_from_data(data: DataProto, idx: int) -> dict[str, Any]:
        row: dict[str, Any] = {}
        non_tensor_batch = data.non_tensor_batch or {}
        for key, values in non_tensor_batch.items():
            try:
                row[key] = values[idx]
            except Exception:
                continue
        extra = row.get("extra_info")
        if isinstance(extra, dict):
            row.update(extra)
        return row
