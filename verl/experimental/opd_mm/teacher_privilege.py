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

"""Teacher-only privileged prompt helpers for OPD-MM logprob distillation."""

from __future__ import annotations

import json
from typing import Any, Optional

import torch

TEACHER_PRIVILEGE_MODE = "opd_mm"


def to_plain(value: Any) -> Any:
    """Convert numpy scalars/0-d arrays to plain Python values when possible."""
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _json_block(value: Any) -> str:
    return json.dumps(to_plain(value), ensure_ascii=False, indent=2, default=str)


def _extract_query(sample_kwargs: dict[str, Any]) -> str:
    raw_prompt = to_plain(sample_kwargs.get("raw_prompt"))
    if isinstance(raw_prompt, list):
        for message in reversed(raw_prompt):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
    return str(sample_kwargs.get("query") or "")


def _extract_gold_answer(sample_kwargs: dict[str, Any]) -> str:
    extra_info = to_plain(sample_kwargs.get("extra_info")) or {}
    if isinstance(extra_info, dict) and extra_info.get("gold_answer") is not None:
        return str(extra_info["gold_answer"])
    reward_model = to_plain(sample_kwargs.get("reward_model")) or {}
    if isinstance(reward_model, dict) and reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    return ""


def should_use_teacher_privilege(sample_kwargs: Optional[dict[str, Any]]) -> bool:
    """Return true when this sample requests OPD-MM teacher-only conditioning."""
    if not sample_kwargs:
        return False
    extra_info = to_plain(sample_kwargs.get("extra_info")) or {}
    if not isinstance(extra_info, dict):
        return False
    return extra_info.get("teacher_privilege_mode") == TEACHER_PRIVILEGE_MODE


def build_teacher_privileged_prompt(
    sample_kwargs: dict[str, Any],
    output_extra_fields: dict[str, Any],
) -> str:
    """Build the teacher-only prompt used for privileged logprob scoring."""
    opd_state = output_extra_fields.get("opd_mm") or {}
    return (
        "You are the privileged OPD-MM teacher for on-policy distillation.\n"
        "You are scoring the student's exact tool-call trajectory, not generating a correction.\n"
        "Use the privileged information below only as teacher-side context.\n"
        "The student policy did not see the gold answer.\n\n"
        f"User question:\n{_extract_query(sample_kwargs)}\n\n"
        f"Gold answer:\n{_extract_gold_answer(sample_kwargs)}\n\n"
        f"Student tool-call history and tool results:\n{_json_block(opd_state)}\n\n"
        "Now assign likelihood to the student's exact continuation."
    )


def align_teacher_outputs_to_student_sequence(
    teacher_ids: torch.Tensor,
    teacher_logprobs: torch.Tensor,
    *,
    teacher_prompt_length: int,
    student_prompt_length: int,
    response_length: int,
    pad_token_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Align teacher-only-prompt logprobs to the student prompt/response layout.

    Distillation loss slices positions that predict response tokens. For a
    sequence prompt + response, those positions are prompt_len - 1 through
    prompt_len + response_len - 2. This helper copies that response-prediction
    slice from the teacher-only sequence into the corresponding positions of
    the student-shaped sequence and leaves prompt-only positions neutral.
    """
    total_length = int(student_prompt_length) + int(response_length)
    ids = torch.full(
        (total_length, *teacher_ids.shape[1:]),
        fill_value=int(pad_token_id),
        dtype=teacher_ids.dtype,
        device=teacher_ids.device,
    )
    logprobs = torch.zeros(
        (total_length, *teacher_logprobs.shape[1:]),
        dtype=teacher_logprobs.dtype,
        device=teacher_logprobs.device,
    )
    if response_length <= 0:
        return ids, logprobs

    src_start = max(0, int(teacher_prompt_length) - 1)
    dst_start = max(0, int(student_prompt_length) - 1)
    length = min(int(response_length), teacher_ids.shape[0] - src_start, total_length - dst_start)
    if length <= 0:
        return ids, logprobs
    ids[dst_start : dst_start + length] = teacher_ids[src_start : src_start + length]
    logprobs[dst_start : dst_start + length] = teacher_logprobs[src_start : src_start + length]
    return ids, logprobs
