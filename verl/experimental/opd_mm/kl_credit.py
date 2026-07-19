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

"""Action-level KL credit assignment helpers for OPD-MM."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable

from verl.experimental.opd_mm.models import ToolAction
from verl.experimental.opd_mm.schema import TrajectoryValidator


def _topk_distribution(ids: Iterable[Any], logprobs: Iterable[Any]) -> dict[int, float]:
    distribution: dict[int, float] = {}
    for token_id, logprob in zip(ids, logprobs, strict=False):
        if token_id is None or logprob is None:
            continue
        try:
            distribution[int(token_id)] = float(logprob)
        except (TypeError, ValueError):
            continue
    return distribution


def normalized_topk_union_kl(
    teacher_ids: Iterable[Any],
    teacher_logprobs: Iterable[Any],
    student_ids: Iterable[Any],
    student_logprobs: Iterable[Any],
    *,
    missing_logprob: float = -10.0,
) -> float:
    """Approximate ``KL(teacher || student)`` on the normalized top-k union.

    This is the same bounded-support calculation used by the OPD-MM KL
    diagnostic: take the union of teacher/student top-k tokens, assign a small
    finite log-probability to a token missing from either side, renormalize both
    distributions on that union, and compute forward KL.
    """
    teacher = _topk_distribution(teacher_ids, teacher_logprobs)
    student = _topk_distribution(student_ids, student_logprobs)
    support = set(teacher) | set(student)
    if not support:
        return float("nan")

    teacher_values = {key: teacher.get(key, float(missing_logprob)) for key in support}
    student_values = {key: student.get(key, float(missing_logprob)) for key in support}
    teacher_norm = math.log(sum(math.exp(value) for value in teacher_values.values()))
    student_norm = math.log(sum(math.exp(value) for value in student_values.values()))
    result = 0.0
    for key in support:
        log_teacher = teacher_values[key] - teacher_norm
        log_student = student_values[key] - student_norm
        probability = math.exp(log_teacher)
        result += probability * (log_teacher - log_student)
    return max(0.0, float(result))


def tokenwise_topk_union_kl(
    teacher_ids: Iterable[Iterable[Any]],
    teacher_logprobs: Iterable[Iterable[Any]],
    student_ids: Iterable[Iterable[Any]],
    student_logprobs: Iterable[Iterable[Any]],
    *,
    missing_logprob: float = -10.0,
) -> list[float]:
    """Compute normalized top-k-union KL for aligned next-token positions."""
    return [
        normalized_topk_union_kl(
            teacher_id_row,
            teacher_logprob_row,
            student_id_row,
            student_logprob_row,
            missing_logprob=missing_logprob,
        )
        for teacher_id_row, teacher_logprob_row, student_id_row, student_logprob_row in zip(
            teacher_ids,
            teacher_logprobs,
            student_ids,
            student_logprobs,
            strict=False,
        )
    ]


def masked_mean(values: Iterable[float], mask: Iterable[Any]) -> float:
    selected = [
        float(value)
        for value, keep in zip(values, mask, strict=False)
        if bool(keep) and math.isfinite(float(value))
    ]
    return sum(selected) / len(selected) if selected else float("nan")


def _canonical_action(action: ToolAction) -> dict[str, Any]:
    arguments = dict(action.arguments)
    if action.tool == "RETRIEVE":
        arguments.setdefault("method", "hybrid")
        arguments.setdefault("top_k", 5)
    elif action.tool == "INSPECT_RAW":
        arguments.setdefault("target", "current_pool")
        arguments.setdefault("instruction", "answer_query_related_visual_details")
    for key, value in list(arguments.items()):
        if isinstance(value, str):
            arguments[key] = value.strip()
    return {"tool": action.tool, "arguments": arguments}


def structured_action_disagreement(
    student_action: dict[str, Any] | None,
    teacher_action: dict[str, Any] | None,
    *,
    allow_inspect_raw: bool = True,
) -> tuple[bool, str]:
    """Return whether two calls differ in tool, normalized arguments, or validity."""
    if not isinstance(student_action, dict):
        return True, "student_unparsed"
    if not isinstance(teacher_action, dict):
        return False, "teacher_unparsed"

    student = ToolAction.from_dict(student_action)
    teacher = ToolAction.from_dict(teacher_action)
    validator = TrajectoryValidator(allow_inspect_raw=allow_inspect_raw)
    try:
        validator._validate_action(student, 0)
    except Exception as exc:
        return True, f"student_invalid:{type(exc).__name__}"
    try:
        validator._validate_action(teacher, 0)
    except Exception as exc:
        return False, f"teacher_invalid:{type(exc).__name__}"

    student_value = _canonical_action(student)
    teacher_value = _canonical_action(teacher)
    if student_value["tool"] != teacher_value["tool"]:
        return True, "tool"
    if json.dumps(student_value["arguments"], ensure_ascii=False, sort_keys=True, default=str) != json.dumps(
        teacher_value["arguments"], ensure_ascii=False, sort_keys=True, default=str
    ):
        return True, "arguments"
    return False, "none"


__all__ = [
    "masked_mean",
    "normalized_topk_union_kl",
    "structured_action_disagreement",
    "tokenwise_topk_union_kl",
]
