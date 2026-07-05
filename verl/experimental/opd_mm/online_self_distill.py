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

"""Online OPD-MM self-distillation hook for verl agent-loop rollouts.

This module is intentionally opt-in. A dataset row enables the hook by setting
extra_info.opd_mm_online_self_distill to true and provides a fully-qualified
teacher class in extra_info.opd_mm_step_teacher_class. AgentLoopWorker calls
maybe_collect_online_step_corrections after the student rollout has executed.
"""

from __future__ import annotations

import json
from typing import Any

from verl.experimental.opd_mm.models import OPDSample, ToolAction
from verl.experimental.opd_mm.step_correction import StepCorrectionCollector
from verl.experimental.opd_mm.teacher_privilege import to_plain
from verl.experimental.opd_mm.tools import hidden_store_from_records
from verl.utils.import_utils import load_class_from_fqn

_TEACHER_CACHE: dict[tuple[str, str], Any] = {}


def _as_dict(value: Any) -> dict[str, Any]:
    value = to_plain(value)
    return value if isinstance(value, dict) else {}


def _teacher_cache_key(class_path: str, kwargs: dict[str, Any]) -> tuple[str, str]:
    return class_path, json.dumps(kwargs, sort_keys=True, default=str)


def _build_teacher(class_path: str, kwargs: dict[str, Any]) -> Any:
    key = _teacher_cache_key(class_path, kwargs)
    if key not in _TEACHER_CACHE:
        teacher_cls = load_class_from_fqn(class_path, "OPD-MM step teacher")
        _TEACHER_CACHE[key] = teacher_cls(**kwargs)
    return _TEACHER_CACHE[key]


def _extract_tools_kwargs(sample_kwargs: dict[str, Any], extra_info: dict[str, Any]) -> dict[str, Any]:
    tools_kwargs = _as_dict(sample_kwargs.get("tools_kwargs"))
    if tools_kwargs:
        return tools_kwargs
    return _as_dict(extra_info.get("tools_kwargs"))


def _extract_query(
    sample_kwargs: dict[str, Any],
    tools_kwargs: dict[str, Any],
    extra_info: dict[str, Any],
) -> str:
    opd_kwargs = _as_dict(tools_kwargs.get("opd_mm"))
    if opd_kwargs.get("query") is not None:
        return str(opd_kwargs["query"])
    raw_prompt = to_plain(sample_kwargs.get("raw_prompt"))
    if isinstance(raw_prompt, list):
        for message in reversed(raw_prompt):
            if not isinstance(message, dict) or message.get("role") != "user":
                continue
            content = message.get("content", "")
            if isinstance(content, str):
                return content
            return json.dumps(content, ensure_ascii=False, default=str)
    if extra_info.get("query") is not None:
        return str(extra_info["query"])
    return str(sample_kwargs.get("query") or "")


def _extract_gold_answer(sample_kwargs: dict[str, Any], extra_info: dict[str, Any]) -> str:
    if extra_info.get("gold_answer") is not None:
        return str(extra_info["gold_answer"])
    reward_model = _as_dict(sample_kwargs.get("reward_model"))
    if reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    reward_model = _as_dict(extra_info.get("reward_model"))
    if reward_model.get("ground_truth") is not None:
        return str(reward_model["ground_truth"])
    return ""


def _extract_student_actions(output_extra_fields: dict[str, Any]) -> list[ToolAction]:
    opd_state = _as_dict(output_extra_fields.get("opd_mm"))
    raw_trace = to_plain(opd_state.get("trace")) or []
    if not isinstance(raw_trace, list):
        return []
    return [ToolAction.from_dict(action) for action in raw_trace if isinstance(action, dict)]


def maybe_collect_online_step_corrections(
    *,
    sample_kwargs: dict[str, Any],
    output_extra_fields: dict[str, Any],
) -> list[dict[str, Any]]:
    """Collect OPD-MM step corrections for an already-executed student rollout.

    The teacher is called on each student-visited state with gold answer,
    previous tool calls, and public tool results. It does not receive a separate
    reward/verifier feedback object unless StepCorrectionCollector is explicitly
    configured elsewhere to include one.
    """
    extra_info = _as_dict(sample_kwargs.get("extra_info"))
    if not extra_info.get("opd_mm_online_self_distill"):
        return []

    teacher_class = extra_info.get("opd_mm_step_teacher_class")
    if not teacher_class:
        return []

    tools_kwargs = _extract_tools_kwargs(sample_kwargs, extra_info)
    opd_kwargs = _as_dict(tools_kwargs.get("opd_mm"))
    records = to_plain(opd_kwargs.get("records") or opd_kwargs.get("memory_records") or [])
    if not isinstance(records, list) or not records:
        return []

    student_actions = _extract_student_actions(output_extra_fields)
    if not student_actions:
        return []

    teacher_kwargs = _as_dict(extra_info.get("opd_mm_step_teacher_kwargs"))
    teacher = _build_teacher(str(teacher_class), teacher_kwargs)
    sample = OPDSample(
        sample_id=str(extra_info.get("sample_id") or extra_info.get("index") or "opd_mm_sample"),
        query=_extract_query(sample_kwargs, tools_kwargs, extra_info),
        gold_answer=_extract_gold_answer(sample_kwargs, extra_info),
        memory_store=hidden_store_from_records(records),
        metadata={
            "question_image": opd_kwargs.get("question_image"),
            "teacher_privileged_context": extra_info.get("teacher_privileged_context"),
        },
    )
    collector = StepCorrectionCollector(teacher=teacher)
    return [correction.to_dict() for correction in collector.collect(sample, student_actions)]
