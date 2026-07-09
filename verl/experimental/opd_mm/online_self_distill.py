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

"""Online OPD-MM self-distillation helpers for verl agent-loop rollouts.

The verl-native path corrects each student-visible state while ToolAgentLoop is
running. The student still executes its own action; verifier and teacher output
are collected beside that state as an SFT example and never control rollout.
"""

from __future__ import annotations

import json
import os
import re
import time
from typing import Any

from verl.experimental.opd_mm.executor import ToolExecutor
from verl.experimental.opd_mm.models import OPDSample, ToolAction
from verl.experimental.opd_mm.retrieval import TurnAwareHybridRetriever
from verl.experimental.opd_mm.step_correction import StepCorrectionCollector
from verl.experimental.opd_mm.teacher_privilege import to_plain
from verl.experimental.opd_mm.schema import TrajectoryValidator
from verl.experimental.opd_mm.tools import OPDToolSession, hidden_store_from_records
from verl.utils.import_utils import load_class_from_fqn

_TEACHER_CACHE: dict[tuple[str, str], Any] = {}
HERMES_TOOL_CALL_XML_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
QWEN3_FUNCTION_RE = re.compile(r"<function=(.*?)</function>|<function=(.*)$", re.DOTALL)
QWEN3_PARAMETER_RE = re.compile(r"<parameter=(.*?)</parameter>|<parameter=(.*)$", re.DOTALL)
_TOOL_CALL_NAME_BY_ACTION = {
    "FILTER": "filter",
    "SORT": "sort",
    "TOPK": "topk",
    "RETRIEVE": "retrieve",
    "EXPAND_NEIGHBORS": "expand_neighbors",
    "INSPECT_RAW": "inspect_raw",
    "STOP": "stop",
}
_VERIFIER_MISSING_EVIDENCE_TYPES = {
    "none",
    "no_public_evidence",
    "irrelevant_evidence",
    "missing_metadata_constraint",
    "candidate_set_too_broad",
    "missing_neighbor_context",
    "missing_temporal_order",
    "missing_raw_visual_detail",
    "incomplete_coverage",
    "insufficient_absence_support",
}


def _truncate_for_dump(value: Any, max_chars: int) -> Any:
    if not isinstance(value, str) or max_chars <= 0 or len(value) <= max_chars:
        return value
    return value[:max_chars] + f"...[truncated {len(value) - max_chars} chars]"


def dump_online_step_correction(
    request: dict[str, Any],
    *,
    teacher_raw_response: str,
    correction: dict[str, Any] | None,
) -> None:
    """Optionally dump frozen-teacher step corrections for OPD-MM debugging."""
    dump_dir = os.getenv("OPD_MM_TEACHER_CORRECTION_DUMP_DIR")
    if not dump_dir:
        return

    try:
        max_chars = int(os.getenv("OPD_MM_TEACHER_CORRECTION_DUMP_MAX_CHARS", "12000") or "12000")
        include_prompt = os.getenv("OPD_MM_TEACHER_CORRECTION_DUMP_INCLUDE_PROMPT", "1") != "0"
        record = {
            "time": time.time(),
            "pid": os.getpid(),
            "request_id": request.get("request_id"),
            "sample_id": request.get("sample_id"),
            "step_index": request.get("step_index"),
            "query": request.get("query"),
            "gold_answer": request.get("gold_answer"),
            "history": request.get("history", []),
            "observation": request.get("observation", {}),
            "student_next_action": request.get("student_next_action"),
            "student_raw_response": _truncate_for_dump(request.get("student_raw_response", ""), max_chars),
            "verifier_raw_response": _truncate_for_dump(request.get("verifier_raw_response", ""), max_chars),
            "verifier_feedback": request.get("verifier_feedback", {}),
            "teacher_raw_response": _truncate_for_dump(teacher_raw_response, max_chars),
            "parsed": correction is not None,
            "teacher_actions": correction.get("teacher_actions", []) if isinstance(correction, dict) else [],
            "teacher_xml_span": correction.get("teacher_xml_span", "") if isinstance(correction, dict) else "",
            "sft_target_xml": correction.get("sft_target_xml", "") if isinstance(correction, dict) else "",
            "stop_gate_applied": correction.get("stop_gate_applied", False) if isinstance(correction, dict) else False,
        }
        if include_prompt:
            record["verifier_prompt"] = _truncate_for_dump(request.get("verifier_prompt", ""), max_chars)
            record["teacher_prompt"] = _truncate_for_dump(request.get("teacher_prompt", ""), max_chars)
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"teacher_corrections_pid{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False, default=str) + "\n")
    except Exception:
        return


def _as_dict(value: Any) -> dict[str, Any]:
    value = to_plain(value)
    return value if isinstance(value, dict) else {}


def _action_dicts(history: list[Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for action in history:
        if isinstance(action, ToolAction):
            actions.append(action.to_dict())
        elif isinstance(action, dict):
            actions.append(action)
    return actions


def _extract_json_object(raw_response: str) -> dict[str, Any]:
    """Extract the first JSON object from strict JSON or lightly wrapped text."""
    text = (raw_response or "").strip()
    if not text:
        raise ValueError("empty verifier response")
    decoder = json.JSONDecoder()
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            parsed, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    raise ValueError("no JSON object found in verifier response")


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return bool(value)


def _coerce_bool_default(value: Any, default: bool) -> bool:
    if value is None:
        return bool(default)
    return _coerce_bool(value)


def _normalize_missing_evidence_type(value: Any) -> str | None:
    if value is None:
        return None
    missing_type = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "sufficient": "none",
        "enough": "none",
        "complete": "none",
        "empty": "no_public_evidence",
        "no_evidence": "no_public_evidence",
        "missing_evidence": "no_public_evidence",
        "no_candidates": "no_public_evidence",
        "wrong_pool": "irrelevant_evidence",
        "wrong_candidates": "irrelevant_evidence",
        "wrong_candidate_set": "irrelevant_evidence",
        "irrelevant_candidates": "irrelevant_evidence",
        "metadata": "missing_metadata_constraint",
        "metadata_constraint": "missing_metadata_constraint",
        "metadata_filter": "missing_metadata_constraint",
        "needs_metadata_filter": "missing_metadata_constraint",
        "filter": "missing_metadata_constraint",
        "too_broad": "candidate_set_too_broad",
        "broad": "candidate_set_too_broad",
        "narrow": "candidate_set_too_broad",
        "needs_narrowing": "candidate_set_too_broad",
        "needs_topk": "candidate_set_too_broad",
        "neighbor_context": "missing_neighbor_context",
        "neighbour_context": "missing_neighbor_context",
        "needs_neighbor_context": "missing_neighbor_context",
        "neighbors": "missing_neighbor_context",
        "neighbours": "missing_neighbor_context",
        "expand_neighbors": "missing_neighbor_context",
        "temporal": "missing_temporal_order",
        "time_order": "missing_temporal_order",
        "needs_temporal_order": "missing_temporal_order",
        "order": "missing_temporal_order",
        "sort": "missing_temporal_order",
        "visual_detail": "missing_raw_visual_detail",
        "raw_visual": "missing_raw_visual_detail",
        "needs_raw_visual_detail": "missing_raw_visual_detail",
        "inspect_raw": "missing_raw_visual_detail",
        "coverage": "incomplete_coverage",
        "more_coverage": "incomplete_coverage",
        "needs_more_coverage": "incomplete_coverage",
        "list_coverage": "incomplete_coverage",
        "absence": "insufficient_absence_support",
        "contradiction": "insufficient_absence_support",
        "negative_evidence": "insufficient_absence_support",
        "needs_absence_support": "insufficient_absence_support",
    }
    missing_type = aliases.get(missing_type, missing_type)
    return missing_type if missing_type in _VERIFIER_MISSING_EVIDENCE_TYPES else None


def _observation_evidence_count(observation: dict[str, Any]) -> int:
    if not isinstance(observation, dict):
        return 0
    value = observation.get("evidence_count")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    for key in ("evidence", "candidate_pool", "candidates", "current_pool", "pool"):
        items = observation.get(key)
        if isinstance(items, list):
            return len(items)
    return 0


def _observation_pool_count(observation: dict[str, Any]) -> int:
    if not isinstance(observation, dict):
        return 0
    value = observation.get("pool_count")
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return int(value)
    for key in ("pool_preview", "candidate_pool", "candidates", "current_pool", "pool"):
        items = observation.get(key)
        if isinstance(items, list):
            return len(items)
    return 0


def _observation_has_candidate_context(observation: dict[str, Any]) -> bool:
    if _observation_evidence_count(observation) > 0:
        return True
    if _observation_pool_count(observation) <= 0:
        return False

    def tool_name(value: Any) -> str:
        if isinstance(value, dict):
            return str(value.get("tool") or value.get("name") or "").upper()
        return str(value or "").upper()

    if tool_name(observation.get("tool")) in {"RETRIEVE", "FILTER"}:
        return True
    trace = observation.get("trace")
    if isinstance(trace, list):
        return any(tool_name(action) in {"RETRIEVE", "FILTER"} for action in trace)
    return False


def _verifier_parse_fallback(error: Exception | str) -> dict[str, Any]:
    message = str(error)
    return {
        "evidence_sufficient": False,
        "missing_evidence_type": "no_public_evidence",
        "reason": f"verifier_parse_error: {message[:240]}",
        "parse_error": message,
    }


_GENERIC_GOLD_ANSWERS = {"yes", "no", "none", "unknown", "not mentioned", "no mention"}


def _normalize_leak_text(value: Any) -> str:
    text = re.sub(r"[^\w\s]", " ", str(value or "").casefold())
    return re.sub(r"\s+", " ", text).strip()


def _gold_answer_fragments(gold_answer: str) -> list[str]:
    fragments: list[str] = []
    text = str(gold_answer or "").strip()
    if text:
        fragments.append(text)
    fragments.extend(part.strip() for part in re.split(r"[,;|\n]+", text) if part.strip())
    seen: set[str] = set()
    result: list[str] = []
    for fragment in fragments:
        normalized = _normalize_leak_text(fragment)
        if len(normalized) < 4 or normalized in _GENERIC_GOLD_ANSWERS or normalized in seen:
            continue
        seen.add(normalized)
        result.append(fragment)
    return result


def _reason_leaks_private_answer(*, reason: str, gold_answer: str, query: str) -> bool:
    reason_norm = _normalize_leak_text(reason)
    query_norm = _normalize_leak_text(query)
    if not reason_norm:
        return False
    if "gold answer" in reason_norm or "private rubric" in reason_norm:
        return True
    for fragment in _gold_answer_fragments(gold_answer):
        fragment_norm = _normalize_leak_text(fragment)
        if not fragment_norm or fragment_norm in query_norm:
            continue
        if fragment_norm in reason_norm:
            return True
        tokens = [token for token in fragment_norm.split() if len(token) > 3]
        if len(tokens) >= 2 and all(token in reason_norm for token in tokens):
            return True
    return False


def _generic_verifier_reason(*, evidence_sufficient: bool, missing_type: str, evidence_count: int) -> str:
    if evidence_sufficient:
        return "Current public evidence appears sufficient for the answer model to answer."
    if evidence_count <= 0:
        return "Current public evidence is insufficient; collect relevant evidence first."
    if missing_type == "missing_metadata_constraint":
        return "Current public evidence does not isolate the requested metadata constraint."
    if missing_type == "missing_temporal_order":
        return "Current public evidence needs timestamp, order, or ranking evidence before answering."
    if missing_type == "candidate_set_too_broad":
        return "Current public evidence needs a smaller focused candidate set before answering."
    if missing_type == "missing_neighbor_context":
        return "Current public evidence needs neighboring dialogue context before answering."
    if missing_type == "missing_raw_visual_detail":
        return "Current public evidence needs raw visual details from retrieved candidates."
    if missing_type == "irrelevant_evidence":
        return "Current public evidence appears off-target; collect a more relevant candidate set."
    if missing_type == "insufficient_absence_support":
        return "Current public evidence needs broader support for absence, conflict, or contradiction."
    return "Current public evidence is insufficient; retrieve more relevant evidence."


def _sanitize_verifier_reason(
    reason: str,
    *,
    gold_answer: str = "",
    query: str = "",
    evidence_sufficient: bool,
    missing_type: str,
    evidence_count: int,
) -> str:
    reason = str(reason or "").strip()
    if _reason_leaks_private_answer(reason=reason, gold_answer=gold_answer, query=query):
        return _generic_verifier_reason(
            evidence_sufficient=evidence_sufficient,
            missing_type=missing_type,
            evidence_count=evidence_count,
        )
    return (reason or _generic_verifier_reason(
        evidence_sufficient=evidence_sufficient,
        missing_type=missing_type,
        evidence_count=evidence_count,
    ))[:1000]


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


def _extract_generation_snapshots(output_extra_fields: dict[str, Any]) -> list[dict[str, Any]]:
    snapshots = to_plain(output_extra_fields.get("opd_mm_generation_snapshots")) or []
    return [snapshot for snapshot in snapshots if isinstance(snapshot, dict)] if isinstance(snapshots, list) else []


def _tool_action_from_function_call(call: dict[str, Any]) -> ToolAction | None:
    name = call.get("name") or call.get("tool")
    if not name:
        return None
    arguments = call.get("arguments") or {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except json.JSONDecodeError:
            arguments = {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolAction(str(name).upper(), dict(arguments))


def _action_call_name(action: ToolAction) -> str:
    return _TOOL_CALL_NAME_BY_ACTION.get(action.tool.upper(), action.tool.lower())


def action_to_hermes_tool_call_xml(action: ToolAction) -> str:
    """Serialize one internal OPD-MM action as executable Hermes XML."""
    payload = {
        "name": _action_call_name(action),
        "arguments": action.arguments,
    }
    return "<tool_call>\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n</tool_call>"


def _format_qwen3_parameter_value(value: Any) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return str(value)


def action_to_qwen3_tool_call_xml(action: ToolAction) -> str:
    """Serialize one internal OPD-MM action as Qwen3/Qwen3.5 function XML."""
    name = _action_call_name(action)
    lines = ["<tool_call>", f"<function={name}>"]
    for key, value in action.arguments.items():
        lines.append(f"<parameter={key}>")
        lines.append(_format_qwen3_parameter_value(value))
        lines.append("</parameter>")
    lines.append("</function>")
    lines.append("</tool_call>")
    return "\n".join(lines)


def action_to_tool_call_xml(action: ToolAction, *, tool_format: str = "qwen3_coder") -> str:
    if tool_format == "hermes":
        return action_to_hermes_tool_call_xml(action)
    return action_to_qwen3_tool_call_xml(action)


def _coerce_qwen3_parameter_value(value: str) -> Any:
    value = value.strip()
    if value.lower() == "true":
        return True
    if value.lower() == "false":
        return False
    try:
        return int(value)
    except ValueError:
        pass
    try:
        parsed = json.loads(value)
        if isinstance(parsed, (dict, list, int, float, bool)):
            return parsed
    except json.JSONDecodeError:
        pass
    return value


def _parse_hermes_action_from_xml(raw_xml: str) -> ToolAction | None:
    match = HERMES_TOOL_CALL_XML_RE.search(raw_xml or "")
    if match is None:
        return None
    body = match.group(1).strip()
    try:
        payload = json.loads(body)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict):
        return None
    name = payload.get("name") or payload.get("tool")
    if not name:
        return None
    arguments = payload.get("arguments") or {}
    if not isinstance(arguments, dict):
        arguments = {}
    return ToolAction(str(name).upper(), dict(arguments))


def _parse_qwen3_action_from_xml(raw_xml: str) -> ToolAction | None:
    outer_match = HERMES_TOOL_CALL_XML_RE.search(raw_xml or "")
    body = outer_match.group(1) if outer_match is not None else raw_xml
    for function_match in QWEN3_FUNCTION_RE.finditer(body or ""):
        function_body = function_match.group(1) or function_match.group(2)
        if not function_body or ">" not in function_body:
            continue
        end_index = function_body.index(">")
        name = function_body[:end_index].strip()
        parameters = function_body[end_index + 1 :]
        arguments: dict[str, Any] = {}
        for parameter_match in QWEN3_PARAMETER_RE.findall(parameters):
            parameter_body = parameter_match[0] or parameter_match[1]
            if not parameter_body or ">" not in parameter_body:
                continue
            parameter_end = parameter_body.index(">")
            key = parameter_body[:parameter_end].strip()
            value = parameter_body[parameter_end + 1 :]
            arguments[key] = _coerce_qwen3_parameter_value(value)
        return ToolAction(str(name).upper(), arguments)
    return None


def extract_canonical_tool_call_xml(
    text: str,
    *,
    allow_inspect_raw: bool = True,
    tool_format: str = "qwen3_coder",
) -> tuple[str, ToolAction, str] | None:
    """Extract and canonicalize the first valid XML tool call.

    Returns ``(canonical_xml, internal_action, raw_xml_span)``. Any preamble,
    suffix, malformed JSON, or invalid OPD-MM action is ignored.
    """
    validator = TrajectoryValidator(allow_inspect_raw=allow_inspect_raw)

    def try_parse(raw_xml: str) -> tuple[str, ToolAction, str] | None:
        action = _parse_qwen3_action_from_xml(raw_xml) or _parse_hermes_action_from_xml(raw_xml)
        if action is None:
            return None
        try:
            validator._validate_action(action, 0)
        except Exception:
            return None
        return action_to_tool_call_xml(action, tool_format=tool_format), action, raw_xml

    for match in HERMES_TOOL_CALL_XML_RE.finditer(text or ""):
        parsed = try_parse(match.group(0))
        if parsed is not None:
            return parsed
    tool_call_start = (text or "").find("<tool_call>")
    if tool_call_start >= 0:
        return try_parse((text or "")[tool_call_start:])
    return None


def build_state_verifier_prompt(
    *,
    query: str,
    gold_answer: str,
    history: list[Any],
    observation: dict[str, Any],
    student_raw_response: str,
    allow_inspect_raw: bool = True,
) -> str:
    """Build a non-leaking state verifier prompt.

    The verifier may see the gold answer as a private rubric, but its JSON
    output is later shown to the teacher and distilled into student behavior.
    Therefore the feedback must describe missing evidence types, not answer
    content.
    """
    missing_types = (
        "none|no_public_evidence|irrelevant_evidence|missing_metadata_constraint|"
        "candidate_set_too_broad|missing_neighbor_context|missing_temporal_order|"
        "missing_raw_visual_detail|incomplete_coverage|insufficient_absence_support"
        if allow_inspect_raw
        else "none|no_public_evidence|irrelevant_evidence|missing_metadata_constraint|"
        "candidate_set_too_broad|missing_neighbor_context|missing_temporal_order|"
        "incomplete_coverage|insufficient_absence_support"
    )
    visual_decision_line = (
        "- missing_raw_visual_detail: candidate evidence exists, but raw visual/media details are still needed."
        if allow_inspect_raw
        else ""
    )
    return f"""You are the OPD-MM state verifier.

Task: compare the user question, current public evidence, and private gold
answer. Decide whether the public evidence is sufficient for a separate answer
model to answer correctly. If not, classify the missing evidence type.

Private rubric: you may use the gold answer only to judge sufficiency. Your JSON
will be shown to the teacher, so never mention gold answer content, gold-only
entities, dates, IDs, labels, list items, or the phrase "gold answer".

Return JSON only:
{{
  "evidence_sufficient": boolean,
  "missing_evidence_type": "{missing_types}",
  "reason": "short non-leaking evidence diagnostic"
}}

Missing evidence types:
- none: public evidence is sufficient.
- no_public_evidence: no usable public evidence/candidates are present.
- irrelevant_evidence: evidence is off-topic, wrong entity, wrong event, or wrong modality.
- missing_metadata_constraint: explicit date/time/author/source/status/modality constraint is not isolated.
- candidate_set_too_broad: evidence is relevant but too broad/noisy to answer confidently.
- missing_neighbor_context: a relevant turn appears present but adjacent dialogue/event context is missing.
- missing_temporal_order: latest/earliest/before/after/order/ranking relation is not supported.
{visual_decision_line}
- incomplete_coverage: list/all/count/multi-fact evidence does not cover the full requested set.
- insufficient_absence_support: absence/conflict/not-mentioned claims lack enough supporting scope.

Decision rules:
- If evidence_count is 0: evidence_sufficient=false and missing_evidence_type=no_public_evidence.
- If evidence_sufficient=true, missing_evidence_type must be none.
- If evidence_sufficient=false, missing_evidence_type must not be none.
- For list/all/count/multi-fact questions, require complete coverage.
- For temporal/order questions, require timestamp/order support.
- For visual-detail questions, require visual evidence, not text guesswork.
- For absence/conflict questions, require enough relevant scope to support absence or contradiction.
- Do not output action recommendations, operation names, parameters, query rewrites, or answer content.

User question:
{query}

Gold answer (private rubric; do not reveal or copy into feedback):
{gold_answer}

Current public evidence state and observations:
{json.dumps(observation, ensure_ascii=False, indent=2, default=str)}
"""


def parse_state_verifier_feedback(
    raw_response: str,
    observation: dict[str, Any],
    *,
    gold_answer: str = "",
    query: str = "",
) -> dict[str, Any]:
    """Parse verifier JSON into a non-leaking evidence-gap diagnostic."""
    try:
        payload = _extract_json_object(raw_response)
    except Exception as error:
        return _verifier_parse_fallback(error)

    try:
        observation = _as_dict(observation)
        evidence_count = _observation_evidence_count(observation)
        has_candidate_context = _observation_has_candidate_context(observation)
        evidence_sufficient = _coerce_bool(payload.get("evidence_sufficient"))
        reason = str(payload.get("reason") or "").strip()
        missing_type = _normalize_missing_evidence_type(
            payload.get("missing_evidence_type")
            if "missing_evidence_type" in payload
            else payload.get("missing_type")
        )
        if missing_type is None:
            if payload.get("missing_evidence_type") is not None or payload.get("missing_type") is not None:
                raise ValueError(
                    f"invalid missing_evidence_type: "
                    f"{payload.get('missing_evidence_type', payload.get('missing_type'))!r}"
                )
            missing_type = "none" if evidence_sufficient else ("no_public_evidence" if evidence_count <= 0 else "incomplete_coverage")

        if evidence_count <= 0:
            evidence_sufficient = False
            missing_type = "no_public_evidence"
            reason = reason or "Need public evidence before answering."
        elif missing_type == "missing_neighbor_context" and not has_candidate_context:
            evidence_sufficient = False
            missing_type = "incomplete_coverage"
            reason = reason or "Need a relevant public candidate before neighboring context can help."
        elif evidence_sufficient and missing_type != "none":
            evidence_sufficient = False
            reason = reason or "The diagnostic says additional public evidence is still missing."
        elif not evidence_sufficient and missing_type == "none":
            missing_type = "incomplete_coverage"
            reason = reason or "Evidence is insufficient, so another tool action is required."
        elif evidence_sufficient:
            missing_type = "none"

        reason = _sanitize_verifier_reason(
            reason,
            gold_answer=gold_answer,
            query=query,
            evidence_sufficient=bool(evidence_sufficient),
            missing_type=missing_type,
            evidence_count=evidence_count,
        )
        return {
            "evidence_sufficient": bool(evidence_sufficient),
            "missing_evidence_type": missing_type,
            "reason": reason[:1000],
            "parse_error": "",
        }
    except Exception as error:
        return _verifier_parse_fallback(error)


def _stop_gate_fallback_action(request: dict[str, Any], verifier_feedback: dict[str, Any]) -> ToolAction:
    """Choose a safe non-STOP target when verifier marks evidence insufficient."""
    missing_type = _normalize_missing_evidence_type(verifier_feedback.get("missing_evidence_type")) or "no_public_evidence"
    observation = _as_dict(request.get("observation"))
    evidence_count = _observation_evidence_count(observation)
    allow_inspect_raw = bool(request.get("allow_inspect_raw", True))
    if missing_type == "missing_neighbor_context" and _observation_has_candidate_context(observation):
        return ToolAction("EXPAND_NEIGHBORS", {"window": 1})
    if missing_type == "missing_raw_visual_detail" and allow_inspect_raw and evidence_count > 0:
        return ToolAction(
            "INSPECT_RAW",
            {
                "target": "current_pool",
                "instruction": "answer_query_related_visual_details",
            },
        )
    if missing_type == "candidate_set_too_broad" and evidence_count > 0:
        return ToolAction("TOPK", {"k": min(max(evidence_count, 1), 5)})
    if missing_type == "missing_temporal_order" and evidence_count > 0:
        return ToolAction("SORT", {"field": "timestamp", "order": "desc"})
    return ToolAction("RETRIEVE", {"method": "hybrid", "top_k": 10, "query": str(request.get("query") or "")})


def build_teacher_correction_prompt(
    *,
    query: str,
    history: list[Any],
    observation: dict[str, Any],
    student_raw_response: str,
    verifier_feedback: dict[str, Any] | None = None,
    gold_answer: str | None = None,
    privileged_context: Any = None,
    allow_inspect_raw: bool = True,
    tool_format: str = "qwen3_coder",
) -> str:
    """Build the one-step teacher prompt for OPD-MM correction.

    ``gold_answer`` is accepted for backward compatibility but intentionally
    ignored. In the verl-native online path, only the verifier sees gold.
    """
    schema = "\n".join(
        line
        for line in TrajectoryValidator(allow_inspect_raw=allow_inspect_raw).schema_text().splitlines()
        if not line.startswith("Return only a JSON array")
        and not line.startswith("uses the original")
        and not line.startswith("timestamp filters")
    )
    if tool_format == "hermes":
        format_name = "Hermes JSON XML"
    else:
        format_name = "Qwen function XML"
    _ = privileged_context
    verifier_feedback = verifier_feedback or {
        "evidence_sufficient": False,
        "missing_evidence_type": "no_public_evidence",
        "reason": "No verifier feedback was provided.",
        "parse_error": "missing_verifier_feedback",
    }
    return f"""You are the OPD-MM teacher for one-step online self-distillation.

Teacher role:
- Correct exactly one next tool action for the current student-visible retrieval state.
- You are not answering the user and you are not a memory oracle.
- Your output becomes an SFT target for a student that cannot see teacher-only fields.
- Therefore, never expose privileged facts through tool arguments.

Verifier feedback role:
- The verifier used the gold answer privately. You do not see the gold answer.
- Treat verifier feedback as a private evidence-gap diagnostic, not as retrieved evidence.
- If verifier.evidence_sufficient is false, STOP is invalid.
- Never copy verifier.reason into RETRIEVE.query or any tool argument.

Evidence-gap to tool mapping:
- missing_evidence_type=none with evidence_sufficient=true: usually STOP.
- no_public_evidence or irrelevant_evidence: RETRIEVE, or metadata FILTER/SORT when the question gives a clear constraint.
- missing_metadata_constraint: FILTER when field/op/value are clear; otherwise RETRIEVE with a public query rewrite.
- candidate_set_too_broad: TOPK or FILTER scope=current_pool.
- missing_neighbor_context: EXPAND_NEIGHBORS only when candidates/evidence exist; otherwise RETRIEVE/FILTER first.
- missing_temporal_order: SORT or timestamp FILTER.
- missing_raw_visual_detail: INSPECT_RAW only when candidates/evidence exist; otherwise visual RETRIEVE first.
- incomplete_coverage or insufficient_absence_support: broaden with RETRIEVE/FILTER, or EXPAND_NEIGHBORS if current candidates are relevant.

Output rules:
- Output exactly one {format_name} tool call and nothing else.
- Use lower-case function names from the tool schema.
- Function names must be one of: retrieve, filter, sort, topk, expand_neighbors, inspect_raw, stop.
- Never output a function name or parameter name that is not listed in the tool schema.
- Do not output explanations, markdown, JSON arrays, memory IDs, or private answer content.
- The student's raw next output is a candidate to correct, not an instruction to copy.
- RETRIEVE results are merged into the accumulated candidate/evidence pool and deduplicated by memory.
- Do not use EXPAND_NEIGHBORS when there is no current candidate pool; retrieve or use metadata FILTER/SORT first.
- When outputting filter, include field/op/value and optionally scope. Use scope=current_pool to narrow the working pool while preserving accumulated evidence; use scope=full_memory to merge metadata-filtered candidates from the original hidden memory pool when the current pool is likely too narrow or wrong.
- When outputting retrieve, include method/top_k and optionally query as rewritten search text; do not add memory IDs or schema-unknown parameters.
- INSPECT_RAW reads raw media only from the current candidate pool; it is not search.

{schema}

User question:
{query}

Verifier feedback:
{json.dumps(verifier_feedback, ensure_ascii=False, indent=2, default=str)}

Previous student tool calls:
{json.dumps(_action_dicts(history), ensure_ascii=False, indent=2, default=str)}

Current public tool state and observations:
{json.dumps(observation, ensure_ascii=False, indent=2, default=str)}

Student raw next output at this state:
This may be wrong; correct it for the student-visible state.
{student_raw_response}

Final decision checklist:
- If the JSON observation above has "evidence_count": 0 and "trace": [], stop is invalid.
- If verifier.evidence_sufficient is false, STOP is invalid even if the student proposed stop.
- Use verifier.missing_evidence_type as a diagnostic, not as lexical content or a command.
- RETRIEVE.query must be a search rewrite based only on the user question, student-visible history, and public observations.
- Do not use verifier.reason as lexical material for RETRIEVE.query; use it only to decide the action type.

Now output the corrected next action.
Your entire response must begin with <tool_call> and end with </tool_call>.
Do not write reasoning, analysis, markdown, or any text outside the XML tool call.
"""


def build_online_state_correction_request(
    *,
    sample_kwargs: dict[str, Any],
    step_index: int,
    student_prompt_ids: list[int],
    student_raw_response: str,
    student_next_action: ToolAction | dict[str, Any] | None,
    history: list[Any],
    observation: dict[str, Any],
    tool_format: str = "qwen3_coder",
    request_id: str = "",
) -> dict[str, Any] | None:
    """Build one correction request from the live student-visible state.

    Unlike the legacy snapshot replay helper, this function neither loads the
    hidden memory store nor re-executes student actions. ``observation`` and
    ``history`` must come directly from the active ToolAgentLoop session.
    """
    extra_info = _as_dict(sample_kwargs.get("extra_info"))
    if not extra_info.get("opd_mm_online_self_distill"):
        return None

    tools_kwargs = _extract_tools_kwargs(sample_kwargs, extra_info)
    opd_kwargs = _as_dict(tools_kwargs.get("opd_mm"))
    skip_initial_value = extra_info.get("opd_mm_skip_initial_correction")
    if skip_initial_value is None:
        skip_initial_value = opd_kwargs.get("skip_initial_correction")
    if int(step_index) == 0 and _coerce_bool_default(skip_initial_value, True):
        return None

    query = _extract_query(sample_kwargs, tools_kwargs, extra_info)
    gold_answer = _extract_gold_answer(sample_kwargs, extra_info)
    allow_inspect_raw = bool(opd_kwargs.get("allow_inspect_raw", True))
    sample_id = str(extra_info.get("sample_id") or extra_info.get("index") or "opd_mm_sample")
    public_observation = _as_dict(observation)
    public_history = _action_dicts(history)
    if isinstance(student_next_action, ToolAction):
        next_action = student_next_action.to_dict()
    elif isinstance(student_next_action, dict):
        next_action = to_plain(student_next_action)
    else:
        next_action = None

    return {
        "request_id": str(request_id or ""),
        "sample_id": sample_id,
        "step_index": int(step_index),
        "query": query,
        "gold_answer": gold_answer,
        "history": public_history,
        "observation": public_observation,
        "student_next_action": next_action,
        "student_raw_response": str(student_raw_response or ""),
        "student_prompt_ids": [int(token) for token in student_prompt_ids],
        "verifier_prompt": build_state_verifier_prompt(
            query=query,
            gold_answer=gold_answer,
            history=public_history,
            observation=public_observation,
            student_raw_response=str(student_raw_response or ""),
            allow_inspect_raw=allow_inspect_raw,
        ),
        "allow_inspect_raw": allow_inspect_raw,
        "tool_format": tool_format,
    }


def build_online_step_correction_requests(
    *,
    sample_kwargs: dict[str, Any],
    output_extra_fields: dict[str, Any],
    tool_format: str = "qwen3_coder",
) -> list[dict[str, Any]]:
    """Legacy helper that reconstructs states from rollout snapshots.

    The live verl-native pipeline no longer calls this function. It remains for
    backward compatibility with saved rollouts and diagnostic tests.
    """
    extra_info = _as_dict(sample_kwargs.get("extra_info"))
    if not extra_info.get("opd_mm_online_self_distill"):
        return []

    tools_kwargs = _extract_tools_kwargs(sample_kwargs, extra_info)
    opd_kwargs = _as_dict(tools_kwargs.get("opd_mm"))
    raw_records = opd_kwargs.get("records")
    if raw_records is None:
        raw_records = opd_kwargs.get("memory_records")
    if raw_records is None:
        raw_records = []
    if hasattr(raw_records, "tolist"):
        raw_records = raw_records.tolist()
    records = to_plain(raw_records)
    if not isinstance(records, list) or not records:
        return []

    snapshots = _extract_generation_snapshots(output_extra_fields)
    if not snapshots:
        return []

    max_steps = int(extra_info.get("opd_mm_step_correction_max_steps") or opd_kwargs.get("max_actions") or 16)
    skip_initial_value = extra_info.get("opd_mm_skip_initial_correction")
    if skip_initial_value is None:
        skip_initial_value = opd_kwargs.get("skip_initial_correction")
    skip_initial_correction = _coerce_bool_default(skip_initial_value, True)
    allow_inspect_raw = bool(opd_kwargs.get("allow_inspect_raw", True))
    sample_id = str(extra_info.get("sample_id") or extra_info.get("index") or "opd_mm_sample")
    sample = OPDSample(
        sample_id=sample_id,
        query=_extract_query(sample_kwargs, tools_kwargs, extra_info),
        gold_answer=_extract_gold_answer(sample_kwargs, extra_info),
        memory_store=hidden_store_from_records(records),
        metadata={
            "question_image": opd_kwargs.get("question_image"),
            "teacher_privileged_context": extra_info.get("teacher_privileged_context"),
        },
    )
    session = OPDToolSession(
        executor=ToolExecutor(retriever=TurnAwareHybridRetriever()),
        memory_store=sample.memory_store,
        query=sample.query,
        question_image=sample.metadata.get("question_image"),
    )

    requests: list[dict[str, Any]] = []
    for step_index, snapshot in enumerate(snapshots[: max(1, max_steps)]):
        if session.stopped:
            break
        history = list(session.trace)
        observation = session.public_state()
        parsed_calls = to_plain(snapshot.get("parsed_tool_calls") or [])
        parsed_actions = [
            action
            for action in (_tool_action_from_function_call(call) for call in parsed_calls if isinstance(call, dict))
            if action is not None
        ]
        student_raw_response = str(snapshot.get("response_text") or "")
        prompt_ids = to_plain(snapshot.get("prompt_ids") or [])
        if not (step_index == 0 and skip_initial_correction):
            requests.append(
                {
                    "sample_id": sample.sample_id,
                    "step_index": step_index,
                    "query": sample.query,
                    "gold_answer": sample.gold_answer,
                    "history": [action.to_dict() for action in history],
                    "observation": observation,
                    "student_next_action": parsed_actions[0].to_dict() if parsed_actions else None,
                    "student_raw_response": student_raw_response,
                    "student_prompt_ids": prompt_ids if isinstance(prompt_ids, list) else [],
                    "verifier_prompt": build_state_verifier_prompt(
                        query=sample.query,
                        gold_answer=sample.gold_answer,
                        history=history,
                        observation=observation,
                        student_raw_response=student_raw_response,
                        allow_inspect_raw=allow_inspect_raw,
                    ),
                    "allow_inspect_raw": allow_inspect_raw,
                    "tool_format": tool_format,
                }
            )
        if not parsed_actions:
            break
        for action in parsed_actions:
            if session.stopped:
                break
            session.execute(action)
    return requests


def finalize_online_step_correction(
    request: dict[str, Any],
    *,
    teacher_raw_response: str,
) -> dict[str, Any] | None:
    """Convert one teacher raw response into a canonical XML SFT correction."""
    parsed = extract_canonical_tool_call_xml(
        teacher_raw_response,
        allow_inspect_raw=bool(request.get("allow_inspect_raw", True)),
        tool_format=str(request.get("tool_format") or "qwen3_coder"),
    )
    if parsed is None:
        return None
    target_xml, teacher_action, raw_xml = parsed
    sample_id = str(request["sample_id"])
    step_index = int(request["step_index"])
    verifier_feedback = _as_dict(request.get("verifier_feedback"))
    stop_gate_applied = False
    if teacher_action.tool == "STOP" and verifier_feedback and not bool(verifier_feedback.get("evidence_sufficient")):
        teacher_action = _stop_gate_fallback_action(request, verifier_feedback)
        target_xml = action_to_tool_call_xml(
            teacher_action,
            tool_format=str(request.get("tool_format") or "qwen3_coder"),
        )
        stop_gate_applied = True
    example = {
        "sample_id": f"{sample_id}:step:{step_index}",
        "input": "",
        "target": target_xml,
        "round_index": 0,
        "metadata": {
            "opd": {
                "mode": "online_step_xml_correction",
                "request_id": request.get("request_id", ""),
                "step_index": step_index,
                "teacher_raw_response": teacher_raw_response,
                "teacher_xml_span": raw_xml,
                "student_raw_response": request.get("student_raw_response", ""),
                "verifier_raw_response": request.get("verifier_raw_response", ""),
                "verifier_feedback": verifier_feedback,
                "stop_gate_applied": stop_gate_applied,
            }
        },
    }
    return {
        "request_id": request.get("request_id", ""),
        "sample_id": sample_id,
        "step_index": step_index,
        "history": request.get("history", []),
        "observation": request.get("observation", {}),
        "feedback": verifier_feedback,
        "teacher_actions": [teacher_action.to_dict()],
        "student_next_action": request.get("student_next_action"),
        "teacher_raw_response": teacher_raw_response,
        "teacher_xml_span": raw_xml,
        "verifier_raw_response": request.get("verifier_raw_response", ""),
        "verifier_feedback": verifier_feedback,
        "stop_gate_applied": stop_gate_applied,
        "sft_prompt_ids": request.get("student_prompt_ids", []),
        "sft_target_xml": target_xml,
        "example": example,
        "metadata": example["metadata"],
    }


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
    raw_records = opd_kwargs.get("records")
    if raw_records is None:
        raw_records = opd_kwargs.get("memory_records")
    if raw_records is None:
        raw_records = []
    if hasattr(raw_records, "tolist"):
        raw_records = raw_records.tolist()
    records = to_plain(raw_records)
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
