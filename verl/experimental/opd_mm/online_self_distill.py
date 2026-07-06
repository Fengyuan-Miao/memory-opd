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
    "INSPECT_RAW": "inspect_raw",
    "STOP": "stop",
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
            "sample_id": request.get("sample_id"),
            "step_index": request.get("step_index"),
            "query": request.get("query"),
            "gold_answer": request.get("gold_answer"),
            "history": request.get("history", []),
            "observation": request.get("observation", {}),
            "student_next_action": request.get("student_next_action"),
            "student_raw_response": _truncate_for_dump(request.get("student_raw_response", ""), max_chars),
            "teacher_raw_response": _truncate_for_dump(teacher_raw_response, max_chars),
            "parsed": correction is not None,
            "teacher_actions": correction.get("teacher_actions", []) if isinstance(correction, dict) else [],
            "teacher_xml_span": correction.get("teacher_xml_span", "") if isinstance(correction, dict) else "",
            "sft_target_xml": correction.get("sft_target_xml", "") if isinstance(correction, dict) else "",
        }
        if include_prompt:
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


def build_teacher_correction_prompt(
    *,
    query: str,
    gold_answer: str,
    history: list[ToolAction],
    observation: dict[str, Any],
    student_raw_response: str,
    privileged_context: Any = None,
    allow_inspect_raw: bool = True,
    tool_format: str = "qwen3_coder",
) -> str:
    """Build the privileged one-step teacher prompt for OPD-MM correction."""
    schema = "\n".join(
        line
        for line in TrajectoryValidator(allow_inspect_raw=allow_inspect_raw).schema_text().splitlines()
        if not line.startswith("Return only a JSON array")
        and not line.startswith("uses the original")
        and not line.startswith("timestamp filters")
    )
    if tool_format == "hermes":
        format_name = "Hermes JSON XML"
        example = '<tool_call>{"name":"retrieve","arguments":{"method":"hybrid","top_k":5}}</tool_call>'
    else:
        format_name = "Qwen function XML"
        example = (
            "<tool_call>\n"
            "<function=retrieve>\n"
            "<parameter=method>\nhybrid\n</parameter>\n"
            "<parameter=top_k>\n5\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )
    privileged = ""
    if privileged_context:
        privileged = (
            "\nAdditional teacher-only context:\n"
            f"{json.dumps(to_plain(privileged_context), ensure_ascii=False, indent=2, default=str)}\n"
        )
    return f"""You are the OPD-MM teacher for one-step online self-distillation.

Teacher role:
- Correct exactly one next tool action for the current student-visible retrieval state.
- You are not answering the user and you are not a memory oracle.
- Your output becomes an SFT target for a student that cannot see teacher-only fields.
- Therefore, never expose privileged facts through tool arguments.

Gold answer role:
- The gold answer is teacher-only supervision/rubric; it is not student-visible evidence.
- Use it only to judge whether current public observations are sufficient and whether the next tool should retrieve, filter, inspect, or stop.
- Do not treat the gold answer as retrieved evidence.
- The gold answer is not a lexical source for RETRIEVE.query; use it as a private rubric, not to reveal answer-only entities in the query.
- Write RETRIEVE.query as if the Gold answer section were hidden: use only the user question, previous student tool calls/results, and current public observations.
- Do not include gold-only answer entities, answer lists, image labels, exact final answers, dates, names, or phrases visible only in the gold answer.
- For broad list questions (for example "what X were mentioned" or "list all X"), RETRIEVE.query must stay generic; never enumerate the gold answer list items in the query.

Output rules:
- Output exactly one {format_name} tool call and nothing else.
- Use lower-case function names from the tool schema.
- Do not output explanations, markdown, JSON arrays, or memory IDs.
- The student's raw next output is a candidate to correct, not an instruction to copy.
- Output stop only when the current public observations/evidence are sufficient for the student to answer.
- Do not output stop solely because the gold answer is known or because the student proposed stop.
- If history/trace is empty and evidence_count is 0, do not output stop; choose RETRIEVE or metadata FILTER/SORT first.
- If evidence_count is 0, prefer RETRIEVE or metadata FILTER/SORT unless previous tool results prove no useful evidence can be collected.
- If the gold answer is "Not mentioned" or similar, the student still needs retrieval/filter observations before stop.
- RETRIEVE results are merged into the accumulated candidate/evidence pool and deduplicated by memory.
- When outputting filter, include field/op/value and optionally scope. Use scope=current_pool to narrow the working pool while preserving accumulated evidence; use scope=full_memory to merge metadata-filtered candidates from the original hidden memory pool when the current pool is likely too narrow or wrong.
- When outputting retrieve, include method/top_k and optionally query as rewritten search text; do not add memory IDs or schema-unknown parameters.

INSPECT_RAW guidance:
- INSPECT_RAW is for checking raw image/media details of records already in the current retrieved candidate pool.
- It is not a retrieval/search action and must not be used to scan the original full memory store.
- If the current public state has no retrieved evidence/candidates yet, prefer RETRIEVE or metadata FILTER/SORT first.
- Use INSPECT_RAW only when visual/raw-media details are needed beyond public summaries/evidence.

Required output format example:
{example}

{schema}

User question:
{query}

Gold answer:
{gold_answer}

Previous student tool calls:
{json.dumps([action.to_dict() for action in history], ensure_ascii=False, indent=2)}

Current public tool state and observations:
{json.dumps(observation, ensure_ascii=False, indent=2, default=str)}
{privileged}
Student raw next output at this state:
This may be wrong; correct it for the student-visible state.
{student_raw_response}

Final decision checklist:
- If the JSON observation above has "evidence_count": 0 and "trace": [], stop is invalid.
- In that empty-evidence initial state, correct a student stop into retrieve/filter/sort/topk.
- A gold answer like "Not mentioned" still requires student-visible retrieval/filter observations before stop.
- RETRIEVE.query must be a non-leaking search rewrite based on the user question, student-visible history, and public observations.
- Before writing RETRIEVE.query, ignore the Gold answer section except as a private grading rubric.
- If the user question asks for a list/category/name that is only revealed by the gold answer, search for the requested category generically instead of inserting the answer.

Now output the corrected next action.
Your entire response must begin with <tool_call> and end with </tool_call>.
Do not write reasoning, analysis, markdown, or any text outside the XML tool call.
"""


def build_online_step_correction_requests(
    *,
    sample_kwargs: dict[str, Any],
    output_extra_fields: dict[str, Any],
    tool_format: str = "qwen3_coder",
) -> list[dict[str, Any]]:
    """Build teacher-generation requests for every student-visited OPD-MM state.

    Unlike ``maybe_collect_online_step_corrections``, this path does not require
    a Python ``StepTeacherPolicy`` class. It is designed for verl's native frozen
    teacher server: AgentLoopWorker generates the teacher response, extracts the
    XML span, and later the trainer turns it into an actor SFT batch.
    """
    extra_info = _as_dict(sample_kwargs.get("extra_info"))
    if not extra_info.get("opd_mm_online_self_distill"):
        return []

    tools_kwargs = _extract_tools_kwargs(sample_kwargs, extra_info)
    opd_kwargs = _as_dict(tools_kwargs.get("opd_mm"))
    records = to_plain(opd_kwargs.get("records") or opd_kwargs.get("memory_records") or [])
    if not isinstance(records, list) or not records:
        return []

    snapshots = _extract_generation_snapshots(output_extra_fields)
    if not snapshots:
        return []

    max_steps = int(extra_info.get("opd_mm_step_correction_max_steps") or opd_kwargs.get("max_actions") or 16)
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
                "teacher_prompt": build_teacher_correction_prompt(
                    query=sample.query,
                    gold_answer=sample.gold_answer,
                    history=history,
                    observation=observation,
                    student_raw_response=student_raw_response,
                    privileged_context=sample.metadata.get("teacher_privileged_context"),
                    allow_inspect_raw=allow_inspect_raw,
                    tool_format=tool_format,
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
    example = {
        "sample_id": f"{sample_id}:step:{step_index}",
        "input": "",
        "target": target_xml,
        "round_index": 0,
        "metadata": {
            "opd": {
                "mode": "online_step_xml_correction",
                "step_index": step_index,
                "teacher_raw_response": teacher_raw_response,
                "teacher_xml_span": raw_xml,
                "student_raw_response": request.get("student_raw_response", ""),
            }
        },
    }
    return {
        "sample_id": sample_id,
        "step_index": step_index,
        "history": request.get("history", []),
        "observation": request.get("observation", {}),
        "feedback": {},
        "teacher_actions": [teacher_action.to_dict()],
        "student_next_action": request.get("student_next_action"),
        "teacher_raw_response": teacher_raw_response,
        "teacher_xml_span": raw_xml,
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
