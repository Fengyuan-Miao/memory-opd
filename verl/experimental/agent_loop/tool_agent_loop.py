# Copyright 2025 Bytedance Ltd. and/or its affiliates
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
import asyncio
import json
import logging
import os
import re
import time
from enum import Enum
from typing import Any, Optional
from uuid import uuid4

import torch
from PIL import Image

from verl.experimental.agent_loop.agent_loop import (
    AgentLoopBase,
    AgentLoopOutput,
    ToolListWrap,
    register,
)
from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser
from verl.experimental.agent_loop.utils import build_gpt_oss_tool_response_text
from verl.tools.function_tool import FunctionTool, normalize_function_tool_return
from verl.tools.schemas import OpenAIFunctionCallSchema, OpenAIFunctionParsedSchema, ToolResponse
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from verl.workers.rollout.replica import TokenOutput

logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

SPEC_DECODE_EXTRA_KEYS = (
    "spec_num_draft_tokens",
    "spec_num_accepted_tokens",
    "spec_num_verify_steps",
)

TOOL_CALL_XML_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
TOOL_CALL_PAYLOAD_RE = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)


def _last_user_text(messages: list[dict[str, Any]]) -> str:
    for message in reversed(messages):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
            return "\n".join(parts)
    return ""


def _dump_student_raw_output(
    *,
    agent_data: "AgentData",
    tokenizer: Any,
    response_ids: list[int],
    distillation_mask: list[int],
    assistant_content: str,
) -> None:
    """Optionally save student raw rollout text for OPD-MM debugging.

    This is intentionally side-effect-only and guarded by an environment
    variable so normal training behavior is unchanged.
    """
    dump_dir = os.getenv("OPD_MM_STUDENT_ROLLOUT_DUMP_DIR")
    if not dump_dir:
        return

    try:
        text = tokenizer.decode(response_ids, skip_special_tokens=False)
        max_chars = int(os.getenv("OPD_MM_STUDENT_ROLLOUT_DUMP_MAX_CHARS", "0") or "0")
        dumped_text = text if max_chars <= 0 else text[:max_chars]
        if max_chars > 0 and len(text) > max_chars:
            dumped_text += f"...[truncated {len(text) - max_chars} chars]"

        tool_spans = [match.span() for match in TOOL_CALL_XML_RE.finditer(text)]
        last_open = text.rfind("<tool_call>")
        last_close = text.rfind("</tool_call>")
        parsed_calls = [
            {
                "name": tool_call.name,
                "arguments": tool_call.arguments,
                "tool_call_id": tool_call.tool_call_id,
            }
            for tool_call in agent_data.tool_calls
        ]
        record = {
            "time": time.time(),
            "pid": os.getpid(),
            "request_id": agent_data.request_id,
            "assistant_turn": agent_data.assistant_turns,
            "token_len": len(response_ids),
            "text_len": len(text),
            "distillation_mask_tokens": int(sum(distillation_mask)) if distillation_mask else 0,
            "has_complete_tool_call_xml": bool(tool_spans),
            "has_unclosed_tool_call_xml": bool(last_open >= 0 and last_open > last_close),
            "tool_call_xml_spans": tool_spans,
            "parsed_tool_calls": parsed_calls,
            "assistant_content": assistant_content,
            "query": (agent_data.tools_kwargs.get("opd_mm", {}) or {}).get("query") or _last_user_text(agent_data.messages),
            "text": dumped_text,
        }
        os.makedirs(dump_dir, exist_ok=True)
        path = os.path.join(dump_dir, f"student_rollout_pid{os.getpid()}.jsonl")
        with open(path, "a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")
    except Exception as exc:
        logger.warning(f"Failed to dump student raw rollout output: {exc}")


def build_tool_call_xml_span_mask(tokenizer: Any, response_ids: list[int]) -> list[int]:
    """Return a token mask that keeps only ``<tool_call>...</tool_call>`` XML spans.

    Tool-capable models may emit natural-language reasoning before a function
    call or malformed suffix text after it. OPD-MM distillation should not
    reinforce that prose: only the executable XML tool-call span is action
    supervision.
    """
    if not response_ids:
        return []

    text = tokenizer.decode(response_ids, skip_special_tokens=False)
    spans = [match.span() for match in TOOL_CALL_XML_RE.finditer(text)]

    # If generation is truncated inside an XML tool call, keep the partial
    # action body but still mask any preamble before the opening tag.
    last_open = text.rfind("<tool_call>")
    last_close = text.rfind("</tool_call>")
    if last_open >= 0 and last_open > last_close:
        spans.append((last_open, len(text)))

    if not spans:
        return [0] * len(response_ids)

    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
    except Exception:
        offsets = None

    mask = [0] * len(response_ids)
    if offsets is not None:
        limit = min(len(response_ids), len(offsets))
        for index in range(limit):
            start, end = offsets[index]
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                mask[index] = 1
        return mask

    # Fallback for tokenizers without offset mappings.
    for span_start, span_end in spans:
        start = len(tokenizer.encode(text[:span_start], add_special_tokens=False))
        end = start + len(tokenizer.encode(text[span_start:span_end], add_special_tokens=False))
        for index in range(max(0, start), min(len(mask), end)):
            mask[index] = 1
    return mask


def build_tool_call_payload_mask(tokenizer: Any, response_ids: list[int]) -> list[int]:
    """Keep the serialized function payload while excluding prose and XML wrappers."""
    if not response_ids:
        return []

    text = tokenizer.decode(response_ids, skip_special_tokens=False)
    spans = [match.span(1) for match in TOOL_CALL_PAYLOAD_RE.finditer(text)]
    last_open = text.rfind("<tool_call>")
    last_close = text.rfind("</tool_call>")
    if last_open >= 0 and last_open > last_close:
        spans.append((last_open + len("<tool_call>"), len(text)))
    if not spans:
        return [0] * len(response_ids)

    try:
        encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.get("offset_mapping")
    except Exception:
        offsets = None

    mask = [0] * len(response_ids)
    if offsets is not None:
        for index, (start, end) in enumerate(offsets[: len(response_ids)]):
            if any(start < span_end and end > span_start for span_start, span_end in spans):
                mask[index] = 1
        return mask

    for span_start, span_end in spans:
        start = len(tokenizer.encode(text[:span_start], add_special_tokens=False))
        end = start + len(tokenizer.encode(text[span_start:span_end], add_special_tokens=False))
        for index in range(max(0, start), min(len(mask), end)):
            mask[index] = 1
    return mask


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    PROCESSING_TOOLS = "processing_tools"
    TERMINATED = "terminated"


class AgentData:
    """Encapsulates all state variables for the agent loop. AgentData is passed to tool calling in case that
    tool may need to access full history state. User can store any tool session data in `extra_fields`."""

    def __init__(
        self,
        messages: list[dict[str, Any]],
        image_data: list[Image.Image],
        video_data: list[tuple[torch.Tensor, dict[str, Any]]],
        audio_data: Optional[list[Any]],
        mm_processor_kwargs: Optional[dict[str, Any]],
        metrics: dict[str, Any],
        request_id: str,
        tools_kwargs: dict[str, Any],
    ):
        self.messages = messages
        self.base_messages = list(messages)
        self.image_data = image_data
        self.video_data = video_data
        self.audio_data = audio_data
        self.mm_processor_kwargs = mm_processor_kwargs or {}
        self.metrics = metrics
        self.request_id = request_id
        self.tools_kwargs = tools_kwargs

        # State variables
        self.prompt_ids: list[int] = []
        self.full_prompt_ids: list[int] = []
        self.response_ids: list[int] = []
        self.response_mask: list[int] = []
        self.distillation_mask: list[int] = []
        self.response_logprobs: list[float] = []
        self.turn_scores: list[float] = []
        self.tool_rewards: list[float] = []
        self.user_turns = 0
        self.assistant_turns = 0

        # Temporary state for tool calls
        self.tool_calls: list[FunctionCall] = []
        self.last_assistant_content: str = ""
        self.online_state_corrector: Any = None
        self.teacher_raw_inspector: Any = None

        self.routed_experts = None

        # Extra fields for dynamic addition, e.g., tool session data
        self.extra_fields: dict[str, Any] = {}


@register("tool_agent")
class ToolAgentLoop(AgentLoopBase):
    def __init__(self, *args, tools: Optional[ToolListWrap] = None, **kwargs):
        """Initialize the tool agent loop.

        Args:
            tools: Tools to use for the tool agent loop.
        """
        super().__init__(*args, **kwargs)

        self.max_user_turns = self.rollout_config.multi_turn.max_user_turns
        self.max_assistant_turns = self.rollout_config.multi_turn.max_assistant_turns
        self.max_parallel_calls = self.rollout_config.multi_turn.max_parallel_calls
        self.max_tool_response_length = self.rollout_config.multi_turn.max_tool_response_length
        self.tool_response_truncate_side = self.rollout_config.multi_turn.tool_response_truncate_side

        tool_list = tools.tools if tools else []
        self.tools = {tool.name: tool for tool in tool_list}
        self.tool_schemas = [tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for tool in tool_list]
        self.tool_parser = ToolParser.get_tool_parser(self.rollout_config.multi_turn.format, self.tokenizer)
        self.tool_parser_name = self.rollout_config.multi_turn.format

        self.prompt_length = self.rollout_config.prompt_length
        self.response_length = self.rollout_config.response_length

    @rollout_trace_op
    async def run(self, sampling_params: dict[str, Any], **kwargs) -> AgentLoopOutput:
        messages = list(kwargs["raw_prompt"])

        # extract multimodal inputs from messages
        multi_modal_data = await self.process_multi_modal_info(messages)
        images = multi_modal_data.get("images")
        videos = multi_modal_data.get("videos")
        audios = multi_modal_data.get("audios")
        mm_processor_kwargs = self._get_mm_processor_kwargs(audios)

        metrics = {}
        request_id = uuid4().hex
        tools_kwargs = kwargs.get("tools_kwargs", {})

        agent_data = AgentData(
            messages=messages,
            image_data=images,
            video_data=videos,
            audio_data=audios,
            mm_processor_kwargs=mm_processor_kwargs,
            metrics=metrics,
            request_id=request_id,
            tools_kwargs=tools_kwargs,
        )
        agent_data.online_state_corrector = kwargs.get("_opd_mm_online_state_corrector")
        agent_data.teacher_raw_inspector = kwargs.get("_opd_mm_teacher_raw_inspector")

        # Per-sample tool selection: filter global tools by extra_info.tool_selection
        extra_info = kwargs.get("extra_info", {}) or {}
        tool_selection = extra_info.get("tool_selection")
        if tool_selection and self.tools:
            selected = {name: self.tools[name] for name in tool_selection if name in self.tools}
            agent_data._active_tools = selected
            agent_data._active_tool_schemas = [
                t.tool_schema.model_dump(exclude_unset=True, exclude_none=True) for t in selected.values()
            ]
        else:
            agent_data._active_tools = self.tools
            agent_data._active_tool_schemas = self.tool_schemas

        # State machine loop
        state = AgentState.PENDING
        while state != AgentState.TERMINATED:
            if state == AgentState.PENDING:
                state = await self._handle_pending_state(agent_data, sampling_params)
            elif state == AgentState.GENERATING:
                state = await self._handle_generating_state(agent_data, sampling_params)
            elif state == AgentState.PROCESSING_TOOLS:
                state = await self._handle_processing_tools_state(agent_data)
            else:
                logger.error(f"Invalid state: {state}")
                state = AgentState.TERMINATED

        # Finalize output
        full_prompt_ids = agent_data.full_prompt_ids or agent_data.prompt_ids
        response_ids = full_prompt_ids[-len(agent_data.response_mask) :] if agent_data.response_mask else []
        prompt_ids = full_prompt_ids[: len(full_prompt_ids) - len(agent_data.response_mask)]
        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        output: AgentLoopOutput = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids[: self.response_length],
            response_mask=agent_data.response_mask[: self.response_length],
            distillation_mask=agent_data.distillation_mask[: self.response_length]
            if agent_data.distillation_mask
            else None,
            multi_modal_data=multi_modal_data,
            mm_processor_kwargs=agent_data.mm_processor_kwargs,
            response_logprobs=agent_data.response_logprobs[: self.response_length]
            if agent_data.response_logprobs
            else None,
            num_turns=agent_data.user_turns + agent_data.assistant_turns + 1,
            metrics=agent_data.metrics,
            routed_experts=(
                agent_data.routed_experts[: len(prompt_ids) + self.response_length]
                if agent_data.routed_experts is not None
                else None
            ),
            extra_fields=agent_data.extra_fields,
        )
        output.extra_fields.update({"turn_scores": agent_data.turn_scores, "tool_rewards": agent_data.tool_rewards})
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: dict[str, Any]) -> AgentState:
        """Handle the pending state: prepare the prompt and start generation."""
        schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
        if self.enable_continuous_token:
            prompt_ids = await self.ct_build_initial_tokens(agent_data.messages, tools=schemas)
        else:
            prompt_ids = await self.apply_chat_template(
                agent_data.messages,
                tools=schemas,
                images=agent_data.image_data,
                videos=agent_data.video_data,
                audios=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        agent_data.prompt_ids = prompt_ids
        if not agent_data.full_prompt_ids:
            agent_data.full_prompt_ids = list(prompt_ids)
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: dict[str, Any], ignore_termination: bool = False
    ) -> AgentState:
        """Handle the generating state: generate model response and check for tool calls."""
        # Inject tool parser stop tokens so generation halts after each tool call
        if self.tool_parser.stop_token_ids:
            stop_token_ids = list(set((sampling_params.get("stop_token_ids") or []) + self.tool_parser.stop_token_ids))
            sampling_params = {**sampling_params, "stop_token_ids": stop_token_ids}

        state_prompt_ids = list(agent_data.prompt_ids)
        with simple_timer("generate_sequences", agent_data.metrics):
            output: TokenOutput = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.prompt_ids,
                sampling_params=sampling_params,
                image_data=agent_data.image_data,
                video_data=agent_data.video_data,
                audio_data=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
        # first time to set num_preempted
        if agent_data.metrics.get("num_preempted") is None:
            agent_data.metrics["num_preempted"] = output.num_preempted if output.num_preempted is not None else -1
        # then add num_preempted to the metrics
        else:
            agent_data.metrics["num_preempted"] += output.num_preempted if output.num_preempted is not None else 0

        if not agent_data.extra_fields:
            agent_data.extra_fields.update(output.extra_fields)
        else:
            # Multi-round calls, only update the maximum max_global_steps.
            max_global_steps = output.extra_fields.get("max_global_steps", None)
            if max_global_steps:
                agent_data.extra_fields["max_global_steps"] = max_global_steps
            for key in SPEC_DECODE_EXTRA_KEYS:
                if key in output.extra_fields and key in agent_data.extra_fields:
                    agent_data.extra_fields[key] = int(agent_data.extra_fields[key]) + int(output.extra_fields[key])

        agent_data.assistant_turns += 1
        agent_data.response_ids = output.token_ids
        if self.enable_continuous_token:
            merge_result, response_mask, response_logprobs = await self.ct_merge_assistant_token(
                agent_data.prompt_ids,
                agent_data.response_ids,
                agent_data.response_mask,
                agent_data.response_logprobs if (agent_data.response_logprobs or output.log_probs) else None,
                assistant_logprobs=output.log_probs if output.log_probs else None,
            )
            agent_data.prompt_ids = merge_result.token_ids
            agent_data.response_mask = response_mask
            if response_logprobs is not None:
                agent_data.response_logprobs = response_logprobs
        else:
            agent_data.prompt_ids += agent_data.response_ids
            if agent_data.full_prompt_ids:
                agent_data.full_prompt_ids += agent_data.response_ids
            agent_data.response_mask += [1] * len(agent_data.response_ids)
            current_distillation_mask = build_tool_call_xml_span_mask(self.tokenizer, agent_data.response_ids)
            agent_data.distillation_mask += current_distillation_mask
            if output.log_probs:
                agent_data.response_logprobs += output.log_probs
        if self.enable_continuous_token:
            current_distillation_mask = []

        if output.routed_experts is not None:
            agent_data.routed_experts = output.routed_experts

        terminate_after_generation = (
            (not ignore_termination and len(agent_data.response_mask) >= self.response_length)
            or (self.max_assistant_turns and agent_data.assistant_turns >= self.max_assistant_turns)
            or (self.max_user_turns and agent_data.user_turns >= self.max_user_turns)
        )

        # Extract tool calls (use per-sample tools if routed)
        active_tools = getattr(agent_data, "_active_tools", self.tools)
        tools = [tool.tool_schema for tool in active_tools.values()]
        assistant_content, agent_data.tool_calls = await self.tool_parser.extract_tool_calls(
            agent_data.response_ids, tools
        )
        agent_data.last_assistant_content = assistant_content or ""
        student_next_action = None
        if agent_data.tool_calls:
            tool_call = agent_data.tool_calls[0]
            try:
                arguments = json.loads(tool_call.arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = {"raw_arguments": tool_call.arguments}
            if isinstance(arguments, dict):
                student_next_action = {"tool": str(tool_call.name).upper(), **arguments}
        self._record_opd_mm_policy_state(
            agent_data=agent_data,
            state_prompt_ids=state_prompt_ids,
            response_ids=agent_data.response_ids,
            response_logprobs=output.log_probs,
            tool_call_mask=build_tool_call_payload_mask(self.tokenizer, agent_data.response_ids),
            student_next_action=student_next_action,
        )
        await self._collect_online_state_correction(
            agent_data=agent_data,
            state_prompt_ids=state_prompt_ids,
            response_ids=agent_data.response_ids,
            assistant_content=assistant_content,
        )
        _dump_student_raw_output(
            agent_data=agent_data,
            tokenizer=self.tokenizer,
            response_ids=agent_data.response_ids,
            distillation_mask=current_distillation_mask,
            assistant_content=assistant_content,
        )
        if self.enable_continuous_token:
            agent_data.messages.append(self._build_assistant_message(assistant_content, agent_data))

        if terminate_after_generation:
            return AgentState.TERMINATED
        if agent_data.tool_calls:
            return AgentState.PROCESSING_TOOLS
        return AgentState.TERMINATED

    @staticmethod
    def _record_opd_mm_policy_state(
        *,
        agent_data: AgentData,
        state_prompt_ids: list[int],
        response_ids: list[int],
        response_logprobs: Optional[list[float]],
        tool_call_mask: list[int],
        student_next_action: Optional[dict[str, Any]],
    ) -> None:
        """Keep the actual refreshed state/action pair for a later GRPO update."""
        enabled = str(os.getenv("OPD_MM_RECORD_POLICY_STATES") or "").strip().lower()
        if enabled not in {"1", "true", "yes", "on"}:
            return
        opd_runtime = (agent_data.tools_kwargs or {}).get("opd_mm")
        if not isinstance(opd_runtime, dict) or not response_ids:
            return
        logprobs = list(response_logprobs or [])
        if len(tool_call_mask) != len(response_ids):
            raise RuntimeError("OPD-MM tool-call mask is not aligned with the sampled action")
        step_index = len(agent_data.extra_fields.get("opd_mm_policy_states", []))
        agent_data.extra_fields.setdefault("opd_mm_policy_states", []).append(
            {
                "step_index": step_index,
                "prompt_ids": [int(token) for token in state_prompt_ids],
                "response_ids": [int(token) for token in response_ids],
                "response_logprobs": [float(value) for value in logprobs],
                "tool_call_mask": [int(bool(value)) for value in tool_call_mask],
                "student_next_action": student_next_action,
            }
        )

    async def _collect_online_state_correction(
        self,
        *,
        agent_data: AgentData,
        state_prompt_ids: list[int],
        response_ids: list[int],
        assistant_content: str,
    ) -> None:
        """Collect one teacher correction from the live student-visible state."""
        corrector = agent_data.online_state_corrector
        if corrector is None:
            return

        observation = agent_data.extra_fields.get("opd_mm")
        if not isinstance(observation, dict):
            observation = {
                "pool_count": 0,
                "evidence_count": 0,
                "pool_preview": [],
                "evidence": [],
                "trace": [],
                "stopped": False,
                "error": "",
                "raw_inspection_calls": 0,
            }
        else:
            observation = dict(observation)

        student_next_action = None
        if agent_data.tool_calls:
            tool_call = agent_data.tool_calls[0]
            try:
                arguments = json.loads(tool_call.arguments)
            except (TypeError, json.JSONDecodeError):
                arguments = {"raw_arguments": tool_call.arguments}
            if isinstance(arguments, dict):
                student_next_action = {"tool": str(tool_call.name).upper(), **arguments}

        multi_modal_data = {}
        if agent_data.image_data is not None:
            multi_modal_data["images"] = agent_data.image_data
        if agent_data.video_data is not None:
            multi_modal_data["videos"] = agent_data.video_data
        if agent_data.audio_data is not None:
            multi_modal_data["audios"] = agent_data.audio_data

        try:
            latest_policy_state = None
            policy_states = agent_data.extra_fields.get("opd_mm_policy_states")
            if isinstance(policy_states, list) and policy_states:
                latest_policy_state = policy_states[-1]
            correction = await corrector(
                {
                    "request_id": agent_data.request_id,
                    "step_index": max(0, agent_data.assistant_turns - 1),
                    "student_prompt_ids": list(state_prompt_ids),
                    "student_raw_response": self.tokenizer.decode(response_ids, skip_special_tokens=False),
                    "student_response_ids": list(response_ids),
                    "student_tool_call_mask": (
                        list(latest_policy_state.get("tool_call_mask") or [])
                        if isinstance(latest_policy_state, dict)
                        else build_tool_call_payload_mask(self.tokenizer, response_ids)
                    ),
                    "student_next_action": student_next_action,
                    "history": observation.get("trace") or [],
                    "observation": observation,
                    "assistant_content": assistant_content,
                    "multi_modal_data": multi_modal_data,
                    "mm_processor_kwargs": agent_data.mm_processor_kwargs,
                    "tool_format": self.tool_parser_name,
                }
            )
        except Exception as exc:
            logger.warning("Failed to collect live OPD-MM state correction: %s", exc)
            return
        if isinstance(correction, dict):
            agent_data.extra_fields.setdefault("opd_mm_step_corrections", []).append(correction)

    async def _handle_processing_tools_state(self, agent_data: AgentData) -> AgentState:
        """Handle the processing tools state: execute tool calls and prepare tool responses."""
        add_messages: list[dict[str, Any]] = []
        new_images_this_turn: list[Any] = []  # Local variable instead of agent_data attribute
        previous_messages = list(agent_data.messages)

        tasks = []
        tool_call_names = []
        for tool_call in agent_data.tool_calls[: self.max_parallel_calls]:
            tasks.append(self._call_tool(tool_call, agent_data.tools_kwargs, agent_data))
            tool_call_names.append(tool_call.name)

        with simple_timer("tool_calls", agent_data.metrics):
            responses = await asyncio.gather(*tasks)

        # Process tool responses and update multi_modal_data
        # Removed: agent_data.new_images_this_turn = []
        terminate_after_tools = False
        for tool_index, (tool_response, tool_reward, tool_metrics) in enumerate(responses):
            tool_call = agent_data.tool_calls[tool_index]
            if isinstance(tool_metrics, dict) and tool_metrics.get("agent_loop_terminate"):
                terminate_after_tools = True

            # Create message from tool response
            if tool_response.image or tool_response.video:
                # Multi-modal content with structured format
                if not getattr(self.processor, "image_processor", None):
                    raise ValueError(
                        "Multimedia data can only be processed by `processor`, but the processor is None. "
                        "This error is often caused if you are using a LLM model but your tool returns multimodal "
                        "data. Plase use a vlm as the base model."
                    )
                content = []
                if tool_response.image:
                    content.append({"type": "image"})
                if tool_response.video:
                    content.append({"type": "video"})
                if tool_response.text:
                    content.append({"type": "text", "text": tool_response.text})
                message = {"role": "tool", "content": content}
            else:
                # Text-only content
                message = {"role": "tool", "content": tool_response.text or ""}
            if tool_call.tool_call_id is not None:
                message["tool_call_id"] = tool_call.tool_call_id

            add_messages.append(message)

            # Handle image data
            if tool_response.image:
                # Add new image data
                if isinstance(tool_response.image, list):
                    # Ensure all elements in the list are valid image objects
                    for img in tool_response.image:
                        if img is not None:  # Add a check to ensure the image is not None
                            new_images_this_turn.append(img)  # Using local variable
                else:
                    # Ensure the image is not None
                    if tool_response.image is not None:
                        new_images_this_turn.append(tool_response.image)  # Using local variable

            # Handle video data
            if tool_response.video:
                # Currently not supported, raise informative error
                logger.warning("Multimedia type 'video' is not currently supported. Only 'image' is supported.")
                raise NotImplementedError(
                    "Multimedia type 'video' is not currently supported. Only 'image' is supported."
                )

            if tool_reward is not None:
                agent_data.tool_rewards.append(tool_reward)

        agent_data.messages.extend(add_messages)
        if terminate_after_tools:
            return AgentState.TERMINATED

        # Rebuild OPD-MM prompts from compact action history and the latest
        # accumulated observation instead of retaining older observations.
        opd_prompt_state = getattr(agent_data, "extra_fields", {}).get("opd_mm_prompt_state")
        if not self.enable_continuous_token and not new_images_this_turn and isinstance(opd_prompt_state, dict):
            from verl.experimental.opd_mm.dataset import opd_messages_for_state

            state_messages = opd_messages_for_state(
                agent_data.base_messages,
                opd_prompt_state.get("action_history") or [],
                opd_prompt_state.get("observation") or {},
            )
            schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
            agent_data.messages = state_messages
            agent_data.prompt_ids = await self.apply_chat_template(
                state_messages,
                tools=schemas,
                images=agent_data.image_data,
                videos=agent_data.video_data,
                audios=agent_data.audio_data,
                mm_processor_kwargs=agent_data.mm_processor_kwargs,
            )
            agent_data.user_turns += 1
            return AgentState.GENERATING

        if self.enable_continuous_token and not new_images_this_turn:
            schemas = getattr(agent_data, "_active_tool_schemas", self.tool_schemas)
            merge_result, response_mask, response_logprobs = await self.ct_merge_non_assistant_msg(
                previous_messages,
                agent_data.messages,
                agent_data.prompt_ids,
                agent_data.response_mask,
                agent_data.response_logprobs if agent_data.response_logprobs else None,
                tools=schemas,
            )
            if len(response_mask) >= self.response_length:
                return AgentState.TERMINATED
            agent_data.prompt_ids = merge_result.token_ids
            agent_data.response_mask = response_mask
            if agent_data.response_logprobs:
                agent_data.response_logprobs = response_logprobs or []
            agent_data.user_turns += 1
            return AgentState.GENERATING
        elif self.tool_parser_name == "gpt-oss":
            logger.info("manually format tool responses for gpt-oss")
            tool_response_text = build_gpt_oss_tool_response_text(add_messages, tool_call_names)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        elif self.tool_parser_name == "gemma4":
            # Gemma4's chat template drops tool responses when passed without the preceding
            # assistant tool_call message. Manually format the response tokens.
            # Format: <|tool_response>response:func_name{value:<|"|>content<|"|>}<tool_response|>
            parts = []
            for msg, name in zip(add_messages, tool_call_names, strict=True):
                content = msg.get("content", "")
                if isinstance(content, list):
                    content = "".join([item.get("text", "") for item in content if item.get("type") == "text"])
                parts.append(f'<|tool_response>response:{name}{{value:<|"|>{content}<|"|>}}<tool_response|>')
            tool_response_text = "".join(parts)
            response_ids = await self.loop.run_in_executor(
                None, lambda: self.tokenizer.encode(tool_response_text, add_special_tokens=False)
            )
        else:
            # Note that we have to pass None to the images and videos if there are no new images / videos
            # to stay compatible with downstream image processing logic!
            images = new_images_this_turn if new_images_this_turn else None
            videos = None
            response_ids = await self.apply_chat_template(
                add_messages,
                images=images,
                videos=videos,
                remove_system_prompt=True,
            )

        if len(agent_data.response_mask) + len(response_ids) >= self.response_length:
            return AgentState.TERMINATED
        # Update prompt_ids and response_mask

        if new_images_this_turn:
            if agent_data.image_data is None:
                agent_data.image_data = []
            elif not isinstance(agent_data.image_data, list):
                agent_data.image_data = [agent_data.image_data]
            for img in new_images_this_turn:
                agent_data.image_data.append(img)

        agent_data.prompt_ids += response_ids
        agent_data.response_mask += [0] * len(response_ids)
        if agent_data.distillation_mask:
            agent_data.distillation_mask += [0] * len(response_ids)
        if agent_data.response_logprobs:
            agent_data.response_logprobs += [0.0] * len(response_ids)
        agent_data.user_turns += 1
        return AgentState.GENERATING

    def _build_assistant_message(self, content: str, agent_data: AgentData) -> dict[str, Any]:
        message: dict[str, Any] = {"role": "assistant", "content": content or ""}
        if not agent_data.tool_calls:
            return message

        tool_calls = []
        for index, tool_call in enumerate(agent_data.tool_calls[: self.max_parallel_calls]):
            function_call, has_decode_error = OpenAIFunctionCallSchema.from_openai_function_parsed_schema(
                OpenAIFunctionParsedSchema(name=tool_call.name, arguments=tool_call.arguments)
            )
            if has_decode_error:
                raise ValueError(
                    f"Invalid tool call arguments for '{tool_call.name}': expected a JSON object string, "
                    f"got {tool_call.arguments!r}"
                )
            tool_call_message = {
                "type": "function",
                "function": function_call.model_dump(),
            }
            if tool_call.tool_call_id is not None:
                tool_call_message["id"] = tool_call.tool_call_id
            tool_calls.append(tool_call_message)
        message["tool_calls"] = tool_calls
        return message

    async def _call_tool(
        self, tool_call: FunctionCall, tools_kwargs: dict[str, Any], agent_data: AgentData
    ) -> tuple[ToolResponse, float, dict]:
        """Call tool and return tool response.

        Dispatches between two contracts:
        - ``FunctionTool``: stateless function-based tool. Invoked directly with
          parsed arguments; no lifecycle.
        - ``BaseTool`` subclass: stateful tool with full lifecycle.
        """
        active_tools = getattr(agent_data, "_active_tools", self.tools)

        # Validate tool name
        tool_name = tool_call.name
        if tool_name not in active_tools:
            available = list(active_tools.keys())
            msg = f"Unknown function '{tool_name}'. Available tools: {available}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        # Validate tool arguments
        try:
            tool_args = json.loads(tool_call.arguments)
        except (json.JSONDecodeError, TypeError) as e:
            msg = f"Invalid JSON in arguments for '{tool_name}': {e}"
            logger.warning(msg)
            return ToolResponse(text=msg), 0.0, {}

        # Execute tool
        tool, instance_id = None, None
        try:
            tool = active_tools[tool_name]

            if isinstance(tool, FunctionTool):
                # Function-based tools have no lifecycle; call directly.
                # Note: tools_kwargs (create_kwargs / release_kwargs) is intentionally
                # ignored here. Function tools are stateless and per-trajectory state
                # injection is not supported by design; use a BaseTool subclass instead.
                raw = await tool.call(tool_args)
                tool_execution_response, tool_reward, res = normalize_function_tool_return(raw)
            else:
                # BaseTool subclass
                kwargs = tools_kwargs.get(tool_name, {})
                instance_id, _ = await tool.create(create_kwargs=kwargs.get("create_kwargs", {}))
                tool_execution_response, tool_reward, res = await tool.execute(
                    instance_id, tool_args, agent_data=agent_data
                )
        except Exception as e:
            logger.warning(f"Error executing tool '{tool_name}': {e}")
            return ToolResponse(text=f"Error executing tool '{tool_name}': {e}"), 0.0, {}
        finally:
            # Only BaseTool instances need release (function tools never set instance_id).
            if tool and instance_id and not isinstance(tool, FunctionTool):
                await tool.release(instance_id)

        tool_response_text = tool_execution_response.text
        if tool_response_text and len(tool_response_text) > self.max_tool_response_length:
            if self.tool_response_truncate_side == "left":
                tool_response_text = "(truncated)..." + tool_response_text[-self.max_tool_response_length :]
            elif self.tool_response_truncate_side == "right":
                tool_response_text = tool_response_text[: self.max_tool_response_length] + "...(truncated)"
            else:
                length = self.max_tool_response_length // 2
                tool_response_text = tool_response_text[:length] + "...(truncated)..." + tool_response_text[-length:]

        # Create ToolResponse from tool execution result
        tool_response_kwargs = {"text": tool_response_text}

        # Add multimedia data if present
        for attr_name in ["image", "video"]:
            if hasattr(tool_execution_response, attr_name):
                attr_value = getattr(tool_execution_response, attr_name)
                if attr_value is not None:
                    tool_response_kwargs[attr_name] = attr_value

        return ToolResponse(**tool_response_kwargs), tool_reward, res
