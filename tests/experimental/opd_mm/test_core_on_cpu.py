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

import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pytest
import torch

from verl import DataProto
from verl.experimental.opd_mm import MemoryRecord, OPDSample, OnPolicyDistiller, PolicyOutput, ToolAction, ToolExecutor
from verl.experimental.opd_mm.dataset import OPD_MM_SYSTEM_PROMPT, opd_messages_for_query, opd_sample_to_rlhf_record
from verl.experimental.opd_mm.online_self_distill import maybe_collect_online_step_corrections
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore, TurnAwareHybridRetriever
from verl.experimental.opd_mm.reward_manager import OPDMMRewardManager
from verl.experimental.opd_mm.schema import TrajectoryValidationError, TrajectoryValidator
from verl.experimental.opd_mm.sft import opd_sft_row_to_verl_record
from verl.experimental.opd_mm.step_correction import StepCorrectionCollector
from verl.experimental.opd_mm.teacher_privilege import (
    align_teacher_outputs_to_student_sequence,
    build_teacher_privileged_prompt,
)
from verl.experimental.opd_mm.tools import OPDFilterTool, hidden_store_from_records, openai_tool_schemas
from verl.tools.tool_registry import load_all_tools


def _records() -> list[MemoryRecord]:
    return [
        MemoryRecord(
            memory_id="m_text_old",
            turn_id="1",
            timestamp="2026-01-01T09:00:00",
            author="user",
            modality="text",
            source_type="conversation",
            summary="The user mentioned an older dog photo.",
            content="I uploaded a dog picture yesterday.",
            metadata={"session_date": "2026-01-01"},
        ),
        MemoryRecord(
            memory_id="m_cat_text",
            turn_id="2",
            timestamp="2026-01-01T10:00:00",
            author="user",
            modality="text",
            source_type="conversation",
            summary="The user described a tabby cat image.",
            content="This is my tabby cat on the sofa.",
            metadata={"session_date": "2026-01-01"},
        ),
        MemoryRecord(
            memory_id="m_cat_image",
            turn_id="2",
            timestamp="2026-01-01T10:00:01",
            author="user",
            modality="image",
            source_type="uploaded_image",
            summary="A tabby cat sitting on a sofa.",
            raw_pointer="images/cat.png",
            metadata={"session_date": "2026-01-01"},
        ),
        MemoryRecord(
            memory_id="m_assistant_image",
            turn_id="3",
            timestamp="2026-01-01T11:00:00",
            author="assistant",
            modality="image",
            source_type="generated_image",
            summary="An assistant-generated chart.",
            raw_pointer="images/chart.png",
        ),
    ]


def test_validator_rejects_memory_ids_and_custom_retrieve_query() -> None:
    validator = TrajectoryValidator(max_actions=4)

    with pytest.raises(TrajectoryValidationError, match="memory IDs"):
        validator.validate([{"tool": "FILTER", "field": "status", "op": "eq", "value": "m_001"}])

    with pytest.raises(TrajectoryValidationError, match="forbidden arguments"):
        validator.validate([{"tool": "RETRIEVE", "query": "custom query", "top_k": 3}])


def test_executor_composes_generic_tools_to_latest_user_image() -> None:
    store = HiddenMemoryStore(_records())
    result = ToolExecutor().run(
        [
            {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
            {"tool": "FILTER", "field": "author", "op": "eq", "value": "user"},
            {"tool": "SORT", "field": "timestamp", "order": "desc"},
            {"tool": "TOPK", "k": 1},
        ],
        query="Which image did I upload last?",
        memory_store=store,
    )

    assert not result.error
    assert result.stopped
    assert result.final_memory_ids == ["m_cat_image"]
    assert result.evidence[0].fields["raw_pointer"] == "images/cat.png"
    assert result.evidence[0].fields["session_date"] == "2026-01-01"


def test_turn_aware_retrieval_returns_text_and_image_from_same_turn() -> None:
    store = HiddenMemoryStore(_records())
    result = ToolExecutor(retriever=TurnAwareHybridRetriever()).run(
        [
            {"tool": "RETRIEVE", "method": "bm25", "top_k": 1},
        ],
        query="tabby cat sofa",
        memory_store=store,
    )

    assert not result.error
    modalities = {item.fields["modality"] for item in result.evidence}
    assert modalities == {"text", "image"}
    assert {item.memory_id for item in result.evidence} == {"m_cat_text", "m_cat_image"}


@dataclass
class FakeAgentData:
    messages: list[dict[str, Any]]
    tools_kwargs: dict[str, Any]
    extra_fields: dict[str, Any] = field(default_factory=dict)


@pytest.mark.asyncio
async def test_verl_native_opd_tools_share_hidden_state_and_hide_ids() -> None:
    records = [record.to_dict() for record in _records()]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Read my latest user image."}],
        tools_kwargs={"opd_mm": {"query": "Read my latest user image.", "records": records}},
    )
    filter_tool = OPDFilterTool(config={"type": "native"}, tool_schema=None)

    await filter_tool.execute("instance", {"field": "modality", "op": "eq", "value": "image"}, agent_data=agent_data)
    response, _, metrics = await filter_tool.execute(
        "instance",
        {"field": "author", "op": "eq", "value": "user"},
        agent_data=agent_data,
    )

    observation = json.loads(response.text)
    assert observation["pool_count"] == 1
    assert observation["pool_preview"][0]["raw_pointer"] == "images/cat.png"
    assert metrics["opd_mm_evidence_count"] == 0
    assert "memory_id" not in json.dumps(observation)
    assert agent_data.extra_fields["opd_mm"]["pool_count"] == 1
    assert "memory_id" not in json.dumps(agent_data.extra_fields["opd_mm"])


def test_tool_config_loads_verl_native_opd_tools() -> None:
    tools = load_all_tools(
        tool_config_path="examples/opd_mm_baseline/opd_mm_tool_config.yaml",
        function_tool_path=None,
    )
    assert [tool.name for tool in tools] == ["filter", "sort", "topk", "retrieve", "inspect_raw", "stop"]


def test_sft_converter_can_emit_native_tool_call_records() -> None:
    record = opd_sft_row_to_verl_record(
        {
            "sample_id": "s1",
            "input": "Find the cat image.",
            "target": json.dumps(
                [
                    {"tool": "RETRIEVE", "method": "bm25", "top_k": 1},
                    {"tool": "STOP"},
                ]
            ),
        },
        include_tools=True,
        native_tool_calls=True,
    )

    assert record["messages"][2]["tool_calls"][0]["function"]["name"] == "retrieve"
    assert [schema["function"]["name"] for schema in record["tools"]] == [
        "filter",
        "sort",
        "topk",
        "retrieve",
        "inspect_raw",
        "stop",
    ]


def test_opd_sample_converts_to_on_policy_distillation_row() -> None:
    sample = OPDSample(
        sample_id="sample-1",
        query="Which image did I upload last?",
        gold_answer="The tabby cat image.",
        memory_store=HiddenMemoryStore(_records()),
        metadata={
            "index": 7,
            "opd_mm_online_self_distill": True,
            "opd_mm_step_teacher_class": "tests.experimental.opd_mm.test_core_on_cpu.FakeStepTeacher",
        },
    )
    row = opd_sample_to_rlhf_record(sample)

    assert row["data_source"] == "opd_mm"
    assert row["agent_name"] == "tool_agent"
    assert row["prompt"] == [
        {"role": "system", "content": OPD_MM_SYSTEM_PROMPT},
        {"role": "user", "content": sample.query},
    ]
    for tool_name in ("RETRIEVE", "FILTER", "SORT", "TOPK", "INSPECT_RAW", "STOP"):
        assert tool_name in row["prompt"][0]["content"]
    assert "READ" not in row["prompt"][0]["content"]
    assert row["extra_info"]["need_tools_kwargs"] is True
    assert row["extra_info"]["teacher_privilege_mode"] == "opd_mm"
    assert row["extra_info"]["tools_kwargs"]["opd_mm"]["query"] == sample.query
    assert row["extra_info"]["tools_kwargs"]["opd_mm"]["records"][0]["memory_id"] == "m_text_old"
    assert row["extra_info"]["gold_answer"] == sample.gold_answer
    assert row["extra_info"]["opd_mm_online_self_distill"] is True
    assert row["extra_info"]["opd_mm_step_teacher_class"].endswith("FakeStepTeacher")


def test_opd_messages_for_query_can_override_or_disable_system_prompt() -> None:
    assert opd_messages_for_query("Where is it?", system_prompt="custom") == [
        {"role": "system", "content": "custom"},
        {"role": "user", "content": "Where is it?"},
    ]
    assert opd_messages_for_query("Where is it?", system_prompt=None) == [
        {"role": "user", "content": "Where is it?"}
    ]


def test_teacher_privileged_prompt_contains_answer_and_tool_results_without_feedback() -> None:
    sample_kwargs = {
        "raw_prompt": [{"role": "user", "content": "Where is the cat?"}],
        "extra_info": {
            "gold_answer": "on the sofa",
            "teacher_privilege_mode": "opd_mm",
            "teacher_feedback": {"correct": False, "reason": "missing visual evidence"},
        },
    }
    prompt = build_teacher_privileged_prompt(
        sample_kwargs,
        {
            "opd_mm": {
                "trace": [{"tool": "RETRIEVE", "method": "hybrid", "top_k": 3}],
                "evidence_count": 1,
            }
        },
    )

    assert "Where is the cat?" in prompt
    assert "on the sofa" in prompt
    assert "RETRIEVE" in prompt
    assert "missing visual evidence" not in prompt
    assert "Student execution feedback" not in prompt


def test_teacher_privileged_logprobs_align_to_student_response_slice() -> None:
    teacher_ids = torch.arange(8).view(8, 1)
    teacher_logprobs = torch.arange(80, 88, dtype=torch.float32).view(8, 1)

    aligned_ids, aligned_logprobs = align_teacher_outputs_to_student_sequence(
        teacher_ids,
        teacher_logprobs,
        teacher_prompt_length=4,
        student_prompt_length=3,
        response_length=3,
        pad_token_id=-1,
    )

    assert aligned_ids.squeeze(-1).tolist() == [-1, -1, 3, 4, 5, -1]
    assert aligned_logprobs.squeeze(-1).tolist() == [0.0, 0.0, 83.0, 84.0, 85.0, 0.0]


class FakeStudent:
    validator = TrajectoryValidator()

    def generate_trace(self, query: str) -> PolicyOutput:
        return PolicyOutput(
            actions=[
                ToolAction("FILTER", {"field": "author", "op": "eq", "value": "assistant"}),
                ToolAction("STOP"),
            ],
            raw_response="student_wrong_trace",
        )


class FakeTeacher:
    def correct(
        self,
        query: str,
        gold_answer: str,
        student_policy: PolicyOutput,
        student_answer: str,
        correct: bool,
        execution=None,
        privileged_context=None,
    ) -> PolicyOutput:
        return PolicyOutput(
            actions=[
                ToolAction("RETRIEVE", {"method": "bm25", "top_k": 1}),
                ToolAction("STOP"),
            ],
            raw_response="teacher_corrected_trace",
        )


class FakeAnswerModel:
    def answer(self, query: str, evidence: list[Any], question_image=None) -> str:
        return " ".join(str(item.fields.get("summary", "")) for item in evidence)


class FakeJudge:
    def evaluate(self, query: str, prediction: str, gold_answer: str) -> tuple[bool, float, str]:
        correct = gold_answer.lower() in prediction.lower()
        return correct, 1.0 if correct else 0.0, "matched" if correct else "missing_gold"


class FakeStepTeacher:
    def correct_next(
        self,
        query: str,
        gold_answer: str,
        history: list[ToolAction],
        observation: dict[str, Any],
        feedback: dict[str, Any],
        privileged_context=None,
    ) -> PolicyOutput:
        if not history:
            actions = [ToolAction("RETRIEVE", {"method": "bm25", "top_k": 1})]
        else:
            actions = [ToolAction("STOP")]
        return PolicyOutput(actions=actions, raw_response="step_teacher")


def test_step_level_correction_collector_labels_student_visited_states() -> None:
    sample = OPDSample(
        sample_id="sample-step",
        query="Find the tabby cat sofa memory.",
        gold_answer="on the sofa",
        memory_store=HiddenMemoryStore(_records()),
    )
    collector = StepCorrectionCollector(
        teacher=FakeStepTeacher(),
        executor=ToolExecutor(retriever=TurnAwareHybridRetriever()),
        answer_model=FakeAnswerModel(),
        judge=FakeJudge(),
    )
    corrections = collector.collect(
        sample,
        [
            ToolAction("RETRIEVE", {"method": "bm25", "top_k": 1}),
            ToolAction("STOP"),
        ],
    )

    assert len(corrections) == 2
    assert corrections[0].history == []
    assert corrections[0].teacher_actions[0].tool == "RETRIEVE"
    assert corrections[1].history[0].tool == "RETRIEVE"
    assert corrections[1].teacher_actions[0].tool == "STOP"
    assert json.loads(corrections[1].example.target)[0]["tool"] == "STOP"
    assert "teacher_feedback" not in corrections[1].example.metadata["opd"]
    assert "missing visual evidence" not in corrections[1].example.input.lower()


def test_online_self_distill_collects_step_corrections_from_rollout_extra_fields() -> None:
    teacher_class = f"{FakeStepTeacher.__module__}.{FakeStepTeacher.__qualname__}"
    corrections = maybe_collect_online_step_corrections(
        sample_kwargs={
            "raw_prompt": [{"role": "user", "content": "Find the tabby cat sofa memory."}],
            "tools_kwargs": {
                "opd_mm": {
                    "query": "Find the tabby cat sofa memory.",
                    "records": [record.to_dict(include_internal_id=True) for record in _records()],
                }
            },
            "extra_info": {
                "gold_answer": "on the sofa",
                "opd_mm_online_self_distill": True,
                "opd_mm_step_teacher_class": teacher_class,
                "sample_id": "online-sample",
            },
        },
        output_extra_fields={
            "opd_mm": {
                "trace": [
                    {"tool": "RETRIEVE", "method": "bm25", "top_k": 1},
                    {"tool": "STOP"},
                ]
            }
        },
    )

    assert len(corrections) == 2
    assert corrections[0]["sample_id"] == "online-sample"
    assert corrections[0]["teacher_actions"][0]["tool"] == "RETRIEVE"
    assert corrections[1]["teacher_actions"][0]["tool"] == "STOP"
    assert corrections[1]["feedback"] == {}
    assert "teacher_feedback" not in corrections[1]["example"]["metadata"]["opd"]


def test_original_on_policy_distiller_runs_verify_and_teacher_correction() -> None:
    sample = OPDSample(
        sample_id="sample-2",
        query="Find the tabby cat sofa memory.",
        gold_answer="tabby cat",
        memory_store=HiddenMemoryStore(_records()),
        metadata={"gold_clue_turn_ids": ["2"]},
    )
    distiller = OnPolicyDistiller(
        student=FakeStudent(),
        teacher=FakeTeacher(),
        executor=ToolExecutor(retriever=TurnAwareHybridRetriever()),
        answer_model=FakeAnswerModel(),
        judge=FakeJudge(),
    )

    rollout = distiller.rollout(sample)

    assert rollout.correct is False
    assert rollout.student_answer == "An assistant-generated chart."
    assert rollout.metadata["teacher_selection_source"] == "llm_teacher"
    assert rollout.teacher_execution is not None
    assert {item.memory_id for item in rollout.teacher_execution.evidence} == {"m_cat_text", "m_cat_image"}
    assert rollout.sft_example.metadata["teacher_candidate_diagnostics"][0]["selected"] is True
    assert json.loads(rollout.sft_example.target)[0]["tool"] == "RETRIEVE"


def test_reward_manager_places_score_on_last_response_token() -> None:
    data = DataProto.from_dict(
        tensors={"response_mask": torch.tensor([[1, 1, 0], [1, 0, 0]])},
        non_tensors={
            "correct": np.array([True, False], dtype=object),
            "extra_info": np.array([{}, {}], dtype=object),
        },
    )
    manager = OPDMMRewardManager(tokenizer=None, num_examine=0, compute_score=None)
    output = manager(data, return_dict=True)

    reward = output["reward_tensor"]
    assert reward.tolist() == [[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]]
    assert output["reward_extra_info"]["opd_mm/correct"] == [1.0, 0.0]


def test_helpers_build_hidden_store_from_dicts_and_schemas() -> None:
    store = hidden_store_from_records([{"memory_id": "m1", "turn_id": "1", "timestamp": "", "author": "user"}])
    assert len(store) == 1
    assert [schema["function"]["name"] for schema in openai_tool_schemas(include_inspect_raw=False)] == [
        "filter",
        "sort",
        "topk",
        "retrieve",
        "stop",
    ]
