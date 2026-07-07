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
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.opd_mm import MemoryRecord, OPDSample, OnPolicyDistiller, PolicyOutput, ToolAction, ToolExecutor
from verl.experimental.opd_mm.dataset import OPD_MM_SYSTEM_PROMPT, opd_messages_for_query, opd_sample_to_rlhf_record
from verl.experimental.opd_mm.online_self_distill import (
    build_teacher_correction_prompt,
    build_online_step_correction_requests,
    dump_online_step_correction,
    extract_canonical_tool_call_xml,
    finalize_online_step_correction,
    maybe_collect_online_step_corrections,
    parse_state_verifier_feedback,
)
from verl.experimental.opd_mm.raw_inspector import RemoteVLLMRawInspector
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore, TurnAwareHybridRetriever
from verl.experimental.opd_mm.reward_manager import OPDMMRewardManager
from verl.experimental.opd_mm.schema import TrajectoryValidationError, TrajectoryValidator
from verl.experimental.opd_mm.sft import opd_sft_row_to_verl_record
from verl.experimental.opd_mm.step_correction import StepCorrectionCollector
from verl.experimental.opd_mm.teacher_privilege import (
    align_teacher_outputs_to_student_sequence,
    build_teacher_privileged_prompt,
)
from verl.experimental.opd_mm.tools import (
    OPDFilterTool,
    OPDRetrieveTool,
    OPDStopTool,
    hidden_store_from_records,
    openai_tool_schemas,
)
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.utils import tensordict_utils as tu
from verl.tools.tool_registry import load_all_tools
from verl.workers.config import ActorConfig, DistillationConfig


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


class FakeRawInspector:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def inspect(
        self,
        raw_pointer: str,
        query: str,
        question_image: str | None = None,
        text_context: str = "",
    ) -> str:
        del query, question_image, text_context
        self.calls.append(raw_pointer)
        return f"observed {raw_pointer}"


class RecordingRetriever:
    def __init__(self) -> None:
        self.queries: list[str] = []

    def retrieve(
        self,
        pool: list[Any],
        query: str,
        store: HiddenMemoryStore,
        method: str = "hybrid",
        top_k: int = 5,
        question_image: str | None = None,
    ) -> list[Any]:
        del store, method, question_image
        self.queries.append(query)
        return pool[:top_k]


def test_remote_vllm_raw_inspector_sends_image_and_returns_text(tmp_path, monkeypatch) -> None:
    image = tmp_path / "cat.jpg"
    image.write_bytes(b"fake-jpeg")
    captured: dict[str, Any] = {}

    class FakeResponse:
        def __enter__(self) -> "FakeResponse":
            return self

        def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
            return None

        def read(self) -> bytes:
            return json.dumps(
                {"choices": [{"message": {"content": "A tabby cat is visible on a sofa."}}]},
                ensure_ascii=False,
            ).encode("utf-8")

    def fake_urlopen(req: Any, timeout: float) -> FakeResponse:
        captured["url"] = req.full_url
        captured["timeout"] = timeout
        captured["payload"] = json.loads(req.data.decode("utf-8"))
        return FakeResponse()

    monkeypatch.setattr("verl.experimental.opd_mm.raw_inspector.urllib_request.urlopen", fake_urlopen)

    inspector = RemoteVLLMRawInspector(
        base_url="http://raw-inspector:8000",
        model="vl-test",
        timeout=12,
        max_tokens=64,
    )
    observation = inspector.inspect(str(image), "What animal is shown?", text_context="The user shared a pet photo.")

    assert observation == "A tabby cat is visible on a sofa."
    assert captured["url"] == "http://raw-inspector:8000/v1/chat/completions"
    assert captured["timeout"] == 12
    assert captured["payload"]["model"] == "vl-test"
    content = captured["payload"]["messages"][1]["content"]
    assert content[0]["type"] == "text"
    assert "What animal is shown?" in content[0]["text"]
    assert content[1]["type"] == "image_url"
    assert content[1]["image_url"]["url"].startswith("data:image/jpeg;base64,")


class QuerySwitchRetriever:
    def retrieve(
        self,
        pool: list[Any],
        query: str,
        store: HiddenMemoryStore,
        method: str = "hybrid",
        top_k: int = 5,
        question_image: str | None = None,
    ) -> list[Any]:
        del pool, method, top_k, question_image
        records = {item.memory.memory_id: item for item in store.initial_pool()}
        if "assistant" in query:
            return [records["m_assistant_image"]]
        return [records["m_cat_text"]]


def test_validator_rejects_memory_ids_and_accepts_rewritten_retrieve_query() -> None:
    validator = TrajectoryValidator(max_actions=4)

    with pytest.raises(TrajectoryValidationError, match="memory IDs"):
        validator.validate([{"tool": "FILTER", "field": "status", "op": "eq", "value": "m_001"}])

    validated_filter = validator.validate(
        [{"tool": "FILTER", "field": "author", "op": "eq", "value": "user", "scope": "full_memory"}]
    )
    assert validated_filter[0].arguments["scope"] == "full_memory"

    with pytest.raises(TrajectoryValidationError, match="invalid FILTER scope"):
        validator.validate([{"tool": "FILTER", "field": "author", "op": "eq", "value": "user", "scope": "all"}])

    validated = validator.validate([{"tool": "RETRIEVE", "query": "custom query", "top_k": 3}])
    assert validated[0].arguments["query"] == "custom query"

    with pytest.raises(TrajectoryValidationError, match="memory IDs"):
        validator.validate([{"tool": "RETRIEVE", "query": "look up m_001", "top_k": 3}])


def test_executor_uses_rewritten_retrieve_query_when_provided() -> None:
    retriever = RecordingRetriever()
    ToolExecutor(retriever=retriever).run(
        [{"tool": "RETRIEVE", "method": "bm25", "query": "rewritten cat sofa", "top_k": 1}],
        query="original user question",
        memory_store=HiddenMemoryStore(_records()),
    )

    assert retriever.queries == ["rewritten cat sofa"]


def test_filter_scope_can_restart_from_full_memory_pool() -> None:
    store = HiddenMemoryStore(_records())

    narrowed_result = ToolExecutor().run(
        [
            {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
            {"tool": "FILTER", "field": "author", "op": "eq", "value": "assistant"},
            {"tool": "FILTER", "field": "author", "op": "eq", "value": "user"},
        ],
        query="Show user memories after a wrong narrow filter.",
        memory_store=store,
    )
    assert narrowed_result.final_memory_ids == []

    reset_result = ToolExecutor().run(
        [
            {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
            {"tool": "FILTER", "field": "author", "op": "eq", "value": "assistant"},
            {"tool": "FILTER", "field": "author", "op": "eq", "value": "user", "scope": "full_memory"},
        ],
        query="Show user memories after resetting the filter scope.",
        memory_store=store,
    )
    assert reset_result.final_memory_ids == ["m_assistant_image", "m_text_old", "m_cat_text", "m_cat_image"]
    assert reset_result.steps[2].pool_before == 1
    assert reset_result.steps[2].pool_after == 4
    assert len({item.memory_id for item in reset_result.evidence}) == len(reset_result.evidence)


def test_repeated_retrieve_merges_and_deduplicates_candidates() -> None:
    result = ToolExecutor(retriever=QuerySwitchRetriever()).run(
        [
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 1, "query": "cat"},
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 1, "query": "cat"},
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 1, "query": "assistant"},
        ],
        query="Find multiple memories.",
        memory_store=HiddenMemoryStore(_records()),
    )

    assert result.final_memory_ids == ["m_cat_text", "m_assistant_image"]
    assert [item.memory_id for item in result.evidence] == ["m_cat_text", "m_assistant_image"]


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


def test_timestamp_filter_accepts_date_only_model_values() -> None:
    records = [
        *_records(),
        MemoryRecord(
            memory_id="m_next_day",
            turn_id="4",
            timestamp="2026-01-02T09:00:00",
            author="user",
            modality="text",
            source_type="conversation",
            summary="A next-day note.",
        ),
    ]
    store = HiddenMemoryStore(records)

    eq_result = ToolExecutor().run(
        [{"tool": "FILTER", "field": "timestamp", "op": "eq", "value": "2026-01-01"}],
        query="Which memories are from 2026-01-01?",
        memory_store=store,
    )
    assert set(eq_result.final_memory_ids) == {
        "m_text_old",
        "m_cat_text",
        "m_cat_image",
        "m_assistant_image",
    }

    contains_result = ToolExecutor().run(
        [{"tool": "FILTER", "field": "timestamp", "op": "contains", "value": "2026/1/1"}],
        query="Which memories are from 2026-01-01?",
        memory_store=store,
    )
    assert set(contains_result.final_memory_ids) == set(eq_result.final_memory_ids)

    before_result = ToolExecutor().run(
        [{"tool": "FILTER", "field": "timestamp", "op": "before", "value": "2026-01-02"}],
        query="Which memories are before 2026-01-02?",
        memory_store=store,
    )
    assert set(before_result.final_memory_ids) == set(eq_result.final_memory_ids)

    after_result = ToolExecutor().run(
        [{"tool": "FILTER", "field": "timestamp", "op": "after", "value": "2026-01-01"}],
        query="Which memories are after 2026-01-01?",
        memory_store=store,
    )
    assert after_result.final_memory_ids == ["m_next_day"]


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
    assert {item.source for item in result.evidence} == {"RETRIEVE"}
    assert result.steps[0].evidence_added == 2


def test_inspect_raw_only_reads_retrieved_candidate_pool() -> None:
    store = HiddenMemoryStore(_records())
    inspector = FakeRawInspector()

    filtered_result = ToolExecutor(raw_inspector=inspector).run(
        [
            {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
            {"tool": "INSPECT_RAW"},
        ],
        query="What is in the cat image?",
        memory_store=store,
    )

    assert inspector.calls == []
    assert filtered_result.steps[1].evidence_added == 0
    assert all("visual_observation" not in item.fields for item in filtered_result.evidence)

    retrieved_inspector = FakeRawInspector()
    retrieved_result = ToolExecutor(
        retriever=TurnAwareHybridRetriever(),
        raw_inspector=retrieved_inspector,
    ).run(
        [
            {"tool": "RETRIEVE", "method": "bm25", "top_k": 1},
            {"tool": "INSPECT_RAW"},
        ],
        query="tabby cat sofa",
        memory_store=store,
    )

    assert retrieved_inspector.calls == ["images/cat.png"]
    assert retrieved_result.steps[1].evidence_added == 1
    assert any(item.source == "INSPECT_RAW" and "visual_observation" in item.fields for item in retrieved_result.evidence)


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
    assert metrics["opd_mm_evidence_count"] == 2
    assert "memory_id" not in json.dumps(observation)
    assert agent_data.extra_fields["opd_mm"]["pool_count"] == 1
    assert agent_data.extra_fields["opd_mm"]["evidence_count"] == 2
    assert "memory_id" not in json.dumps(agent_data.extra_fields["opd_mm"])


@pytest.mark.asyncio
async def test_verl_native_filter_scope_can_restart_hidden_pool() -> None:
    records = [record.to_dict() for record in _records()]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Find user memories."}],
        tools_kwargs={"opd_mm": {"query": "Find user memories.", "records": records}},
    )
    filter_tool = OPDFilterTool(config={"type": "native"}, tool_schema=None)

    await filter_tool.execute("instance", {"field": "modality", "op": "eq", "value": "image"}, agent_data=agent_data)
    await filter_tool.execute("instance", {"field": "author", "op": "eq", "value": "assistant"}, agent_data=agent_data)
    response, _, _ = await filter_tool.execute(
        "instance",
        {"field": "author", "op": "eq", "value": "user", "scope": "full_memory"},
        agent_data=agent_data,
    )

    observation = json.loads(response.text)
    assert observation["pool_count"] == 4
    assert {item["author"] for item in observation["pool_preview"]} == {"assistant", "user"}
    assert observation["evidence_count"] == 4


@pytest.mark.asyncio
async def test_retrieve_tool_adds_public_evidence_before_inspect_raw() -> None:
    records = [record.to_dict() for record in _records()]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Find the tabby cat on the sofa."}],
        tools_kwargs={"opd_mm": {"query": "Find the tabby cat on the sofa.", "records": records}},
    )
    retrieve_tool = OPDRetrieveTool(config={"type": "native"}, tool_schema=None)

    response, _, metrics = await retrieve_tool.execute(
        "instance",
        {"method": "bm25", "top_k": 1},
        agent_data=agent_data,
    )

    observation = json.loads(response.text)
    assert observation["evidence_count"] == 2
    assert metrics["opd_mm_evidence_count"] == 2
    assert {item["source"] for item in observation["new_evidence"]} == {"RETRIEVE"}
    assert {item["modality"] for item in observation["new_evidence"]} == {"text", "image"}
    assert all("visual_observation" not in item for item in observation["new_evidence"])
    assert "memory_id" not in json.dumps(observation)


@pytest.mark.asyncio
async def test_opd_stop_tool_requests_agent_loop_termination() -> None:
    records = [record.to_dict() for record in _records()]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Find the tabby cat on the sofa."}],
        tools_kwargs={"opd_mm": {"query": "Find the tabby cat on the sofa.", "records": records}},
    )
    stop_tool = OPDStopTool(config={"type": "native"}, tool_schema=None)

    response, _, metrics = await stop_tool.execute("instance", {}, agent_data=agent_data)

    observation = json.loads(response.text)
    assert observation["stopped"] is True
    assert agent_data.extra_fields["opd_mm"]["stopped"] is True
    assert metrics["opd_mm_terminate"] is True
    assert metrics["agent_loop_terminate"] is True


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
    assert "scope=full_memory" in row["prompt"][0]["content"]
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


def _minimal_distillation_batch() -> TensorDict:
    data = TensorDict(
        {
            "prompts": torch.tensor([[1, 2]], dtype=torch.long),
            "responses": torch.tensor([[3, 0]], dtype=torch.long),
            "response_mask": torch.tensor([[1, 0]], dtype=torch.long),
            "input_ids": torch.tensor([[1, 2, 3, 0]], dtype=torch.long),
            "attention_mask": torch.tensor([[1, 1, 1, 0]], dtype=torch.long),
            "position_ids": torch.tensor([[0, 1, 2, 0]], dtype=torch.long),
        },
        batch_size=1,
    )
    tu.assign_non_tensor(data, dp_size=1, batch_num_tokens=1, global_batch_size=1)
    return data


def test_opd_mm_xml_sft_loss_uses_marker_when_mask_is_missing() -> None:
    data = _minimal_distillation_batch()
    tu.assign_non_tensor(data, opd_mm_sft_batch=True)
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    model_output = {"log_probs": torch.tensor([0.0, -0.5, -0.25])}

    loss, metrics = distillation_ppo_loss(
        config,
        DistillationConfig(),
        model_output=model_output,
        data=data,
    )

    assert loss.item() == pytest.approx(0.5)
    assert metrics["distillation/opd_mm_sft_tokens"].values == [1.0]


def test_opd_mm_batch_without_teacher_logprobs_returns_zero_loss() -> None:
    data = _minimal_distillation_batch()
    tu.assign_non_tensor(data, data_source="opd_mm")
    config = ActorConfig(strategy="fsdp", rollout_n=1, ppo_micro_batch_size_per_gpu=1)
    model_output = {"log_probs": torch.tensor([0.0, -0.5, -0.25])}

    loss, metrics = distillation_ppo_loss(
        config,
        DistillationConfig(),
        model_output=model_output,
        data=data,
    )

    assert loss.item() == pytest.approx(0.0)
    assert metrics["distillation/opd_mm_no_supervision_batches"].values == [1.0]


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


def test_teacher_xml_correction_extracts_only_canonical_tool_call() -> None:
    parsed = extract_canonical_tool_call_xml(
        "Reasoning first.\n"
        "<tool_call>\n"
        "<function=retrieve>\n"
        "<parameter=method>\nhybrid\n</parameter>\n"
        "<parameter=top_k>\n5\n</parameter>\n"
        "</function>\n"
        "</tool_call>\nDone."
    )

    assert parsed is not None
    target_xml, action, raw_xml = parsed
    assert action.tool == "RETRIEVE"
    assert action.arguments == {"method": "hybrid", "top_k": 5}
    assert raw_xml.startswith("<tool_call>")
    assert target_xml == (
        "<tool_call>\n"
        "<function=retrieve>\n"
        "<parameter=method>\n"
        "hybrid\n"
        "</parameter>\n"
        "<parameter=top_k>\n"
        "5\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )


def test_teacher_xml_correction_accepts_rewritten_retrieve_query() -> None:
    parsed = extract_canonical_tool_call_xml(
        "<tool_call>\n"
        "<function=retrieve>\n"
        "<parameter=method>\nhybrid\n</parameter>\n"
        "<parameter=top_k>\n10\n</parameter>\n"
        "<parameter=query>\npark walk dog YYYY-MM-DD\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    assert parsed is not None
    target_xml, action, _ = parsed
    assert action.tool == "RETRIEVE"
    assert action.arguments == {"method": "hybrid", "top_k": 10, "query": "park walk dog YYYY-MM-DD"}
    assert "<parameter=query>\npark walk dog YYYY-MM-DD\n</parameter>" in target_xml


def test_teacher_xml_correction_recovers_unclosed_rewritten_retrieve_query() -> None:
    parsed = extract_canonical_tool_call_xml(
        "<tool_call>\n"
        "<function=retrieve>\n"
        "<parameter=method>\nhybrid</parameter>\n"
        "<parameter=top_k>\n5</parameter>\n"
        "<parameter=query>\nMaria travel Paris date YYYY-MM-DD</parameter>\n"
        "</parameter>\n"
    )

    assert parsed is not None
    target_xml, action, _ = parsed
    assert action.arguments == {"method": "hybrid", "top_k": 5, "query": "Maria travel Paris date YYYY-MM-DD"}
    assert target_xml.endswith("</tool_call>")


def test_online_xml_correction_requests_include_invalid_student_state() -> None:
    requests = build_online_step_correction_requests(
        sample_kwargs={
            "raw_prompt": [{"role": "user", "content": "Find the tabby cat sofa memory."}],
            "tools_kwargs": {
                "opd_mm": {
                    "query": "Find the tabby cat sofa memory.",
                    "records": [record.to_dict(include_internal_id=True) for record in _records()],
                }
            },
            "extra_info": {
                "gold_answer": "SECRET_GOLD_ANSWER",
                "opd_mm_online_self_distill": True,
                "sample_id": "online-xml-sample",
            },
        },
        output_extra_fields={
            "opd_mm_generation_snapshots": [
                {
                    "prompt_ids": [11, 22, 33],
                    "response_text": "I should search the memories, but I forgot the XML.",
                    "parsed_tool_calls": [],
                }
            ]
        },
    )

    assert len(requests) == 1
    assert requests[0]["student_prompt_ids"] == [11, 22, 33]
    assert requests[0]["student_next_action"] is None
    assert "teacher_prompt" not in requests[0]
    assert "You are the OPD-MM state verifier" in requests[0]["verifier_prompt"]
    assert "Private rubric:" in requests[0]["verifier_prompt"]
    assert "Tool/action guide:" in requests[0]["verifier_prompt"]
    assert "retrieve(method=bm25|dense|vision|hybrid" in requests[0]["verifier_prompt"]
    assert "filter(field=modality|author|source_type|timestamp|status" in requests[0]["verifier_prompt"]
    assert "inspect_raw(target=current_pool" in requests[0]["verifier_prompt"]
    assert "SECRET_GOLD_ANSWER" in requests[0]["verifier_prompt"]

    verifier_feedback = {
        "evidence_sufficient": False,
        "reason": "Need public retrieval evidence before answering.",
        "recommended_next_action": "retrieve",
        "parse_error": "",
    }
    requests[0]["verifier_feedback"] = verifier_feedback
    requests[0]["teacher_prompt"] = build_teacher_correction_prompt(
        query=requests[0]["query"],
        history=requests[0]["history"],
        observation=requests[0]["observation"],
        student_raw_response=requests[0]["student_raw_response"],
        verifier_feedback=verifier_feedback,
        gold_answer="SECRET_GOLD_ANSWER",
        allow_inspect_raw=requests[0]["allow_inspect_raw"],
        tool_format=requests[0]["tool_format"],
    )
    assert "Teacher role:" in requests[0]["teacher_prompt"]
    assert "Correct exactly one next tool action" in requests[0]["teacher_prompt"]
    assert "Verifier feedback role:" in requests[0]["teacher_prompt"]
    assert "The verifier saw the gold answer; you did not." in requests[0]["teacher_prompt"]
    assert "Gold answer:" not in requests[0]["teacher_prompt"]
    assert "SECRET_GOLD_ANSWER" not in requests[0]["teacher_prompt"]
    assert "Never copy verifier reason into RETRIEVE.query" in requests[0]["teacher_prompt"]
    assert "not an instruction to copy" in requests[0]["teacher_prompt"]
    assert "history/trace is empty and evidence_count is 0, do not output stop" in requests[0]["teacher_prompt"]
    assert 'If the JSON observation above has "evidence_count": 0 and "trace": [], stop is invalid' in requests[
        0
    ]["teacher_prompt"]
    assert "If verifier.evidence_sufficient is false, STOP is invalid" in requests[0]["teacher_prompt"]
    assert "Do not use verifier.reason as lexical material for RETRIEVE.query" in requests[0]["teacher_prompt"]
    assert "scope=full_memory" in requests[0]["teacher_prompt"]
    assert "optionally query as rewritten search text" in requests[0]["teacher_prompt"]

    correction = finalize_online_step_correction(
        requests[0],
        teacher_raw_response='<tool_call>{"name":"retrieve","arguments":{"method":"bm25","top_k":1}}</tool_call>',
    )
    assert correction is not None
    assert correction["teacher_actions"][0]["tool"] == "RETRIEVE"
    assert correction["sft_prompt_ids"] == [11, 22, 33]
    assert correction["sft_target_xml"] == (
        "<tool_call>\n"
        "<function=retrieve>\n"
        "<parameter=method>\n"
        "bm25\n"
        "</parameter>\n"
        "<parameter=top_k>\n"
        "1\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    assert correction["feedback"] == verifier_feedback


def test_state_verifier_feedback_parser_accepts_wrapped_json_and_blocks_stop() -> None:
    feedback = parse_state_verifier_feedback(
        "```json\n"
        '{"evidence_sufficient": false, "reason": "Need broader public evidence.", '
        '"recommended_next_action": "STOP"}\n'
        "```",
        {"evidence_count": 3},
    )

    assert feedback["evidence_sufficient"] is False
    assert feedback["recommended_next_action"] == "retrieve"
    assert feedback["parse_error"] == ""


def test_state_verifier_feedback_sanitizes_gold_answer_leakage() -> None:
    feedback = parse_state_verifier_feedback(
        '{"evidence_sufficient": false, '
        "\"reason\": \"No evidence found mentioning Lena's brother or a cat named Miso.\", "
        '"recommended_next_action": "retrieve"}',
        {"evidence_count": 0},
        gold_answer="Miso",
        query="What is the name of Lena’s brother’s cat?",
    )

    assert feedback["evidence_sufficient"] is False
    assert feedback["recommended_next_action"] == "retrieve"
    assert "Miso" not in feedback["reason"]
    assert "gold answer" not in feedback["reason"].lower()
    assert feedback["reason"] == "Current public evidence is insufficient; collect relevant evidence first."


def test_state_verifier_feedback_parser_falls_back_on_invalid_action() -> None:
    feedback = parse_state_verifier_feedback(
        '{"evidence_sufficient": true, "reason": "Looks enough.", "recommended_next_action": "jump"}',
        {"evidence_count": 2},
    )

    # "jump" is intentionally not a verifier action; use the safe non-leaking fallback.
    assert feedback["evidence_sufficient"] is False
    assert feedback["recommended_next_action"] == "retrieve"
    assert "invalid recommended_next_action" in feedback["parse_error"]


def test_online_xml_correction_stop_gate_replaces_insufficient_teacher_stop() -> None:
    request = {
        "sample_id": "stop-gate",
        "step_index": 0,
        "query": "What dance styles were mentioned?",
        "history": [],
        "observation": {"evidence_count": 0, "trace": []},
        "student_raw_response": "<tool_call><function=stop></function></tool_call>",
        "student_prompt_ids": [7, 8, 9],
        "allow_inspect_raw": True,
        "tool_format": "qwen3_coder",
        "verifier_raw_response": '{"evidence_sufficient": false, "reason": "Need list coverage.", '
        '"recommended_next_action": "retrieve"}',
        "verifier_feedback": {
            "evidence_sufficient": False,
            "reason": "Need list coverage.",
            "recommended_next_action": "retrieve",
            "parse_error": "",
        },
    }

    correction = finalize_online_step_correction(
        request,
        teacher_raw_response="<tool_call>\n<function=stop>\n</function>\n</tool_call>",
    )

    assert correction is not None
    assert correction["stop_gate_applied"] is True
    assert correction["teacher_actions"][0]["tool"] == "RETRIEVE"
    assert "<function=retrieve>" in correction["sft_target_xml"]


def test_online_teacher_correction_dump_writes_jsonl(tmp_path, monkeypatch) -> None:
    request = {
        "sample_id": "sample-dump",
        "step_index": 0,
        "query": "Find the cat memory.",
        "gold_answer": "cat",
        "teacher_prompt": "teacher prompt",
        "student_raw_response": "student raw",
        "student_prompt_ids": [1, 2, 3],
        "allow_inspect_raw": True,
        "tool_format": "qwen3_coder",
    }
    teacher_raw = (
        "<tool_call>\n"
        "<function=retrieve>\n"
        "<parameter=method>\nbm25\n</parameter>\n"
        "<parameter=top_k>\n1\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )
    correction = finalize_online_step_correction(request, teacher_raw_response=teacher_raw)
    monkeypatch.setenv("OPD_MM_TEACHER_CORRECTION_DUMP_DIR", str(tmp_path))

    dump_online_step_correction(request, teacher_raw_response=teacher_raw, correction=correction)

    files = list(tmp_path.glob("teacher_corrections_pid*.jsonl"))
    assert len(files) == 1
    record = json.loads(files[0].read_text(encoding="utf-8"))
    assert record["sample_id"] == "sample-dump"
    assert record["parsed"] is True
    assert record["teacher_actions"][0]["tool"] == "RETRIEVE"
    assert "<function=retrieve>" in record["sft_target_xml"]


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
    schemas = openai_tool_schemas(include_inspect_raw=False)
    assert [schema["function"]["name"] for schema in schemas] == [
        "filter",
        "sort",
        "topk",
        "retrieve",
        "stop",
    ]
    filter_scope = schemas[0]["function"]["parameters"]["properties"]["scope"]
    assert filter_scope["enum"] == ["current_pool", "full_memory"]
