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
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from verl import DataProto
from verl.experimental.agent_loop.agent_loop import AgentLoopWorker
from verl.experimental.agent_loop.tool_agent_loop import AgentState, ToolAgentLoop
from verl.experimental.agent_loop.tool_parser import FunctionCall
from verl.experimental.opd_mm import MemoryRecord, OPDSample, OnPolicyDistiller, PolicyOutput, ToolAction, ToolExecutor
from verl.experimental.opd_mm.dataset import (
    OPD_MM_SYSTEM_PROMPT,
    opd_messages_for_query,
    opd_messages_for_state,
    opd_sample_to_rlhf_record,
)
from verl.experimental.opd_mm.online_self_distill import (
    build_online_state_correction_request,
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
    OPDExpandNeighborsTool,
    OPDFilterTool,
    OPDInspectRawTool,
    OPDRetrieveTool,
    OPDStopTool,
    hidden_store_from_records,
    openai_tool_schemas,
)
from verl.trainer.distillation.losses import distillation_ppo_loss
from verl.tools.schemas import ToolResponse
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
            source_type="dialogue_turn",
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
            source_type="dialogue_turn",
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
            source_type="dialogue_image",
            summary="A tabby cat sitting on a sofa.",
            raw_pointer="images/cat.png",
            metadata={"session_date": "2026-01-01", "image_id": "D1:IMG_001"},
        ),
        MemoryRecord(
            memory_id="m_assistant_image",
            turn_id="3",
            timestamp="2026-01-01T11:00:00",
            author="assistant",
            modality="image",
            source_type="dialogue_turn",
            summary="An assistant-generated chart.",
            raw_pointer="images/chart.png",
            metadata={"image_id": "D1:IMG_002"},
        ),
    ]


def _neighbor_records() -> list[MemoryRecord]:
    return [
        MemoryRecord(
            memory_id="n_prev",
            turn_id="scenario_a:D1:1",
            timestamp="2026-02-01T09:00:00",
            author="user",
            modality="text",
            source_type="dialogue_turn",
            summary="The previous turn contains setup context.",
            content="Previous context before the middle clue.",
            metadata={"scenario": "scenario_a", "session_id": "D1", "round_id": "D1:1"},
        ),
        MemoryRecord(
            memory_id="n_mid",
            turn_id="scenario_a:D1:2",
            timestamp="2026-02-01T09:01:00",
            author="user",
            modality="text",
            source_type="dialogue_turn",
            summary="The middle turn mentions the unique middle clue.",
            content="The unique middle clue is here.",
            metadata={"scenario": "scenario_a", "session_id": "D1", "round_id": "D1:2"},
        ),
        MemoryRecord(
            memory_id="n_next",
            turn_id="scenario_a:D1:3",
            timestamp="2026-02-01T09:02:00",
            author="assistant",
            modality="text",
            source_type="dialogue_turn",
            summary="The next turn contains follow-up context.",
            content="Follow-up context after the middle clue.",
            metadata={"scenario": "scenario_a", "session_id": "D1", "round_id": "D1:3"},
        ),
        MemoryRecord(
            memory_id="n_far",
            turn_id="scenario_a:D1:5",
            timestamp="2026-02-01T09:04:00",
            author="assistant",
            modality="text",
            source_type="dialogue_turn",
            summary="A farther turn should not be added with window one.",
            metadata={"scenario": "scenario_a", "session_id": "D1", "round_id": "D1:5"},
        ),
        MemoryRecord(
            memory_id="n_other_scenario",
            turn_id="scenario_b:D1:3",
            timestamp="2026-02-01T09:02:00",
            author="assistant",
            modality="text",
            source_type="dialogue_turn",
            summary="Same session label but different scenario.",
            metadata={"scenario": "scenario_b", "session_id": "D1", "round_id": "D1:3"},
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


class PoolAwareRetriever:
    def __init__(self) -> None:
        self.source_pool_ids: list[list[str]] = []

    def retrieve(
        self,
        pool: list[Any],
        query: str,
        store: HiddenMemoryStore,
        method: str = "hybrid",
        top_k: int = 5,
        question_image: str | None = None,
    ) -> list[Any]:
        del query, method, top_k, question_image
        self.source_pool_ids.append([item.memory.memory_id for item in pool])
        records = {item.memory.memory_id: item for item in store.initial_pool()}
        return [records["m_cat_text"]] if len(self.source_pool_ids) == 1 else list(pool[:1])


def test_validator_rejects_memory_ids_and_accepts_rewritten_retrieve_query() -> None:
    validator = TrajectoryValidator(max_actions=4)

    with pytest.raises(TrajectoryValidationError, match="memory IDs"):
        validator.validate([{"tool": "FILTER", "field": "status", "op": "eq", "value": "m_001"}])

    validated_filter = validator.validate(
        [{"tool": "FILTER", "field": "modality", "op": "eq", "value": "image", "scope": "full_memory"}]
    )
    assert validated_filter[0].arguments["scope"] == "full_memory"

    with pytest.raises(TrajectoryValidationError, match="invalid FILTER field"):
        validator.validate(
            [{"tool": "FILTER", "field": "author", "op": "eq", "value": "user", "scope": "full_memory"}]
        )

    with pytest.raises(TrajectoryValidationError, match="invalid FILTER scope"):
        validator.validate([{"tool": "FILTER", "field": "modality", "op": "eq", "value": "image", "scope": "all"}])

    invalid_filter_values = (
        ("modality", "dialogue"),
        ("source_type", "dialogue"),
        ("source_type", "user"),
        ("source_type", "MEMORY"),
        ("source_type", "assistant"),
        ("status", "completed"),
    )
    for field, value in invalid_filter_values:
        with pytest.raises(TrajectoryValidationError, match=f"invalid FILTER value for {field}"):
            validator.validate(
                [{"tool": "FILTER", "field": field, "op": "eq", "value": value, "scope": "full_memory"}]
            )

    validated = validator.validate([{"tool": "RETRIEVE", "query": "custom query", "top_k": 3}])
    assert validated[0].arguments["query"] == "custom query"

    with pytest.raises(TrajectoryValidationError, match="memory IDs"):
        validator.validate([{"tool": "RETRIEVE", "query": "look up m_001", "top_k": 3}])

    with pytest.raises(TrajectoryValidationError, match="unknown arguments"):
        validator.validate([{"tool": "RETRIEVE", "query": "custom query", "top_k": 3, "scope": "full_memory"}])


def test_validator_accepts_expand_neighbors_and_rejects_invalid_arguments() -> None:
    validator = TrajectoryValidator(max_actions=4)

    validated = validator.validate([{"tool": "EXPAND_NEIGHBORS", "window": 2}])
    assert validated[0].tool == "EXPAND_NEIGHBORS"
    assert validated[0].arguments == {"window": 2}

    for bad_window in (0, 4, "1", True):
        with pytest.raises(TrajectoryValidationError, match="window must be one of"):
            validator.validate([{"tool": "EXPAND_NEIGHBORS", "window": bad_window}])

    with pytest.raises(TrajectoryValidationError, match="unknown arguments"):
        validator.validate([{"tool": "EXPAND_NEIGHBORS", "window": 1, "extra": "nope"}])

    with pytest.raises(TrajectoryValidationError, match="forbidden arguments"):
        validator.validate([{"tool": "EXPAND_NEIGHBORS", "window": 1, "memory_id": "safe"}])


def test_opd_state_prompt_keeps_action_history_and_only_latest_observation() -> None:
    base = opd_messages_for_query("Where did Maya travel?")
    first = opd_messages_for_state(
        base,
        [{"tool": "RETRIEVE", "method": "bm25", "top_k": 5}],
        {"pool_count": 5, "evidence_preview": [{"content": "old retrieval result"}]},
    )
    second = opd_messages_for_state(
        base,
        [
            {"tool": "RETRIEVE", "method": "bm25", "top_k": 5},
            {"tool": "FILTER", "field": "timestamp", "op": "eq", "value": "2024-09-01", "scope": "full_memory"},
        ],
        {"pool_count": 2, "evidence_preview": [{"content": "latest refreshed result"}]},
    )

    assert "old retrieval result" in first[-1]["content"]
    assert "old retrieval result" not in second[-1]["content"]
    assert "latest refreshed result" in second[-1]["content"]
    assert '"tool":"RETRIEVE"' in second[-1]["content"]
    assert '"tool":"FILTER"' in second[-1]["content"]
    assert len(second) == len(base)


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
            {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image", "scope": "full_memory"},
            {
                "tool": "FILTER",
                "field": "source_type",
                "op": "eq",
                "value": "dialogue_turn",
                "scope": "current_pool",
            },
            {
                "tool": "FILTER",
                "field": "source_type",
                "op": "eq",
                "value": "dialogue_image",
                "scope": "current_pool",
            },
        ],
        query="Show user memories after a wrong narrow filter.",
        memory_store=store,
    )
    assert narrowed_result.final_memory_ids == []

    reset_result = ToolExecutor().run(
        [
            {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image", "scope": "full_memory"},
            {
                "tool": "FILTER",
                "field": "source_type",
                "op": "eq",
                "value": "dialogue_turn",
                "scope": "current_pool",
            },
            {"tool": "FILTER", "field": "source_type", "op": "eq", "value": "dialogue_image", "scope": "full_memory"},
        ],
        query="Show user memories after resetting the filter scope.",
        memory_store=store,
    )
    assert reset_result.final_memory_ids == ["m_cat_image"]
    assert reset_result.steps[2].pool_before == 1
    assert reset_result.steps[2].pool_after == 1
    assert len({item.memory_id for item in reset_result.evidence}) == len(reset_result.evidence)


def test_repeated_retrieve_replaces_pool_by_default() -> None:
    result = ToolExecutor(retriever=QuerySwitchRetriever()).run(
        [
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 1, "query": "cat"},
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 1, "query": "cat"},
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 1, "query": "assistant"},
        ],
        query="Find multiple memories.",
        memory_store=HiddenMemoryStore(_records()),
    )

    assert result.final_memory_ids == ["m_assistant_image"]
    assert [item.memory_id for item in result.evidence] == ["m_assistant_image"]


def test_repeated_retrieve_always_searches_original_memory_store() -> None:
    retriever = PoolAwareRetriever()
    result = ToolExecutor(retriever=retriever).run(
        [
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 1, "query": "first"},
            {"tool": "RETRIEVE", "method": "hybrid", "top_k": 1, "query": "second"},
        ],
        query="Find multiple memories.",
        memory_store=HiddenMemoryStore(_records()),
    )

    assert not result.error
    assert retriever.source_pool_ids[0] == ["m_text_old", "m_cat_text", "m_cat_image", "m_assistant_image"]
    assert retriever.source_pool_ids[1] == ["m_text_old", "m_cat_text", "m_cat_image", "m_assistant_image"]
    assert result.final_memory_ids == ["m_text_old"]


def test_executor_composes_generic_tools_to_latest_user_image() -> None:
    store = HiddenMemoryStore(_records())
    result = ToolExecutor().run(
        [
            {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image", "scope": "full_memory"},
            {
                "tool": "FILTER",
                "field": "source_type",
                "op": "eq",
                "value": "dialogue_image",
                "scope": "current_pool",
            },
            {"tool": "SORT", "field": "timestamp", "order": "desc"},
            {"tool": "TOPK", "k": 1},
        ],
        query="Which image did I upload last?",
        memory_store=store,
    )

    assert not result.error
    assert result.stopped
    assert result.final_memory_ids == ["m_cat_image"]
    assert result.evidence[0].fields["content"] == "A tabby cat sitting on a sofa."
    assert result.evidence[0].fields["image_id"] == "D1:IMG_001"
    assert "raw_pointer" not in result.evidence[0].fields
    assert "summary" not in result.evidence[0].fields
    assert "turn_id" not in result.evidence[0].fields
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
            source_type="dialogue_turn",
            summary="A next-day note.",
        ),
    ]
    store = HiddenMemoryStore(records)

    eq_result = ToolExecutor().run(
        [{"tool": "FILTER", "field": "timestamp", "op": "eq", "value": "2026-01-01", "scope": "full_memory"}],
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
        [
            {
                "tool": "FILTER",
                "field": "timestamp",
                "op": "contains",
                "value": "2026/1/1",
                "scope": "full_memory",
            }
        ],
        query="Which memories are from 2026-01-01?",
        memory_store=store,
    )
    assert set(contains_result.final_memory_ids) == set(eq_result.final_memory_ids)

    before_result = ToolExecutor().run(
        [{"tool": "FILTER", "field": "timestamp", "op": "before", "value": "2026-01-02", "scope": "full_memory"}],
        query="Which memories are before 2026-01-02?",
        memory_store=store,
    )
    assert set(before_result.final_memory_ids) == set(eq_result.final_memory_ids)

    after_result = ToolExecutor().run(
        [{"tool": "FILTER", "field": "timestamp", "op": "after", "value": "2026-01-01", "scope": "full_memory"}],
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
    assert {item.source for item in result.evidence} == {"MEMORY"}
    assert result.steps[0].evidence_added == 2


def test_expand_neighbors_refreshes_evidence_from_expanded_pool() -> None:
    store = HiddenMemoryStore(_neighbor_records())
    result = ToolExecutor(retriever=TurnAwareHybridRetriever(context_window=0)).run(
        [
            {"tool": "RETRIEVE", "method": "bm25", "top_k": 1, "query": "unique middle clue"},
            {"tool": "EXPAND_NEIGHBORS", "window": 1},
        ],
        query="Need surrounding context for the middle clue.",
        memory_store=store,
    )

    assert not result.error
    assert {item.memory_id for item in result.evidence} == {"n_prev", "n_mid", "n_next"}
    assert "n_far" not in {item.memory_id for item in result.evidence}
    assert "n_other_scenario" not in {item.memory_id for item in result.evidence}
    assert result.evidence[0].memory_id == "n_mid"
    assert {item.source for item in result.evidence} == {"MEMORY"}
    assert result.steps[1].evidence_added == 2


def test_expand_neighbors_requires_existing_candidate_pool_and_does_not_expose_full_memory_on_error() -> None:
    result = ToolExecutor().run(
        [{"tool": "EXPAND_NEIGHBORS", "window": 1}],
        query="Expand before any search.",
        memory_store=HiddenMemoryStore(_neighbor_records()),
    )

    assert "EXPAND_NEIGHBORS requires an existing candidate pool" in result.error
    assert result.evidence == []
    assert result.steps[0].error


def test_inspect_raw_only_reads_retrieved_candidate_pool() -> None:
    store = HiddenMemoryStore(_records())
    inspector = FakeRawInspector()

    filtered_result = ToolExecutor(raw_inspector=inspector).run(
        [
            {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image", "scope": "full_memory"},
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
async def test_tool_agent_loop_rebuilds_opd_prompt_without_old_observations() -> None:
    loop = object.__new__(ToolAgentLoop)
    loop.max_parallel_calls = 1
    loop.enable_continuous_token = False
    loop.tool_parser_name = "qwen3_coder"
    loop.response_length = 2048
    loop.tool_schemas = []
    captured: dict[str, Any] = {}

    async def fake_call_tool(tool_call: Any, tools_kwargs: dict[str, Any], agent_data: Any) -> tuple[Any, ...]:
        del tool_call, tools_kwargs, agent_data
        return ToolResponse(text="latest tool response"), 0.0, {}

    async def fake_apply_chat_template(messages: list[dict[str, Any]], **kwargs: Any) -> list[int]:
        captured["messages"] = messages
        captured["kwargs"] = kwargs
        return [101, 102, 103]

    loop._call_tool = fake_call_tool
    loop.apply_chat_template = fake_apply_chat_template
    base_messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "Find Maya's trip."},
    ]
    agent_data = SimpleNamespace(
        tool_calls=[SimpleNamespace(name="retrieve", tool_call_id=None)],
        tools_kwargs={},
        metrics={},
        messages=[*base_messages, {"role": "tool", "content": "stale retrieval observation"}],
        base_messages=base_messages,
        extra_fields={
            "opd_mm_prompt_state": {
                "action_history": [{"tool": "RETRIEVE", "method": "bm25", "top_k": 5}],
                "observation": {"pool_count": 2, "evidence_preview": [{"content": "latest result"}]},
            }
        },
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs={},
        tool_rewards=[],
        user_turns=0,
        prompt_ids=[1, 2],
        response_mask=[],
        distillation_mask=[],
        response_logprobs=[],
    )

    state = await ToolAgentLoop._handle_processing_tools_state(loop, agent_data)

    assert state == AgentState.GENERATING
    assert agent_data.prompt_ids == [101, 102, 103]
    assert agent_data.messages == captured["messages"]
    prompt_text = json.dumps(captured["messages"], ensure_ascii=False)
    assert "latest result" in prompt_text
    assert "stale retrieval observation" not in prompt_text
    assert '"tool":"RETRIEVE"' in captured["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_verl_native_opd_tools_share_hidden_state_and_hide_ids() -> None:
    records = [record.to_dict() for record in _records()]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Read my latest user image."}],
        tools_kwargs={"opd_mm": {"query": "Read my latest user image.", "records": records}},
    )
    filter_tool = OPDFilterTool(config={"type": "native"}, tool_schema=None)

    await filter_tool.execute(
        "instance",
        {"field": "modality", "op": "eq", "value": "image", "scope": "full_memory"},
        agent_data=agent_data,
    )
    response, _, metrics = await filter_tool.execute(
        "instance",
        {"field": "source_type", "op": "eq", "value": "dialogue_image", "scope": "current_pool"},
        agent_data=agent_data,
    )

    observation = json.loads(response.text)
    assert observation["pool_count"] == 1
    assert observation["pool_preview"][0]["content"] == "A tabby cat sitting on a sofa."
    assert observation["pool_preview"][0]["image_id"] == "D1:IMG_001"
    assert observation["evidence_preview"][0]["image_id"] == "D1:IMG_001"
    assert "raw_pointer" not in observation["pool_preview"][0]
    assert "summary" not in observation["pool_preview"][0]
    assert "source_type" not in observation["pool_preview"][0]
    assert "turn_id" not in observation["pool_preview"][0]
    assert metrics["opd_mm_evidence_count"] == 1
    assert "memory_id" not in json.dumps(observation)
    assert agent_data.extra_fields["opd_mm"]["pool_count"] == 1
    assert agent_data.extra_fields["opd_mm"]["evidence_count"] == 1
    assert agent_data.extra_fields["opd_mm"]["evidence"][0]["image_id"] == "D1:IMG_001"
    assert "source" not in agent_data.extra_fields["opd_mm"]["evidence"][0]
    assert "author" not in agent_data.extra_fields["opd_mm"]["evidence"][0]
    assert "memory_id" not in json.dumps(agent_data.extra_fields["opd_mm"])


@pytest.mark.asyncio
async def test_verl_native_filter_scope_can_restart_hidden_pool() -> None:
    records = [record.to_dict() for record in _records()]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Find user memories."}],
        tools_kwargs={"opd_mm": {"query": "Find user memories.", "records": records}},
    )
    filter_tool = OPDFilterTool(config={"type": "native"}, tool_schema=None)

    await filter_tool.execute(
        "instance",
        {"field": "modality", "op": "eq", "value": "image", "scope": "full_memory"},
        agent_data=agent_data,
    )
    await filter_tool.execute(
        "instance",
        {"field": "source_type", "op": "eq", "value": "dialogue_turn", "scope": "current_pool"},
        agent_data=agent_data,
    )
    response, _, _ = await filter_tool.execute(
        "instance",
        {"field": "source_type", "op": "eq", "value": "dialogue_image", "scope": "full_memory"},
        agent_data=agent_data,
    )

    observation = json.loads(response.text)
    assert observation["pool_count"] == 1
    assert observation["evidence_count"] == 1


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
    assert observation["new_evidence_count"] == 2
    assert all("source" not in item and "author" not in item for item in observation["evidence_preview"])
    assert {item["modality"] for item in observation["evidence_preview"]} == {"text", "image"}
    image_preview = [item for item in observation["evidence_preview"] if item["modality"] == "image"]
    text_preview = [item for item in observation["evidence_preview"] if item["modality"] == "text"]
    assert image_preview[0]["image_id"] == "D1:IMG_001"
    assert "image_id" not in text_preview[0]
    assert all("visual_observation" not in item for item in observation["evidence_preview"])
    assert "memory_id" not in json.dumps(observation)
    assert "last_action" not in observation
    prompt_state = agent_data.extra_fields["opd_mm_prompt_state"]
    assert prompt_state["action_history"] == [{"tool": "RETRIEVE", "method": "bm25", "top_k": 1}]
    assert prompt_state["observation"] == observation


@pytest.mark.asyncio
async def test_tool_observation_stays_bounded_when_refreshed_evidence_grows() -> None:
    records = [
        MemoryRecord(
            memory_id=f"bounded_{index}",
            turn_id=f"turn_{index}",
            timestamp=f"2026-03-{index + 1:02d}T10:00:00",
            author="user",
            modality="text",
            source_type="dialogue_turn",
            summary=f"summary {index}",
            content=f"memory {index} " + ("x" * 1000),
        ).to_dict()
        for index in range(20)
    ]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Find all memories."}],
        tools_kwargs={"opd_mm": {"query": "Find all memories.", "records": records, "vector_store_dir": None}},
    )
    filter_tool = OPDFilterTool(config={"type": "native"}, tool_schema=None)

    response, _, _ = await filter_tool.execute(
        "instance",
        {"field": "modality", "op": "eq", "value": "text", "scope": "full_memory"},
        agent_data=agent_data,
    )

    observation = json.loads(response.text)
    assert observation["pool_count"] == 20
    assert observation["evidence_count"] == 20
    assert len(observation["pool_preview"]) == 3
    assert len(observation["evidence_preview"]) == 4
    assert all(len(item["content"]) <= 234 for item in observation["pool_preview"])
    assert all(len(item["content"]) <= 234 for item in observation["evidence_preview"])
    assert "new_evidence" not in observation
    assert "last_action" not in observation
    assert len(response.text) < 4000


@pytest.mark.asyncio
async def test_inspect_raw_can_use_async_teacher_service_callback() -> None:
    records = [record.to_dict() for record in _records()]
    calls: list[dict[str, Any]] = []

    async def teacher_inspect(payload: dict[str, Any]) -> str:
        calls.append(payload)
        return "Teacher sees a tabby cat sitting on a sofa."

    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "What is in the cat image?"}],
        tools_kwargs={
            "opd_mm": {
                "query": "What is in the cat image?",
                "records": records,
                "vector_store_dir": None,
                "raw_inspector_backend": "teacher",
            }
        },
    )
    agent_data.teacher_raw_inspector = teacher_inspect
    retrieve_tool = OPDRetrieveTool(config={"type": "native", "raw_inspector_backend": "teacher"}, tool_schema=None)
    inspect_tool = OPDInspectRawTool(config={"type": "native", "raw_inspector_backend": "teacher"}, tool_schema=None)

    await retrieve_tool.execute(
        "instance",
        {"method": "bm25", "top_k": 1, "query": "tabby cat sofa"},
        agent_data=agent_data,
    )
    response, _, metrics = await inspect_tool.execute(
        "instance",
        {
            "target": "current_pool",
            "instruction": "answer_query_related_visual_details",
        },
        agent_data=agent_data,
    )

    observation = json.loads(response.text)
    assert len(calls) == 1
    assert calls[0]["raw_pointer"] == "images/cat.png"
    assert calls[0]["query"] == "What is in the cat image?"
    assert observation["error"] == ""
    assert observation["new_evidence_count"] == 1
    assert "source" not in observation["evidence_preview"][-1]
    assert "author" not in observation["evidence_preview"][-1]
    assert observation["evidence_preview"][-1]["image_id"] == "D1:IMG_001"
    assert observation["evidence_preview"][-1]["visual_observation"] == "Teacher sees a tabby cat sitting on a sofa."
    assert metrics["agent_loop_terminate"] is False


@pytest.mark.asyncio
async def test_verl_native_expand_neighbors_observation_is_visible_next_step() -> None:
    records = [record.to_dict() for record in _neighbor_records()]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Find context around the unique middle clue."}],
        tools_kwargs={
            "opd_mm": {
                "query": "Find context around the unique middle clue.",
                "records": records,
                "vector_store_dir": None,
            }
        },
    )
    retrieve_tool = OPDRetrieveTool(config={"type": "native"}, tool_schema=None)
    expand_tool = OPDExpandNeighborsTool(config={"type": "native"}, tool_schema=None)

    await retrieve_tool.execute("instance", {"method": "bm25", "top_k": 1, "query": "unique middle clue"}, agent_data=agent_data)
    response, _, metrics = await expand_tool.execute("instance", {"window": 1}, agent_data=agent_data)

    observation = json.loads(response.text)
    assert observation["tool"] == "EXPAND_NEIGHBORS"
    assert observation["pool_count"] == 3
    assert observation["evidence_count"] == 3
    assert observation["new_evidence_count"] == 2
    assert all("source" not in item and "author" not in item for item in observation["evidence_preview"])
    assert {item["content"] for item in observation["evidence_preview"]} == {
        "The unique middle clue is here.",
        "Previous context before the middle clue.",
        "Follow-up context after the middle clue.",
    }
    assert metrics["opd_mm_terminate"] is False
    assert [item["tool"] for item in agent_data.extra_fields["opd_mm"]["trace"]] == [
        "RETRIEVE",
        "EXPAND_NEIGHBORS",
    ]
    assert "memory_id" not in json.dumps(observation)


@pytest.mark.asyncio
async def test_verl_native_expand_neighbors_without_candidates_errors_and_terminates() -> None:
    records = [record.to_dict() for record in _neighbor_records()]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Expand immediately."}],
        tools_kwargs={"opd_mm": {"query": "Expand immediately.", "records": records, "vector_store_dir": None}},
    )
    expand_tool = OPDExpandNeighborsTool(config={"type": "native"}, tool_schema=None)

    response, _, metrics = await expand_tool.execute("instance", {"window": 1}, agent_data=agent_data)

    observation = json.loads(response.text)
    assert "EXPAND_NEIGHBORS requires an existing candidate pool" in observation["error"]
    assert observation["evidence_count"] == 0
    assert metrics["agent_loop_terminate"] is True


@pytest.mark.asyncio
async def test_verl_native_max_action_forces_stop() -> None:
    records = [record.to_dict() for record in _records()]
    agent_data = FakeAgentData(
        messages=[{"role": "user", "content": "Find the tabby cat on the sofa."}],
        tools_kwargs={"opd_mm": {"query": "Find the tabby cat on the sofa.", "records": records}},
    )
    retrieve_tool = OPDRetrieveTool(config={"type": "native"}, tool_schema=None)

    for _ in range(7):
        response, _, metrics = await retrieve_tool.execute(
            "instance",
            {"method": "bm25", "top_k": 1},
            agent_data=agent_data,
        )
        observation = json.loads(response.text)
        assert observation["tool"] == "RETRIEVE"
        assert metrics["agent_loop_terminate"] is False

    response, _, metrics = await retrieve_tool.execute(
        "instance",
        {"method": "bm25", "top_k": 1},
        agent_data=agent_data,
    )
    observation = json.loads(response.text)
    assert observation["tool"] == "STOP"
    assert observation["stopped"] is True
    assert observation["error"] == ""
    assert metrics["agent_loop_terminate"] is True
    assert len(agent_data.extra_fields["opd_mm"]["trace"]) == 8
    assert agent_data.extra_fields["opd_mm"]["trace"][-1]["tool"] == "STOP"


@pytest.mark.asyncio
async def test_tool_agent_loop_collects_correction_from_each_live_state() -> None:
    payloads: list[dict[str, Any]] = []

    async def corrector(payload: dict[str, Any]) -> dict[str, Any]:
        payloads.append(payload)
        return {
            "step_index": payload["step_index"],
            "sft_prompt_ids": payload["student_prompt_ids"],
            "sft_target_xml": "<tool_call><function=stop></function></tool_call>",
        }

    loop = SimpleNamespace(
        tokenizer=SimpleNamespace(decode=lambda ids, skip_special_tokens=False: f"response-{ids[-1]}"),
        tool_parser_name="qwen3_coder",
    )
    agent_data = SimpleNamespace(
        online_state_corrector=corrector,
        extra_fields={
            "opd_mm": {
                "pool_count": 1,
                "evidence_count": 1,
                "trace": [{"tool": "RETRIEVE", "method": "bm25", "top_k": 1}],
                "evidence": [{"source": "RETRIEVE", "summary": "first evidence"}],
            }
        },
        tool_calls=[FunctionCall(name="stop", arguments="{}")],
        request_id="live-request",
        assistant_turns=2,
        image_data=None,
        video_data=None,
        audio_data=None,
        mm_processor_kwargs={},
    )

    await ToolAgentLoop._collect_online_state_correction(
        loop,
        agent_data=agent_data,
        state_prompt_ids=[1, 2, 3],
        response_ids=[4],
        assistant_content="",
    )
    agent_data.extra_fields["opd_mm"] = {
        "pool_count": 3,
        "evidence_count": 3,
        "trace": [
            {"tool": "RETRIEVE", "method": "bm25", "top_k": 1},
            {"tool": "EXPAND_NEIGHBORS", "window": 1},
        ],
        "evidence": [{"source": "EXPAND_NEIGHBORS", "summary": "neighbor evidence"}],
    }
    agent_data.tool_calls = [FunctionCall(name="expand_neighbors", arguments='{"window": 1}')]
    agent_data.assistant_turns = 3
    await ToolAgentLoop._collect_online_state_correction(
        loop,
        agent_data=agent_data,
        state_prompt_ids=[1, 2, 3, 4, 5],
        response_ids=[6],
        assistant_content="",
    )

    assert len(payloads) == 2
    assert payloads[0]["step_index"] == 1
    assert payloads[0]["observation"]["evidence_count"] == 1
    assert payloads[0]["student_next_action"] == {"tool": "STOP"}
    assert payloads[1]["step_index"] == 2
    assert payloads[1]["observation"]["evidence_count"] == 3
    assert payloads[1]["history"][-1]["tool"] == "EXPAND_NEIGHBORS"
    assert payloads[1]["student_prompt_ids"] == [1, 2, 3, 4, 5]
    assert len(agent_data.extra_fields["opd_mm_step_corrections"]) == 2


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
    assert [tool.name for tool in tools] == [
        "filter",
        "sort",
        "topk",
        "retrieve",
        "expand_neighbors",
        "inspect_raw",
        "stop",
    ]
    inspect_tool = next(tool for tool in tools if tool.name == "inspect_raw")
    assert inspect_tool.config["raw_inspector_backend"] == "teacher"
    assert "raw_inspector_url" not in inspect_tool.config


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
        "expand_neighbors",
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
    for tool_name in ("RETRIEVE", "FILTER", "SORT", "TOPK", "EXPAND_NEIGHBORS", "INSPECT_RAW", "STOP"):
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
                ToolAction(
                    "FILTER",
                    {
                        "field": "timestamp",
                        "op": "eq",
                        "value": "2026-01-01T11:00:00",
                        "scope": "full_memory",
                    },
                ),
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
        return " ".join(str(item.fields.get("content") or item.fields.get("summary", "")) for item in evidence)


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


def test_teacher_xml_correction_accepts_expand_neighbors_tool_call() -> None:
    parsed = extract_canonical_tool_call_xml(
        "<tool_call>\n"
        "<function=expand_neighbors>\n"
        "<parameter=window>\n1\n</parameter>\n"
        "</function>\n"
        "</tool_call>"
    )

    assert parsed is not None
    target_xml, action, _ = parsed
    assert action.tool == "EXPAND_NEIGHBORS"
    assert action.arguments == {"window": 1}
    assert "<function=expand_neighbors>" in target_xml


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


def test_teacher_xml_correction_repairs_missing_parameter_closes_and_aliases() -> None:
    parsed = extract_canonical_tool_call_xml(
        "<tool_call>\n"
        "<function=filter>\n"
        "<parameter=field>\n"
        "session_date\n"
        "<parameter=op>\n"
        "equals\n"
        "<parameter=value>\n"
        "2024-09-28\n"
        "<parameter=scope>\n"
        "current_pool\n"
        "</parameter>\n"
        "</function>\n"
        "</tool_call><|im_end|>"
    )

    assert parsed is not None
    target_xml, action, _ = parsed
    assert action.tool == "FILTER"
    assert action.arguments == {
        "field": "timestamp",
        "op": "eq",
        "value": "2024-09-28",
        "scope": "current_pool",
    }
    assert "<parameter=field>\ntimestamp\n</parameter>" in target_xml
    assert "<parameter=op>\neq\n</parameter>" in target_xml


def test_teacher_xml_correction_rejects_defaultable_empty_calls() -> None:
    assert extract_canonical_tool_call_xml(
        "<tool_call>\n<function=retrieve>\n</function>\n</tool_call>"
    ) is None
    assert extract_canonical_tool_call_xml(
        "<tool_call>\n<function=inspect_raw>\n</function>\n</tool_call>"
    ) is None


def test_online_xml_correction_drops_invalid_filter_value() -> None:
    correction = finalize_online_step_correction(
        {
            "sample_id": "invalid-filter-value",
            "step_index": 2,
            "allow_inspect_raw": True,
            "tool_format": "qwen3_coder",
        },
        teacher_raw_response=(
            "<tool_call>\n"
            "<function=filter>\n"
            "<parameter=field>\nsource_type\n</parameter>\n"
            "<parameter=op>\neq\n</parameter>\n"
            "<parameter=value>\nuser\n</parameter>\n"
            "<parameter=scope>\ncurrent_pool\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        ),
    )

    assert correction is None


def test_teacher_xml_correction_recovers_explicit_nonstandard_argument_lines() -> None:
    retrieve = extract_canonical_tool_call_xml(
        "<tool_call>\n"
        "<function=retrieve>\n"
        "method=hybrid,\n"
        "top_k=10,\n"
        "query=Maya baking alone child help conversation\n"
        "</function>\n"
        "</tool_call>"
    )
    inspect = extract_canonical_tool_call_xml(
        "<tool_call>\n"
        "<function=inspect_raw>\n"
        'target="current_pool",\n'
        'instruction="answer_query_related_visual_details"\n'
        "</function>\n"
        "</tool_call>"
    )
    mixed_filter = extract_canonical_tool_call_xml(
        "<tool_call>\n"
        "<function=filter>\n"
        "<field=session_date>\n"
        "op=eq\n"
        "value=2024-09-28\n"
        "scope=current_pool\n"
        "</function>\n"
        "</tool_call>"
    )

    assert retrieve is not None
    assert retrieve[1].arguments == {
        "method": "hybrid",
        "top_k": 10,
        "query": "Maya baking alone child help conversation",
    }
    assert inspect is not None
    assert inspect[1].arguments == {
        "target": "current_pool",
        "instruction": "answer_query_related_visual_details",
    }
    assert mixed_filter is not None
    assert mixed_filter[1].arguments == {
        "field": "timestamp",
        "op": "eq",
        "value": "2024-09-28",
        "scope": "current_pool",
    }


def test_live_online_state_request_uses_current_state_without_snapshot_replay() -> None:
    request = build_online_state_correction_request(
        sample_kwargs={
            "raw_prompt": [{"role": "user", "content": "What happened after the relevant turn?"}],
            "tools_kwargs": {
                "opd_mm": {
                    "query": "What happened after the relevant turn?",
                    "allow_inspect_raw": True,
                }
            },
            "extra_info": {
                "gold_answer": "SECRET_GOLD_ANSWER",
                "opd_mm_online_self_distill": True,
                "sample_id": "live-state-sample",
            },
        },
        request_id="request-live",
        step_index=1,
        student_prompt_ids=[11, 22, 33],
        student_raw_response="<tool_call><function=stop></function></tool_call>",
        student_next_action=ToolAction("STOP"),
        history=[ToolAction("RETRIEVE", {"method": "bm25", "top_k": 1})],
        observation={
            "pool_count": 1,
            "evidence_count": 1,
            "trace": [{"tool": "RETRIEVE", "method": "bm25", "top_k": 1}],
            "evidence": [{"source": "RETRIEVE", "summary": "Public evidence."}],
        },
    )

    assert request is not None
    assert request["request_id"] == "request-live"
    assert request["step_index"] == 1
    assert request["student_prompt_ids"] == [11, 22, 33]
    assert request["history"] == [{"tool": "RETRIEVE", "method": "bm25", "top_k": 1}]
    assert request["observation"]["evidence_count"] == 1
    assert request["student_next_action"] == {"tool": "STOP"}
    assert "SECRET_GOLD_ANSWER" in request["verifier_prompt"]


def test_live_online_state_request_includes_initial_state_by_default() -> None:
    request = build_online_state_correction_request(
        sample_kwargs={
            "raw_prompt": [{"role": "user", "content": "Find the relevant memory."}],
            "tools_kwargs": {"opd_mm": {"query": "Find the relevant memory."}},
            "extra_info": {
                "gold_answer": "SECRET_GOLD_ANSWER",
                "opd_mm_online_self_distill": True,
            "sample_id": "stage0-sample",
            },
        },
        request_id="request-stage0",
        step_index=0,
        student_prompt_ids=[11, 22, 33],
        student_raw_response="<tool_call><function=retrieve></function></tool_call>",
        student_next_action=ToolAction("RETRIEVE", {"method": "hybrid", "top_k": 5}),
        history=[],
        observation={"pool_count": 0, "evidence_count": 0, "trace": []},
    )

    assert request is not None
    assert request["step_index"] == 0
    assert request["observation"]["evidence_count"] == 0


def test_live_online_state_request_can_skip_initial_state_explicitly() -> None:
    request = build_online_state_correction_request(
        sample_kwargs={
            "raw_prompt": [{"role": "user", "content": "Find the relevant memory."}],
            "tools_kwargs": {"opd_mm": {"query": "Find the relevant memory."}},
            "extra_info": {
                "gold_answer": "SECRET_GOLD_ANSWER",
                "opd_mm_online_self_distill": True,
                "opd_mm_skip_initial_correction": True,
                "sample_id": "skip-stage0-sample",
            },
        },
        request_id="request-stage0-skip",
        step_index=0,
        student_prompt_ids=[11, 22, 33],
        student_raw_response="<tool_call><function=retrieve></function></tool_call>",
        student_next_action=ToolAction("RETRIEVE", {"method": "hybrid", "top_k": 5}),
        history=[],
        observation={"pool_count": 0, "evidence_count": 0, "trace": []},
    )

    assert request is None


@pytest.mark.asyncio
async def test_agent_loop_worker_generates_verifier_and_teacher_for_one_live_state() -> None:
    class FakeTeacherServer:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def generate_teacher_response_single(self, **kwargs: Any) -> list[int]:
            self.calls.append(kwargs)
            return [101] if len(self.calls) == 1 else [202]

    teacher_server = FakeTeacherServer()
    encoded_tools: list[Any] = []

    def encode_teacher_prompt(prompt: str, **kwargs: Any) -> list[int]:
        encoded_tools.append(kwargs.get("tools"))
        return [len(prompt)]

    def decode(token_ids: list[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        if token_ids == [101]:
            return json.dumps(
                {
                    "evidence_sufficient": False,
                    "reason": "Current public evidence needs neighboring dialogue context before answering.",
                    "missing_evidence_type": "missing_neighbor_context",
                }
            )
        return (
            "<tool_call>\n"
            "<function=expand_neighbors>\n"
            "<parameter=window>\n1\n</parameter>\n"
            "</function>\n"
            "</tool_call>"
        )

    worker = SimpleNamespace(
        teacher_key="data_source",
        teacher_server_manager=teacher_server,
        tokenizer=SimpleNamespace(decode=decode),
        rollout_config=SimpleNamespace(multi_turn=SimpleNamespace(format="qwen3_coder")),
        _encode_opd_mm_teacher_prompt=encode_teacher_prompt,
    )
    correction = await AgentLoopWorker._generate_opd_mm_online_state_correction(
        worker,
        sample_kwargs={
            "data_source": "opd_mm",
            "raw_prompt": [{"role": "user", "content": "What happened next?"}],
            "tools_kwargs": {"opd_mm": {"query": "What happened next?"}},
            "extra_info": {
                "gold_answer": "private answer",
                "opd_mm_online_self_distill": True,
                "sample_id": "live-worker-sample",
            },
        },
        state_payload={
            "request_id": "live-worker-request",
            "step_index": 1,
            "student_prompt_ids": [7, 8, 9],
            "student_raw_response": "<tool_call><function=stop></function></tool_call>",
            "student_next_action": {"tool": "STOP"},
            "history": [{"tool": "RETRIEVE", "method": "bm25", "top_k": 1}],
            "observation": {
                "pool_count": 1,
                "evidence_count": 1,
                "trace": [{"tool": "RETRIEVE", "method": "bm25", "top_k": 1}],
                "evidence": [{"source": "RETRIEVE", "summary": "public evidence"}],
            },
            "tool_format": "qwen3_coder",
            "multi_modal_data": {},
            "mm_processor_kwargs": {},
        },
    )

    assert correction is not None
    assert len(teacher_server.calls) == 2
    assert encoded_tools[0] is None
    teacher_tools = encoded_tools[1]
    assert [tool["function"]["name"] for tool in teacher_tools] == [
        "filter",
        "sort",
        "topk",
        "retrieve",
        "expand_neighbors",
        "inspect_raw",
        "stop",
    ]
    assert correction["step_index"] == 1
    assert correction["sft_prompt_ids"] == [7, 8, 9]
    assert correction["teacher_actions"] == [{"tool": "EXPAND_NEIGHBORS", "window": 1}]
    assert correction["feedback"]["missing_evidence_type"] == "missing_neighbor_context"


@pytest.mark.asyncio
async def test_agent_loop_worker_raw_inspection_uses_teacher_vllm(tmp_path, monkeypatch) -> None:
    image_path = tmp_path / "memory.png"
    from PIL import Image

    Image.new("RGB", (4, 4), color="red").save(image_path)

    class FakeTeacherServer:
        def __init__(self) -> None:
            self.calls: list[dict[str, Any]] = []

        async def generate_teacher_response_single(self, **kwargs: Any) -> list[int]:
            self.calls.append(kwargs)
            return [303]

    teacher_server = FakeTeacherServer()
    monkeypatch.setattr(
        "verl.experimental.agent_loop.agent_loop.apply_chat_template",
        lambda *args, **kwargs: "formatted multimodal prompt",
    )
    monkeypatch.setattr(
        "verl.experimental.agent_loop.agent_loop.build_multimodal_processor_inputs",
        lambda *args, **kwargs: {"input_ids": [[11, 22, 33]]},
    )
    worker = SimpleNamespace(
        processor=object(),
        teacher_key="data_source",
        teacher_server_manager=teacher_server,
        tokenizer=SimpleNamespace(
            decode=lambda token_ids, skip_special_tokens=True: "A red square is visible."
        ),
        _get_mm_processor_kwargs=lambda: {},
    )

    observation = await AgentLoopWorker._generate_opd_mm_teacher_raw_inspection(
        worker,
        sample_kwargs={"data_source": "opd_mm"},
        payload={
            "raw_pointer": str(image_path),
            "query": "What color is visible?",
            "question_image": None,
            "text_context": "A color sample.",
        },
        inspector_config={"raw_inspector_max_tokens": 64, "raw_inspector_temperature": 0.0},
    )

    assert observation == "A red square is visible."
    assert len(teacher_server.calls) == 1
    assert teacher_server.calls[0]["prompt_ids"] == [11, 22, 33]
    assert len(teacher_server.calls[0]["multi_modal_data"]["images"]) == 1
    assert teacher_server.calls[0]["sampling_params"]["max_tokens"] == 64


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
                    "opd_mm_skip_initial_correction": False,
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
    assert "Private answer rubric:" in requests[0]["verifier_prompt"]
    assert "Current public evidence state and observations" in requests[0]["verifier_prompt"]
    assert "Tool semantics to consider" not in requests[0]["verifier_prompt"]
    assert "retrieve(method=bm25|dense|vision|hybrid" not in requests[0]["verifier_prompt"]
    assert "filter(field=modality|author|source_type|timestamp|status" not in requests[0]["verifier_prompt"]
    assert "expand_neighbors(window=1|2|3" not in requests[0]["verifier_prompt"]
    assert "inspect_raw(target=current_pool" not in requests[0]["verifier_prompt"]
    assert "SECRET_GOLD_ANSWER" in requests[0]["verifier_prompt"]

    verifier_feedback = {
        "evidence_sufficient": False,
        "missing_evidence_type": "no_public_evidence",
        "reason": "Need public retrieval evidence before answering.",
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
    assert "Produce exactly one next tool action" in requests[0]["teacher_prompt"]
    assert "Verifier feedback is a private diagnostic, not evidence" in requests[0]["teacher_prompt"]
    assert "The verifier saw the answer rubric; you did not." in requests[0]["teacher_prompt"]
    assert "Gold answer:" not in requests[0]["teacher_prompt"]
    assert "SECRET_GOLD_ANSWER" not in requests[0]["teacher_prompt"]
    assert "Do not copy" in requests[0]["teacher_prompt"]
    assert "verifier.reason" in requests[0]["teacher_prompt"]
    assert "Use evidence_sufficient as" in requests[0]["teacher_prompt"]
    assert "when false, choose a non-STOP tool" in requests[0]["teacher_prompt"]
    assert "when true and the observation has no" in requests[0]["teacher_prompt"]
    assert "For STOP, emit exactly" not in requests[0]["teacher_prompt"]
    assert "schema-described tool" in requests[0]["teacher_prompt"]
    assert "Do not repeat an" in requests[0]["teacher_prompt"]
    assert "identical action" in requests[0]["teacher_prompt"]
    assert "Allowed calls and arguments:" not in requests[0]["teacher_prompt"]
    assert "Output contract" not in requests[0]["teacher_prompt"]
    assert "chat template supplies the tool descriptions" in requests[0]["teacher_prompt"]

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
        '"missing_evidence_type": "none"}\n'
        "```",
        {"evidence_count": 3},
    )

    assert feedback["evidence_sufficient"] is False
    assert feedback["missing_evidence_type"] == "incomplete_coverage"
    assert feedback["parse_error"] == ""


def test_state_verifier_feedback_sanitizes_gold_answer_leakage() -> None:
    feedback = parse_state_verifier_feedback(
        '{"evidence_sufficient": false, '
        "\"reason\": \"No evidence found mentioning Lena's brother or a cat named Miso.\", "
        '"missing_evidence_type": "no_public_evidence"}',
        {"evidence_count": 0},
        gold_answer="Miso",
        query="What is the name of Lena’s brother’s cat?",
    )

    assert feedback["evidence_sufficient"] is False
    assert feedback["missing_evidence_type"] == "no_public_evidence"
    assert "Miso" not in feedback["reason"]
    assert "gold answer" not in feedback["reason"].lower()
    assert feedback["reason"] == "Current public evidence is insufficient; collect relevant evidence first."


def test_state_verifier_feedback_distinguishes_nonempty_irrelevant_evidence() -> None:
    feedback = parse_state_verifier_feedback(
        '{"evidence_sufficient": false, "reason": "No usable evidence for the requested event.", '
        '"missing_evidence_type": "no_public_evidence"}',
        {"evidence_count": 3, "pool_count": 3},
    )

    assert feedback["evidence_sufficient"] is False
    assert feedback["missing_evidence_type"] == "irrelevant_evidence"
    assert feedback["parse_error"] == ""


def test_state_verifier_feedback_parser_falls_back_on_invalid_missing_evidence_type() -> None:
    feedback = parse_state_verifier_feedback(
        '{"evidence_sufficient": true, "reason": "Looks enough.", "missing_evidence_type": "jump"}',
        {"evidence_count": 2},
    )

    # "jump" is intentionally not a verifier missing-evidence type; use the safe non-leaking fallback.
    assert feedback["evidence_sufficient"] is False
    assert feedback["missing_evidence_type"] == "no_public_evidence"
    assert "invalid missing_evidence_type" in feedback["parse_error"]


def test_state_verifier_feedback_parser_accepts_expand_neighbors_when_candidates_exist() -> None:
    feedback = parse_state_verifier_feedback(
        '{"evidence_sufficient": false, "reason": "Need neighboring dialogue context.", '
        '"missing_evidence_type": "missing_neighbor_context"}',
        {"evidence_count": 1, "pool_count": 1},
    )

    assert feedback["evidence_sufficient"] is False
    assert feedback["missing_evidence_type"] == "missing_neighbor_context"
    assert feedback["parse_error"] == ""


def test_state_verifier_feedback_parser_falls_back_from_expand_neighbors_on_empty_state() -> None:
    feedback = parse_state_verifier_feedback(
        '{"evidence_sufficient": false, "reason": "Need neighboring dialogue context.", '
        '"missing_evidence_type": "missing_neighbor_context"}',
        {"evidence_count": 0, "pool_count": 5, "trace": [{"tool": "SORT"}]},
    )

    assert feedback["evidence_sufficient"] is False
    assert feedback["missing_evidence_type"] == "no_public_evidence"
    assert feedback["parse_error"] == ""


def test_online_xml_correction_drops_insufficient_teacher_stop() -> None:
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
        '"missing_evidence_type": "incomplete_coverage"}',
        "verifier_feedback": {
            "evidence_sufficient": False,
            "missing_evidence_type": "incomplete_coverage",
            "reason": "Need list coverage.",
            "parse_error": "",
        },
    }

    correction = finalize_online_step_correction(
        request,
        teacher_raw_response="<tool_call>\n<function=stop>\n</function>\n</tool_call>",
    )

    assert correction is None


def test_online_xml_correction_drops_insufficient_teacher_stop_with_candidates() -> None:
    request = {
        "sample_id": "stop-gate-expand",
        "step_index": 1,
        "query": "What happened around the relevant turn?",
        "history": [{"tool": "RETRIEVE", "method": "bm25", "top_k": 1}],
        "observation": {"evidence_count": 1, "pool_count": 1, "trace": [{"tool": "RETRIEVE"}]},
        "student_raw_response": "<tool_call><function=stop></function></tool_call>",
        "student_prompt_ids": [7, 8, 9],
        "allow_inspect_raw": True,
        "tool_format": "qwen3_coder",
        "verifier_feedback": {
            "evidence_sufficient": False,
            "missing_evidence_type": "missing_neighbor_context",
            "reason": "Need neighboring dialogue context.",
            "parse_error": "",
        },
    }

    correction = finalize_online_step_correction(
        request,
        teacher_raw_response="<tool_call>\n<function=stop>\n</function>\n</tool_call>",
    )

    assert correction is None


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
        "expand_neighbors",
        "stop",
    ]
    filter_scope = schemas[0]["function"]["parameters"]["properties"]["scope"]
    filter_value_description = schemas[0]["function"]["parameters"]["properties"]["value"]["description"]
    assert filter_scope["enum"] == ["current_pool", "full_memory"]
    assert "source_type uses dialogue_turn or dialogue_image" in filter_value_description
    assert "modality uses text or image" in filter_value_description
    assert "MEMORY/user/assistant" not in filter_value_description
    assert "scope" in schemas[0]["function"]["parameters"]["required"]
    assert schemas[3]["function"]["parameters"]["required"] == ["method", "top_k"]
    inspect_schema = openai_tool_schemas(include_inspect_raw=True)[5]
    assert inspect_schema["function"]["parameters"]["required"] == ["target", "instruction"]
    assert "scope" not in schemas[3]["function"]["parameters"]["properties"]
