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
from verl.experimental.opd_mm import MemoryRecord, ToolExecutor
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore, TurnAwareHybridRetriever
from verl.experimental.opd_mm.reward_manager import OPDMMRewardManager
from verl.experimental.opd_mm.schema import TrajectoryValidationError, TrajectoryValidator
from verl.experimental.opd_mm.sft import opd_sft_row_to_verl_record
from verl.experimental.opd_mm.tools import OPDFilterTool, OPDReadTool, hidden_store_from_records, openai_tool_schemas
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


def test_executor_composes_generic_tools_to_read_latest_user_image() -> None:
    store = HiddenMemoryStore(_records())
    result = ToolExecutor().run(
        [
            {"tool": "FILTER", "field": "modality", "op": "eq", "value": "image"},
            {"tool": "FILTER", "field": "author", "op": "eq", "value": "user"},
            {"tool": "SORT", "field": "timestamp", "order": "desc"},
            {"tool": "TOPK", "k": 1},
            {"tool": "READ", "fields": ["summary", "raw_pointer", "timestamp"]},
        ],
        query="Which image did I upload last?",
        memory_store=store,
    )

    assert not result.error
    assert result.stopped
    assert result.final_memory_ids == ["m_cat_image"]
    assert result.evidence[0].fields["raw_pointer"] == "images/cat.png"
    assert result.evidence[0].fields["session_date"] == "2026-01-01"


def test_turn_aware_retrieval_reads_text_and_image_from_same_turn() -> None:
    store = HiddenMemoryStore(_records())
    result = ToolExecutor(retriever=TurnAwareHybridRetriever()).run(
        [
            {"tool": "RETRIEVE", "method": "bm25", "top_k": 1},
            {"tool": "READ", "fields": ["summary", "modality", "raw_pointer"]},
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
    read_tool = OPDReadTool(config={"type": "native"}, tool_schema=None)

    await filter_tool.execute("instance", {"field": "modality", "op": "eq", "value": "image"}, agent_data=agent_data)
    await filter_tool.execute("instance", {"field": "author", "op": "eq", "value": "user"}, agent_data=agent_data)
    response, _, metrics = await read_tool.execute(
        "instance",
        {"fields": ["summary", "raw_pointer"]},
        agent_data=agent_data,
    )

    observation = json.loads(response.text)
    assert observation["evidence_count"] == 1
    assert metrics["opd_mm_evidence_count"] == 1
    assert "memory_id" not in json.dumps(observation)
    assert agent_data.extra_fields["opd_mm"]["evidence_count"] == 1
    assert "memory_id" not in json.dumps(agent_data.extra_fields["opd_mm"])


def test_tool_config_loads_verl_native_opd_tools() -> None:
    tools = load_all_tools(
        tool_config_path="examples/opd_mm_baseline/opd_mm_tool_config.yaml",
        function_tool_path=None,
    )
    assert [tool.name for tool in tools] == ["filter", "sort", "topk", "retrieve", "read", "inspect_raw", "stop"]


def test_sft_converter_can_emit_native_tool_call_records() -> None:
    record = opd_sft_row_to_verl_record(
        {
            "sample_id": "s1",
            "input": "Find the cat image.",
            "target": json.dumps(
                [
                    {"tool": "RETRIEVE", "method": "bm25", "top_k": 1},
                    {"tool": "READ", "fields": ["summary"]},
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
        "read",
        "inspect_raw",
        "stop",
    ]


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
        "read",
        "stop",
    ]
