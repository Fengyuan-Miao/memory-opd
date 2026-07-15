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

"""Dataset conversion helpers for OPD-MM on-policy distillation.

verl's OPD trainer expects RLHF-style rows. During rollout, RLHFDataset moves
extra_info.tools_kwargs into the DataProto non-tensor batch. ToolAgentLoop then
passes those kwargs into OPD-MM tools, while the prompt itself remains memory
free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from verl.experimental.opd_mm.models import MemoryRecord, OPDSample
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore

DEFAULT_DATA_SOURCE = "opd_mm"
DEFAULT_AGENT_NAME = "tool_agent"
OPD_MM_SYSTEM_PROMPT = """You are an OPD-MM multimodal memory retrieval planner. Select exactly one available tool action per turn to
gather enough public evidence for the user's question from a hidden memory store. Use only the user question,
executed action history, and current public observation. Do not invent or expose hidden memory IDs.

Treat the current observation as authoritative: current pool and evidence supersede earlier observations rather than
accumulating across steps. Choose an action that addresses the unresolved evidence need, and do not repeat an
unchanged action without a state-based reason. Stop only when current evidence is sufficient or the observation
reports an unrecoverable error; inference has no gold-aware validator. A public image_id may be used when the
question asks for an image or image ID.
"""


def memory_records_from_store(store: HiddenMemoryStore | Iterable[MemoryRecord]) -> list[MemoryRecord]:
    """Extract records from a HiddenMemoryStore or iterable of MemoryRecord."""
    if isinstance(store, HiddenMemoryStore):
        return list(store._records)
    return list(store)


def memory_records_to_dicts(records: Iterable[MemoryRecord]) -> list[dict[str, Any]]:
    """Serialize MemoryRecord objects for hidden tools_kwargs storage."""
    return [record.to_dict(include_internal_id=True) for record in records]


def opd_messages_for_query(query: str, system_prompt: str | None = OPD_MM_SYSTEM_PROMPT) -> list[dict[str, str]]:
    """Build the student-visible OPD-MM prompt messages."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": query})
    return messages


def opd_messages_for_state(
    base_messages: list[dict[str, Any]],
    action_history: list[dict[str, Any]],
    observation: dict[str, Any],
) -> list[dict[str, Any]]:
    """Build an OmniMem-style next-action prompt from the latest state only.

    Earlier tool observations are deliberately excluded.  The model receives
    the compact executed-action list and one authoritative refreshed
    observation, matching the original interactive OPD prompt structure.
    """
    messages = []
    for message in base_messages:
        copied = dict(message)
        if isinstance(copied.get("content"), list):
            copied["content"] = [dict(item) if isinstance(item, dict) else item for item in copied["content"]]
        messages.append(copied)
    state_text = (
        "\n\nExecuted action history:\n"
        + json.dumps(action_history, ensure_ascii=False, separators=(",", ":"), default=str)
        + "\n\nCurrent refreshed observation:\n"
        + json.dumps(observation, ensure_ascii=False, separators=(",", ":"), default=str)
        + "\n\nChoose the next tool action."
    )
    for message in reversed(messages):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            message["content"] = content + state_text
            break
        if isinstance(content, list):
            content.append({"type": "text", "text": state_text.lstrip()})
            break
    else:
        messages.append({"role": "user", "content": state_text.lstrip()})
    return messages


def opd_sample_to_rlhf_record(
    sample: OPDSample,
    *,
    data_source: str = DEFAULT_DATA_SOURCE,
    agent_name: str = DEFAULT_AGENT_NAME,
    prompt_key: str = "prompt",
    index: int = 0,
) -> dict[str, Any]:
    """Convert an OPDSample into one verl RLHF/OPD row.

    The row is intended for verl.trainer.main_ppo with distillation.enabled=True.
    Teacher routing uses data_source by default, so the teacher model key should
    match the data_source value unless distillation.teacher_key is overridden.
    """
    records = memory_records_from_store(sample.memory_store)
    tools_kwargs = {
        "opd_mm": {
            "query": sample.query,
            "records": memory_records_to_dicts(records),
            "question_image": sample.metadata.get("question_image"),
            "allow_inspect_raw": sample.metadata.get("allow_inspect_raw", True),
            "max_raw_inspections": sample.metadata.get("max_raw_inspections", 3),
        }
    }
    extra_info = dict(sample.metadata.get("extra_info") or {})
    extra_info.update(
        {
            "index": sample.metadata.get("index", index),
            "tools_kwargs": tools_kwargs,
            "need_tools_kwargs": True,
            "gold_answer": sample.gold_answer,
            "sample_id": sample.sample_id,
            "teacher_privilege_mode": "opd_mm",
        }
    )
    for key in (
        "opd_mm_online_self_distill",
        "opd_mm_step_teacher_class",
        "opd_mm_step_teacher_kwargs",
        "opd_mm_step_verifier_kwargs",
        "opd_mm_skip_initial_correction",
    ):
        if key in sample.metadata:
            extra_info[key] = sample.metadata[key]
    system_prompt = sample.metadata.get("opd_mm_system_prompt", OPD_MM_SYSTEM_PROMPT)
    if sample.metadata.get("include_opd_mm_system_prompt", True) is False:
        system_prompt = None
    return {
        "data_source": sample.metadata.get("data_source", data_source),
        "agent_name": sample.metadata.get("agent_name", agent_name),
        prompt_key: opd_messages_for_query(sample.query, system_prompt=system_prompt),
        "reward_model": sample.metadata.get("reward_model", {"style": "rule", "ground_truth": sample.gold_answer}),
        "extra_info": extra_info,
    }


def iter_opd_rlhf_records(
    samples: Iterable[OPDSample],
    *,
    data_source: str = DEFAULT_DATA_SOURCE,
    agent_name: str = DEFAULT_AGENT_NAME,
    prompt_key: str = "prompt",
) -> Iterable[dict[str, Any]]:
    """Yield verl RLHF/OPD records from OPDSample objects."""
    for index, sample in enumerate(samples):
        yield opd_sample_to_rlhf_record(
            sample,
            data_source=data_source,
            agent_name=agent_name,
            prompt_key=prompt_key,
            index=index,
        )


def write_opd_rlhf_jsonl(
    samples: Iterable[OPDSample],
    output_path: str | Path,
    *,
    data_source: str = DEFAULT_DATA_SOURCE,
    agent_name: str = DEFAULT_AGENT_NAME,
    prompt_key: str = "prompt",
) -> Path:
    """Write OPD-MM samples as verl RLHF/OPD JSONL rows."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in iter_opd_rlhf_records(
            samples,
            data_source=data_source,
            agent_name=agent_name,
            prompt_key=prompt_key,
        ):
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return output


def write_opd_rlhf_parquet(
    samples: Iterable[OPDSample],
    output_path: str | Path,
    *,
    data_source: str = DEFAULT_DATA_SOURCE,
    agent_name: str = DEFAULT_AGENT_NAME,
    prompt_key: str = "prompt",
) -> Path:
    """Write OPD-MM samples as verl RLHF/OPD parquet rows."""
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = list(iter_opd_rlhf_records(samples, data_source=data_source, agent_name=agent_name, prompt_key=prompt_key))

    import pandas as pd

    pd.DataFrame(rows).to_parquet(output, index=False)
    return output
