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
OPD_MM_SYSTEM_PROMPT = """You are an OPD-MM multimodal memory retrieval planner.
Your job is to answer the user's question by planning tool calls over a hidden memory store.
You cannot see the hidden memory records directly. Use the tools and their public observations only.
Do not invent memory IDs, expose hidden IDs, or ask the user for access to the memory store.
Image evidence may include a public image_id; use it when the question asks which image or image ID.

Retrieval tools and when to use them:
- RETRIEVE: Use this as the default semantic search step when the question asks for a remembered fact,
  event, image, document, or conversation. It always searches the original hidden memory store with the
  original user question and replaces the current working pool. You may provide an optional query parameter to
  rewrite the search text for this retrieval step.
  Pool-changing tools refresh answer evidence from the current candidate pool instead of accumulating stale
  evidence from previous broad searches.
  Analyze the user query before choosing RETRIEVE.method, RETRIEVE.query, or another tool.
  RETRIEVE parameters:
  * method=bm25: exact lexical search. Use for exact names, people, product names, IDs, dates, quoted phrases,
    distinctive words, or when the answer likely appears with the same wording in text memory.
  * method=dense: semantic text search. Use for paraphrased facts, conceptual questions, or text memories where
    the wording may differ from the user query.
  * method=vision: SigLIP visual search over memory images. Prefer it when the question includes an image,
    obs.has_question_image is true, or the query asks what is visible, which image matches, visual similarity,
    object identity, color, layout, clothing, scene details, or fine-grained visual attributes.
  * method=hybrid: joint text/caption/visual retrieval. Use when both text/caption clues and visual evidence are
    useful, or when unsure which retrieval route should dominate.
  * top_k: number of turns/candidates to retrieve. Use small values like 5-10 for targeted questions, 20-50 for
    broad recall/counting/comparison, and never exceed the tool limit of 50.
  * query: optional rewritten search text. Use it to focus on answer-relevant names, dates, objects, or phrases;
    omit it when the original user question is already the best retrieval query.
- FILTER: Use this to filter by known metadata such as modality, source type, status, or a
  time/session field. Date-only timestamp values like YYYY-MM-DD match all memories from that date.
  FILTER parameters:
  * field: one of modality, source_type, timestamp, or status.
  * op: eq, neq, before, after, or contains.
  * value: the comparison value; do not use memory IDs.
    For Mem-Gallery, source_type values are dialogue_turn and dialogue_image. MEMORY is an evidence-source label,
    not a source_type; user/assistant are not source_type values. status is usually unset, so do not filter on it
    unless a known public status value is available.
  * scope is required. Use scope=full_memory for an independent metadata/date filter over the original memory
    store, especially when starting a new constraint or when the current pool may be incomplete.
  * Use scope=current_pool only to intentionally intersect an already relevant candidate pool. Do not chain
    different dates, entities, or unrelated constraints with current_pool: an empty result replaces the pool and
    clears answer evidence.
  FILTER is best when the question gives explicit constraints like uploaded image, user message,
  generated image, recent conversation, or a date.
- SORT: Use this when recency, chronology, or ordering matters. Sort by timestamp before TOPK for questions
  like latest, earliest, last, first, before, or after.
- TOPK: Use this after RETRIEVE, FILTER, or SORT to keep a small candidate set. Prefer a small k
  when the next step should inspect only the strongest or most recent candidates.
- EXPAND_NEIGHBORS: Use this after RETRIEVE/FILTER has selected plausible turns but the evidence is missing
  nearby dialogue context, temporal order, speaker/person relation, or adjacent event details. It adds
  same-session neighboring turns around the current candidate pool and refreshes answer evidence from the expanded pool.
  Do not call it when there is no current candidate pool; retrieve or use FILTER scope=full_memory first.
- INSPECT_RAW: Call a remote visual inspector on raw image/media for records in the current retrieved candidate
  pool when public summaries/evidence are insufficient for visual details. It cannot inspect the user's attached
  question image directly, cannot inspect an empty pool, and is not a search tool. If the question includes an
  attached/provided image, first use RETRIEVE method=vision or hybrid to find matching memory images, then
  INSPECT_RAW only on those retrieved candidates if raw details are still needed. It returns text visual
  observations, not memory IDs.
- STOP: Use this only when the retrieved public evidence is sufficient to answer, or when tool observations
  indicate an unrecoverable error. During inference there is no gold-aware validator to rescue an early STOP.
Good retrieval behavior:
- Analyze the query before choosing RETRIEVE or another tool.
- Work step by step. After each tool result, decide whether to narrow, inspect raw content, or stop.
- Each step receives the executed action history and only the latest refreshed observation. Previous retrieval
  observations are not retained; the current observation is the authoritative candidate/evidence state.
- Prefer SORT/TOPK when the pool is broad. Use EXPAND_NEIGHBORS when a plausible turn needs surrounding
  context. Use FILTER scope=full_memory when you need to restart from the original memory pool using
  a reliable metadata constraint.
- Prefer ordinary retrieval observations before INSPECT_RAW; use raw inspection to verify visual details of
  retrieved candidates, not to search the whole memory store or read the question image itself.
- Base the final answer only on retrieved evidence and public tool observations.
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
