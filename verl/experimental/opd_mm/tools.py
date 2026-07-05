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

"""verl-native tool adapters for the OPD-MM hidden-memory executor.

These tools keep per-trajectory state through the agent_data object supplied by
ToolAgentLoop. That lets FILTER, SORT, TOPK, RETRIEVE, INSPECT_RAW, and
STOP behave like the original OPD-MM sequential action space while still
exposing OpenAI function schemas to verl.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Optional

from verl.experimental.opd_mm.executor import ToolExecutor
from verl.experimental.opd_mm.models import EvidenceItem, ExecutionStep, MemoryRecord, PoolItem, ToolAction
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore, HybridRetriever
from verl.experimental.opd_mm.schema import (
    FILTER_FIELDS,
    FILTER_OPS,
    INSPECT_INSTRUCTIONS,
    INSPECT_TARGETS,
    RETRIEVAL_METHODS,
    SORT_FIELDS,
    SORT_ORDERS,
    TrajectoryValidator,
)
from verl.tools.base_tool import BaseTool
from verl.tools.schemas import OpenAIFunctionToolSchema, ToolResponse

_SESSION_ATTR = "_opd_mm_tool_session"


def _property(type_: str | list[str], description: str, enum: Optional[list[Any]] = None) -> dict[str, Any]:
    value: dict[str, Any] = {"type": type_, "description": description}
    if enum is not None:
        value["enum"] = enum
    return value


def _schema(
    name: str,
    description: str,
    properties: dict[str, dict[str, Any]],
    required: list[str],
) -> OpenAIFunctionToolSchema:
    return OpenAIFunctionToolSchema.model_validate(
        {
            "type": "function",
            "function": {
                "name": name,
                "description": description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }
    )


def memory_record_from_dict(value: dict[str, Any], index: int = 0) -> MemoryRecord:
    """Build a MemoryRecord from a plain dictionary."""
    known = {
        "memory_id",
        "turn_id",
        "timestamp",
        "author",
        "modality",
        "source_type",
        "summary",
        "content",
        "raw_pointer",
        "status",
        "metadata",
    }
    metadata = dict(value.get("metadata") or {})
    for key, item in value.items():
        if key not in known:
            metadata[key] = item
    return MemoryRecord(
        memory_id=str(value.get("memory_id", f"opd_memory_{index}")),
        turn_id=str(value.get("turn_id", index)),
        timestamp=str(value.get("timestamp", "")),
        author=str(value.get("author", "")),
        modality=str(value.get("modality", "text")),
        source_type=str(value.get("source_type", "memory")),
        summary=str(value.get("summary", "") or ""),
        content=str(value.get("content", "") or ""),
        raw_pointer=value.get("raw_pointer"),
        status=str(value.get("status", "active")),
        metadata=metadata,
    )


def hidden_store_from_records(records: list[dict[str, Any] | MemoryRecord]) -> HiddenMemoryStore:
    """Build a HiddenMemoryStore from MemoryRecord objects or dictionaries."""
    built = [
        record if isinstance(record, MemoryRecord) else memory_record_from_dict(record, i)
        for i, record in enumerate(records)
    ]
    return HiddenMemoryStore(built)


def _sanitize_evidence(items: list[EvidenceItem]) -> list[dict[str, Any]]:
    sanitized = []
    for item in items:
        data = item.to_dict()
        data.pop("memory_id", None)
        sanitized.append(data)
    return sanitized


def _sanitize_pool_preview(items: list[PoolItem], max_items: int = 5) -> list[dict[str, Any]]:
    """Return an ID-free preview of the current hidden pool for tool observations."""
    preview = []
    for item in items[:max_items]:
        memory = item.memory
        entry = {
            "summary": memory.summary,
            "content": memory.content[:360] if memory.content else None,
            "timestamp": memory.timestamp,
            "turn_id": memory.turn_id,
            "author": memory.author,
            "modality": memory.modality,
            "source_type": memory.source_type,
            "raw_pointer": memory.raw_pointer,
            "session_date": memory.metadata.get("session_date"),
        }
        if item.score:
            entry["retrieval_score"] = item.score
        preview.append({key: value for key, value in entry.items() if value is not None})
    return preview


@dataclass
class OPDToolSession:
    """Per-trajectory state shared by OPD-MM tools."""

    executor: ToolExecutor
    memory_store: HiddenMemoryStore
    query: str
    question_image: Optional[str] = None
    pool: list[PoolItem] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    steps: list[ExecutionStep] = field(default_factory=list)
    trace: list[ToolAction] = field(default_factory=list)
    raw_calls: int = 0
    stopped: bool = False
    error: str = ""

    def __post_init__(self) -> None:
        if not self.pool:
            self.pool = self.memory_store.initial_pool()

    def execute(self, action: ToolAction) -> dict[str, Any]:
        """Execute one validated action against the current hidden pool."""
        before = len(self.pool)
        evidence_before = len(self.evidence)
        step_error = ""

        if self.stopped:
            return self._observation(action, [], "trajectory already stopped")

        try:
            self.executor.validator._validate_action(action, len(self.trace))
            if action.tool == "FILTER":
                self.pool = self.executor._filter(self.pool, **action.arguments)
            elif action.tool == "SORT":
                self.pool = self.executor._sort(self.pool, **action.arguments)
            elif action.tool == "TOPK":
                self.pool = self.executor._topk_turns(self.pool, action.arguments["k"])
            elif action.tool == "RETRIEVE":
                self.pool = self.executor.retriever.retrieve(
                    self.pool,
                    query=self.query,
                    store=self.memory_store,
                    method=action.arguments.get("method", "hybrid"),
                    top_k=action.arguments.get("top_k", 5),
                    question_image=self.question_image,
                )
            elif action.tool == "INSPECT_RAW":
                remaining = max(0, self.executor.max_raw_inspections - self.raw_calls)
                inspected = self.executor._inspect_raw(
                    self.pool,
                    self.query,
                    remaining,
                    question_image=self.question_image,
                )
                self.raw_calls += len(inspected)
                self.evidence.extend(inspected)
            elif action.tool == "STOP":
                self.stopped = True
        except Exception as exc:
            step_error = str(exc)
            self.error = step_error

        self.trace.append(action)
        self.steps.append(
            ExecutionStep(
                index=len(self.steps),
                action=action,
                pool_before=before,
                pool_after=len(self.pool),
                evidence_added=len(self.evidence) - evidence_before,
                error=step_error,
            )
        )
        new_evidence = self.evidence[evidence_before:]
        return self._observation(action, new_evidence, step_error)

    def _observation(self, action: ToolAction, new_evidence: list[EvidenceItem], error: str) -> dict[str, Any]:
        return {
            "tool": action.tool,
            "pool_count": len(self.pool),
            "evidence_count": len(self.evidence),
            "pool_preview": _sanitize_pool_preview(self.pool),
            "new_evidence": _sanitize_evidence(new_evidence),
            "stopped": self.stopped,
            "error": error,
        }

    def public_state(self) -> dict[str, Any]:
        """Return serializable, ID-free state for AgentLoopOutput.extra_fields."""
        return {
            "pool_count": len(self.pool),
            "evidence_count": len(self.evidence),
            "pool_preview": _sanitize_pool_preview(self.pool),
            "evidence": _sanitize_evidence(self.evidence),
            "trace": [action.to_dict() for action in self.trace],
            "stopped": self.stopped,
            "error": self.error,
            "raw_inspection_calls": self.raw_calls,
        }


class OPDBaseTool(BaseTool):
    """Base class for one OPD-MM action exposed as a verl native tool."""

    tool_name = ""
    description = ""
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []

    def __init__(self, config: dict, tool_schema: OpenAIFunctionToolSchema | None = None):
        super().__init__(config or {}, tool_schema or self.get_openai_tool_schema())

    def get_openai_tool_schema(self) -> OpenAIFunctionToolSchema:
        return _schema(self.tool_name, self.description, self.properties, self.required)

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        session = self._session(agent_data)
        action = self._action(parameters)
        observation = session.execute(action)
        if agent_data is not None and hasattr(agent_data, "extra_fields"):
            agent_data.extra_fields["opd_mm"] = session.public_state()
        return ToolResponse(text=json.dumps(observation, ensure_ascii=False)), 0.0, {
            "opd_mm_pool_count": observation["pool_count"],
            "opd_mm_evidence_count": observation["evidence_count"],
        }

    def _action(self, parameters: dict[str, Any]) -> ToolAction:
        return ToolAction(self.tool_name.upper(), dict(parameters))

    def _session(self, agent_data: Any) -> OPDToolSession:
        if agent_data is not None and hasattr(agent_data, _SESSION_ATTR):
            return getattr(agent_data, _SESSION_ATTR)

        runtime = dict(self.config or {})
        if agent_data is not None:
            tools_kwargs = getattr(agent_data, "tools_kwargs", {}) or {}
            runtime.update(tools_kwargs.get("opd_mm", {}) or {})
            runtime.update(tools_kwargs.get(self.name, {}) or {})

        store = runtime.get("memory_store")
        if store is None:
            store = hidden_store_from_records(runtime.get("records") or runtime.get("memory_records") or [])
        if not isinstance(store, HiddenMemoryStore):
            raise TypeError("OPD-MM tools require a HiddenMemoryStore or records in tools_kwargs['opd_mm']")

        query = runtime.get("query") or runtime.get("raw_query") or self._query_from_agent_data(agent_data)
        session = OPDToolSession(
            executor=ToolExecutor(
                retriever=runtime.get("retriever") or HybridRetriever(),
                raw_inspector=runtime.get("raw_inspector"),
                validator=runtime.get("validator") or TrajectoryValidator(
                    max_actions=int(runtime.get("max_actions", 8)),
                    max_top_k=int(runtime.get("max_top_k", 50)),
                    allow_inspect_raw=bool(runtime.get("allow_inspect_raw", True)),
                ),
                max_raw_inspections=int(runtime.get("max_raw_inspections", 3)),
            ),
            memory_store=store,
            query=str(query or ""),
            question_image=runtime.get("question_image"),
        )
        if agent_data is not None:
            setattr(agent_data, _SESSION_ATTR, session)
        return session

    @staticmethod
    def _query_from_agent_data(agent_data: Any) -> str:
        messages = getattr(agent_data, "messages", []) if agent_data is not None else []
        for message in reversed(messages):
            if isinstance(message, dict) and message.get("role") == "user":
                content = message.get("content", "")
                if isinstance(content, str):
                    return content
        return ""


class OPDFilterTool(OPDBaseTool):
    tool_name = "filter"
    description = "Filter the hidden current memory pool by a metadata field."
    properties = {
        "field": _property("string", "The memory field to filter.", sorted(FILTER_FIELDS)),
        "op": _property("string", "The comparison operator.", sorted(FILTER_OPS)),
        "value": _property(["string", "number", "boolean"], "The comparison value. Do not use memory IDs."),
    }
    required = ["field", "op", "value"]


class OPDSortTool(OPDBaseTool):
    tool_name = "sort"
    description = "Sort the current hidden memory pool."
    properties = {
        "field": _property("string", "The field to sort by.", sorted(SORT_FIELDS)),
        "order": _property("string", "Sort order.", sorted(SORT_ORDERS)),
    }
    required = ["field", "order"]


class OPDTopKTool(OPDBaseTool):
    tool_name = "topk"
    description = "Keep the top k turns from the current hidden memory pool."
    properties = {"k": _property("integer", "Positive number of turns to keep.")}
    required = ["k"]


class OPDRetrieveTool(OPDBaseTool):
    tool_name = "retrieve"
    description = "Rank the current hidden pool against the original user query."
    properties = {
        "method": _property("string", "Retrieval method. Do not provide a custom query.", sorted(RETRIEVAL_METHODS)),
        "top_k": _property("integer", "Positive number of turns to retrieve."),
    }
    required: list[str] = []


class OPDInspectRawTool(OPDBaseTool):
    tool_name = "inspect_raw"
    description = "Opt-in raw visual inspection for images in the current hidden pool."
    properties = {
        "target": _property("string", "Inspection target.", sorted(INSPECT_TARGETS)),
        "instruction": _property("string", "Inspection instruction.", sorted(INSPECT_INSTRUCTIONS)),
    }
    required: list[str] = []


class OPDStopTool(OPDBaseTool):
    tool_name = "stop"
    description = "Stop the OPD-MM retrieval trajectory once enough evidence is collected."
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []


OPD_TOOL_CLASSES = [
    OPDFilterTool,
    OPDSortTool,
    OPDTopKTool,
    OPDRetrieveTool,
    OPDInspectRawTool,
    OPDStopTool,
]


def openai_tool_schemas(include_inspect_raw: bool = True) -> list[dict[str, Any]]:
    """Return OpenAI tool schemas for OPD-MM tools."""
    classes = (
        OPD_TOOL_CLASSES
        if include_inspect_raw
        else [cls for cls in OPD_TOOL_CLASSES if cls is not OPDInspectRawTool]
    )
    return [
        _schema(cls.tool_name, cls.description, cls.properties, cls.required).model_dump(
            exclude_unset=True, exclude_none=True
        )
        for cls in classes
    ]


__all__ = [
    "OPDBaseTool",
    "OPDFilterTool",
    "OPDInspectRawTool",
    "OPDRetrieveTool",
    "OPDStopTool",
    "OPDToolSession",
    "OPDTopKTool",
    "OPD_TOOL_CLASSES",
    "OPDSortTool",
    "hidden_store_from_records",
    "memory_record_from_dict",
    "openai_tool_schemas",
]
