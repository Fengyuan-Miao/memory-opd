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
ToolAgentLoop. That lets FILTER, SORT, TOPK, RETRIEVE, EXPAND_NEIGHBORS, DROP,
INSPECT_RAW, and STOP share one bounded working evidence pool
while still exposing OpenAI function schemas to verl.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Optional

from verl.experimental.opd_mm.executor import DEFAULT_MAX_POOL_SIZE, ToolExecutor
from verl.experimental.opd_mm.models import EvidenceItem, ExecutionStep, MemoryRecord, PoolItem, ToolAction
from verl.experimental.opd_mm.raw_inspector import DEFAULT_RAW_INSPECTOR_URL, RemoteVLLMRawInspector
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore, TurnAwareHybridRetriever
from verl.experimental.opd_mm.schema import (
    DEFAULT_MAX_ACTIONS,
    EXPAND_NEIGHBOR_WINDOWS,
    FILTER_FIELDS,
    FILTER_OPS,
    FILTER_SCOPES,
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
_AUTO_VECTOR_STORE = object()
DEFAULT_VECTOR_STORE_DIR = "dataset/mem_gallery/opd_mm_store"
DEFAULT_DENSE_MODEL_PATH = "/home/miaofy/data/pretrained_models/all-MiniLM-L6-v2"
DEFAULT_VISION_MODEL_PATH = "/home/miaofy/data/pretrained_models/SigLIP-Base-Patch16-384"
DEFAULT_HYBRID_MODEL_PATH = "/home/miaofy/data/pretrained_models/gme-Qwen2-VL-2B-Instruct"
DEFAULT_RAW_INSPECTOR_TIMEOUT = 60.0
DEFAULT_RAW_INSPECTOR_MAX_TOKENS = 256
OBSERVATION_TEXT_MAX_CHARS = 220
OBSERVATION_CATALOG_TEXT_MAX_CHARS = 96
OBSERVATION_POOL_PREVIEW_ITEMS = 3
OBSERVATION_EVIDENCE_PREVIEW_ITEMS = 4


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


def _optional_str(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    return "" if text.lower() in {"", "none", "null", "false", "0"} else text


@lru_cache(maxsize=16)
def _cached_remote_raw_inspector(
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    max_tokens: int,
    temperature: float,
) -> RemoteVLLMRawInspector:
    return RemoteVLLMRawInspector(
        base_url=base_url,
        model=model or None,
        api_key=api_key or None,
        timeout=timeout,
        max_tokens=max_tokens,
        temperature=temperature,
    )


def _raw_inspector_from_runtime(runtime: dict[str, Any]) -> Any:
    if runtime.get("raw_inspector") is not None:
        return runtime["raw_inspector"]
    backend = _optional_str(os.getenv("OPD_MM_RAW_INSPECTOR_BACKEND") or runtime.get("raw_inspector_backend"))
    if backend.lower() == "teacher":
        return None
    if not bool(runtime.get("allow_inspect_raw", True)):
        return None
    base_url = (
        _optional_str(runtime.get("raw_inspector_url"))
        or _optional_str(os.getenv("OPD_MM_RAW_INSPECTOR_URL"))
        or DEFAULT_RAW_INSPECTOR_URL
    )
    if not base_url:
        return None
    return _cached_remote_raw_inspector(
        base_url,
        _optional_str(runtime.get("raw_inspector_model")) or _optional_str(os.getenv("OPD_MM_RAW_INSPECTOR_MODEL")),
        _optional_str(runtime.get("raw_inspector_api_key")) or _optional_str(os.getenv("OPD_MM_RAW_INSPECTOR_API_KEY")),
        float(runtime.get("raw_inspector_timeout") or os.getenv("OPD_MM_RAW_INSPECTOR_TIMEOUT") or DEFAULT_RAW_INSPECTOR_TIMEOUT),
        int(runtime.get("raw_inspector_max_tokens") or os.getenv("OPD_MM_RAW_INSPECTOR_MAX_TOKENS") or DEFAULT_RAW_INSPECTOR_MAX_TOKENS),
        float(runtime.get("raw_inspector_temperature") or os.getenv("OPD_MM_RAW_INSPECTOR_TEMPERATURE") or 0.0),
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


class _LazyEncoder:
    """Proxy that loads a heavy query encoder only when a vector method is used."""

    def __init__(self, loader: Any):
        self._loader = loader
        self._encoder: Any = None

    def _get(self) -> Any:
        if self._encoder is None:
            self._encoder = self._loader()
        return self._encoder

    def __getattr__(self, name: str) -> Any:
        return getattr(self._get(), name)


@lru_cache(maxsize=16)
def _cached_vector_index(index_dir: str) -> Any:
    from verl.experimental.opd_mm.vector_index import DiskVectorIndex

    return DiskVectorIndex.load(index_dir)


@lru_cache(maxsize=8)
def _cached_dense_encoder(model_path: str, device: str) -> Any:
    from verl.experimental.opd_mm.vector_index import MiniLMTextEncoder

    return MiniLMTextEncoder(model_path, device=device)


@lru_cache(maxsize=8)
def _cached_vision_encoder(model_path: str, device: str) -> Any:
    from verl.experimental.opd_mm.vector_index import SigLIPVisionEncoder

    return SigLIPVisionEncoder(model_path, device=device)


@lru_cache(maxsize=4)
def _cached_hybrid_encoder(model_path: str, device: str) -> Any:
    from verl.experimental.opd_mm.vector_index import GMEQwen2VLUnifiedEncoder

    return GMEQwen2VLUnifiedEncoder(model_path, device=device)


def _path_text(value: Any) -> str:
    return str(Path(str(value)).expanduser())


def _resolve_vector_store_dir(value: Any = _AUTO_VECTOR_STORE) -> Optional[Path]:
    if value is None or value is False:
        return None
    if value is _AUTO_VECTOR_STORE:
        value = os.getenv("OPD_MM_VECTOR_STORE_DIR") or DEFAULT_VECTOR_STORE_DIR
    if not value:
        return None
    root = Path(str(value)).expanduser()
    if not root.exists():
        return None
    if not (root / "indexes").exists():
        return None
    return root


def _load_index(root: Path, name: str) -> Any:
    index_dir = root / "indexes" / name
    if not (index_dir / "embeddings.npy").exists() or not (index_dir / "items.jsonl").exists():
        return None
    return _cached_vector_index(_path_text(index_dir))


def _model_path(
    *,
    configured: Any,
    env_name: str,
    default: str,
    index: Any,
) -> Optional[str]:
    if configured:
        return _path_text(configured)
    env_value = os.getenv(env_name)
    if env_value:
        return _path_text(env_value)
    manifest_value = (getattr(index, "manifest", None) or {}).get("model_path") if index is not None else None
    if manifest_value and Path(str(manifest_value)).expanduser().exists():
        return _path_text(manifest_value)
    if Path(default).expanduser().exists():
        return _path_text(default)
    return None


def _index_overlaps_records(indexes: list[Any], records: list[MemoryRecord]) -> bool:
    memory_ids = {record.memory_id for record in records}
    if not memory_ids:
        return False
    for index in indexes:
        if index is None:
            continue
        row_by_memory_id = getattr(index, "_row_by_memory_id", {})
        if any(memory_id in row_by_memory_id for memory_id in memory_ids):
            return True
    return False


def _indexed_store_from_records(
    records: list[MemoryRecord],
    *,
    vector_store_dir: Any = _AUTO_VECTOR_STORE,
    dense_model_path: Any = None,
    vision_model_path: Any = None,
    hybrid_model_path: Any = None,
    vector_device: str = "cuda:0",
    require_overlap: bool = True,
) -> Optional[HiddenMemoryStore]:
    root = _resolve_vector_store_dir(vector_store_dir)
    if root is None:
        return None

    dense_index = _load_index(root, "dense")
    vision_index = _load_index(root, "vision")
    hybrid_index = _load_index(root, "hybrid")
    indexes = [dense_index, vision_index, hybrid_index]
    if not any(indexes):
        return None
    if require_overlap and not _index_overlaps_records(indexes, records):
        return None

    from verl.experimental.opd_mm.vector_index import DiskIndexedHiddenMemoryStore

    device = str(vector_device or os.getenv("OPD_MM_RETRIEVER_DEVICE") or "cuda:0")
    dense_path = _model_path(
        configured=dense_model_path,
        env_name="OPD_MM_DENSE_MODEL_PATH",
        default=DEFAULT_DENSE_MODEL_PATH,
        index=dense_index,
    )
    vision_path = _model_path(
        configured=vision_model_path,
        env_name="OPD_MM_VISION_MODEL_PATH",
        default=DEFAULT_VISION_MODEL_PATH,
        index=vision_index,
    )
    hybrid_path = _model_path(
        configured=hybrid_model_path,
        env_name="OPD_MM_HYBRID_MODEL_PATH",
        default=DEFAULT_HYBRID_MODEL_PATH,
        index=hybrid_index,
    )
    dense_encoder = (
        _LazyEncoder(lambda: _cached_dense_encoder(dense_path, device))
        if dense_index is not None and dense_path
        else None
    )
    vision_encoder = (
        _LazyEncoder(lambda: _cached_vision_encoder(vision_path, device))
        if vision_index is not None and vision_path
        else None
    )
    hybrid_encoder = (
        _LazyEncoder(lambda: _cached_hybrid_encoder(hybrid_path, device))
        if hybrid_index is not None and hybrid_path
        else None
    )
    return DiskIndexedHiddenMemoryStore(
        records,
        dense_index=dense_index,
        vision_index=vision_index,
        hybrid_index=hybrid_index,
        dense_query_encoder=dense_encoder,
        vision_query_encoder=vision_encoder,
        hybrid_query_encoder=hybrid_encoder,
    )


def hidden_store_from_records(
    records: list[dict[str, Any] | MemoryRecord],
    *,
    vector_store_dir: Any = _AUTO_VECTOR_STORE,
    dense_model_path: Any = None,
    vision_model_path: Any = None,
    hybrid_model_path: Any = None,
    vector_device: str = "cuda:0",
) -> HiddenMemoryStore:
    """Build a HiddenMemoryStore from records, attaching disk vector indexes when available."""
    built = [
        record if isinstance(record, MemoryRecord) else memory_record_from_dict(record, i)
        for i, record in enumerate(records)
    ]
    indexed_store = _indexed_store_from_records(
        built,
        vector_store_dir=vector_store_dir,
        dense_model_path=dense_model_path,
        vision_model_path=vision_model_path,
        hybrid_model_path=hybrid_model_path,
        vector_device=vector_device,
        require_overlap=vector_store_dir is _AUTO_VECTOR_STORE,
    )
    if indexed_store is not None:
        return indexed_store
    return HiddenMemoryStore(built)


def _sanitize_evidence(
    items: list[EvidenceItem], evidence_ids_by_memory: dict[str, str]
) -> list[dict[str, Any]]:
    sanitized = []
    for item in items:
        data = item.to_dict()
        data.pop("memory_id", None)
        data.pop("source", None)
        data.pop("author", None)
        data["evidence_id"] = evidence_ids_by_memory[item.memory_id]
        sanitized.append(data)
    return sanitized


def _clip_text(value: Any, max_chars: int = OBSERVATION_TEXT_MAX_CHARS) -> Any:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "...(truncated)"


def _sanitize_evidence_preview(
    items: list[EvidenceItem],
    evidence_ids_by_memory: dict[str, str],
    max_items: int = OBSERVATION_EVIDENCE_PREVIEW_ITEMS,
) -> list[dict[str, Any]]:
    """Return a compact evidence preview with trajectory-local public IDs."""
    preview = []
    for item in items[-max_items:]:
        data = item.to_dict()
        data.pop("memory_id", None)
        fields = data.get("fields") if isinstance(data.get("fields"), dict) else data
        entry: dict[str, Any] = {"evidence_id": evidence_ids_by_memory[item.memory_id]}
        for key in (
            "image_id",
            "content",
            "visual_observation",
            "linked_text_context",
            "timestamp",
            "session_date",
            "modality",
            "retrieval_score",
        ):
            value = fields.get(key)
            if value not in (None, ""):
                entry[key] = _clip_text(value)
        preview.append({key: value for key, value in entry.items() if value not in (None, "")})
    return preview


def _sanitize_pool_preview(
    items: list[PoolItem],
    evidence_ids_by_memory: dict[str, str],
    max_items: int = OBSERVATION_POOL_PREVIEW_ITEMS,
) -> list[dict[str, Any]]:
    """Return an internal-ID-free preview of the current working pool."""
    preview = []
    for item in items[:max_items]:
        memory = item.memory
        content = memory.content or memory.summary
        entry = {
            "evidence_id": evidence_ids_by_memory[memory.memory_id],
            "image_id": memory.public_image_id(),
            "content": _clip_text(content) if content else None,
            "timestamp": memory.timestamp,
            "modality": memory.modality,
            "session_date": memory.metadata.get("session_date"),
        }
        if item.score:
            entry["retrieval_score"] = item.score
        preview.append({key: value for key, value in entry.items() if value is not None})
    return preview


def _sanitize_evidence_catalog(
    pool: list[PoolItem],
    evidence: list[EvidenceItem],
    evidence_ids_by_memory: dict[str, str],
) -> list[dict[str, Any]]:
    """Return every droppable candidate with enough public content for relevance decisions."""
    fields_by_memory: dict[str, dict[str, Any]] = {}
    for item in evidence:
        fields_by_memory.setdefault(item.memory_id, {}).update(item.fields)

    catalog = []
    for item in pool:
        memory = item.memory
        fields = fields_by_memory.get(memory.memory_id, {})
        entry: dict[str, Any] = {
            "evidence_id": evidence_ids_by_memory[memory.memory_id],
            "content": _clip_text(
                fields.get("content") or memory.content or memory.summary,
                OBSERVATION_CATALOG_TEXT_MAX_CHARS,
            ),
            "visual_observation": _clip_text(
                fields.get("visual_observation"),
                OBSERVATION_CATALOG_TEXT_MAX_CHARS,
            ),
            "image_id": memory.public_image_id(),
            "timestamp": memory.timestamp,
            "modality": memory.modality,
        }
        catalog.append({key: value for key, value in entry.items() if value not in (None, "")})
    return catalog


def _update_dynamic_drop_schema(agent_data: Any, evidence_ids: list[str]) -> None:
    """Constrain the next DROP call to IDs visible in the current observation."""
    active_tools = getattr(agent_data, "_active_tools", None)
    if not isinstance(active_tools, dict):
        return
    schemas = []
    for tool in active_tools.values():
        schema = tool.tool_schema.model_dump(exclude_unset=True, exclude_none=True)
        if schema.get("function", {}).get("name") == "drop":
            items = schema["function"]["parameters"]["properties"]["evidence_ids"].setdefault("items", {})
            items["enum"] = list(evidence_ids)
        schemas.append(schema)
    agent_data._active_tool_schemas = schemas


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
    pool_has_candidates: bool = False
    max_actions_reached: bool = False
    evidence_ids_by_memory: dict[str, str] = field(default_factory=dict)
    evidence_revision: int = 0
    last_drop_revision: int = 0
    pool_overflow_count: int = 0
    drop_calls: int = 0
    dropped_evidence_count: int = 0

    def __post_init__(self) -> None:
        if not self.pool:
            self.pool = self.memory_store.initial_pool()

    def execute(self, action: ToolAction) -> dict[str, Any]:
        """Execute one validated action against the current hidden pool."""
        before = len(self.pool)
        step_error = ""
        new_evidence: list[EvidenceItem] = []
        self.pool_overflow_count = 0

        if self.stopped:
            return self._observation(action, [], "trajectory already stopped")

        try:
            if len(self.trace) >= self.executor.validator.max_actions - 1 and action.tool != "STOP":
                self.max_actions_reached = True
                action = ToolAction("STOP")
            self.executor.validator._validate_action(action, len(self.trace))
            if action.tool == "FILTER":
                scope = action.arguments["scope"]
                source_pool = self.executor._filter_source_pool(
                    self.pool,
                    self.memory_store,
                    scope,
                )
                filtered = self.executor._filter(
                    source_pool,
                    field=action.arguments["field"],
                    op=action.arguments["op"],
                    value=action.arguments["value"],
                )
                if scope == "full_memory":
                    self.pool, self.pool_overflow_count = self.executor._merge_discovery_pool(
                        self.pool, filtered, self.pool_has_candidates
                    )
                else:
                    self.pool = filtered[: self.executor.max_pool_size]
                self.pool_has_candidates = True
                new_evidence = self.executor._refresh_evidence_from_pool(
                    self.evidence, self.pool, source="FILTER"
                )
            elif action.tool == "SORT":
                self.pool = self.executor._sort(self.pool, **action.arguments)
                if self.pool_has_candidates:
                    new_evidence = self.executor._refresh_evidence_from_pool(
                        self.evidence, self.pool, source="SORT"
                    )
            elif action.tool == "TOPK":
                self.pool = self.executor._topk_turns(self.pool, action.arguments["k"])
                if self.pool_has_candidates:
                    new_evidence = self.executor._refresh_evidence_from_pool(
                        self.evidence, self.pool, source="TOPK"
                    )
            elif action.tool == "RETRIEVE":
                retrieve_query = action.arguments.get("query") or self.query
                retrieved = self.executor.retriever.retrieve(
                    self.memory_store.initial_pool(),
                    query=retrieve_query,
                    store=self.memory_store,
                    method=action.arguments.get("method", "hybrid"),
                    top_k=action.arguments.get("top_k", 5),
                    question_image=self.question_image,
                )
                self.pool, self.pool_overflow_count = self.executor._merge_discovery_pool(
                    self.pool, retrieved, self.pool_has_candidates
                )
                self.pool_has_candidates = True
                new_evidence = self.executor._refresh_evidence_from_pool(
                    self.evidence, self.pool, source="RETRIEVE"
                )
            elif action.tool == "EXPAND_NEIGHBORS":
                if not self.pool_has_candidates or not self.pool:
                    raise ValueError("EXPAND_NEIGHBORS requires an existing candidate pool")
                expanded = self.executor._expand_neighbors(
                    self.pool,
                    self.memory_store,
                    action.arguments["window"],
                )
                self.pool, self.pool_overflow_count = self.executor._merge_discovery_pool(
                    self.pool,
                    expanded,
                    self.pool_has_candidates,
                    prioritize_incoming=False,
                )
                self.pool_has_candidates = True
                new_evidence = self.executor._refresh_evidence_from_pool(
                    self.evidence, self.pool, source="EXPAND_NEIGHBORS"
                )
            elif action.tool == "DROP":
                if self.evidence_revision <= self.last_drop_revision:
                    raise ValueError("DROP requires evidence added or enriched since the previous DROP")
                self.pool = self.executor._drop_pool(
                    self.pool,
                    action.arguments["evidence_ids"],
                    self.evidence_ids_by_memory,
                )
                self.pool_has_candidates = True
                self.executor._refresh_evidence_from_pool(self.evidence, self.pool, source="DROP")
                self.last_drop_revision = self.evidence_revision
                self.drop_calls += 1
                self.dropped_evidence_count += len(action.arguments["evidence_ids"])
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
                new_evidence = inspected
            elif action.tool == "STOP":
                self.stopped = True
        except Exception as exc:
            step_error = str(exc)
            self.error = step_error

        if self.pool_has_candidates:
            self.executor._ensure_public_evidence_ids(self.pool, self.evidence_ids_by_memory)
        if new_evidence:
            self.evidence_revision += 1

        self.trace.append(action)
        self.steps.append(
            ExecutionStep(
                index=len(self.steps),
                action=action,
                pool_before=before,
                pool_after=len(self.pool),
                evidence_added=len(new_evidence),
                error=step_error,
            )
        )
        return self._observation(action, new_evidence, step_error)

    async def execute_inspect_raw_with_teacher(self, action: ToolAction, inspect_fn: Any) -> dict[str, Any]:
        """Execute INSPECT_RAW using the async verl teacher service callback."""
        before = len(self.pool)
        step_error = ""
        inspected: list[EvidenceItem] = []

        if self.stopped:
            return self._observation(action, [], "trajectory already stopped")

        if len(self.trace) >= self.executor.validator.max_actions - 1 and action.tool != "STOP":
            self.max_actions_reached = True
            action = ToolAction("STOP")
            self.stopped = True
            self.trace.append(action)
            self.steps.append(
                ExecutionStep(
                    index=len(self.steps),
                    action=action,
                    pool_before=before,
                    pool_after=len(self.pool),
                    evidence_added=0,
                    error="",
                )
            )
            return self._observation(action, [], "")

        try:
            self.executor.validator._validate_action(action, len(self.trace))
            remaining = max(0, self.executor.max_raw_inspections - self.raw_calls)
            inspect_pool = [item for item in self.pool if item.retrieved]
            text_by_turn = self.executor._text_context_by_turn(inspect_pool)
            for item in inspect_pool:
                if len(inspected) >= remaining:
                    break
                pointer = item.memory.raw_pointer
                if not pointer:
                    continue
                context = text_by_turn.get(item.memory.turn_id, "")
                visual_observation = await inspect_fn(
                    {
                        "raw_pointer": pointer,
                        "query": self.query,
                        "question_image": self.question_image,
                        "text_context": context,
                    }
                )
                fields = {
                    "visual_observation": str(visual_observation or ""),
                    "linked_text_context": context,
                    "image_label": f"context={context[:220]}",
                    "session_date": item.memory.metadata.get("session_date"),
                    "timestamp": item.memory.timestamp,
                }
                image_id = item.memory.public_image_id()
                if image_id:
                    fields["image_id"] = image_id
                inspected.append(
                    EvidenceItem(
                        memory_id=item.memory.memory_id,
                        fields=fields,
                        source="INSPECT_RAW",
                    )
                )
            self.raw_calls += len(inspected)
            self.evidence.extend(inspected)
            if inspected:
                self.evidence_revision += 1
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
                evidence_added=len(inspected),
                error=step_error,
            )
        )
        return self._observation(action, inspected, step_error)

    def _observation(self, action: ToolAction, new_evidence: list[EvidenceItem], error: str) -> dict[str, Any]:
        """Return a bounded snapshot of the current accumulated state.

        ToolAgentLoop already keeps the assistant tool call in message history,
        so repeating its full arguments here only grows the prompt.  Candidate
        and evidence previews have fixed item and text limits regardless of the
        size of the current pool.
        """
        visible_pool = self.pool if self.pool_has_candidates else []
        self.executor._ensure_public_evidence_ids(visible_pool, self.evidence_ids_by_memory)
        new_evidence_ids = list(
            dict.fromkeys(
                self.evidence_ids_by_memory[item.memory_id]
                for item in new_evidence
                if item.memory_id in self.evidence_ids_by_memory
            )
        )
        observation = {
            "refresh_state": False,
            "tool": action.tool,
            "pool_count": len(visible_pool),
            "pool_capacity": self.executor.max_pool_size,
            "pool_overflow_count": self.pool_overflow_count,
            "evidence_count": len(self.evidence),
            "evidence_memory_count": len(visible_pool),
            "pool_preview": _sanitize_pool_preview(visible_pool, self.evidence_ids_by_memory),
            "new_evidence_count": len(new_evidence),
            "new_evidence_ids": new_evidence_ids,
            "evidence_revision": self.evidence_revision,
            "last_drop_revision": self.last_drop_revision,
            "drop_calls": self.drop_calls,
            "dropped_evidence_count": self.dropped_evidence_count,
            "evidence_catalog": _sanitize_evidence_catalog(
                visible_pool,
                self.evidence,
                self.evidence_ids_by_memory,
            ),
            "evidence_preview": _sanitize_evidence_preview(
                self.evidence,
                self.evidence_ids_by_memory,
            ),
            "stopped": self.stopped,
            "error": _clip_text(error),
        }
        if action.tool == "DROP" and not error:
            observation["dropped_evidence_ids"] = [
                str(value).strip() for value in action.arguments.get("evidence_ids", [])
            ]
        return observation

    def public_state(self) -> dict[str, Any]:
        """Return serializable public state for AgentLoopOutput.extra_fields."""
        visible_pool = self.pool if self.pool_has_candidates else []
        self.executor._ensure_public_evidence_ids(visible_pool, self.evidence_ids_by_memory)
        return {
            "query": self.query,
            "pool_count": len(visible_pool),
            "pool_capacity": self.executor.max_pool_size,
            "pool_overflow_count": self.pool_overflow_count,
            "evidence_count": len(self.evidence),
            "evidence_memory_count": len(visible_pool),
            "pool_preview": _sanitize_pool_preview(visible_pool, self.evidence_ids_by_memory),
            "evidence_catalog": _sanitize_evidence_catalog(
                visible_pool,
                self.evidence,
                self.evidence_ids_by_memory,
            ),
            "evidence": _sanitize_evidence(self.evidence, self.evidence_ids_by_memory),
            "evidence_revision": self.evidence_revision,
            "last_drop_revision": self.last_drop_revision,
            "drop_calls": self.drop_calls,
            "dropped_evidence_count": self.dropped_evidence_count,
            "trace": [action.to_dict() for action in self.trace],
            "stopped": self.stopped,
            "error": self.error,
            "raw_inspection_calls": self.raw_calls,
            "max_actions_reached": self.max_actions_reached,
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
        if agent_data is not None:
            _update_dynamic_drop_schema(
                agent_data,
                [item["evidence_id"] for item in observation.get("evidence_catalog", [])],
            )
        if agent_data is not None and hasattr(agent_data, "extra_fields"):
            agent_data.extra_fields["opd_mm"] = session.public_state()
            agent_data.extra_fields["opd_mm_prompt_state"] = {
                "action_history": [item.to_dict() for item in session.trace],
                "observation": observation,
            }
        terminate_agent_loop = bool(observation["stopped"] or observation["error"])
        return ToolResponse(text=json.dumps(observation, ensure_ascii=False)), 0.0, {
            "opd_mm_pool_count": observation["pool_count"],
            "opd_mm_evidence_count": observation["evidence_count"],
            "opd_mm_drop_calls": observation["drop_calls"],
            "opd_mm_dropped_evidence_count": observation["dropped_evidence_count"],
            "opd_mm_terminate": terminate_agent_loop,
            "agent_loop_terminate": terminate_agent_loop,
        }

    def _action(self, parameters: dict[str, Any]) -> ToolAction:
        return ToolAction(self.tool_name.upper(), dict(parameters))

    def _runtime(self, agent_data: Any) -> dict[str, Any]:
        runtime = dict(self.config or {})
        if agent_data is not None:
            tools_kwargs = getattr(agent_data, "tools_kwargs", {}) or {}
            runtime.update(tools_kwargs.get("opd_mm", {}) or {})
            runtime.update(tools_kwargs.get(self.name, {}) or {})
        return runtime

    def _session(self, agent_data: Any) -> OPDToolSession:
        if agent_data is not None and hasattr(agent_data, _SESSION_ATTR):
            return getattr(agent_data, _SESSION_ATTR)

        runtime = self._runtime(agent_data)

        store = runtime.get("memory_store")
        if store is None:
            store = hidden_store_from_records(
                runtime.get("records") or runtime.get("memory_records") or [],
                vector_store_dir=(
                    runtime.get("vector_store_dir")
                    or runtime.get("index_store_dir")
                    or runtime.get("memory_store_dir")
                    or _AUTO_VECTOR_STORE
                ),
                dense_model_path=runtime.get("dense_model_path"),
                vision_model_path=runtime.get("vision_model_path"),
                hybrid_model_path=runtime.get("hybrid_model_path"),
                vector_device=str(runtime.get("vector_device") or runtime.get("retriever_device") or "cuda:0"),
            )
        if not isinstance(store, HiddenMemoryStore):
            raise TypeError("OPD-MM tools require a HiddenMemoryStore or records in tools_kwargs['opd_mm']")

        query = runtime.get("query") or runtime.get("raw_query") or self._query_from_agent_data(agent_data)
        session = OPDToolSession(
            executor=ToolExecutor(
                retriever=runtime.get("retriever") or TurnAwareHybridRetriever(),
                raw_inspector=_raw_inspector_from_runtime(runtime),
                validator=runtime.get("validator") or TrajectoryValidator(
                    max_actions=int(runtime.get("max_actions", DEFAULT_MAX_ACTIONS)),
                    max_top_k=int(runtime.get("max_top_k", 50)),
                    allow_inspect_raw=bool(runtime.get("allow_inspect_raw", True)),
                ),
                max_raw_inspections=int(runtime.get("max_raw_inspections", 3)),
                max_pool_size=int(runtime.get("max_pool_size", DEFAULT_MAX_POOL_SIZE)),
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
    description = (
        "Filter hidden memories by metadata. current_pool narrows the working pool; "
        "full_memory collects matching candidates from the original memory when the current pool is too narrow."
    )
    properties = {
        "field": _property("string", "The memory field to filter.", sorted(FILTER_FIELDS)),
        "op": _property("string", "The comparison operator.", sorted(FILTER_OPS)),
        "value": _property(
            ["string", "number", "boolean"],
            "Allowed values by field: modality uses text or image; source_type uses dialogue_turn or "
            "dialogue_image; status uses active; timestamp uses a public YYYY-MM-DD date or timestamp. "
            "Do not use memory IDs.",
        ),
        "scope": _property(
            "string",
            "Required filter scope. Use full_memory for an independent metadata/date filter over the original "
            "memory store. Use current_pool only to intentionally intersect the existing candidates; unrelated "
            "or mutually exclusive current_pool filters can empty the pool.",
            sorted(FILTER_SCOPES),
        ),
    }
    required = ["field", "op", "value", "scope"]


class OPDSortTool(OPDBaseTool):
    tool_name = "sort"
    description = "Sort the current candidate pool when timestamp, recency, score, or turn order matters."
    properties = {
        "field": _property("string", "The field to sort by.", sorted(SORT_FIELDS)),
        "order": _property("string", "Sort order.", sorted(SORT_ORDERS)),
    }
    required = ["field", "order"]


class OPDTopKTool(OPDBaseTool):
    tool_name = "topk"
    description = "Keep the strongest k turns from the current candidate pool after retrieval, filtering, or sorting."
    properties = {"k": _property("integer", "Positive number of candidate turns to keep.")}
    required = ["k"]


class OPDRetrieveTool(OPDBaseTool):
    tool_name = "retrieve"
    description = (
        "Rank hidden memories against the original user query or an optional rewritten query. "
        "Always searches the original hidden memory store and replaces the working pool."
    )
    properties = {
        "method": _property(
            "string",
            "Retrieval route: bm25 for exact names, IDs, dates, or phrases; dense for paraphrased semantic text; "
            "vision for image matching or visual attributes; hybrid when text/caption and visual signals are both "
            "materially relevant.",
            sorted(RETRIEVAL_METHODS),
        ),
        "top_k": _property(
            "integer",
            "Positive number of turns to retrieve (1-50); use larger values for broad coverage and smaller values "
            "for a focused candidate set.",
        ),
        "query": _property(
            "string",
            "Optional rewritten search text for this retrieval step. Omit to use the original user query.",
        ),
    }
    required = ["method", "top_k"]


class OPDExpandNeighborsTool(OPDBaseTool):
    tool_name = "expand_neighbors"
    description = (
        "Expand the current candidate pool with neighboring turns from the same session. "
        "Use only after retrieval/filtering has selected relevant candidates."
    )
    properties = {
        "window": _property(
            "integer",
            "Neighbor distance in turns. Must be 1, 2, or 3.",
            sorted(EXPAND_NEIGHBOR_WINDOWS),
        ),
    }
    required = ["window"]


class OPDDropTool(OPDBaseTool):
    tool_name = "drop"
    description = (
        "Remove clearly irrelevant, duplicate, or conflicting memories from the current candidate pool using "
        "their public evidence_id values. Submit all removals in one call. Do not call again until a later action "
        "adds or enriches evidence; omit this action when the current evidence is already useful."
    )
    properties = {
        "evidence_ids": {
            "type": "array",
            "description": "Non-empty unique list of public E1/E2/... IDs from the current evidence_catalog.",
            "items": {"type": "string", "pattern": "^E[1-9][0-9]*$"},
            "minItems": 1,
            "uniqueItems": True,
        }
    }
    required = ["evidence_ids"]


class OPDInspectRawTool(OPDBaseTool):
    tool_name = "inspect_raw"
    description = (
        "Opt-in raw visual inspection for images/media in the current retrieved candidate pool. "
        "Use only after retrieve/filter has selected candidate images. This cannot inspect the user's "
        "attached question image, cannot inspect an empty pool, and does not search the original full memory store."
    )
    properties = {
        "target": _property("string", "Inspection target.", sorted(INSPECT_TARGETS)),
        "instruction": _property("string", "Inspection instruction.", sorted(INSPECT_INSTRUCTIONS)),
    }
    required = ["target", "instruction"]

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        runtime = self._runtime(agent_data)
        backend = _optional_str(
            runtime.get("raw_inspector_backend") or os.getenv("OPD_MM_RAW_INSPECTOR_BACKEND")
        ).lower()
        if backend != "teacher":
            return await super().execute(instance_id, parameters, **kwargs)

        session = self._session(agent_data)
        action = self._action(parameters)
        inspect_fn = getattr(agent_data, "teacher_raw_inspector", None) if agent_data is not None else None
        if inspect_fn is None:
            async def unavailable(_: dict[str, Any]) -> str:
                raise RuntimeError("teacher raw inspector is unavailable")

            inspect_fn = unavailable
        observation = await session.execute_inspect_raw_with_teacher(action, inspect_fn)
        if agent_data is not None:
            _update_dynamic_drop_schema(
                agent_data,
                [item["evidence_id"] for item in observation.get("evidence_catalog", [])],
            )
        if agent_data is not None and hasattr(agent_data, "extra_fields"):
            agent_data.extra_fields["opd_mm"] = session.public_state()
            agent_data.extra_fields["opd_mm_prompt_state"] = {
                "action_history": [item.to_dict() for item in session.trace],
                "observation": observation,
            }
        terminate_agent_loop = bool(observation["stopped"] or observation["error"])
        return ToolResponse(text=json.dumps(observation, ensure_ascii=False)), 0.0, {
            "opd_mm_pool_count": observation["pool_count"],
            "opd_mm_evidence_count": observation["evidence_count"],
            "opd_mm_drop_calls": observation["drop_calls"],
            "opd_mm_dropped_evidence_count": observation["dropped_evidence_count"],
            "opd_mm_terminate": terminate_agent_loop,
            "agent_loop_terminate": terminate_agent_loop,
        }


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
    OPDExpandNeighborsTool,
    OPDDropTool,
    OPDInspectRawTool,
    OPDStopTool,
]


def openai_tool_schemas(
    include_inspect_raw: bool = True,
    evidence_ids: Optional[list[str]] = None,
) -> list[dict[str, Any]]:
    """Return OpenAI tool schemas for OPD-MM tools."""
    classes = (
        OPD_TOOL_CLASSES
        if include_inspect_raw
        else [cls for cls in OPD_TOOL_CLASSES if cls is not OPDInspectRawTool]
    )
    schemas = [
        _schema(cls.tool_name, cls.description, cls.properties, cls.required).model_dump(
            exclude_unset=True, exclude_none=True
        )
        for cls in classes
    ]
    if evidence_ids is not None:
        for schema in schemas:
            if schema["function"]["name"] != "drop":
                continue
            schema["function"]["parameters"]["properties"]["evidence_ids"]["items"]["enum"] = list(
                evidence_ids
            )
    return schemas


__all__ = [
    "OPDBaseTool",
    "OPDExpandNeighborsTool",
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
