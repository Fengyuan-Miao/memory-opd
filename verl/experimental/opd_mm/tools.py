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
ToolAgentLoop. SEARCH_METADATA, RETRIEVE, and EXPAND_NEIGHBORS build a bounded,
high-recall candidate pool. A fixed model-based semantic layer then selects
the smaller answer-evidence view exposed to the policy.
"""

from __future__ import annotations

import inspect
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
from verl.experimental.opd_mm.semantic_selector import RemoteSemanticEvidenceSelector, SemanticSelection
from verl.experimental.opd_mm.schema import (
    DEFAULT_MAX_ACTIONS,
    EXPAND_NEIGHBOR_WINDOWS,
    INSPECT_INSTRUCTIONS,
    INSPECT_TARGETS,
    METADATA_SEARCH_FIELDS,
    METADATA_SEARCH_OPS,
    RETRIEVAL_METHODS,
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
DEFAULT_EVIDENCE_SELECTOR_TIMEOUT = 120.0
DEFAULT_EVIDENCE_SELECTOR_MAX_TOKENS = 512
OBSERVATION_TEXT_MAX_CHARS = 220


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
        _optional_str(runtime.get("raw_inspector_model"))
        or _optional_str(os.getenv("OPD_MM_RAW_INSPECTOR_MODEL")),
        _optional_str(runtime.get("raw_inspector_api_key"))
        or _optional_str(os.getenv("OPD_MM_RAW_INSPECTOR_API_KEY")),
        float(
            runtime.get("raw_inspector_timeout")
            or os.getenv("OPD_MM_RAW_INSPECTOR_TIMEOUT")
            or DEFAULT_RAW_INSPECTOR_TIMEOUT
        ),
        int(
            runtime.get("raw_inspector_max_tokens")
            or os.getenv("OPD_MM_RAW_INSPECTOR_MAX_TOKENS")
            or DEFAULT_RAW_INSPECTOR_MAX_TOKENS
        ),
        float(runtime.get("raw_inspector_temperature") or os.getenv("OPD_MM_RAW_INSPECTOR_TEMPERATURE") or 0.0),
    )


@lru_cache(maxsize=16)
def _cached_remote_evidence_selector(
    base_url: str,
    model: str,
    api_key: str,
    timeout: float,
    max_tokens: int,
    retries: int,
) -> RemoteSemanticEvidenceSelector:
    return RemoteSemanticEvidenceSelector(
        base_url=base_url,
        model=model,
        api_key=api_key,
        timeout=timeout,
        max_tokens=max_tokens,
        retries=retries,
    )


def _evidence_selector_from_runtime(runtime: dict[str, Any]) -> Any:
    if runtime.get("evidence_selector") is not None:
        return runtime["evidence_selector"]
    if not bool(runtime.get("semantic_evidence_selection", True)):
        return None
    backend = _optional_str(
        os.getenv("OPD_MM_EVIDENCE_SELECTOR_BACKEND") or runtime.get("evidence_selector_backend")
    ).lower()
    if backend == "teacher":
        return None
    base_url = (
        _optional_str(runtime.get("evidence_selector_base_url"))
        or _optional_str(os.getenv("OPD_MM_EVIDENCE_SELECTOR_BASE_URL"))
        or _optional_str(os.getenv("OPD_MM_VERIFIER_BASE_URL"))
        or _optional_str(os.getenv("OPD_MM_OUTCOME_BASE_URL"))
    )
    model = (
        _optional_str(runtime.get("evidence_selector_model"))
        or _optional_str(os.getenv("OPD_MM_EVIDENCE_SELECTOR_MODEL"))
        or _optional_str(os.getenv("OPD_MM_VERIFIER_MODEL"))
        or _optional_str(os.getenv("OPD_MM_OUTCOME_MODEL"))
        or _optional_str(os.getenv("OUTCOME_SERVED_MODEL"))
    )
    if not base_url or not model:
        return None
    return _cached_remote_evidence_selector(
        base_url,
        model,
        _optional_str(runtime.get("evidence_selector_api_key"))
        or _optional_str(os.getenv("OPD_MM_EVIDENCE_SELECTOR_API_KEY"))
        or "EMPTY",
        float(
            runtime.get("evidence_selector_timeout")
            or os.getenv("OPD_MM_EVIDENCE_SELECTOR_TIMEOUT")
            or DEFAULT_EVIDENCE_SELECTOR_TIMEOUT
        ),
        int(
            runtime.get("evidence_selector_max_tokens")
            or os.getenv("OPD_MM_EVIDENCE_SELECTOR_MAX_TOKENS")
            or DEFAULT_EVIDENCE_SELECTOR_MAX_TOKENS
        ),
        int(runtime.get("evidence_selector_retries") or os.getenv("OPD_MM_EVIDENCE_SELECTOR_RETRIES") or 2),
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
    if metadata.get("scenario_file"):
        metadata["scenario_file"] = _resolve_dataset_asset_path(metadata["scenario_file"])
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
        raw_pointer=_resolve_dataset_asset_path(value.get("raw_pointer")),
        status=str(value.get("status", "active")),
        metadata=metadata,
    )


def _resolve_dataset_asset_path(value: Any) -> Any:
    """Rebase serialized dataset assets onto the current dataset root.

    Prepared parquet/JSONL files may have been built in a different checkout
    and therefore contain absolute image/dialog paths from that machine.
    """
    if not isinstance(value, (str, Path)) or not str(value):
        return value
    path = Path(str(value)).expanduser()
    if path.exists() or not path.is_absolute():
        return str(path)
    dataset_root = os.getenv("OPD_MM_DATASET_ROOT")
    if not dataset_root:
        return str(path)
    root = Path(dataset_root).expanduser()
    parts = path.parts
    for marker in ("image", "dialog"):
        if marker in parts:
            return str(root.joinpath(*parts[parts.index(marker) :]))
    return str(path)


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


def _sanitize_evidence(items: list[EvidenceItem]) -> list[dict[str, Any]]:
    """Expose one complete public record per memory.

    A memory can have both a MEMORY item and an INSPECT_EVIDENCE_IMAGE item internally.
    Merge those fields into the same public entry so the model never receives
    duplicate copies of one memory.
    """
    sanitized_by_memory: dict[str, dict[str, Any]] = {}
    for item in items:
        data = item.to_dict()
        data.pop("memory_id", None)
        data.pop("source", None)
        data.pop("author", None)
        entry = sanitized_by_memory.setdefault(item.memory_id, {})
        entry.update({key: value for key, value in data.items() if value not in (None, "")})
    return list(sanitized_by_memory.values())


def _clip_text(value: Any, max_chars: int = OBSERVATION_TEXT_MAX_CHARS) -> Any:
    if not isinstance(value, str) or len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "...(truncated)"


@dataclass
class OPDToolSession:
    """Per-trajectory state shared by OPD-MM tools."""

    executor: ToolExecutor
    memory_store: HiddenMemoryStore
    query: str
    question_image: Optional[str] = None
    evidence_selector: Any = None
    pool: list[PoolItem] = field(default_factory=list)
    evidence: list[EvidenceItem] = field(default_factory=list)
    steps: list[ExecutionStep] = field(default_factory=list)
    trace: list[ToolAction] = field(default_factory=list)
    raw_calls: int = 0
    stopped: bool = False
    error: str = ""
    pool_has_candidates: bool = False
    max_actions_reached: bool = False
    evidence_revision: int = 0
    pool_overflow_count: int = 0
    semantic_filter_status: str = "not_run"
    semantic_filter_error: str = ""
    semantic_filter_candidate_count: int = 0
    semantic_filter_selected_count: int = 0
    discovery_signatures: set[str] = field(default_factory=set)
    stagnant_discovery_signatures: set[str] = field(default_factory=set)
    search_exhausted: bool = False

    def __post_init__(self) -> None:
        if not self.pool:
            self.pool = self.memory_store.initial_pool()

    def execute(self, action: ToolAction, *, defer_semantic_filter: bool = False) -> dict[str, Any]:
        """Execute one validated action against the current hidden pool."""
        before = len(self.pool)
        evidence_ids_before = {item.memory_id for item in self.evidence}
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
            if action.tool == "SEARCH_METADATA":
                filtered = self.executor._search_metadata(
                    self.memory_store.initial_pool(),
                    field=action.arguments["field"],
                    op=action.arguments["op"],
                    value=action.arguments["value"],
                )
                self.pool, self.pool_overflow_count = self.executor._merge_discovery_pool(
                    self.pool, filtered, self.pool_has_candidates
                )
                self.pool_has_candidates = True
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
            elif action.tool == "EXPAND_NEIGHBORS":
                relevant_pool = self._evidence_pool()
                if not relevant_pool:
                    raise ValueError("EXPAND_NEIGHBORS requires existing answer evidence")
                expanded = self.executor._expand_neighbors(
                    relevant_pool,
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
            elif action.tool == "INSPECT_EVIDENCE_IMAGE":
                remaining = max(0, self.executor.max_raw_inspections - self.raw_calls)
                inspected = self.executor._inspect_raw(
                    self._evidence_pool(),
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

        if (
            not step_error
            and action.tool in {"SEARCH_METADATA", "RETRIEVE", "EXPAND_NEIGHBORS"}
            and not defer_semantic_filter
        ):
            new_evidence = self._apply_semantic_selection(
                [f"C{index}" for index in range(1, len(self.pool) + 1)],
                status="fallback_all_unconfigured",
                error="semantic selector was not invoked by this synchronous caller",
            )
        if new_evidence:
            self.evidence_revision += 1
        if (
            not step_error
            and not defer_semantic_filter
            and action.tool in {"SEARCH_METADATA", "RETRIEVE", "EXPAND_NEIGHBORS"}
        ):
            self._record_discovery_progress(action, evidence_ids_before)

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

    async def execute_with_semantic_filter(self, action: ToolAction) -> dict[str, Any]:
        """Execute one action and screen discovery results before exposing evidence."""
        evidence_ids_before = {item.memory_id for item in self.evidence}
        observation = self.execute(action, defer_semantic_filter=True)
        executed_action = self.trace[-1] if self.trace else action
        if (
            observation.get("error")
            or observation.get("stopped")
            or executed_action.tool not in {"SEARCH_METADATA", "RETRIEVE", "EXPAND_NEIGHBORS"}
        ):
            return observation

        candidates = self._selector_candidates()
        if self.evidence_selector is None:
            selection = SemanticSelection(
                [str(item["candidate_id"]) for item in candidates],
                status="fallback_all_unconfigured",
                error="semantic evidence selector is not configured",
            )
        else:
            try:
                selector = getattr(self.evidence_selector, "select", self.evidence_selector)
                try:
                    result = selector(
                        query=self.query,
                        candidates=candidates,
                        question_image=self.question_image,
                        action=executed_action.to_dict(),
                    )
                except TypeError as exc:
                    if "unexpected keyword argument" not in str(exc):
                        raise
                    result = selector(query=self.query, candidates=candidates)
                if inspect.isawaitable(result):
                    result = await result
                if isinstance(result, SemanticSelection):
                    selection = result
                elif isinstance(result, list):
                    selection = SemanticSelection([str(value) for value in result])
                else:
                    raise TypeError("semantic evidence selector must return SemanticSelection or list[str]")
            except Exception as exc:
                selection = SemanticSelection(
                    [str(item["candidate_id"]) for item in candidates],
                    status="fallback_all_error",
                    error=str(exc),
                )

        if not selection.selected_candidate_ids and self.evidence:
            previous_memory_ids = {item.memory_id for item in self.evidence}
            preserved_ids = [
                f"C{index}"
                for index, item in enumerate(self.pool, start=1)
                if item.memory.memory_id in previous_memory_ids
            ]
            if preserved_ids:
                selection = SemanticSelection(
                    preserved_ids,
                    status="preserved_previous_on_empty",
                    error="selector returned no candidates; preserved prior evidence",
                )

        new_evidence = self._apply_semantic_selection(
            selection.selected_candidate_ids,
            status=selection.status,
            error=selection.error,
        )
        if new_evidence:
            self.evidence_revision += 1
        if self.steps:
            self.steps[-1].evidence_added = len(new_evidence)
        self._record_discovery_progress(executed_action, evidence_ids_before)
        return self._observation(executed_action, new_evidence, "")

    def _record_discovery_progress(self, action: ToolAction, evidence_ids_before: set[str]) -> None:
        """Track when distinct searches stop surfacing new relevant memories."""
        signature = f"{action.tool}:{json.dumps(action.arguments, ensure_ascii=False, sort_keys=True, default=str)}"
        self.discovery_signatures.add(signature)
        evidence_ids_after = {item.memory_id for item in self.evidence}
        if evidence_ids_after - evidence_ids_before:
            self.stagnant_discovery_signatures.clear()
            self.search_exhausted = False
            return
        self.stagnant_discovery_signatures.add(signature)
        self.search_exhausted = len(self.stagnant_discovery_signatures) >= 2

    def _selector_candidates(self) -> list[dict[str, Any]]:
        candidates = []
        for index, item in enumerate(self.pool, start=1):
            fields = self.executor._pool_evidence([item], source="CANDIDATE")[0].fields
            candidates.append({"candidate_id": f"C{index}", **fields})
        return candidates

    def _apply_semantic_selection(
        self,
        selected_candidate_ids: list[str],
        *,
        status: str,
        error: str,
    ) -> list[EvidenceItem]:
        candidate_by_id = {f"C{index}": item for index, item in enumerate(self.pool, start=1)}
        selected_ids = list(dict.fromkeys(str(value).strip() for value in selected_candidate_ids))
        unknown = [value for value in selected_ids if value not in candidate_by_id]
        if unknown:
            selected_ids = list(candidate_by_id)
            status = "fallback_all_invalid_ids"
            error = f"selector returned unknown candidate IDs: {unknown[:8]}"
        selected_pool = [candidate_by_id[value] for value in selected_ids]
        before_signature = self._evidence_signature()
        new_evidence = self.executor._refresh_evidence_from_pool(
            self.evidence,
            selected_pool,
            source="SEMANTIC_SELECTION",
        )
        self.semantic_filter_status = str(status or "ok")
        self.semantic_filter_error = str(error or "")
        self.semantic_filter_candidate_count = len(self.pool)
        self.semantic_filter_selected_count = len(selected_pool)
        if self._evidence_signature() != before_signature and not new_evidence:
            self.evidence_revision += 1
        return new_evidence

    def _evidence_pool(self) -> list[PoolItem]:
        memory_ids = {item.memory_id for item in self.evidence}
        return [item for item in self.pool if item.memory.memory_id in memory_ids]

    def _evidence_signature(self) -> tuple[Any, ...]:
        return tuple(
            (item.memory_id, item.source, json.dumps(item.fields, ensure_ascii=False, sort_keys=True, default=str))
            for item in self.evidence
        )

    async def execute_inspect_raw_with_teacher(self, action: ToolAction, inspect_fn: Any) -> dict[str, Any]:
        """Execute INSPECT_EVIDENCE_IMAGE using the async verl teacher service callback."""
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
            inspect_pool = self._evidence_pool()
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
                        source="INSPECT_EVIDENCE_IMAGE",
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
        """Return the current accumulated state with one entry per memory.

        ToolAgentLoop already keeps the assistant tool call in message history,
        so repeating its full arguments here only grows the prompt. The pool's
        capacity bounds this complete evidence list without text truncation.
        """
        visible_pool = self.pool if self.pool_has_candidates else []
        new_evidence_count = len({item.memory_id for item in new_evidence})
        public_evidence = _sanitize_evidence(self.evidence)
        observation = {
            "refresh_state": False,
            "tool": action.tool,
            "pool_count": len(visible_pool),
            "pool_capacity": self.executor.max_pool_size,
            "pool_overflow_count": self.pool_overflow_count,
            "evidence_count": len(public_evidence),
            "evidence_memory_count": len(public_evidence),
            "new_evidence_count": new_evidence_count,
            "evidence_revision": self.evidence_revision,
            "semantic_filter_status": self.semantic_filter_status,
            "semantic_filter_candidate_count": self.semantic_filter_candidate_count,
            "semantic_filter_selected_count": self.semantic_filter_selected_count,
            "semantic_filter_error": _clip_text(self.semantic_filter_error),
            "distinct_search_count": len(self.discovery_signatures),
            "stagnant_search_count": len(self.stagnant_discovery_signatures),
            "search_exhausted": self.search_exhausted,
            # The pool remains private. This is the sole model-visible,
            # semantically screened answer-evidence representation.
            "evidence": public_evidence,
            "stopped": self.stopped,
            "error": _clip_text(error),
        }
        return observation

    def public_state(self) -> dict[str, Any]:
        """Return serializable public state for AgentLoopOutput.extra_fields."""
        visible_pool = self.pool if self.pool_has_candidates else []
        public_evidence = _sanitize_evidence(self.evidence)
        return {
            "query": self.query,
            "pool_count": len(visible_pool),
            "pool_capacity": self.executor.max_pool_size,
            "pool_overflow_count": self.pool_overflow_count,
            "evidence_count": len(public_evidence),
            "evidence_memory_count": len(public_evidence),
            "evidence": public_evidence,
            "evidence_revision": self.evidence_revision,
            "semantic_filter_status": self.semantic_filter_status,
            "semantic_filter_candidate_count": self.semantic_filter_candidate_count,
            "semantic_filter_selected_count": self.semantic_filter_selected_count,
            "semantic_filter_error": self.semantic_filter_error,
            "distinct_search_count": len(self.discovery_signatures),
            "stagnant_search_count": len(self.stagnant_discovery_signatures),
            "search_exhausted": self.search_exhausted,
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
        observation = await session.execute_with_semantic_filter(action)
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
            "opd_mm_semantic_filter_fallback": float(
                str(observation["semantic_filter_status"]).startswith("fallback_all")
            ),
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
        teacher_selector = getattr(agent_data, "teacher_evidence_selector", None) if agent_data is not None else None
        if teacher_selector is not None:
            runtime["evidence_selector"] = teacher_selector

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
            question_image=_resolve_dataset_asset_path(runtime.get("question_image")),
            evidence_selector=_evidence_selector_from_runtime(runtime),
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


class OPDSearchMetadataTool(OPDBaseTool):
    tool_name = "search_metadata"
    description = (
        "Search the complete hidden memory store by public metadata and merge deduplicated matches into the "
        "private candidate pool. An internal semantic selector then exposes only question-relevant answer evidence."
    )
    properties = {
        "field": _property(
            "string",
            "Metadata field. modality is only text/image; status is only active; timestamp is only a public ISO date.",
            sorted(METADATA_SEARCH_FIELDS),
        ),
        "op": _property(
            "string",
            "Field-specific operator: modality uses eq/neq; status uses eq; timestamp uses eq/contains/before/after.",
            sorted(METADATA_SEARCH_OPS),
        ),
        "value": _property(
            ["string", "number", "boolean"],
            "Literal metadata value only: text/image for modality, active for status, or a public YYYY, YYYY-MM, "
            "YYYY-MM-DD, or ISO timestamp. Never put a topic, entity, event, or memory ID here; use RETRIEVE for those.",
        ),
    }
    required = ["field", "op", "value"]


class OPDRetrieveTool(OPDBaseTool):
    tool_name = "retrieve"
    description = (
        "Rank hidden memories against the original user query or an optional rewritten query. "
        "Always searches the original hidden memory store and fuses deduplicated results into the bounded working "
        "pool. Memories supported by multiple retrieval steps are prioritized, then an internal semantic selector "
        "exposes only candidates relevant to answering the question."
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
        "Expansion is anchored on current answer evidence, and the expanded pool is semantically screened again. "
        "Use only after retrieval/filtering has produced relevant evidence."
    )
    properties = {
        "window": _property(
            "integer",
            "Neighbor distance in turns. Must be 1, 2, or 3.",
            sorted(EXPAND_NEIGHBOR_WINDOWS),
        ),
    }
    required = ["window"]


class OPDInspectEvidenceImageTool(OPDBaseTool):
    tool_name = "inspect_evidence_image"
    description = (
        "Inspect raw visual details only for images already present in current public answer evidence. "
        "This action cannot search the hidden memory store or inspect the user's attached question image."
    )
    properties = {
        "target": _property("string", "Images in current public answer evidence.", sorted(INSPECT_TARGETS)),
        "instruction": _property("string", "Inspection instruction.", sorted(INSPECT_INSTRUCTIONS)),
    }
    required = ["target", "instruction"]

    async def execute(self, instance_id: str, parameters: dict[str, Any], **kwargs) -> tuple[ToolResponse, float, dict]:
        agent_data = kwargs.get("agent_data")
        runtime = self._runtime(agent_data)
        backend = _optional_str(
            os.getenv("OPD_MM_RAW_INSPECTOR_BACKEND") or runtime.get("raw_inspector_backend")
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
            "opd_mm_semantic_filter_fallback": float(
                str(observation["semantic_filter_status"]).startswith("fallback_all")
            ),
            "opd_mm_terminate": terminate_agent_loop,
            "agent_loop_terminate": terminate_agent_loop,
        }


class OPDStopTool(OPDBaseTool):
    tool_name = "stop"
    description = "Stop the OPD-MM retrieval trajectory once enough evidence is collected."
    properties: dict[str, dict[str, Any]] = {}
    required: list[str] = []


OPD_TOOL_CLASSES = [
    OPDSearchMetadataTool,
    OPDRetrieveTool,
    OPDExpandNeighborsTool,
    OPDInspectEvidenceImageTool,
    OPDStopTool,
]


def openai_tool_schemas(
    include_inspect_raw: bool = True,
) -> list[dict[str, Any]]:
    """Return OpenAI tool schemas for OPD-MM tools."""
    classes = (
        OPD_TOOL_CLASSES
        if include_inspect_raw
        else [cls for cls in OPD_TOOL_CLASSES if cls is not OPDInspectEvidenceImageTool]
    )
    return [
        _schema(cls.tool_name, cls.description, cls.properties, cls.required).model_dump(
            exclude_unset=True, exclude_none=True
        )
        for cls in classes
    ]


__all__ = [
    "OPDBaseTool",
    "OPDExpandNeighborsTool",
    "OPDSearchMetadataTool",
    "OPDInspectEvidenceImageTool",
    "OPDRetrieveTool",
    "OPDStopTool",
    "OPDToolSession",
    "OPD_TOOL_CLASSES",
    "hidden_store_from_records",
    "memory_record_from_dict",
    "openai_tool_schemas",
]
