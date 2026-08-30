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
from verl.experimental.opd_mm.models import (
    EventCandidate,
    EventEvidence,
    EvidenceItem,
    ExecutionStep,
    MemoryRecord,
    PoolItem,
    ToolAction,
)
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
    parts = path.parts
    candidates: list[Path] = []

    # Prefer the same repository-relative dataset path. This handles mixed
    # validation batches (for example Mem-Gallery plus MMem) without rebasing
    # every serialized path onto whichever dataset happens to be the training
    # root.
    if "dataset" in parts:
        dataset_index = parts.index("dataset")
        repository_root = Path(__file__).resolve().parents[3]
        candidates.append(repository_root.joinpath(*parts[dataset_index:]))

    configured_roots: list[Path] = []
    raw_roots = os.getenv("OPD_MM_DATASET_ROOTS", "")
    if raw_roots:
        configured_roots.extend(
            Path(item).expanduser() for item in raw_roots.split(os.pathsep) if item
        )
    dataset_root = os.getenv("OPD_MM_DATASET_ROOT")
    if dataset_root:
        configured_roots.append(Path(dataset_root).expanduser())

    for root in configured_roots:
        # Preserve the complete suffix below the configured dataset directory,
        # including an intermediate ``data`` component when one exists.
        root_name = root.name
        if root_name in parts:
            root_index = len(parts) - 1 - tuple(reversed(parts)).index(root_name)
            candidates.append(root.joinpath(*parts[root_index + 1 :]))
        for marker in ("image", "dialog"):
            if marker not in parts:
                continue
            marker_index = parts.index(marker)
            suffix = parts[marker_index:]
            candidates.append(root.joinpath(*suffix))
            if root.name != "data":
                candidates.append(root.joinpath("data", *suffix))

    seen: set[str] = set()
    for candidate in candidates:
        normalized = str(candidate)
        if normalized in seen:
            continue
        seen.add(normalized)
        if candidate.exists():
            return normalized

    # Keep the original path when no verified local replacement exists. This
    # preserves a useful error instead of manufacturing a plausible-looking
    # but invalid path.
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
    pool: list[EventCandidate] = field(default_factory=list)
    evidence: list[EventEvidence] = field(default_factory=list)
    steps: list[ExecutionStep] = field(default_factory=list)
    trace: list[ToolAction] = field(default_factory=list)
    raw_calls: int = 0
    terminated: bool = False
    policy_stopped: bool = False
    stopped: bool = False
    termination_reason: str = ""
    error: str = ""
    pool_has_candidates: bool = False
    max_actions_reached: bool = False
    evidence_revision: int = 0
    pool_revision: int = 0
    pool_overflow_count: int = 0
    semantic_filter_status: str = "not_run"
    semantic_filter_error: str = ""
    semantic_filter_candidate_count: int = 0
    semantic_filter_selected_count: int = 0
    discovery_signatures: set[str] = field(default_factory=set)
    discovery_attempt_count: int = 0
    retrieval_methods_tried: set[str] = field(default_factory=set)
    rewritten_queries: set[str] = field(default_factory=set)
    metadata_fields_tried: set[str] = field(default_factory=set)
    modalities_searched: set[str] = field(default_factory=set)
    neighbor_windows_tried: set[int] = field(default_factory=set)
    consecutive_no_gain_count: int = 0
    blocked_action_count: int = 0
    last_action_status: str = "not_run"
    blocked_action_reason: str = ""
    inspected_image_memory_ids: set[str] = field(default_factory=set)
    inspected_image_ids: set[str] = field(default_factory=set)
    visual_observations: dict[str, str] = field(default_factory=dict)
    candidate_ids: dict[str, str] = field(default_factory=dict)
    evidence_ids: dict[str, str] = field(default_factory=dict)
    session_aliases: dict[str, str] = field(default_factory=dict)
    image_aliases: dict[str, str] = field(default_factory=dict)
    no_gain_action_revisions: dict[str, tuple[int, int]] = field(default_factory=dict)

    def execute(self, action: ToolAction, *, defer_semantic_filter: bool = False) -> dict[str, Any]:
        """Execute one action against the event-level private working pool."""
        before = len(self.pool)
        pool_keys_before = {item.event_key for item in self.pool}
        pool_signature_before = self._pool_signature()
        evidence_signature_before = self._evidence_signature()
        step_error = ""
        changed_evidence: list[EventEvidence] = []
        self.pool_overflow_count = 0
        self.last_action_status = "executed"
        self.blocked_action_reason = ""

        if self.terminated:
            return self._observation(action, [], "trajectory already terminated")
        if len(self.trace) >= self.executor.validator.max_actions:
            self._terminate("budget_exhausted")
            return self._observation(action, [], "")

        try:
            self.executor.validator._validate_action(action, len(self.trace))
            blocked_reason = self._guardrail_reason(action)
            if blocked_reason:
                self._record_blocked_action(action, before, blocked_reason)
                return self._finalize_action_observation(action, [], "")

            if action.tool in {"SEARCH_METADATA", "RETRIEVE", "EXPAND_NEIGHBORS"}:
                incoming = self._discover_events(action)
                self.discovery_attempt_count += 1
                self._record_search_dimensions(action, incoming)
                self.pool, self.pool_overflow_count = self._merge_event_pool(
                    self.pool,
                    incoming,
                    self.pool_has_candidates,
                    prioritize_incoming=action.tool != "EXPAND_NEIGHBORS",
                )
                self.pool_has_candidates = True
                if self._pool_signature() != pool_signature_before:
                    self.pool_revision += 1
            elif action.tool == "INSPECT_EVIDENCE_IMAGE":
                changed_evidence = self._inspect_evidence_images_sync()
            elif action.tool == "STOP":
                self._terminate("policy_stop")
        except Exception as exc:
            step_error = str(exc)
            self.error = step_error
            self._terminate("tool_error")

        if (
            not step_error
            and action.tool in {"SEARCH_METADATA", "RETRIEVE", "EXPAND_NEIGHBORS"}
            and not defer_semantic_filter
        ):
            changed_evidence = self._apply_semantic_selection(
                [self._candidate_id(item.event_key) for item in self.pool],
                status="fallback_all_unconfigured",
                error="semantic selector was not invoked by this synchronous caller",
            )
        if (
            not step_error
            and not defer_semantic_filter
            and action.tool in {"SEARCH_METADATA", "RETRIEVE", "EXPAND_NEIGHBORS"}
        ):
            self._record_discovery_progress(action, pool_keys_before, evidence_signature_before)

        self.trace.append(action)
        self.steps.append(
            ExecutionStep(
                index=len(self.steps),
                action=action,
                pool_before=before,
                pool_after=len(self.pool),
                evidence_added=len(changed_evidence),
                error=step_error,
            )
        )
        return self._finalize_action_observation(action, changed_evidence, step_error)

    async def execute_with_semantic_filter(self, action: ToolAction) -> dict[str, Any]:
        """Execute one action and screen discovery results before exposing evidence."""
        pool_keys_before = {item.event_key for item in self.pool}
        evidence_signature_before = self._evidence_signature()
        observation = self.execute(action, defer_semantic_filter=True)
        executed_action = self.trace[-1] if self.trace else action
        if (
            observation.get("error")
            or self.last_action_status == "blocked"
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
            previous_event_keys = {item.event_key for item in self.evidence}
            preserved_ids = [
                self._candidate_id(item.event_key)
                for item in self.pool
                if item.event_key in previous_event_keys
            ]
            if preserved_ids:
                selection = SemanticSelection(
                    preserved_ids,
                    status="preserved_previous_on_empty",
                    error="selector returned no candidates; preserved prior evidence",
                )

        changed_evidence = self._apply_semantic_selection(
            selection.selected_candidate_ids,
            status=selection.status,
            error=selection.error,
        )
        if self.steps:
            self.steps[-1].evidence_added = len(changed_evidence)
        self._record_discovery_progress(executed_action, pool_keys_before, evidence_signature_before)
        return self._observation(executed_action, changed_evidence, "")

    def _discover_events(self, action: ToolAction) -> list[EventCandidate]:
        if action.tool == "SEARCH_METADATA":
            records = self.executor._search_metadata(
                self.memory_store.initial_pool(),
                field=action.arguments["field"],
                op=action.arguments["op"],
                value=action.arguments["value"],
            )
            return self.executor.group_event_candidates(records, self.memory_store, hydrate_turn=True)
        if action.tool == "RETRIEVE":
            records = self.executor.retriever.retrieve(
                self.memory_store.initial_pool(),
                query=action.arguments.get("query") or self.query,
                store=self.memory_store,
                method=action.arguments.get("method", "hybrid"),
                top_k=action.arguments.get("top_k", 5),
                question_image=self.question_image,
            )
            return self.executor.group_event_candidates(records, self.memory_store, hydrate_turn=True)
        return self.executor.expand_neighbor_events(self._evidence_pool(), self.memory_store, action.arguments["window"])

    def _merge_event_pool(
        self,
        existing: list[EventCandidate],
        incoming: list[EventCandidate],
        has_existing_candidates: bool,
        *,
        prioritize_incoming: bool,
    ) -> tuple[list[EventCandidate], int]:
        if not has_existing_candidates:
            return list(incoming[: self.executor.max_pool_size]), max(0, len(incoming) - self.executor.max_pool_size)
        by_key = {item.event_key: item for item in existing}
        for event in incoming:
            previous = by_key.get(event.event_key)
            if previous is None:
                by_key[event.event_key] = event
                continue
            records = {item.memory.memory_id: item for item in previous.records}
            records.update({item.memory.memory_id: item for item in event.records})
            previous.records = sorted(records.values(), key=lambda item: (item.memory.timestamp, item.memory.memory_id))
            previous.score = event.score or previous.score
            previous.retrieved = previous.retrieved or event.retrieved

        existing_ranks = {item.event_key: rank for rank, item in enumerate(existing, start=1)}
        incoming_ranks = {item.event_key: rank for rank, item in enumerate(incoming, start=1)}
        stable_order = {
            key: index
            for index, key in enumerate(dict.fromkeys([item.event_key for item in existing + incoming]))
        }

        def fusion_key(key: str) -> tuple[float, int, int, int]:
            ranks = [rank for rank in (existing_ranks.get(key), incoming_ranks.get(key)) if rank is not None]
            preferred = incoming_ranks if prioritize_incoming else existing_ranks
            return (-sum(1.0 / (60.0 + rank) for rank in ranks), 0 if key in preferred else 1, min(ranks), stable_order[key])

        ranked = [by_key[key] for key in sorted(by_key, key=fusion_key)]
        return ranked[: self.executor.max_pool_size], max(0, len(ranked) - self.executor.max_pool_size)

    def _record_discovery_progress(
        self,
        action: ToolAction,
        pool_keys_before: set[str],
        evidence_signature_before: tuple[Any, ...],
    ) -> None:
        """Track objective search attempts without claiming answer absence."""
        signature = self._action_signature(action)
        self.discovery_signatures.add(signature)
        # Private retrieval hits that the semantic selector rejects are not
        # answer-level progress.  Count gain only when the public evidence
        # obtains a new event or an existing event is materially updated.
        # This prevents endless broad-pool churn from making absence searches
        # look as if they are still progressing.
        gained = self._evidence_signature() != evidence_signature_before
        if gained:
            self.consecutive_no_gain_count = 0
            self.no_gain_action_revisions.pop(signature, None)
        else:
            self.consecutive_no_gain_count += 1
            self.no_gain_action_revisions[signature] = (self.pool_revision, self.evidence_revision)

    def _record_search_dimensions(
        self,
        action: ToolAction,
        incoming: list[EventCandidate],
    ) -> None:
        if action.tool == "RETRIEVE":
            method = str(action.arguments.get("method") or "hybrid").lower()
            self.retrieval_methods_tried.add(method)
            self.modalities_searched.update({"text", "image"} if method == "hybrid" else ({"image"} if method == "vision" else {"text"}))
            rewritten = " ".join(str(action.arguments.get("query") or "").lower().split())
            original = " ".join(self.query.lower().split())
            if rewritten and rewritten != original:
                self.rewritten_queries.add(rewritten)
        elif action.tool == "SEARCH_METADATA":
            field_name = str(action.arguments.get("field") or "")
            self.metadata_fields_tried.add(field_name)
            if field_name == "modality":
                self.modalities_searched.add(str(action.arguments.get("value") or "").lower())
            else:
                self.modalities_searched.update({"text", "image"})
        elif action.tool == "EXPAND_NEIGHBORS":
            self.neighbor_windows_tried.add(int(action.arguments["window"]))
            self.modalities_searched.update(
                record.memory.modality
                for event in incoming
                for record in event.records
                if record.memory.modality in {"text", "image"}
            )

    def _search_progress(self) -> dict[str, Any]:
        state = "unsearched" if self.discovery_attempt_count == 0 else (
            "stalled" if self.consecutive_no_gain_count >= 2 else "progressing"
        )
        return {
            "discovery_attempt_count": self.discovery_attempt_count,
            "distinct_discovery_count": len(self.discovery_signatures),
            "retrieval_methods_tried": sorted(self.retrieval_methods_tried),
            "rewritten_query_count": len(self.rewritten_queries),
            "metadata_fields_tried": sorted(self.metadata_fields_tried),
            "modalities_searched": sorted(self.modalities_searched),
            "neighbor_windows_tried": sorted(self.neighbor_windows_tried),
            "inspected_evidence_image_count": len(self.inspected_image_ids),
            "consecutive_no_gain_count": self.consecutive_no_gain_count,
            "state": state,
        }

    @staticmethod
    def _action_signature(action: ToolAction) -> str:
        return f"{action.tool}:{json.dumps(action.arguments, ensure_ascii=False, sort_keys=True, default=str)}"

    def _guardrail_reason(self, action: ToolAction) -> str:
        if action.tool == "EXPAND_NEIGHBORS" and not self.evidence:
            return "requires_current_evidence"
        if action.tool == "INSPECT_EVIDENCE_IMAGE" and not self._uninspected_image_records():
            return "requires_uninspected_evidence_image"
        if action.tool in {"SEARCH_METADATA", "RETRIEVE", "EXPAND_NEIGHBORS"}:
            revisions = self.no_gain_action_revisions.get(self._action_signature(action))
            if revisions == (self.pool_revision, self.evidence_revision):
                return "unchanged_no_gain_action"
        return ""

    def _record_blocked_action(self, action: ToolAction, pool_before: int, reason: str) -> None:
        self.blocked_action_count += 1
        self.last_action_status = "blocked"
        self.blocked_action_reason = reason
        if action.tool in {"SEARCH_METADATA", "RETRIEVE", "EXPAND_NEIGHBORS"}:
            # A blocked duplicate is objectively another consecutive step
            # without candidate/evidence gain.  It must not count as a new or
            # distinct search route, but it should allow search_progress to
            # report that the already explored frontier has stalled.
            self.consecutive_no_gain_count += 1
        self.trace.append(action)
        self.steps.append(
            ExecutionStep(
                index=len(self.steps),
                action=action,
                pool_before=pool_before,
                pool_after=len(self.pool),
                evidence_added=0,
                status="blocked",
                blocked_reason=reason,
                error="",
            )
        )

    def _selector_candidates(self) -> list[dict[str, Any]]:
        candidates = []
        for event in self.pool:
            public = self._event_evidence(event, allocate_evidence_id=False).to_public_dict()
            candidates.append({"candidate_id": self._candidate_id(event.event_key), **public})
        return candidates

    def _apply_semantic_selection(
        self,
        selected_candidate_ids: list[str],
        *,
        status: str,
        error: str,
    ) -> list[EventEvidence]:
        candidate_by_id = {self._candidate_id(item.event_key): item for item in self.pool}
        selected_ids = list(dict.fromkeys(str(value).strip() for value in selected_candidate_ids))
        unknown = [value for value in selected_ids if value not in candidate_by_id]
        if unknown:
            selected_ids = list(candidate_by_id)
            status = "fallback_all_invalid_ids"
            error = f"selector returned unknown candidate IDs: {unknown[:8]}"
        selected_pool = [candidate_by_id[value] for value in selected_ids]
        before_signature = self._evidence_signature()
        before_by_key = {item.event_key: item.to_public_dict() for item in self.evidence}
        self.evidence = [self._event_evidence(event) for event in selected_pool]
        self.semantic_filter_status = str(status or "ok")
        self.semantic_filter_error = str(error or "")
        self.semantic_filter_candidate_count = len(self.pool)
        self.semantic_filter_selected_count = len(selected_pool)
        changed = [
            item for item in self.evidence if before_by_key.get(item.event_key) != item.to_public_dict()
        ]
        if self._evidence_signature() != before_signature:
            self.evidence_revision += 1
        return changed

    def _evidence_pool(self) -> list[EventCandidate]:
        event_keys = {item.event_key for item in self.evidence}
        return [item for item in self.pool if item.event_key in event_keys]

    def _evidence_signature(self) -> tuple[Any, ...]:
        # Selector output order is not evidence gain.  Only a new event or a
        # content/inspection update should advance the public evidence state.
        return tuple(
            sorted(
                (
                    item.event_key,
                    json.dumps(item.to_public_dict(), ensure_ascii=False, sort_keys=True, default=str),
                )
                for item in self.evidence
            )
        )

    def _pool_signature(self) -> tuple[Any, ...]:
        return tuple((item.event_key, tuple(item.memory_ids), float(item.score)) for item in self.pool)

    def _candidate_id(self, event_key: str) -> str:
        if event_key not in self.candidate_ids:
            self.candidate_ids[event_key] = f"C{len(self.candidate_ids) + 1}"
        return self.candidate_ids[event_key]

    def _evidence_id(self, event_key: str) -> str:
        if event_key not in self.evidence_ids:
            self.evidence_ids[event_key] = f"E{len(self.evidence_ids) + 1}"
        return self.evidence_ids[event_key]

    def _event_evidence(
        self,
        event: EventCandidate,
        *,
        allocate_evidence_id: bool = True,
    ) -> EventEvidence:
        contents: list[str] = []
        modalities: set[str] = set()
        images: list[dict[str, Any]] = []
        image_memory_ids: list[str] = []
        timestamp = ""
        session_alias = ""
        turn_index: int | None = None
        for item in event.records:
            memory = item.memory
            modality = str(memory.modality or "").lower()
            if modality:
                modalities.add(modality)
            if not timestamp or memory.timestamp < timestamp:
                timestamp = memory.timestamp
            session_turn = self.executor._memory_session_turn_key(memory)
            if session_turn is not None:
                session_key, turn_index = session_turn
                if session_key not in self.session_aliases:
                    self.session_aliases[session_key] = f"S{len(self.session_aliases) + 1}"
                session_alias = self.session_aliases[session_key]
            content = self.executor._public_content(memory)
            if modality == "image":
                image_memory_ids.append(memory.memory_id)
                image: dict[str, Any] = {
                    "image_id": self._public_image_key(memory),
                    "visual_observation": self.visual_observations.get(memory.memory_id),
                }
                if content:
                    image["description"] = content
                images.append(image)
            elif content and content not in contents:
                contents.append(content)
        return EventEvidence(
            event_key=event.event_key,
            evidence_id=(self._evidence_id(event.event_key) if allocate_evidence_id else ""),
            session_alias=session_alias,
            turn_index=turn_index,
            timestamp=timestamp,
            modalities=sorted(modalities, key=lambda value: ({"text": 0, "image": 1}.get(value, 2), value)),
            content="\n".join(contents),
            images=images,
            member_memory_ids=event.memory_ids,
            image_memory_ids=image_memory_ids,
        )

    def _uninspected_image_records(self) -> list[PoolItem]:
        if self.raw_calls >= self.executor.max_raw_inspections:
            return []
        records: list[PoolItem] = []
        for event in self._evidence_pool():
            for item in event.records:
                if (
                    str(item.memory.modality).lower() == "image"
                    and item.memory.raw_pointer
                    and self._public_image_key(item.memory) not in self.inspected_image_ids
                ):
                    records.append(item)
        return records

    def _inspect_evidence_images_sync(self) -> list[EventEvidence]:
        if self.executor.raw_inspector is None:
            raise ValueError("INSPECT_EVIDENCE_IMAGE requires a configured inspector")
        before = self._evidence_signature()
        before_by_key = {item.event_key: item.to_public_dict() for item in self.evidence}
        remaining = self.executor.max_raw_inspections - self.raw_calls
        for item in self._uninspected_image_records()[:remaining]:
            event = next(event for event in self.evidence if item.memory.memory_id in event.image_memory_ids)
            result = self.executor.raw_inspector.inspect(
                item.memory.raw_pointer or "",
                self.query,
                question_image=self.question_image,
                text_context=event.content,
            )
            self.visual_observations[item.memory.memory_id] = str(result or "")
            self.inspected_image_memory_ids.add(item.memory.memory_id)
            self.inspected_image_ids.add(self._public_image_key(item.memory))
            self.raw_calls += 1
        self.evidence = [self._event_evidence(event) for event in self._evidence_pool()]
        if self._evidence_signature() != before:
            self.evidence_revision += 1
            return [
                item
                for item in self.evidence
                if before_by_key.get(item.event_key) != item.to_public_dict()
            ]
        return []

    async def execute_inspect_raw_with_teacher(self, action: ToolAction, inspect_fn: Any) -> dict[str, Any]:
        """Execute INSPECT_EVIDENCE_IMAGE using the async verl teacher service callback."""
        before = len(self.pool)
        if self.terminated:
            return self._observation(action, [], "trajectory already terminated")
        if len(self.trace) >= self.executor.validator.max_actions:
            self._terminate("budget_exhausted")
            return self._observation(action, [], "")
        step_error = ""
        before_signature = self._evidence_signature()
        before_by_key = {item.event_key: item.to_public_dict() for item in self.evidence}
        try:
            self.executor.validator._validate_action(action, len(self.trace))
            blocked_reason = self._guardrail_reason(action)
            if blocked_reason:
                self._record_blocked_action(action, before, blocked_reason)
                return self._finalize_action_observation(action, [], "")
            remaining = self.executor.max_raw_inspections - self.raw_calls
            for item in self._uninspected_image_records()[:remaining]:
                event = next(event for event in self.evidence if item.memory.memory_id in event.image_memory_ids)
                visual_observation = await inspect_fn(
                    {
                        "raw_pointer": item.memory.raw_pointer,
                        "query": self.query,
                        "question_image": self.question_image,
                        "text_context": event.content,
                    }
                )
                self.visual_observations[item.memory.memory_id] = str(visual_observation or "")
                self.inspected_image_memory_ids.add(item.memory.memory_id)
                self.inspected_image_ids.add(self._public_image_key(item.memory))
                self.raw_calls += 1
            self.evidence = [self._event_evidence(event) for event in self._evidence_pool()]
            if self._evidence_signature() != before_signature:
                self.evidence_revision += 1
        except Exception as exc:
            step_error = str(exc)
            self.error = step_error
            self._terminate("tool_error")

        self.trace.append(action)
        changed = [
            item
            for item in self.evidence
            if before_by_key.get(item.event_key) != item.to_public_dict()
        ]
        self.steps.append(
            ExecutionStep(
                index=len(self.steps),
                action=action,
                pool_before=before,
                pool_after=len(self.pool),
                evidence_added=len(changed),
                error=step_error,
            )
        )
        return self._finalize_action_observation(action, changed, step_error)

    def _public_image_key(self, memory: MemoryRecord) -> str:
        public_id = str(memory.public_image_id() or "").strip()
        if public_id:
            return public_id
        if memory.memory_id not in self.image_aliases:
            self.image_aliases[memory.memory_id] = f"IMG{len(self.image_aliases) + 1}"
        return self.image_aliases[memory.memory_id]

    def _terminate(self, reason: str) -> None:
        self.terminated = True
        self.termination_reason = reason
        self.policy_stopped = reason == "policy_stop"
        self.stopped = self.policy_stopped
        self.max_actions_reached = reason == "budget_exhausted"

    def _finalize_action_observation(
        self,
        action: ToolAction,
        changed_evidence: list[EventEvidence],
        error: str,
    ) -> dict[str, Any]:
        if len(self.trace) >= self.executor.validator.max_actions and not self.terminated:
            self._terminate("budget_exhausted")
        return self._observation(action, changed_evidence, error)

    def available_tool_names(self) -> list[str]:
        names = ["search_metadata", "retrieve"]
        if self.evidence:
            names.append("expand_neighbors")
        if self._uninspected_image_records():
            names.append("inspect_evidence_image")
        names.append("stop")
        return names

    def _observation(self, action: ToolAction, changed_evidence: list[EventEvidence], error: str) -> dict[str, Any]:
        """Return one authoritative accumulated event-level observation."""
        public_evidence = [item.to_public_dict() for item in self.evidence]
        record_count = sum(len(item.member_memory_ids) for item in self.evidence)
        observation = {
            "refresh_state": False,
            "tool": action.tool,
            "pool_count": len(self.pool),
            "pool_capacity": self.executor.max_pool_size,
            "pool_overflow_count": self.pool_overflow_count,
            "evidence_count": len(public_evidence),
            "evidence_event_count": len(public_evidence),
            "evidence_record_count": record_count,
            "evidence_memory_count": record_count,
            "new_evidence_count": len(changed_evidence),
            "evidence_revision": self.evidence_revision,
            "semantic_filter_status": self.semantic_filter_status,
            "semantic_filter_candidate_count": self.semantic_filter_candidate_count,
            "semantic_filter_selected_count": self.semantic_filter_selected_count,
            "semantic_filter_error": _clip_text(self.semantic_filter_error),
            "search_progress": self._search_progress(),
            "question_image_attached": bool(self.question_image),
            "available_tools": self.available_tool_names(),
            "last_action_status": self.last_action_status,
            "blocked_action_reason": self.blocked_action_reason,
            "blocked_action_count": self.blocked_action_count,
            "evidence": public_evidence,
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "policy_stopped": self.policy_stopped,
            "stopped": self.stopped,
            "max_actions_reached": self.max_actions_reached,
            "error": _clip_text(error),
        }
        return observation

    def public_state(self) -> dict[str, Any]:
        """Return serializable public state for AgentLoopOutput.extra_fields."""
        public_evidence = [item.to_public_dict() for item in self.evidence]
        record_count = sum(len(item.member_memory_ids) for item in self.evidence)
        return {
            "query": self.query,
            "pool_count": len(self.pool),
            "pool_capacity": self.executor.max_pool_size,
            "pool_overflow_count": self.pool_overflow_count,
            "evidence_count": len(public_evidence),
            "evidence_event_count": len(public_evidence),
            "evidence_record_count": record_count,
            "evidence_memory_count": record_count,
            "evidence": public_evidence,
            "evidence_revision": self.evidence_revision,
            "semantic_filter_status": self.semantic_filter_status,
            "semantic_filter_candidate_count": self.semantic_filter_candidate_count,
            "semantic_filter_selected_count": self.semantic_filter_selected_count,
            "semantic_filter_error": self.semantic_filter_error,
            "search_progress": self._search_progress(),
            "question_image_attached": bool(self.question_image),
            "available_tools": self.available_tool_names(),
            "last_action_status": self.last_action_status,
            "blocked_action_reason": self.blocked_action_reason,
            "blocked_action_count": self.blocked_action_count,
            "trace": [action.to_dict() for action in self.trace],
            "steps": [step.to_dict() for step in self.steps],
            "terminated": self.terminated,
            "termination_reason": self.termination_reason,
            "policy_stopped": self.policy_stopped,
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
        terminate_agent_loop = bool(observation["terminated"] or observation["error"])
        return ToolResponse(text=json.dumps(observation, ensure_ascii=False)), 0.0, {
            "opd_mm_pool_count": observation["pool_count"],
            "opd_mm_evidence_count": observation["evidence_count"],
            "opd_mm_evidence_event_count": observation["evidence_event_count"],
            "opd_mm_evidence_record_count": observation["evidence_record_count"],
            "opd_mm_blocked_action_count": observation["blocked_action_count"],
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
            "Metadata field. modality is only text/image; status is only active; timestamp uses public time values.",
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
            "YYYY-MM-DD, ISO timestamp, or exact timestamp shown in evidence. Never put a topic, entity, event, "
            "or memory ID here; use RETRIEVE for those.",
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
        "Use only after a discovery action has produced relevant evidence."
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
        terminate_agent_loop = bool(observation["terminated"] or observation["error"])
        return ToolResponse(text=json.dumps(observation, ensure_ascii=False)), 0.0, {
            "opd_mm_pool_count": observation["pool_count"],
            "opd_mm_evidence_count": observation["evidence_count"],
            "opd_mm_evidence_event_count": observation["evidence_event_count"],
            "opd_mm_evidence_record_count": observation["evidence_record_count"],
            "opd_mm_blocked_action_count": observation["blocked_action_count"],
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
    available_tool_names: Optional[list[str] | set[str]] = None,
) -> list[dict[str, Any]]:
    """Return OpenAI tool schemas for OPD-MM tools."""
    classes = (
        OPD_TOOL_CLASSES
        if include_inspect_raw
        else [cls for cls in OPD_TOOL_CLASSES if cls is not OPDInspectEvidenceImageTool]
    )
    allowed = set(available_tool_names) if available_tool_names is not None else None
    return [
        _schema(cls.tool_name, cls.description, cls.properties, cls.required).model_dump(
            exclude_unset=True, exclude_none=True
        )
        for cls in classes
        if allowed is None or cls.tool_name in allowed
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
