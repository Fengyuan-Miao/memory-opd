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

"""Strict executable action schema for OPD-MM policies."""

from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterable, List

from .models import ToolAction


ALLOWED_TOOLS = {
    "FILTER",
    "SORT",
    "TOPK",
    "RETRIEVE",
    "EXPAND_NEIGHBORS",
    "DROP",
    "INSPECT_RAW",
    "STOP",
}
DEFAULT_MAX_ACTIONS = 10
FILTER_FIELDS = {"modality", "source_type", "timestamp", "status"}
FILTER_OPS = {"eq", "neq", "before", "after", "contains"}
FILTER_SCOPES = {"current_pool", "full_memory"}
FILTER_VALUE_ENUMS = {
    "modality": {"image", "text"},
    "source_type": {"dialogue_image", "dialogue_turn"},
    "status": {"active"},
}
SORT_FIELDS = {"timestamp", "turn_id", "score"}
SORT_ORDERS = {"asc", "desc"}
RETRIEVAL_METHODS = {"bm25", "dense", "vision", "hybrid"}
EXPAND_NEIGHBOR_WINDOWS = {1, 2, 3}
INSPECT_TARGETS = {"current_pool"}
INSPECT_INSTRUCTIONS = {"answer_query_related_visual_details"}
FORBIDDEN_ARGUMENT_KEYS = {
    "memory_id",
    "memory_ids",
    "candidate_id",
    "candidate_ids",
    "search_query",
}
MEMORY_ID_PATTERN = re.compile(r"\b(?:m|memory|mau)[-_]?\d+\b", re.IGNORECASE)
EVIDENCE_ID_PATTERN = re.compile(r"^E[1-9]\d*$")

TOOL_SCHEMA_TEXT = """Allowed executable tools:
FILTER(field=modality|source_type|timestamp|status,
       op=eq|neq|before|after|contains, value=...,
       scope=current_pool|full_memory)
SORT(field=timestamp|turn_id|score, order=asc|desc)
TOPK(k=positive integer)
RETRIEVE(method=bm25|dense|vision|hybrid, top_k=positive integer,
         query=optional rewritten search text)
EXPAND_NEIGHBORS(window=1|2|3)
DROP(evidence_ids=[public evidence IDs])
STOP()

Return only a JSON array of tool calls. Do not emit memory IDs. RETRIEVE uses
the original user query by default; optionally provide query to rewrite the
search text for the current retrieval step. Every RETRIEVE searches the
original hidden memory store and merges deduplicated results into the current
candidate pool. For timestamp filters, date-only values
such as YYYY-MM-DD match all memory timestamps from that date. FILTER scope is
required: use full_memory for an independent metadata/date filter over the
original store and merge its results; use current_pool only for an intentional
intersection with the current candidates. Chaining unrelated current_pool filters can empty the pool.
For Mem-Gallery, source_type values are dialogue_turn and dialogue_image;
modality values are text and image; status value is active.
EXPAND_NEIGHBORS adds
nearby turns around the current candidate pool; use it only after a retrieve or
filter step has selected relevant candidates. DROP removes clearly irrelevant,
duplicate, or conflicting current evidence by its public evidence_id. Submit all
such IDs in one call, and do not call DROP again until evidence is added or enriched."""


def build_tool_schema(allow_inspect_raw: bool = True) -> str:
    lines = TOOL_SCHEMA_TEXT.splitlines()
    if allow_inspect_raw:
        stop_index = lines.index("STOP()")
        lines[stop_index:stop_index] = [
            "INSPECT_RAW(target=current_pool,",
            "            instruction=answer_query_related_visual_details)",
            "# INSPECT_RAW calls a visual inspector only for the current retrieved candidate pool;",
            "# it cannot inspect the user's attached question image, cannot inspect an empty pool,",
            "# and is not a search over the original full memory store. Use vision/hybrid RETRIEVE first.",
        ]
    return "\n".join(lines)


class TrajectoryValidationError(ValueError):
    pass


class TrajectoryValidator:
    def __init__(
        self,
        max_actions: int = DEFAULT_MAX_ACTIONS,
        max_top_k: int = 50,
        allow_inspect_raw: bool = True,
    ):
        self.max_actions = max(1, int(max_actions))
        self.max_top_k = max(1, int(max_top_k))
        self.allow_inspect_raw = bool(allow_inspect_raw)

    def schema_text(self) -> str:
        return build_tool_schema(self.allow_inspect_raw)

    def validate(self, values: Iterable[Dict[str, Any] | ToolAction]) -> List[ToolAction]:
        actions = [
            value if isinstance(value, ToolAction) else ToolAction.from_dict(value)
            for value in values
        ]
        if not actions:
            raise TrajectoryValidationError("trajectory is empty")
        if len(actions) > self.max_actions:
            raise TrajectoryValidationError(
                f"trajectory has {len(actions)} actions; maximum is {self.max_actions}"
            )

        validated = []
        for index, action in enumerate(actions):
            self._validate_action(action, index)
            if action.tool == "STOP" and index != len(actions) - 1:
                raise TrajectoryValidationError("STOP must be the final action")
            validated.append(action)
        if validated[-1].tool != "STOP":
            if len(validated) >= self.max_actions:
                raise TrajectoryValidationError("trajectory must end with STOP")
            validated.append(ToolAction("STOP"))
        return validated

    def _validate_action(self, action: ToolAction, index: int) -> None:
        if action.tool not in ALLOWED_TOOLS:
            raise TrajectoryValidationError(
                f"action {index}: unsupported tool {action.tool!r}"
            )
        if action.tool == "INSPECT_RAW" and not self.allow_inspect_raw:
            raise TrajectoryValidationError(
                f"action {index}: INSPECT_RAW is unavailable in this run"
            )
        forbidden = FORBIDDEN_ARGUMENT_KEYS & set(action.arguments)
        if forbidden:
            raise TrajectoryValidationError(
                f"action {index}: forbidden arguments {sorted(forbidden)}"
            )
        for value in action.arguments.values():
            if MEMORY_ID_PATTERN.search(json.dumps(value, ensure_ascii=False)):
                raise TrajectoryValidationError(
                    f"action {index}: memory IDs are not allowed"
                )

        validator = getattr(self, f"_validate_{action.tool.lower()}")
        validator(action.arguments, index)

    @staticmethod
    def _require_exact_keys(
        arguments: Dict[str, Any],
        required: set[str],
        optional: set[str],
        index: int,
    ) -> None:
        missing = required - set(arguments)
        unknown = set(arguments) - required - optional
        if missing:
            raise TrajectoryValidationError(
                f"action {index}: missing arguments {sorted(missing)}"
            )
        if unknown:
            raise TrajectoryValidationError(
                f"action {index}: unknown arguments {sorted(unknown)}"
            )

    def _validate_filter(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, {"field", "op", "value", "scope"}, set(), index)
        if args["field"] not in FILTER_FIELDS:
            raise TrajectoryValidationError(f"action {index}: invalid FILTER field")
        if args["op"] not in FILTER_OPS:
            raise TrajectoryValidationError(f"action {index}: invalid FILTER op")
        if not isinstance(args["value"], (str, int, float, bool)):
            raise TrajectoryValidationError(f"action {index}: invalid FILTER value")
        allowed_values = FILTER_VALUE_ENUMS.get(args["field"])
        if allowed_values is not None and args["value"] not in allowed_values:
            raise TrajectoryValidationError(
                f"action {index}: invalid FILTER value for {args['field']}; "
                f"expected one of {sorted(allowed_values)}"
            )
        if args["field"] == "timestamp" and (not isinstance(args["value"], str) or not args["value"].strip()):
            raise TrajectoryValidationError(f"action {index}: invalid FILTER timestamp value")
        if args["scope"] not in FILTER_SCOPES:
            raise TrajectoryValidationError(f"action {index}: invalid FILTER scope")

    def _validate_sort(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, {"field", "order"}, set(), index)
        if args["field"] not in SORT_FIELDS or args["order"] not in SORT_ORDERS:
            raise TrajectoryValidationError(f"action {index}: invalid SORT arguments")

    def _validate_topk(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, {"k"}, set(), index)
        self._validate_k(args["k"], index, "k")

    def _validate_retrieve(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, set(), {"method", "top_k", "query"}, index)
        if args.get("method", "hybrid") not in RETRIEVAL_METHODS:
            raise TrajectoryValidationError(f"action {index}: invalid RETRIEVE method")
        self._validate_k(args.get("top_k", 5), index, "top_k")
        if "query" in args and (not isinstance(args["query"], str) or not args["query"].strip()):
            raise TrajectoryValidationError(f"action {index}: invalid RETRIEVE query")

    def _validate_expand_neighbors(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, {"window"}, set(), index)
        window = args["window"]
        if not isinstance(window, int) or isinstance(window, bool) or window not in EXPAND_NEIGHBOR_WINDOWS:
            raise TrajectoryValidationError(f"action {index}: window must be one of {sorted(EXPAND_NEIGHBOR_WINDOWS)}")

    def _validate_drop(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, {"evidence_ids"}, set(), index)
        evidence_ids = args["evidence_ids"]
        if not isinstance(evidence_ids, list) or not evidence_ids:
            raise TrajectoryValidationError(f"action {index}: evidence_ids must be a non-empty list")
        if len(evidence_ids) != len(set(evidence_ids)):
            raise TrajectoryValidationError(f"action {index}: evidence_ids must be unique")
        if any(
            not isinstance(value, str) or EVIDENCE_ID_PATTERN.fullmatch(value.strip()) is None
            for value in evidence_ids
        ):
            raise TrajectoryValidationError(f"action {index}: invalid public evidence_id")

    def _validate_inspect_raw(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, set(), {"target", "instruction"}, index)
        if args.get("target", "current_pool") not in INSPECT_TARGETS:
            raise TrajectoryValidationError(f"action {index}: invalid INSPECT_RAW target")
        if (
            args.get("instruction", "answer_query_related_visual_details")
            not in INSPECT_INSTRUCTIONS
        ):
            raise TrajectoryValidationError(
                f"action {index}: invalid INSPECT_RAW instruction"
            )

    def _validate_stop(self, args: Dict[str, Any], index: int) -> None:
        self._require_exact_keys(args, set(), set(), index)

    def _validate_k(self, value: Any, index: int, name: str) -> None:
        if not isinstance(value, int) or isinstance(value, bool):
            raise TrajectoryValidationError(f"action {index}: {name} must be an integer")
        if value <= 0 or value > self.max_top_k:
            raise TrajectoryValidationError(
                f"action {index}: {name} must be between 1 and {self.max_top_k}"
            )
