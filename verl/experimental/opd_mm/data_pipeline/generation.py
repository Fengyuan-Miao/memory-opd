"""Model-backed generation stages for the MMem construction pipeline.

This module creates a canonical episode artifact from a small generation
request.  Hidden plans are persisted only under the construction work
directory.  The returned artifact contains exclusively finalized dialogue,
verified image observations, extracted facts, and QA candidates that pass the
deterministic Base-v0 acceptance pipeline.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable

from .image_generation import DEFAULT_IMAGE_MODEL, GPTImageClient, resize_image_bytes
from .multimodal_generation import MultimodalResponsesClient
from .pipeline import build_episode
from .prompts import (
    ASSISTANT_SIMULATOR_CONTRACT,
    DIALOGUE_LEAKAGE_CHECKER_CONTRACT,
    EPISODE_PLANNER_CONTRACT,
    EVENT_GRAPH_CONTRACT,
    FULL_STATE_AUDITOR_CONTRACT,
    IMAGE_CONTRACT_GENERATOR_CONTRACT,
    IMAGE_CONTRACT_REPAIR_CONTRACT,
    IMAGE_PROMPT_COMPILER_CONTRACT,
    QA_JUDGE_CONTRACT,
    QA_REALIZER_CONTRACT,
    QA_SPEC_GENERATOR_CONTRACT,
    STATE_EXTRACTOR_CONTRACT,
    USER_SIMULATOR_CONTRACT,
    VISUAL_VERIFIER_CONTRACT,
)


DEFAULT_TASK_RATIOS = {
    "FR": 0.128,
    "VS": 0.179,
    "TTL": 0.197,
    "TR": 0.072,
    "VR": 0.102,
    "MR": 0.120,
    "KR": 0.047,
    "CD": 0.047,
    "AR": 0.108,
}


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(_json(value) + "\n", encoding="utf-8")
    temporary.replace(path)


def _nonempty(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _string_list(value: Any) -> list[str]:
    return [str(item) for item in _list(value) if _nonempty(str(item))]


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_filename(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _task_quota(total: int, ratios: dict[str, float]) -> dict[str, int]:
    if total < 1:
        raise ValueError("qa_count must be positive")
    names = list(DEFAULT_TASK_RATIOS)
    weights = {name: max(float(ratios.get(name, 0.0)), 0.0) for name in names}
    denominator = sum(weights.values())
    if denominator <= 0:
        raise ValueError("task ratios must have a positive sum")
    exact = {name: total * weights[name] / denominator for name in names}
    quota = {name: int(exact[name]) for name in names}
    remaining = total - sum(quota.values())
    for name in sorted(names, key=lambda item: exact[item] - quota[item], reverse=True)[:remaining]:
        quota[name] += 1
    return {name: count for name, count in quota.items() if count}


@dataclass(frozen=True)
class GenerationRequest:
    episode_id: str
    dataset: str = "mmem_v2"
    language: str = "en"
    session_count: int = 4
    time_span_days: int = 30
    rounds_per_session_min: int = 4
    rounds_per_session_max: int = 7
    images_per_session_min: int = 0
    images_per_session_max: int = 2
    qa_count: int = 12
    task_ratios: dict[str, float] = field(default_factory=lambda: dict(DEFAULT_TASK_RATIOS))
    scenario_constraints: list[str] = field(default_factory=list)
    existing_cluster_summaries: list[str] = field(default_factory=list)
    seed: int = 20260901

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "GenerationRequest":
        if not isinstance(value, dict):
            raise ValueError("generation request must be an object")
        episode_id = str(value.get("episode_id") or "").strip()
        if not episode_id:
            raise ValueError("generation request requires episode_id")
        if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", episode_id) is None:
            raise ValueError("episode_id may contain only letters, digits, underscores, and hyphens")
        request = cls(
            episode_id=episode_id,
            dataset=str(value.get("dataset") or "mmem_v2"),
            language=str(value.get("language") or "en"),
            session_count=int(value.get("session_count", 4)),
            time_span_days=int(value.get("time_span_days", 30)),
            rounds_per_session_min=int(value.get("rounds_per_session_min", 4)),
            rounds_per_session_max=int(value.get("rounds_per_session_max", 7)),
            images_per_session_min=int(value.get("images_per_session_min", 0)),
            images_per_session_max=int(value.get("images_per_session_max", 2)),
            qa_count=int(value.get("qa_count", 12)),
            task_ratios={
                str(key).upper(): float(item)
                for key, item in dict(value.get("task_ratios") or DEFAULT_TASK_RATIOS).items()
            },
            scenario_constraints=_string_list(value.get("scenario_constraints")),
            existing_cluster_summaries=_string_list(value.get("existing_cluster_summaries")),
            seed=int(value.get("seed", 20260901)),
        )
        if request.session_count < 1:
            raise ValueError("session_count must be positive")
        if request.rounds_per_session_min < 1 or request.rounds_per_session_max < request.rounds_per_session_min:
            raise ValueError("invalid rounds_per_session range")
        if request.images_per_session_min < 0 or request.images_per_session_max < request.images_per_session_min:
            raise ValueError("invalid images_per_session range")
        if request.time_span_days < 1 or request.qa_count < 1:
            raise ValueError("time_span_days and qa_count must be positive")
        unknown_tasks = sorted(set(request.task_ratios) - set(DEFAULT_TASK_RATIOS))
        if unknown_tasks:
            raise ValueError(f"unknown task ratio keys: {unknown_tasks}")
        visual_weight = sum(request.task_ratios.get(task, 0.0) for task in ("VS", "TTL", "VR"))
        if visual_weight > 0 and request.images_per_session_max == 0:
            raise ValueError("VS/TTL/VR task ratios require images_per_session_max > 0")
        return request

    def planner_input(self) -> dict[str, Any]:
        return {
            **asdict(self),
            "task_hook_quota": _task_quota(self.qa_count, self.task_ratios),
        }


@dataclass(frozen=True)
class GenerationConfig:
    stage_retries: int = 2
    image_candidates: int = 1
    image_repair_rounds: int = 1
    qa_paraphrase_candidates: int = 2
    run_full_state_audit: bool = False
    overwrite_images: bool = False
    resume: bool = False


@dataclass(frozen=True)
class GeneratedEpisode:
    artifact: dict[str, Any]
    artifact_path: Path
    work_dir: Path
    accepted_qa_count: int
    rejected_qa_count: int


class GenerationStageError(RuntimeError):
    pass


class MMemGenerationPipeline:
    """Orchestrate isolated Terra stages, GPT Image generation, and hard acceptance."""

    def __init__(
        self,
        *,
        multimodal_client: MultimodalResponsesClient,
        image_client: GPTImageClient,
        config: GenerationConfig | None = None,
    ) -> None:
        self.multimodal = multimodal_client
        self.image_client = image_client
        self.config = config or GenerationConfig()
        self._stage_root = Path(".")
        self._stage_counts: dict[str, int] = {}

    def _record_stage(self, stage: str, attempt: int, prompt: dict[str, Any], output: dict[str, Any]) -> None:
        count = self._stage_counts.get(stage, 0)
        self._stage_counts[stage] = count + 1
        root = self._stage_root / "model_stages" / stage
        _write_json(root / f"{count:04d}_attempt{attempt}_input.json", prompt)
        _write_json(root / f"{count:04d}_attempt{attempt}_output.json", output)

    def _generate_checked(
        self,
        stage: str,
        contract: str,
        payload: dict[str, Any],
        validator: Callable[[dict[str, Any]], list[str]],
        *,
        image_paths: Iterable[str | Path] = (),
    ) -> dict[str, Any]:
        current = dict(payload)
        last_errors: list[str] = []
        for attempt in range(self.config.stage_retries + 1):
            value, _ = self.multimodal.generate_json(
                task_contract=contract,
                prompt=_json(current),
                image_paths=image_paths,
                max_output_tokens=8192,
            )
            self._record_stage(stage, attempt, current, value)
            errors = validator(value)
            if not errors:
                return value
            last_errors = errors
            current = {
                **payload,
                "repair_request": {
                    "validation_errors": errors,
                    "previous_output": value,
                    "instruction": "Return a corrected complete object; do not explain the repair.",
                },
            }
        raise GenerationStageError(f"{stage} failed validation: {last_errors}")

    @staticmethod
    def _validate_blueprint(value: dict[str, Any], request: GenerationRequest) -> list[str]:
        errors = []
        if value.get("episode_id") != request.episode_id:
            errors.append("episode_id must exactly match the request")
        sessions = _list(value.get("session_plan"))
        if len(sessions) != request.session_count:
            errors.append(f"session_plan must contain exactly {request.session_count} sessions")
        ids = [str(item.get("session_id") or "") for item in sessions if isinstance(item, dict)]
        if len(ids) != len(set(ids)) or any(not item for item in ids):
            errors.append("session IDs must be non-empty and unique")
        for session in sessions:
            if not isinstance(session, dict):
                errors.append("every session_plan item must be an object")
                continue
            rounds = _as_int(session.get("target_rounds"))
            images = _as_int(session.get("target_image_count"), -1)
            if not request.rounds_per_session_min <= rounds <= request.rounds_per_session_max:
                errors.append(f"target_rounds for {session.get('session_id')} is outside the request range")
            if not request.images_per_session_min <= images <= request.images_per_session_max:
                errors.append(f"target_image_count for {session.get('session_id')} is outside the request range")
            if not _nonempty(session.get("date")):
                errors.append(f"session {session.get('session_id')} has no date")
        if not isinstance(value.get("persona"), dict):
            errors.append("persona must be an object")
        return errors

    @staticmethod
    def _validate_event_graph(value: dict[str, Any], blueprint: dict[str, Any]) -> list[str]:
        errors = []
        sessions = {str(item["session_id"]): item for item in _list(blueprint.get("session_plan"))}
        events = _list(value.get("events"))
        ids = [str(item.get("event_id") or "") for item in events if isinstance(item, dict)]
        if len(ids) != len(set(ids)) or any(not item for item in ids):
            errors.append("event IDs must be non-empty and unique")
        for session_id, session in sessions.items():
            rows = [item for item in events if isinstance(item, dict) and item.get("session_id") == session_id]
            if len(rows) != _as_int(session.get("target_rounds")):
                errors.append(f"{session_id} must have exactly {session.get('target_rounds')} events")
            turns = sorted(_as_int(item.get("turn_index")) for item in rows)
            if turns != list(range(1, len(rows) + 1)):
                errors.append(f"{session_id} turn_index values must be contiguous from 1")
        facts = _list(value.get("planned_facts"))
        fact_ids = [str(item.get("fact_id") or "") for item in facts if isinstance(item, dict)]
        if len(fact_ids) != len(set(fact_ids)) or any(not item for item in fact_ids):
            errors.append("planned fact IDs must be non-empty and unique")
        known_facts = set(fact_ids)
        assigned_facts: list[str] = []
        for event in events:
            if not isinstance(event, dict):
                continue
            assigned_facts.extend(_string_list(event.get("fact_ids_to_express")))
        unknown_assignments = sorted(set(assigned_facts) - known_facts)
        repeated_assignments = sorted({item for item in assigned_facts if assigned_facts.count(item) > 1})
        if unknown_assignments:
            errors.append(f"events reference unknown planned facts: {unknown_assignments}")
        if repeated_assignments:
            errors.append(f"planned facts must be expressed in one event only: {repeated_assignments}")
        for session_id in sessions:
            rows = sorted(
                [item for item in events if isinstance(item, dict) and item.get("session_id") == session_id],
                key=lambda item: _as_int(item.get("turn_index")),
            )
            for left, right in zip(rows, rows[1:]):
                left_items = _string_list(left.get("allowed_user_information"))
                right_items = _string_list(right.get("allowed_user_information"))
                for first in left_items:
                    first_tokens = set(re.findall(r"[a-z0-9]+", first.casefold()))
                    for second in right_items:
                        second_tokens = set(re.findall(r"[a-z0-9]+", second.casefold()))
                        union = first_tokens | second_tokens
                        if len(union) >= 5 and len(first_tokens & second_tokens) / len(union) >= 0.8:
                            errors.append(f"{session_id} adjacent events repeat the same allowed information")
                            break
                    else:
                        continue
                    break
        image_needs = _list(value.get("image_needs"))
        image_ids = [str(item.get("image_id") or "") for item in image_needs if isinstance(item, dict)]
        if len(image_ids) != len(set(image_ids)) or any(not item for item in image_ids):
            errors.append("image IDs must be non-empty and unique")
        known_events = set(ids)
        known_images = set(image_ids)
        for hook in _list(value.get("task_hooks")):
            if not isinstance(hook, dict):
                errors.append("every task hook must be an object")
                continue
            unknown_hook_facts = sorted(set(_string_list(hook.get("target_fact_ids"))) - known_facts)
            unknown_hook_events = sorted(set(_string_list(hook.get("target_event_ids"))) - known_events)
            unknown_hook_images = sorted(set(_string_list(hook.get("target_image_ids"))) - known_images)
            if unknown_hook_facts or unknown_hook_events or unknown_hook_images:
                errors.append(
                    f"hook {hook.get('hook_id')} has unknown targets: "
                    f"facts={unknown_hook_facts}, events={unknown_hook_events}, images={unknown_hook_images}"
                )
        return errors

    @staticmethod
    def _normalize_image_ids(graph: dict[str, Any], episode_id: str) -> dict[str, Any]:
        """Replace model-authored semantic image names with deterministic opaque IDs."""

        image_needs = _list(graph.get("image_needs"))
        mapping = {
            str(item.get("image_id")): (
                "IMG_"
                + hashlib.blake2s(
                    f"{episode_id}:image:{index}".encode("utf-8"),
                    digest_size=5,
                ).hexdigest()
            )
            for index, item in enumerate(image_needs)
            if isinstance(item, dict) and item.get("image_id")
        }

        def replace(value: Any) -> Any:
            if isinstance(value, str):
                return mapping.get(value, value)
            if isinstance(value, list):
                return [replace(item) for item in value]
            if isinstance(value, dict):
                return {key: replace(item) for key, item in value.items()}
            return value

        normalized = replace(graph)
        return dict(normalized) if isinstance(normalized, dict) else graph

    @staticmethod
    def _validate_image_contracts(value: dict[str, Any], graph: dict[str, Any]) -> list[str]:
        contracts = _list(value.get("image_contracts"))
        ordered_expected = [
            str(item.get("image_id"))
            for item in _list(graph.get("image_needs"))
            if isinstance(item, dict)
        ]
        expected = set(ordered_expected)
        image_position = {image_id: index for index, image_id in enumerate(ordered_expected)}
        actual = {str(item.get("image_id")) for item in contracts if isinstance(item, dict)}
        errors = []
        if actual != expected:
            errors.append(
                f"image contracts must cover exactly image_needs; "
                f"expected={sorted(expected)}, actual={sorted(actual)}"
            )
        for contract in contracts:
            if not isinstance(contract, dict):
                errors.append("every image contract must be an object")
                continue
            if not _list(contract.get("must_be_visible")):
                errors.append(f"{contract.get('image_id')} needs at least one must_be_visible fact")
            critical = _list(contract.get("task_critical_constraints"))
            visible_signatures = {
                _json(item) for item in _list(contract.get("must_be_visible")) if isinstance(item, dict)
            }
            if not critical:
                errors.append(f"{contract.get('image_id')} needs task_critical_constraints")
            elif len(critical) > 3:
                errors.append(f"{contract.get('image_id')} has more than three task-critical constraints")
            elif any(not isinstance(item, dict) or _json(item) not in visible_signatures for item in critical):
                errors.append(f"{contract.get('image_id')} task-critical constraints must be exact visible constraints")
            if contract.get("role") not in {"memory", "query", "both"}:
                errors.append(f"{contract.get('image_id')} has invalid role")
            image_id = str(contract.get("image_id") or "")
            for reference_id in _string_list(contract.get("reference_image_ids")):
                if reference_id not in expected:
                    errors.append(f"{image_id} references unknown image {reference_id}")
                elif image_position[reference_id] >= image_position.get(image_id, -1):
                    errors.append(f"{image_id} reference {reference_id} must be an earlier image")
        return errors

    @staticmethod
    def _validate_repaired_contract(value: dict[str, Any], original: dict[str, Any]) -> list[str]:
        repaired = value.get("image_contract")
        if not isinstance(repaired, dict):
            return ["image_contract must be an object"]
        errors = []
        for key in ("image_id", "role", "event_id"):
            if repaired.get(key) != original.get(key):
                errors.append(f"repaired contract must preserve {key}")
        for key in ("entities", "qa_hook_ids", "reference_image_ids"):
            if set(_string_list(repaired.get(key))) != set(_string_list(original.get(key))):
                errors.append(f"repaired contract must preserve {key}")
        original_critical = {
            _json(item) for item in _list(original.get("task_critical_constraints")) if isinstance(item, dict)
        }
        repaired_critical = {
            _json(item) for item in _list(repaired.get("task_critical_constraints")) if isinstance(item, dict)
        }
        repaired_visible = {
            _json(item) for item in _list(repaired.get("must_be_visible")) if isinstance(item, dict)
        }
        if repaired_critical != original_critical or not repaired_critical <= repaired_visible:
            errors.append("repaired contract must preserve every task-critical constraint as visible")
        if not _list(repaired.get("must_be_visible")):
            errors.append("repaired contract requires at least one must_be_visible fact")
        if len(_list(repaired.get("must_be_visible"))) > 4:
            errors.append("repaired contract must contain at most four must_be_visible facts")
        if not isinstance(repaired.get("information_partition"), dict):
            errors.append("repaired contract must preserve an information_partition")
        return errors

    @staticmethod
    def _validate_prompt(value: dict[str, Any]) -> list[str]:
        errors = []
        if not _nonempty(value.get("prompt")):
            errors.append("prompt is empty")
        if not _nonempty(value.get("public_retrieval_description")):
            errors.append("public_retrieval_description is empty")
        return errors

    @staticmethod
    def _validate_visual_shape(value: dict[str, Any]) -> list[str]:
        errors = []
        if not isinstance(value.get("hard_gate_passed"), bool):
            errors.append("hard_gate_passed must be boolean")
        if value.get("hard_gate_passed") is True and not _list(value.get("verified_visual_facts")):
            errors.append("a passing verdict requires verified visual facts")
        if value.get("hard_gate_passed") is True and not _nonempty(value.get("public_retrieval_description")):
            errors.append("visual verifier returned no public description")
        return errors

    @classmethod
    def _validate_visual_contract(cls, value: dict[str, Any], contract: dict[str, Any]) -> list[str]:
        errors = cls._validate_visual_shape(value)
        if value.get("hard_gate_passed") is not True:
            errors.append(f"visual hard gate failed: {value.get('failure_reasons')}")
        description = str(value.get("public_retrieval_description") or "").casefold()
        raw_partition = contract.get("information_partition")
        partition = dict(raw_partition) if isinstance(raw_partition, dict) else {}
        visual_only = {str(item).casefold() for item in _list(partition.get("visual_only"))}
        forbidden_values = []
        for fact in _list(contract.get("must_be_visible")):
            if not isinstance(fact, dict):
                continue
            predicate = str(fact.get("predicate") or "").casefold()
            if predicate in visual_only or f"{fact.get('subject')}.{predicate}".casefold() in visual_only:
                forbidden_values.append(str(fact.get("object") or "").strip())
        raw_public_contract = contract.get("public_representation_contract")
        public_contract = dict(raw_public_contract) if isinstance(raw_public_contract, dict) else {}
        forbidden_values.extend(_string_list(public_contract.get("forbidden_in_retrieval_description")))
        leaked = [item for item in forbidden_values if len(item) >= 3 and item.casefold() in description]
        if leaked:
            errors.append(f"public retrieval description leaks visual-only values: {leaked}")
        return errors

    def _plan(self, request: GenerationRequest) -> tuple[dict[str, Any], dict[str, Any]]:
        blueprint_path = self._stage_root / "private_plan" / "episode_blueprint.json"
        graph_path = self._stage_root / "private_plan" / "event_graph.json"
        if self.config.resume and blueprint_path.is_file() and graph_path.is_file():
            blueprint = json.loads(blueprint_path.read_text(encoding="utf-8"))
            graph = json.loads(graph_path.read_text(encoding="utf-8"))
            errors = [
                *self._validate_blueprint(blueprint, request),
                *self._validate_event_graph(graph, blueprint),
            ]
            if errors:
                raise GenerationStageError(f"saved private plan is incompatible with request: {errors}")
            return blueprint, graph
        planner_input = request.planner_input()
        blueprint = self._generate_checked(
            "episode_planner",
            EPISODE_PLANNER_CONTRACT,
            planner_input,
            lambda value: self._validate_blueprint(value, request),
        )
        graph = self._generate_checked(
            "event_graph",
            EVENT_GRAPH_CONTRACT,
            {"blueprint": blueprint, "task_hook_quota": planner_input["task_hook_quota"]},
            lambda value: self._validate_event_graph(value, blueprint),
        )
        graph = self._normalize_image_ids(graph, request.episode_id)
        normalized_errors = self._validate_event_graph(graph, blueprint)
        if normalized_errors:
            raise GenerationStageError(f"opaque image ID normalization failed: {normalized_errors}")
        _write_json(self._stage_root / "private_plan" / "episode_blueprint.json", blueprint)
        _write_json(self._stage_root / "private_plan" / "event_graph.json", graph)
        return blueprint, graph

    def _image_contracts(self, blueprint: dict[str, Any], graph: dict[str, Any]) -> list[dict[str, Any]]:
        if not _list(graph.get("image_needs")):
            return []
        contract_path = self._stage_root / "private_plan" / "image_contracts.json"
        if self.config.resume and contract_path.is_file():
            contracts = json.loads(contract_path.read_text(encoding="utf-8"))
            wrapped = {"image_contracts": contracts}
            errors = self._validate_image_contracts(wrapped, graph)
            if errors:
                raise GenerationStageError(f"saved image contracts are incompatible with event graph: {errors}")
            return [dict(item) for item in contracts]
        result = self._generate_checked(
            "image_contracts",
            IMAGE_CONTRACT_GENERATOR_CONTRACT,
            {
                "scenario": blueprint.get("scenario"),
                "persona": blueprint.get("persona"),
                "recurring_entities": blueprint.get("recurring_entities"),
                "events": graph.get("events"),
                "task_hooks": graph.get("task_hooks"),
                "image_needs": graph.get("image_needs"),
            },
            lambda value: self._validate_image_contracts(value, graph),
        )
        contracts = [dict(item) for item in result["image_contracts"]]
        _write_json(self._stage_root / "private_plan" / "image_contracts.json", contracts)
        return contracts

    def _generate_images(self, contracts: list[dict[str, Any]]) -> list[dict[str, Any]]:
        cached_path = self._stage_root / "observed" / "verified_images.json"
        if self.config.resume and cached_path.is_file():
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            if isinstance(cached, list) and all(
                isinstance(item, dict) and Path(str(item.get("path") or "")).is_file()
                for item in cached
            ):
                return [dict(item) for item in cached]
        images = []
        image_manifests = []
        image_root = self._stage_root / "images"
        image_root.mkdir(parents=True, exist_ok=True)
        for contract in contracts:
            image_id = str(contract["image_id"])
            safe_image_id = _safe_filename(image_id)
            selected: tuple[Path, dict[str, Any], dict[str, Any]] | None = None
            active_contract = dict(contract)
            for repair_round in range(self.config.image_repair_rounds + 1):
                explicit_reference_ids = set(_string_list(active_contract.get("reference_image_ids")))
                active_entities = set(_string_list(active_contract.get("entities")))
                references = []
                for previous_image in images:
                    previous_id = str(previous_image.get("image_id") or "")
                    previous_contract = next(
                        (item for item in contracts if str(item.get("image_id") or "") == previous_id),
                        {},
                    )
                    shares_identity = bool(
                        active_entities & set(_string_list(previous_contract.get("entities")))
                    )
                    if previous_id in explicit_reference_ids or (not explicit_reference_ids and shares_identity):
                        references.append(previous_image)
                prompt_spec = self._generate_checked(
                    "image_prompt_compiler",
                    IMAGE_PROMPT_COMPILER_CONTRACT,
                    {
                        "image_contract": active_contract,
                        "generation_mode": "image_edit" if references else "text_to_image",
                        "reference_images": [self._visible_image_state(item) for item in references],
                    },
                    self._validate_prompt,
                )
                negative = str(prompt_spec.get("negative_prompt") or "").strip()
                image_prompt = str(prompt_spec["prompt"])
                if negative:
                    image_prompt += f"\nAvoid: {negative}"
                round_failures = []
                # The runtime budget is authoritative so a generated contract
                # cannot reduce fault tolerance to one candidate.
                for candidate_index in range(max(self.config.image_candidates, 1)):
                    filename = f"{safe_image_id}.r{repair_round:02d}.candidate_{candidate_index:02d}.png"
                    path = image_root / filename
                    if self.config.overwrite_images or not path.is_file():
                        if references:
                            data = self.image_client.edit(
                                image_prompt,
                                reference_paths=[item["path"] for item in references],
                            )
                        else:
                            data = self.image_client.generate(image_prompt)
                        data = resize_image_bytes(data)
                        path.write_bytes(data)
                    image_inputs = [
                        {"position": 1, "role": "candidate", "image_id": image_id},
                        *[
                            {
                                "position": position,
                                "role": "earlier_reference",
                                "image_id": str(item["image_id"]),
                            }
                            for position, item in enumerate(references, 2)
                        ],
                    ]
                    verified = self._generate_checked(
                        "visual_verifier",
                        VISUAL_VERIFIER_CONTRACT,
                        {
                            "image_contract": active_contract,
                            "image_inputs": image_inputs,
                            "repair_round": repair_round,
                            "candidate_index": candidate_index,
                            "proposed_public_description": prompt_spec.get("public_retrieval_description"),
                        },
                        self._validate_visual_shape,
                        image_paths=[path, *(item["path"] for item in references)],
                    )
                    visual_errors = self._validate_visual_contract(verified, active_contract)
                    if visual_errors:
                        rejection = {
                            "image_id": image_id,
                            "repair_round": repair_round,
                            "candidate_index": candidate_index,
                            "errors": visual_errors,
                        }
                        round_failures.append(rejection)
                        report_name = f"{safe_image_id}_r{repair_round:02d}_{candidate_index:02d}.json"
                        _write_json(self._stage_root / "reports" / "image_rejections" / report_name, rejection)
                        continue
                    selected = (
                        path,
                        verified,
                        {
                            "generator_model": DEFAULT_IMAGE_MODEL,
                            "generation_mode": "image_edit" if references else "text_to_image",
                            "reference_image_ids": [str(item["image_id"]) for item in references],
                            "repair_round": repair_round,
                            "candidate_index": candidate_index,
                        },
                    )
                    break
                if selected is not None:
                    break
                if repair_round >= self.config.image_repair_rounds:
                    continue
                repaired = self._generate_checked(
                    "image_contract_repair",
                    IMAGE_CONTRACT_REPAIR_CONTRACT,
                    {"image_contract": active_contract, "candidate_failures": round_failures},
                    lambda value: self._validate_repaired_contract(value, active_contract),
                )
                active_contract = dict(repaired["image_contract"])
                contract.clear()
                contract.update(active_contract)
                _write_json(self._stage_root / "private_plan" / "image_contracts.json", contracts)
            if selected is None:
                raise GenerationStageError(f"no image candidate passed visual verification for {image_id}")
            candidate_path, verified, lineage = selected
            final_path = image_root / f"{safe_image_id}.png"
            if candidate_path.resolve() != final_path.resolve():
                shutil.copy2(candidate_path, final_path)
            visual_facts = []
            verifier_checks = [
                item for item in _list(verified.get("checks")) if isinstance(item, dict)
            ]
            for index, fact in enumerate(_list(contract.get("must_be_visible")), 1):
                if not isinstance(fact, dict) or not _nonempty(fact.get("predicate")):
                    continue
                matching_check = next(
                    (
                        item
                        for item in verifier_checks
                        if str(item.get("subject") or "") == str(fact.get("subject") or "")
                        and str(item.get("predicate") or "") == str(fact.get("predicate") or "")
                        and item.get("passed") is True
                    ),
                    None,
                )
                visual_facts.append(
                    {
                        "visual_fact_id": f"VF_{image_id}_{index:02d}",
                        "subject": str(fact.get("subject") or ""),
                        "predicate": str(fact["predicate"]),
                        "value": fact.get("object"),
                        "confidence": float(
                            (matching_check or {}).get("confidence", 1.0) or 1.0
                        ),
                    }
                )
            images.append(
                {
                    "image_id": image_id,
                    "path": str(final_path.resolve()),
                    "role": str(contract.get("role") or "memory"),
                    "public_retrieval_description": str(verified["public_retrieval_description"]),
                    "private_verified_visual_facts": visual_facts,
                }
            )
            image_manifests.append(
                {
                    "image_id": image_id,
                    **lineage,
                    "candidate_path": str(candidate_path.resolve()),
                    "final_path": str(final_path.resolve()),
                    "task_critical_constraints": list(contract.get("task_critical_constraints") or []),
                    "verified_visual_facts": visual_facts,
                    "uncertain_visual_facts": list(verified.get("uncertain_facts") or []),
                    "verifier_checks": list(verified.get("checks") or []),
                    "accepted": True,
                }
            )
        _write_json(self._stage_root / "private_plan" / "image_contracts.json", contracts)
        _write_json(self._stage_root / "observed" / "verified_images.json", images)
        _write_json(self._stage_root / "observed" / "image_manifests.json", image_manifests)
        return images

    @staticmethod
    def _visible_image_state(image: dict[str, Any]) -> dict[str, Any]:
        return {
            "image_id": image["image_id"],
            "public_retrieval_description": image["public_retrieval_description"],
            "verified_visual_facts": image["private_verified_visual_facts"],
        }

    def _generate_dialogue(
        self,
        blueprint: dict[str, Any],
        graph: dict[str, Any],
        images: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        cached_path = self._stage_root / "observed" / "conversation.json"
        if self.config.resume and cached_path.is_file():
            cached = json.loads(cached_path.read_text(encoding="utf-8"))
            sessions = cached.get("sessions") if isinstance(cached, dict) else None
            if isinstance(sessions, list) and sessions:
                return [dict(item) for item in sessions]
        image_by_id = {str(item["image_id"]): item for item in images}
        events_by_session: dict[str, list[dict[str, Any]]] = {}
        for event in _list(graph.get("events")):
            events_by_session.setdefault(str(event["session_id"]), []).append(event)
        history: list[dict[str, Any]] = []
        sessions = []
        contract_by_image = {
            str(item["image_id"]): item
            for item in json.loads((self._stage_root / "private_plan" / "image_contracts.json").read_text())
        } if (self._stage_root / "private_plan" / "image_contracts.json").is_file() else {}
        for session_plan in _list(blueprint.get("session_plan")):
            session_id = str(session_plan["session_id"])
            rows = sorted(events_by_session.get(session_id, []), key=lambda item: int(item["turn_index"]))
            final_events = []
            for event_plan in rows:
                planned_image_ids = [
                    image_id
                    for image_id in _string_list(event_plan.get("image_ids"))
                    if image_id in image_by_id and image_by_id[image_id].get("role") in {"memory", "both"}
                ]
                attached_images = [image_by_id[image_id] for image_id in planned_image_ids]
                partitions = {
                    image_id: contract_by_image.get(image_id, {}).get("information_partition", {})
                    for image_id in planned_image_ids
                }
                repair_history: list[dict[str, Any]] = []
                event: dict[str, Any] | None = None
                for dialogue_attempt in range(self.config.stage_retries + 1):
                    user_payload = {
                        "persona": blueprint.get("persona"),
                        "past_conversation": history,
                        "session": session_plan,
                        "current_event": event_plan,
                        "current_images": [self._visible_image_state(item) for item in attached_images],
                        "information_partitions": partitions,
                    }
                    if repair_history:
                        user_payload["repair_history"] = repair_history
                    user_value = self._generate_checked(
                        "user_simulator",
                        USER_SIMULATOR_CONTRACT,
                        user_payload,
                        lambda value: [] if _nonempty(value.get("user")) else ["user message is empty"],
                        image_paths=[item["path"] for item in attached_images],
                    )
                    user = str(user_value["user"]).strip()
                    assistant_value = self._generate_checked(
                        "assistant_simulator",
                        ASSISTANT_SIMULATOR_CONTRACT,
                        {
                            "past_conversation": history,
                            "current_user_message": user,
                            "attached_image_ids": planned_image_ids,
                            "repair_history": repair_history,
                        },
                        lambda value: [] if _nonempty(value.get("assistant")) else ["assistant reply is empty"],
                        image_paths=[item["path"] for item in attached_images],
                    )
                    event = {
                        "event_id": str(event_plan["event_id"]),
                        "session_id": session_id,
                        "turn_index": int(event_plan["turn_index"]),
                        "timestamp": str(event_plan["timestamp"]),
                        "user": user,
                        "assistant": str(assistant_value["assistant"]).strip(),
                        "image_ids": planned_image_ids,
                    }
                    if not partitions:
                        break
                    leakage = self._generate_checked(
                        "dialogue_leakage_checker",
                        DIALOGUE_LEAKAGE_CHECKER_CONTRACT,
                        {
                            "event": event,
                            "image_information_partitions": partitions,
                            "allowed_user_information": event_plan.get("allowed_user_information", []),
                        },
                        lambda value: [] if isinstance(value.get("passed"), bool) else ["passed must be boolean"],
                    )
                    if leakage["passed"] is True:
                        break
                    repair_history.append(
                        {
                            "leaks": leakage.get("leaks", []),
                            "instruction": leakage.get("repair_instruction") or "Remove image-only information.",
                            "dialogue_attempt": dialogue_attempt,
                        }
                    )
                    event = None
                if event is None:
                    raise GenerationStageError(f"dialogue leakage could not be repaired for {event_plan['event_id']}")
                final_events.append(event)
                history.append(event)
            sessions.append({"session_id": session_id, "date": str(session_plan["date"]), "events": final_events})
        _write_json(self._stage_root / "observed" / "conversation.json", {"sessions": sessions})
        return sessions

    @staticmethod
    def _validate_fact_batch(
        value: dict[str, Any],
        *,
        sessions: list[dict[str, Any]],
        images: list[dict[str, Any]],
        existing_fact_ids: set[str],
    ) -> list[str]:
        errors = []
        event_by_id = {
            str(event["event_id"]): event
            for session in sessions
            for event in _list(session.get("events"))
        }
        visual_to_image = {
            str(fact["visual_fact_id"]): str(image["image_id"])
            for image in images
            for fact in _list(image.get("private_verified_visual_facts"))
            if isinstance(fact, dict) and fact.get("visual_fact_id")
        }
        known = set(existing_fact_ids)
        facts = _list(value.get("observed_facts"))
        for index, fact in enumerate(facts):
            if not isinstance(fact, dict):
                errors.append(f"observed_facts[{index}] is not an object")
                continue
            fact_id = str(fact.get("fact_id") or "")
            if not fact_id or fact_id in known:
                errors.append(f"invalid or duplicate fact_id {fact_id!r}")
                continue
            for key in ("subject", "predicate", "valid_from_session"):
                if not _nonempty(fact.get(key)):
                    errors.append(f"{fact_id}.{key} is empty")
            for related in [*_string_list(fact.get("supersedes")), *_string_list(fact.get("contradicts"))]:
                if related not in known:
                    errors.append(f"{fact_id} relates to unknown or later fact {related}")
            provenance = _list(fact.get("observed_provenance"))
            if not provenance:
                errors.append(f"{fact_id} has no observed provenance")
            for source in provenance:
                if not isinstance(source, dict):
                    errors.append(f"{fact_id} has malformed provenance")
                    continue
                event_id = str(source.get("event_id") or "")
                event = event_by_id.get(event_id)
                if event is None:
                    errors.append(f"{fact_id} references unknown event {event_id}")
                    continue
                content = f"User: {event['user']}\nAssistant: {event['assistant']}"
                for span in _string_list(source.get("text_spans")):
                    if span not in content:
                        errors.append(f"{fact_id} span is not verbatim in {event_id}: {span!r}")
                for visual_id in _string_list(source.get("visual_fact_ids")):
                    if visual_id not in visual_to_image:
                        errors.append(f"{fact_id} references unknown visual fact {visual_id}")
                    elif visual_to_image[visual_id] not in _string_list(event.get("image_ids")):
                        errors.append(f"{fact_id} visual fact {visual_id} is not attached to {event_id}")
            known.add(fact_id)
        return errors

    def _extract_facts(
        self,
        sessions: list[dict[str, Any]],
        images: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        facts_path = self._stage_root / "observed" / "observed_facts.json"
        if self.config.resume and facts_path.is_file():
            cached = json.loads(facts_path.read_text(encoding="utf-8"))
            facts = [dict(item) for item in cached] if isinstance(cached, list) else []
        else:
            facts = []
            for session_index, session in enumerate(sessions):
                visible_image_ids = {
                    image_id
                    for event in _list(session.get("events"))
                    for image_id in _string_list(event.get("image_ids"))
                }
                relevant_images = [
                    self._visible_image_state(item)
                    for item in images
                    if item["image_id"] in visible_image_ids
                ]
                value = self._generate_checked(
                    "state_extractor",
                    STATE_EXTRACTOR_CONTRACT,
                    {
                        "session": session,
                        "verified_visual_observations": relevant_images,
                        "existing_observed_facts": facts,
                        "fact_id_prefix": f"F{session_index + 1:02d}_",
                    },
                    lambda item: self._validate_fact_batch(
                        item,
                        sessions=sessions,
                        images=images,
                        existing_fact_ids={str(fact["fact_id"]) for fact in facts},
                    ),
                )
                facts.extend(dict(item) for item in value["observed_facts"])
            _write_json(facts_path, facts)
        audit: dict[str, Any] = {"enabled": self.config.run_full_state_audit}
        audit_path = self._stage_root / "reports" / "state_audit.json"
        if self.config.resume and audit_path.is_file():
            cached_audit = json.loads(audit_path.read_text(encoding="utf-8"))
            if isinstance(cached_audit, dict):
                return facts, cached_audit
        if self.config.run_full_state_audit:
            full = self._generate_checked(
                "full_state_auditor",
                FULL_STATE_AUDITOR_CONTRACT,
                {
                    "sessions": sessions,
                    "verified_visual_observations": [self._visible_image_state(item) for item in images],
                    "fact_id_prefix": "AUDIT_",
                },
                lambda item: self._validate_fact_batch(
                    item,
                    sessions=sessions,
                    images=images,
                    existing_fact_ids=set(),
                ),
            )
            def signatures(rows: list[dict[str, Any]]) -> set[str]:
                return {
                    _json([item.get("subject"), item.get("predicate"), item.get("object")])
                    for item in rows
                }
            incremental_signatures = signatures(facts)
            audit_signatures = signatures(_list(full.get("observed_facts")))
            audit = {
                "enabled": True,
                "incremental_fact_count": len(facts),
                "audit_fact_count": len(_list(full.get("observed_facts"))),
                "missing_from_audit": sorted(incremental_signatures - audit_signatures),
                "additional_in_audit": sorted(audit_signatures - incremental_signatures),
                "audit_did_not_overwrite_incremental": True,
            }
        _write_json(audit_path, audit)
        return facts, audit

    @staticmethod
    def _canonical_qa(spec: dict[str, Any], question_text: str) -> dict[str, Any]:
        return {
            "qa_id": str(spec.get("qa_id") or ""),
            "task": str(spec.get("task") or "").upper(),
            "memory_cutoff": dict(spec.get("memory_cutoff") or {"mode": "episode_end"}),
            "question_modality": str(spec.get("question_modality") or "text"),
            "question_text": question_text,
            "question_image_ids": _string_list(spec.get("question_image_ids")),
            "answer": str(spec.get("answer") or ""),
            "canonical_answer": spec.get("canonical_answer", spec.get("answer")),
            "answer_type": str(spec.get("answer_type") or "text"),
            # Evidence is derived from the structured oracle and observed
            # provenance. Model-proposed sets are only generation hints and
            # must never override deterministic minimal-evidence mining.
            "required_evidence_sets": [],
            "required_visual_fact_ids": _string_list(spec.get("required_visual_fact_ids")),
            "supporting_event_ids": [],
            "hard_negatives": _string_list(spec.get("hard_negatives")),
            "answer_function": dict(spec.get("answer_function") or {}),
            "task_oracle": dict(spec.get("task_oracle") or {}),
        }

    @staticmethod
    def _validate_qa_specs(
        value: dict[str, Any],
        quota: dict[str, int],
        observed_episode: dict[str, Any] | None = None,
    ) -> list[str]:
        specs = _list(value.get("qa_specs"))
        errors = []
        observed_episode = observed_episode or {}
        known_images = {
            str(item.get("image_id"))
            for item in _list(observed_episode.get("images"))
            if isinstance(item, dict) and item.get("image_id")
        }
        image_roles = {
            str(item.get("image_id")): str(item.get("role") or "memory")
            for item in _list(observed_episode.get("images"))
            if isinstance(item, dict) and item.get("image_id")
        }
        dialogue_text = "\n".join(
            f"{event.get('user', '')}\n{event.get('assistant', '')}"
            for session in _list(observed_episode.get("sessions"))
            if isinstance(session, dict)
            for event in _list(session.get("events"))
            if isinstance(event, dict)
        ).casefold()
        visual_values = {
            str(fact.get("visual_fact_id")): fact.get("value")
            for image in _list(observed_episode.get("images"))
            if isinstance(image, dict)
            for fact in _list(image.get("private_verified_visual_facts"))
            if isinstance(fact, dict) and fact.get("visual_fact_id")
        }
        if len(specs) != sum(quota.values()):
            errors.append(f"qa_specs must contain exactly {sum(quota.values())} items")
        ids = [str(item.get("qa_id") or "") for item in specs if isinstance(item, dict)]
        if len(ids) != len(set(ids)) or any(not item for item in ids):
            errors.append("QA IDs must be non-empty and unique")
        actual = {
            task: sum(1 for item in specs if isinstance(item, dict) and str(item.get("task") or "").upper() == task)
            for task in quota
        }
        if actual != quota:
            errors.append(f"QA task counts must match quota; expected={quota}, actual={actual}")
        for item in specs:
            if not isinstance(item, dict):
                errors.append("every QA specification must be an object")
                continue
            if not _nonempty(item.get("answer")) or not isinstance(item.get("task_oracle"), dict):
                errors.append(f"{item.get('qa_id')} lacks an answer or structured task_oracle")
                continue
            task = str(item.get("task") or "").upper()
            oracle = dict(item["task_oracle"])
            if task == "VS":
                image_id = str(oracle.get("image_id") or "")
                if oracle.get("kind") != "image_lookup" or str(item.get("answer") or "") != image_id:
                    errors.append(f"{item.get('qa_id')} VS answer must exactly equal image_lookup.image_id")
                question_images = _string_list(item.get("question_image_ids"))
                query_image_id = str(oracle.get("query_image_id") or "")
                if question_images:
                    if item.get("question_modality") not in {"image", "text+image"}:
                        errors.append(f"{item.get('qa_id')} VS query image requires image modality")
                    if query_image_id not in question_images or any(
                        image_roles.get(value) != "query" for value in question_images
                    ):
                        errors.append(f"{item.get('qa_id')} VS must use a separate query-only image")
                    if not _string_list(oracle.get("matching_visual_predicates")):
                        errors.append(f"{item.get('qa_id')} VS query-image lookup requires matching predicates")
                elif item.get("question_modality") != "text" or query_image_id:
                    errors.append(f"{item.get('qa_id')} VS without a query image must use text modality")
                negative_images = set(_string_list(item.get("hard_negatives"))) & known_images
                if len(known_images) > 1 and not (negative_images - {image_id}):
                    errors.append(f"{item.get('qa_id')} VS requires a different historical-image hard negative")
                for visual_id in _string_list(item.get("required_visual_fact_ids")):
                    raw_value = visual_values.get(visual_id)
                    normalized = str(raw_value).strip().casefold()
                    if normalized and normalized not in {"true", "false", "yes", "no"}:
                        pattern = rf"(?<!\w){re.escape(normalized)}(?!\w)"
                        if re.search(pattern, dialogue_text):
                            errors.append(
                                f"{item.get('qa_id')} VS visual target {visual_id} is directly stated in dialogue"
                            )
            if task == "AR":
                scope = _string_list(oracle.get("closed_world_scope"))
                missing = str(oracle.get("missing_predicate") or "")
                anchors = _string_list(oracle.get("topic_anchor_event_ids"))
                if oracle.get("kind") != "absence_in_closed_scope" or not missing or missing not in scope:
                    errors.append(f"{item.get('qa_id')} AR scope must contain missing_predicate")
                if not anchors:
                    errors.append(f"{item.get('qa_id')} AR requires topic_anchor_event_ids")
                intent = str(item.get("question_intent") or "").casefold()
                if re.search(r"\b(whether|is|does|did|has|have|can)\b", intent):
                    errors.append(f"{item.get('qa_id')} AR must ask for a missing value, not a binary claim")
            if observed_episode:
                try:
                    provisional = MMemGenerationPipeline._canonical_qa(
                        item,
                        "Which recorded result satisfies this request?",
                    )
                    provisional_result = build_episode(
                        {**observed_episode, "qa_candidates": [provisional]}
                    )
                    if not provisional_result.accepted_qas:
                        certificate = (
                            provisional_result.certificates[0]
                            if provisional_result.certificates
                            else {}
                        )
                        messages = [
                            str(issue.get("message") or issue.get("code") or "invalid QA spec")
                            for issue in certificate.get("issues", [])
                            if isinstance(issue, dict)
                        ]
                        errors.append(
                            f"{item.get('qa_id')} fails deterministic oracle precheck: "
                            + "; ".join(messages[:6])
                        )
                except Exception as exc:
                    errors.append(
                        f"{item.get('qa_id')} deterministic oracle precheck failed: "
                        f"{type(exc).__name__}: {exc}"
                    )
        return errors

    def _generate_qas(
        self,
        request: GenerationRequest,
        base_artifact: dict[str, Any],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        quota = _task_quota(request.qa_count, request.task_ratios)
        spec_result = self._generate_checked(
            "qa_spec_generator",
            QA_SPEC_GENERATOR_CONTRACT,
            {
                "task_quota": quota,
                "observed_episode": base_artifact,
                "instruction": "Every target, answer, and oracle field must be derived from observed_episode only.",
            },
            # The validator sees only the finalized observed episode, never
            # the private plan.
            lambda value: self._validate_qa_specs(value, quota, base_artifact),
        )
        accepted: list[dict[str, Any]] = []
        rejected: list[dict[str, Any]] = []
        pending_judgment: list[dict[str, Any]] = []
        for spec in _list(spec_result.get("qa_specs")):
            if not isinstance(spec, dict):
                continue
            realized = self._generate_checked(
                "qa_realizer",
                QA_REALIZER_CONTRACT,
                {
                    "qa_specification": spec,
                    "supporting_events": [
                        event
                        for session in base_artifact["sessions"]
                        for event in session["events"]
                        if event["event_id"] in {
                            item
                            for group in _list(spec.get("required_evidence_sets"))
                            for item in _string_list(group)
                        }
                    ],
                    "candidate_count": self.config.qa_paraphrase_candidates,
                },
                lambda value: [] if _list(value.get("candidates")) else ["candidates must be non-empty"],
            )
            deterministic_candidates = []
            deterministic_failures = []
            for candidate in _list(realized.get("candidates")):
                question = str(candidate.get("question_text") or "").strip() if isinstance(candidate, dict) else ""
                if not question:
                    continue
                qa = self._canonical_qa(spec, question)
                trial = {**base_artifact, "qa_candidates": [qa]}
                try:
                    result = build_episode(trial)
                except Exception as exc:
                    deterministic_failures.append({"question_text": question, "error": f"{type(exc).__name__}: {exc}"})
                    continue
                if result.accepted_qas:
                    deterministic_candidates.append(dict(result.accepted_qas[0]))
                else:
                    certificate = result.certificates[0] if result.certificates else {}
                    deterministic_failures.append({"question_text": question, "certificate": certificate})
            if not deterministic_candidates:
                rejected.append(
                    {
                        "qa_spec": spec,
                        "reason": "no deterministic-valid realization",
                        "failures": deterministic_failures,
                    }
                )
                continue
            pending_judgment.append(
                {
                    "qa_id": str(spec.get("qa_id") or ""),
                    "qa_specification": {
                        **spec,
                        "required_evidence_sets": [],
                        "supporting_event_ids": [],
                    },
                    "candidates": deterministic_candidates,
                }
            )
        if pending_judgment:
            expected_ids = [item["qa_id"] for item in pending_judgment]

            def validate_batch_judgment(value: dict[str, Any]) -> list[str]:
                judgments = _list(value.get("judgments"))
                actual_ids = [
                    str(item.get("qa_id") or "") for item in judgments if isinstance(item, dict)
                ]
                errors = []
                if actual_ids != expected_ids:
                    errors.append("judge must return one judgment per QA in input order")
                for item in judgments:
                    if not isinstance(item, dict) or not isinstance(item.get("accepted"), bool):
                        errors.append("every judgment requires boolean accepted")
                    elif item["accepted"] and not isinstance(item.get("selected_index"), int):
                        errors.append("accepted judgment requires selected_index")
                return errors

            judged_batch = self._generate_checked(
                "qa_judge_batch",
                QA_JUDGE_CONTRACT,
                {"observed_episode": base_artifact, "items": pending_judgment},
                validate_batch_judgment,
            )
            judgments = {
                str(item["qa_id"]): item
                for item in _list(judged_batch.get("judgments"))
                if isinstance(item, dict) and item.get("qa_id")
            }
            for pending in pending_judgment:
                judgment = judgments.get(pending["qa_id"], {})
                index = judgment.get("selected_index")
                candidates = pending["candidates"]
                if judgment.get("accepted") is True and isinstance(index, int) and 0 <= index < len(candidates):
                    accepted.append(candidates[index])
                else:
                    rejected.append(
                        {
                            "qa_spec": pending["qa_specification"],
                            "reason": "qa_judge_rejected",
                            "judge": judgment,
                        }
                    )
        _write_json(self._stage_root / "observed" / "qa_candidates.json", accepted)
        _write_json(self._stage_root / "reports" / "generation_rejected_qas.json", rejected)
        return accepted, rejected

    def generate(
        self,
        request: GenerationRequest,
        *,
        work_root: str | Path,
    ) -> GeneratedEpisode:
        self._stage_root = Path(work_root).resolve() / request.episode_id
        self._stage_root.mkdir(parents=True, exist_ok=True)
        self._stage_counts.clear()
        if self.config.resume:
            model_stage_root = self._stage_root / "model_stages"
            if model_stage_root.is_dir():
                for stage_root in model_stage_root.iterdir():
                    indices = []
                    for output_path in stage_root.glob("*_output.json"):
                        prefix = output_path.name.split("_attempt", 1)[0]
                        if prefix.isdigit():
                            indices.append(int(prefix))
                    if indices:
                        self._stage_counts[stage_root.name] = max(indices) + 1
        _write_json(self._stage_root / "generation_request.json", request.planner_input())
        blueprint, graph = self._plan(request)
        contracts = self._image_contracts(blueprint, graph)
        images = self._generate_images(contracts)
        sessions = self._generate_dialogue(blueprint, graph, images)
        facts, _ = self._extract_facts(sessions, images)
        base_artifact = {
            "schema_version": "mmem-v2.0",
            "dataset": request.dataset,
            "episode_id": request.episode_id,
            "character_profile": dict(blueprint.get("persona") or {}),
            "sessions": sessions,
            "images": images,
            "observed_facts": facts,
            "qa_candidates": [],
        }
        qas, rejected = self._generate_qas(request, base_artifact)
        artifact = {**base_artifact, "qa_candidates": qas}
        final = build_episode(artifact)
        if not final.accepted_qas:
            raise GenerationStageError("generation produced no QA that passed deterministic acceptance")
        accepted_ids = {str(item["qa_id"]) for item in final.accepted_qas}
        artifact["qa_candidates"] = [item for item in qas if str(item["qa_id"]) in accepted_ids]
        artifact_path = self._stage_root / f"{request.episode_id}.episode.json"
        _write_json(artifact_path, artifact)
        _write_json(self._stage_root / "reports" / "deterministic_manifest.json", final.manifest())
        return GeneratedEpisode(
            artifact=artifact,
            artifact_path=artifact_path,
            work_dir=self._stage_root,
            accepted_qa_count=len(artifact["qa_candidates"]),
            rejected_qa_count=len(rejected) + len(final.rejected_qas),
        )
