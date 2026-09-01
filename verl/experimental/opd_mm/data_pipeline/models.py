"""Canonical artifact models for the MMem Base-v0 data pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


class ArtifactValidationError(ValueError):
    """Raised when an episode artifact violates the construction contract."""

    def __init__(self, errors: list[str]):
        self.errors = errors
        super().__init__("; ".join(errors))


def _text(value: Any, path: str, errors: list[str]) -> str:
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{path} must be a non-empty string")
        return ""
    return value.strip()


def _string_list(value: Any, path: str, errors: list[str]) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(item, str) and item.strip() for item in value):
        errors.append(f"{path} must be a list of non-empty strings")
        return []
    return [item.strip() for item in value]


@dataclass(frozen=True)
class Event:
    event_id: str
    session_id: str
    turn_index: int
    timestamp: str
    user: str
    assistant: str
    image_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Any, path: str, errors: list[str]) -> "Event":
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            value = {}
        turn_index = value.get("turn_index")
        if not isinstance(turn_index, int) or isinstance(turn_index, bool) or turn_index < 1:
            errors.append(f"{path}.turn_index must be a positive integer")
            turn_index = 1
        return cls(
            event_id=_text(value.get("event_id"), f"{path}.event_id", errors),
            session_id=_text(value.get("session_id"), f"{path}.session_id", errors),
            turn_index=turn_index,
            timestamp=_text(value.get("timestamp"), f"{path}.timestamp", errors),
            user=_text(value.get("user"), f"{path}.user", errors),
            assistant=_text(value.get("assistant"), f"{path}.assistant", errors),
            image_ids=tuple(_string_list(value.get("image_ids", []), f"{path}.image_ids", errors)),
        )

    @property
    def content(self) -> str:
        return f"User: {self.user}\nAssistant: {self.assistant}"

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["image_ids"] = list(self.image_ids)
        return value


@dataclass(frozen=True)
class Session:
    session_id: str
    date: str
    events: tuple[Event, ...]

    @classmethod
    def from_dict(cls, value: Any, path: str, errors: list[str]) -> "Session":
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            value = {}
        session_id = _text(value.get("session_id"), f"{path}.session_id", errors)
        raw_events = value.get("events")
        if not isinstance(raw_events, list) or not raw_events:
            errors.append(f"{path}.events must be a non-empty list")
            raw_events = []
        events = tuple(
            Event.from_dict(item, f"{path}.events[{index}]", errors)
            for index, item in enumerate(raw_events)
        )
        for event in events:
            if event.session_id and event.session_id != session_id:
                errors.append(
                    f"{path}: event {event.event_id!r} belongs to {event.session_id!r}, "
                    f"expected {session_id!r}"
                )
        return cls(
            session_id=session_id,
            date=_text(value.get("date"), f"{path}.date", errors),
            events=events,
        )

    def to_dict(self) -> dict[str, Any]:
        return {"session_id": self.session_id, "date": self.date, "events": [event.to_dict() for event in self.events]}


@dataclass(frozen=True)
class VisualFact:
    visual_fact_id: str
    predicate: str
    value: Any
    subject: str = ""
    confidence: float | None = None
    verifier_agreement: int | None = None

    @classmethod
    def from_dict(cls, value: Any, path: str, errors: list[str]) -> "VisualFact":
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            value = {}
        confidence = value.get("confidence")
        agreement = value.get("verifier_agreement")
        return cls(
            visual_fact_id=_text(value.get("visual_fact_id"), f"{path}.visual_fact_id", errors),
            predicate=_text(value.get("predicate"), f"{path}.predicate", errors),
            value=value.get("value"),
            subject=str(value.get("subject") or ""),
            confidence=float(confidence) if isinstance(confidence, (int, float)) else None,
            verifier_agreement=int(agreement) if isinstance(agreement, int) else None,
        )


@dataclass(frozen=True)
class ImageState:
    image_id: str
    path: str
    public_retrieval_description: str
    private_verified_visual_facts: tuple[VisualFact, ...] = ()
    role: str = "memory"

    @classmethod
    def from_dict(cls, value: Any, path: str, errors: list[str]) -> "ImageState":
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            value = {}
        raw_facts = value.get("private_verified_visual_facts", [])
        if not isinstance(raw_facts, list):
            errors.append(f"{path}.private_verified_visual_facts must be a list")
            raw_facts = []
        role = str(value.get("role") or "memory")
        if role not in {"memory", "query", "both"}:
            errors.append(f"{path}.role must be memory, query, or both")
        return cls(
            image_id=_text(value.get("image_id"), f"{path}.image_id", errors),
            path=_text(value.get("path"), f"{path}.path", errors),
            public_retrieval_description=str(value.get("public_retrieval_description") or "").strip(),
            private_verified_visual_facts=tuple(
                VisualFact.from_dict(item, f"{path}.private_verified_visual_facts[{index}]", errors)
                for index, item in enumerate(raw_facts)
            ),
            role=role,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "image_id": self.image_id,
            "path": self.path,
            "public_retrieval_description": self.public_retrieval_description,
            "private_verified_visual_facts": [asdict(fact) for fact in self.private_verified_visual_facts],
            "role": self.role,
        }


@dataclass(frozen=True)
class Provenance:
    event_id: str
    text_spans: tuple[str, ...] = ()
    visual_fact_ids: tuple[str, ...] = ()

    @classmethod
    def from_dict(cls, value: Any, path: str, errors: list[str]) -> "Provenance":
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            value = {}
        return cls(
            event_id=_text(value.get("event_id"), f"{path}.event_id", errors),
            text_spans=tuple(_string_list(value.get("text_spans", []), f"{path}.text_spans", errors)),
            visual_fact_ids=tuple(
                _string_list(value.get("visual_fact_ids", []), f"{path}.visual_fact_ids", errors)
            ),
        )


@dataclass(frozen=True)
class ObservedFact:
    fact_id: str
    subject: str
    predicate: str
    object: Any
    epistemic_status: str
    lifecycle_status: str
    valid_from_session: str
    valid_to_session: str | None
    supersedes: tuple[str, ...]
    contradicts: tuple[str, ...]
    observed_provenance: tuple[Provenance, ...]

    @classmethod
    def from_dict(cls, value: Any, path: str, errors: list[str]) -> "ObservedFact":
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            value = {}
        raw_provenance = value.get("observed_provenance")
        if not isinstance(raw_provenance, list) or not raw_provenance:
            errors.append(f"{path}.observed_provenance must be a non-empty list")
            raw_provenance = []
        lifecycle = str(value.get("lifecycle_status") or "active")
        if lifecycle not in {"active", "superseded", "retracted"}:
            errors.append(f"{path}.lifecycle_status is invalid")
        return cls(
            fact_id=_text(value.get("fact_id"), f"{path}.fact_id", errors),
            subject=_text(value.get("subject"), f"{path}.subject", errors),
            predicate=_text(value.get("predicate"), f"{path}.predicate", errors),
            object=value.get("object"),
            epistemic_status=str(value.get("epistemic_status") or "asserted"),
            lifecycle_status=lifecycle,
            valid_from_session=_text(value.get("valid_from_session"), f"{path}.valid_from_session", errors),
            valid_to_session=(str(value["valid_to_session"]) if value.get("valid_to_session") else None),
            supersedes=tuple(_string_list(value.get("supersedes", []), f"{path}.supersedes", errors)),
            contradicts=tuple(_string_list(value.get("contradicts", []), f"{path}.contradicts", errors)),
            observed_provenance=tuple(
                Provenance.from_dict(item, f"{path}.observed_provenance[{index}]", errors)
                for index, item in enumerate(raw_provenance)
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        for key in ("supersedes", "contradicts"):
            value[key] = list(value[key])
        value["observed_provenance"] = [
            {
                "event_id": provenance.event_id,
                "text_spans": list(provenance.text_spans),
                "visual_fact_ids": list(provenance.visual_fact_ids),
            }
            for provenance in self.observed_provenance
        ]
        return value


@dataclass(frozen=True)
class MemoryCutoff:
    mode: str = "episode_end"
    session_id: str | None = None
    event_id: str | None = None
    timestamp: str | None = None

    @classmethod
    def from_dict(cls, value: Any, path: str, errors: list[str]) -> "MemoryCutoff":
        if value is None:
            value = {"mode": "episode_end"}
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            value = {}
        mode = str(value.get("mode") or "episode_end")
        if mode not in {"episode_end", "session_end", "event", "timestamp"}:
            errors.append(f"{path}.mode is invalid")
        return cls(
            mode=mode,
            session_id=str(value["session_id"]) if value.get("session_id") else None,
            event_id=str(value["event_id"]) if value.get("event_id") else None,
            timestamp=str(value["timestamp"]) if value.get("timestamp") else None,
        )

    def to_dict(self) -> dict[str, Any]:
        return {key: value for key, value in asdict(self).items() if value is not None}


@dataclass(frozen=True)
class QACandidate:
    qa_id: str
    task: str
    memory_cutoff: MemoryCutoff
    question_modality: str
    question_text: str
    question_image_ids: tuple[str, ...]
    answer: str
    canonical_answer: Any
    answer_type: str
    required_evidence_sets: tuple[tuple[str, ...], ...]
    required_visual_fact_ids: tuple[str, ...]
    supporting_event_ids: tuple[str, ...]
    hard_negatives: tuple[str, ...]
    answer_function: dict[str, Any]
    task_oracle: dict[str, Any]
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, value: Any, path: str, errors: list[str]) -> "QACandidate":
        if not isinstance(value, dict):
            errors.append(f"{path} must be an object")
            value = {}
        raw_sets = value.get("required_evidence_sets", [])
        if not isinstance(raw_sets, list) or not all(isinstance(item, list) for item in raw_sets):
            errors.append(f"{path}.required_evidence_sets must be a list of event-id lists")
            raw_sets = []
        task = str(value.get("task") or "").upper()
        if task not in {"FR", "TR", "KR", "AR", "VS", "VR", "MR", "CD", "TTL"}:
            errors.append(f"{path}.task is invalid")
        oracle = value.get("task_oracle")
        if not isinstance(oracle, dict) or not oracle.get("kind"):
            errors.append(f"{path}.task_oracle must contain kind")
            oracle = {}
        answer_function = value.get("answer_function", {})
        if not isinstance(answer_function, dict):
            errors.append(f"{path}.answer_function must be an object")
            answer_function = {}
        reserved = {
            "qa_id", "task", "memory_cutoff", "question_modality", "question_text", "question_image_ids",
            "answer", "canonical_answer", "answer_type", "required_evidence_sets", "required_visual_fact_ids", "supporting_event_ids",
            "hard_negatives", "answer_function", "task_oracle",
        }
        metadata = dict(value.get("metadata") or {})
        metadata.update({key: item for key, item in value.items() if key not in reserved and key != "metadata"})
        return cls(
            qa_id=_text(value.get("qa_id"), f"{path}.qa_id", errors),
            task=task,
            memory_cutoff=MemoryCutoff.from_dict(value.get("memory_cutoff"), f"{path}.memory_cutoff", errors),
            question_modality=str(value.get("question_modality") or "text"),
            question_text=_text(value.get("question_text"), f"{path}.question_text", errors),
            question_image_ids=tuple(
                _string_list(value.get("question_image_ids", []), f"{path}.question_image_ids", errors)
            ),
            answer=_text(value.get("answer"), f"{path}.answer", errors),
            canonical_answer=value.get("canonical_answer", value.get("answer")),
            answer_type=str(value.get("answer_type") or "text"),
            required_evidence_sets=tuple(tuple(str(event_id) for event_id in item) for item in raw_sets),
            required_visual_fact_ids=tuple(
                _string_list(value.get("required_visual_fact_ids", []), f"{path}.required_visual_fact_ids", errors)
            ),
            supporting_event_ids=tuple(
                _string_list(value.get("supporting_event_ids", []), f"{path}.supporting_event_ids", errors)
            ),
            hard_negatives=tuple(_string_list(value.get("hard_negatives", []), f"{path}.hard_negatives", errors)),
            answer_function=answer_function,
            task_oracle=oracle,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "qa_id": self.qa_id,
            "task": self.task,
            "memory_cutoff": self.memory_cutoff.to_dict(),
            "question_modality": self.question_modality,
            "question_text": self.question_text,
            "question_image_ids": list(self.question_image_ids),
            "answer": self.answer,
            "canonical_answer": self.canonical_answer,
            "answer_type": self.answer_type,
            "required_evidence_sets": [list(item) for item in self.required_evidence_sets],
            "required_visual_fact_ids": list(self.required_visual_fact_ids),
            "supporting_event_ids": list(self.supporting_event_ids),
            "hard_negatives": list(self.hard_negatives),
            "answer_function": self.answer_function,
            "task_oracle": self.task_oracle,
            **self.metadata,
        }


@dataclass(frozen=True)
class Episode:
    schema_version: str
    dataset: str
    episode_id: str
    character_profile: dict[str, Any]
    sessions: tuple[Session, ...]
    images: tuple[ImageState, ...]
    observed_facts: tuple[ObservedFact, ...]
    qa_candidates: tuple[QACandidate, ...]
    source_root: str = ""

    @classmethod
    def from_dict(cls, value: Any, *, source_root: str | Path = "") -> "Episode":
        errors: list[str] = []
        if not isinstance(value, dict):
            raise ArtifactValidationError(["root must be an object"])
        forbidden = sorted(set(value) & {"planned_facts", "planned_evidence", "planned_provenance"})
        if forbidden:
            errors.append(
                f"planned state cannot enter the observed artifact: {forbidden}; keep it in a separate planning file"
            )
        raw_sessions = value.get("sessions")
        raw_images = value.get("images", [])
        raw_facts = value.get("observed_facts", [])
        raw_qas = value.get("qa_candidates", [])
        collections = (
            (raw_sessions, "sessions"),
            (raw_images, "images"),
            (raw_facts, "observed_facts"),
            (raw_qas, "qa_candidates"),
        )
        for raw, name in collections:
            if not isinstance(raw, list):
                errors.append(f"root.{name} must be a list")
        sessions = tuple(
            Session.from_dict(item, f"root.sessions[{index}]", errors)
            for index, item in enumerate(raw_sessions if isinstance(raw_sessions, list) else [])
        )
        images = tuple(
            ImageState.from_dict(item, f"root.images[{index}]", errors)
            for index, item in enumerate(raw_images if isinstance(raw_images, list) else [])
        )
        facts = tuple(
            ObservedFact.from_dict(item, f"root.observed_facts[{index}]", errors)
            for index, item in enumerate(raw_facts if isinstance(raw_facts, list) else [])
        )
        qas = tuple(
            QACandidate.from_dict(item, f"root.qa_candidates[{index}]", errors)
            for index, item in enumerate(raw_qas if isinstance(raw_qas, list) else [])
        )
        profile = value.get("character_profile")
        if not isinstance(profile, dict):
            errors.append("root.character_profile must be an object")
            profile = {}
        episode = cls(
            schema_version=str(value.get("schema_version") or "mmem-v2.0"),
            dataset=str(value.get("dataset") or "mmem_v2"),
            episode_id=_text(value.get("episode_id"), "root.episode_id", errors),
            character_profile=profile,
            sessions=sessions,
            images=images,
            observed_facts=facts,
            qa_candidates=qas,
            source_root=str(source_root),
        )
        if errors:
            raise ArtifactValidationError(errors)
        return episode

    @property
    def events(self) -> tuple[Event, ...]:
        return tuple(event for session in self.sessions for event in session.events)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "dataset": self.dataset,
            "episode_id": self.episode_id,
            "character_profile": self.character_profile,
            "sessions": [session.to_dict() for session in self.sessions],
            "images": [image.to_dict() for image in self.images],
            "observed_facts": [fact.to_dict() for fact in self.observed_facts],
            "qa_candidates": [qa.to_dict() for qa in self.qa_candidates],
        }
