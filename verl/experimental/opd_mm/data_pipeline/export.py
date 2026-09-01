"""Deterministic Mem-Gallery and OPD-MM release adapters."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Iterable

from .models import Episode


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _safe(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _public_profile(profile: dict[str, Any]) -> dict[str, Any]:
    """Project a private generation persona onto the minimal public identity."""

    return {
        key: profile[key]
        for key in ("name",)
        if isinstance(profile.get(key), str) and str(profile[key]).strip()
    }


def _resolve_source(episode: Episode, value: str) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = Path(episode.source_root) / path
    return path.resolve()


def export_mem_gallery_episode(
    episode: Episode,
    accepted_qas: Iterable[dict[str, Any]],
    output_root: str | Path,
) -> Path:
    """Export one canonical episode to the layout consumed by OPD-MM."""

    root = Path(output_root)
    scenario = _safe(episode.episode_id)
    image_dir = root / "data" / "image" / scenario
    image_dir.mkdir(parents=True, exist_ok=True)
    image_by_id = {image.image_id: image for image in episode.images}
    relative_image: dict[str, str] = {}
    for image in episode.images:
        source = _resolve_source(episode, image.path)
        if not source.is_file():
            raise FileNotFoundError(f"image {image.image_id!r} does not exist: {source}")
        suffix = source.suffix.lower() if source.suffix else ".png"
        filename = f"{_safe(image.image_id)}{suffix}"
        destination = image_dir / filename
        if source != destination.resolve():
            shutil.copy2(source, destination)
        relative_image[image.image_id] = f"../image/{scenario}/{filename}"

    event_by_id = {event.event_id: event for event in episode.events}
    sessions = []
    for session in episode.sessions:
        dialogues = []
        for event in session.events:
            item: dict[str, Any] = {
                "round": event.event_id,
                "timestamp": event.timestamp,
                "user": event.user,
                "assistant": event.assistant,
            }
            if event.image_ids:
                item["image_id"] = list(event.image_ids)
                item["input_image"] = [relative_image[image_id] for image_id in event.image_ids]
            dialogues.append(item)
        sessions.append({"session_id": session.session_id, "date": session.date, "dialogues": dialogues})

    public_qas = []
    for qa_index, qa in enumerate(accepted_qas):
        evidence_sets = qa.get("required_evidence_sets") or []
        primary = list(evidence_sets[0]) if evidence_sets else []
        session_ids = list(
            dict.fromkeys(event_by_id[event_id].session_id for event_id in primary if event_id in event_by_id)
        )
        sample_id = f"{episode.dataset}:{episode.episode_id}:{qa['qa_id']}"
        public: dict[str, Any] = {
            "dataset": episode.dataset,
            "sample_id": sample_id,
            "scenario": episode.episode_id,
            "schema_version": episode.schema_version,
            "evidence_unit": "event",
            "point": qa["task"],
            "question": qa["question_text"],
            "answer": qa["answer"],
            "canonical_answer": qa.get("canonical_answer", qa["answer"]),
            "answer_type": qa.get("answer_type", "text"),
            "memory_cutoff": qa.get("memory_cutoff", {"mode": "episode_end"}),
            "session_id": session_ids,
            "clue": primary,
            "required_evidence_sets": evidence_sets,
            "required_visual_fact_ids": list(qa.get("required_visual_fact_ids") or []),
            "supporting_event_ids": list(qa.get("supporting_event_ids") or []),
            "hard_negatives": list(qa.get("hard_negatives") or []),
            "hard_negative_event_ids": list(qa.get("hard_negative_event_ids") or []),
            "hard_negative_image_ids": list(qa.get("hard_negative_image_ids") or []),
        }
        if qa["task"] == "AR":
            oracle = dict(qa.get("task_oracle") or {})
            public["answerable"] = False
            public["evidence_scope"] = {
                "closed_world_scope": list(oracle.get("closed_world_scope") or []),
                "topic_anchor_event_ids": list(oracle.get("topic_anchor_event_ids") or []),
                "memory_cutoff": public["memory_cutoff"],
                "require_full_history_scan": True,
            }
        question_images = qa.get("question_image_ids") or []
        if question_images:
            image_id = str(question_images[0])
            public["question_image"] = relative_image[image_id]
        public_qas.append(public)
    output = root / "data" / "dialog" / f"{scenario}.json"
    _write_json(
        output,
        {
            "character_profile": _public_profile(episode.character_profile),
            "multi_session_dialogues": sessions,
            "human-annotated QAs": public_qas,
            "annotation_provenance": {
                "generated_by": "model_pipeline",
                "validated_by": ["deterministic_oracle", "qa_quality_judge"],
                "human_reviewed": False,
            },
        },
    )
    return output


def export_opd_mm_store(dataset_root: str | Path, output_dir: str | Path, *, dataset_name: str) -> dict[str, Any]:
    """Write record/QA tables without building hardware-specific vector indexes."""

    from verl.experimental.opd_mm.mem_gallery import (
        load_mem_gallery_qas,
        load_mem_gallery_records,
        memory_records_to_jsonl,
        qas_to_jsonl,
    )

    root = Path(dataset_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records = load_mem_gallery_records(root, dataset_name=dataset_name)
    qas = load_mem_gallery_qas(root, dataset_name=dataset_name)
    memory_records_to_jsonl(records, output / "records.jsonl")
    qas_to_jsonl(qas, output / "qas.jsonl")
    parquet_written = False
    parquet_error = None
    try:
        import pandas as pd

        pd.DataFrame([record.to_dict(include_internal_id=True) for record in records]).to_parquet(
            output / "records.parquet", index=False
        )
        pd.DataFrame(qas).to_parquet(output / "qas.parquet", index=False)
        parquet_written = True
    except Exception as exc:  # pragma: no cover - depends on optional parquet stack
        parquet_error = f"{type(exc).__name__}: {exc}"
    manifest = {
        "dataset": dataset_name,
        "dataset_root": str(root.resolve()),
        "schema_version": "mmem-v2.0",
        "evidence_unit": "event",
        "record_count": len(records),
        "qa_count": len(qas),
        "parquet_written": parquet_written,
        "parquet_error": parquet_error,
        "indexes": {},
    }
    _write_json(output / "manifest.json", manifest)
    return manifest
