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

"""Mem-Gallery conversion helpers for the OPD-MM memory store."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from verl.experimental.opd_mm.models import MemoryRecord

MEM_GALLERY_DATASET = "mem_gallery"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def _load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _safe_id(value: Any) -> str:
    return str(value or "").replace("/", "_").replace(":", "_").replace(" ", "_")


def _scenario_turn_id(scenario: str, round_id: str) -> str:
    return f"{scenario}:{round_id}"


def _resolve_image_path(dataset_root: Path, relative_path: str | None) -> str | None:
    if not relative_path:
        return None
    value = str(relative_path)
    if value.startswith("../image/"):
        value = value[len("../image/") :]
    path = dataset_root / "data" / "image" / value
    return str(path.resolve())


def _dialogue_text(round_data: dict[str, Any]) -> str:
    user_text = str(round_data.get("user") or "").strip()
    assistant_text = str(round_data.get("assistant") or "").strip()
    chunks = []
    if user_text:
        chunks.append(f"User: {user_text}")
    if assistant_text:
        chunks.append(f"Assistant: {assistant_text}")
    return "\n".join(chunks)


def _summary(text: str, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _base_metadata(
    *,
    scenario: str,
    scenario_file: Path,
    profile: dict[str, Any],
    session: dict[str, Any],
    round_data: dict[str, Any],
    turn_index: int,
) -> dict[str, Any]:
    round_id = str(round_data.get("round") or "")
    session_id = str(session.get("session_id") or "")
    return {
        "dataset": MEM_GALLERY_DATASET,
        "scenario": scenario,
        "scenario_file": str(scenario_file),
        "character_profile": profile,
        "session_id": session_id,
        "session_date": str(session.get("date") or ""),
        "round_id": round_id,
        "local_turn_id": round_id,
        "turn_index": turn_index,
        "user": str(round_data.get("user") or ""),
        "assistant": str(round_data.get("assistant") or ""),
    }


def load_mem_gallery_records(dataset_root: str | Path) -> list[MemoryRecord]:
    """Convert Mem-Gallery dialogue rounds/images into OPD-MM memory records."""
    root = Path(dataset_root)
    records: list[MemoryRecord] = []
    for scenario_file in sorted((root / "data" / "dialog").glob("*.json")):
        scenario = scenario_file.stem
        value = _load_json(scenario_file)
        profile = dict(value.get("character_profile") or {})
        for session in value.get("multi_session_dialogues", []):
            for turn_index, round_data in enumerate(
                session.get("dialogues", []),
                start=1,
            ):
                round_id = str(round_data.get("round") or f"round_{turn_index}")
                turn_id = _scenario_turn_id(scenario, round_id)
                timestamp = f"{session.get('date', '')}T{turn_index:04d}"
                metadata = _base_metadata(
                    scenario=scenario,
                    scenario_file=scenario_file,
                    profile=profile,
                    session=session,
                    round_data=round_data,
                    turn_index=turn_index,
                )
                image_ids = list(round_data.get("image_id") or [])
                image_paths = list(round_data.get("input_image") or [])
                image_captions = list(round_data.get("image_caption") or [])
                metadata.update(
                    {
                        "image_ids": image_ids,
                        "input_images": image_paths,
                        "image_captions": image_captions,
                    }
                )

                content = _dialogue_text(round_data)
                if content:
                    records.append(
                        MemoryRecord(
                            memory_id=(
                                f"{MEM_GALLERY_DATASET}:{_safe_id(scenario)}:"
                                f"{_safe_id(round_id)}:text"
                            ),
                            turn_id=turn_id,
                            timestamp=timestamp,
                            author="dialogue",
                            modality="text",
                            source_type="dialogue_turn",
                            summary=_summary(content),
                            content=content,
                            metadata=dict(metadata),
                        )
                    )

                for image_index, relative_image in enumerate(image_paths):
                    image_id = (
                        str(image_ids[image_index])
                        if image_index < len(image_ids)
                        else f"{round_id}:IMG_{image_index + 1:03d}"
                    )
                    caption = (
                        str(image_captions[image_index])
                        if image_index < len(image_captions)
                        else ""
                    )
                    image_metadata = dict(metadata)
                    image_metadata.update(
                        {
                            "image_index": image_index,
                            "image_id": image_id,
                            "relative_image_path": relative_image,
                        }
                    )
                    records.append(
                        MemoryRecord(
                            memory_id=(
                                f"{MEM_GALLERY_DATASET}:{_safe_id(scenario)}:"
                                f"{_safe_id(image_id)}:image"
                            ),
                            turn_id=turn_id,
                            timestamp=timestamp,
                            author="user",
                            modality="image",
                            source_type="dialogue_image",
                            summary=_summary(caption),
                            content=caption,
                            raw_pointer=_resolve_image_path(root, relative_image),
                            metadata=image_metadata,
                        )
                    )
    return records


def load_mem_gallery_qas(dataset_root: str | Path) -> list[dict[str, Any]]:
    """Load Mem-Gallery human annotated QAs with resolved support metadata."""
    root = Path(dataset_root)
    qas: list[dict[str, Any]] = []
    for scenario_file in sorted((root / "data" / "dialog").glob("*.json")):
        scenario = scenario_file.stem
        value = _load_json(scenario_file)
        for index, qa in enumerate(value.get("human-annotated QAs", [])):
            clues = list(qa.get("clue") or [])
            question_image = _resolve_image_path(root, qa.get("question_image"))
            qas.append(
                {
                    "sample_id": (
                        f"{MEM_GALLERY_DATASET}:{_safe_id(scenario)}:"
                        f"qa:{index:04d}"
                    ),
                    "dataset": MEM_GALLERY_DATASET,
                    "scenario": scenario,
                    "scenario_file": str(scenario_file),
                    "qa_index": index,
                    "point": qa.get("point"),
                    "question": qa.get("question"),
                    "answer": qa.get("answer"),
                    "gold_answer": qa.get("answer"),
                    "session_id": list(qa.get("session_id") or []),
                    "clue": clues,
                    "support_turn_ids": [
                        _scenario_turn_id(scenario, str(clue))
                        for clue in clues
                    ],
                    "question_image": question_image,
                    "question_image_relative": qa.get("question_image"),
                    "raw_qa": qa,
                }
            )
    return qas


def memory_records_to_jsonl(records: Iterable[MemoryRecord], path: str | Path) -> Path:
    """Write MemoryRecord objects as JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(
                json.dumps(record.to_dict(include_internal_id=True), ensure_ascii=False)
                + "\n"
            )
    return output


def qas_to_jsonl(qas: Iterable[dict[str, Any]], path: str | Path) -> Path:
    """Write Mem-Gallery QA dictionaries as JSONL."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as handle:
        for qa in qas:
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")
    return output
