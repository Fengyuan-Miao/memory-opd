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

"""STARK conversion helpers for the OPD-MM memory store."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from verl.experimental.opd_mm.models import MemoryRecord

STARK_DATASET = "stark"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
SESSION_COUNT = 6
SESSION_LIST_FIELDS = (
    "speakers",
    "utterances",
    "rationales",
    "image_descriptions",
    "image_sources",
    "keywords",
    "image_id_from_mobile",
    "images_key",
    "images_module_name",
)
PERSONA_FIELDS = (
    "name",
    "age",
    "gender",
    "birthplace",
    "residence",
    "human_face_description",
    "human_face_image_key",
    "persona_category",
    "persona_sentence",
    "persona_entity_key",
    "persona_entity_value",
    "persona_commonsense_relation",
    "persona_commonsense_inference",
)
_UNICODE_SURROGATE_PAIR = re.compile(
    r"\\u([dD][89aAbB][0-9a-fA-F]{2})\\u([dD][c-fC-F][0-9a-fA-F]{2})"
)
_UNICODE_ESCAPE = re.compile(r"\\u([0-9a-fA-F]{4})")
_DATE_PREFIX = re.compile(r"^(\d{4})[./-](\d{2})[./-](\d{2})")


def _decode_unicode_escapes(value: Any) -> str:
    """Decode STARK's occasionally double-escaped ``\\uXXXX`` strings."""

    text = str(value or "")

    def replace_pair(match: re.Match[str]) -> str:
        high = int(match.group(1), 16)
        low = int(match.group(2), 16)
        return chr(0x10000 + ((high - 0xD800) << 10) + (low - 0xDC00))

    text = _UNICODE_SURROGATE_PAIR.sub(replace_pair, text)
    return _UNICODE_ESCAPE.sub(lambda match: chr(int(match.group(1), 16)), text)


def _json_list(value: Any) -> list[Any]:
    if value in (None, ""):
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return parsed if isinstance(parsed, list) else []


def _safe_id(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "")).strip("_") or "unknown"


def _summary(text: str, limit: int = 220) -> str:
    value = " ".join(str(text or "").split())
    if len(value) <= limit:
        return value
    return value[: limit - 1].rstrip() + "…"


def _normalize_date(value: Any) -> str:
    text = str(value or "").strip()
    match = _DATE_PREFIX.match(text)
    if match:
        return "-".join(match.groups())
    return text.replace(".", "-")


def _value_at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else ""


def _flatten_image_keys(value: Any) -> list[str]:
    if value in (None, ""):
        return []
    if isinstance(value, str):
        return [value]
    if not isinstance(value, list):
        return []
    flattened: list[str] = []
    for item in value:
        flattened.extend(_flatten_image_keys(item))
    return flattened


def _normalize_string_list(value: Any) -> list[str]:
    """Normalize STARK fields that alternate between a string and a list."""

    if value in (None, ""):
        return []
    if isinstance(value, list):
        normalized: list[str] = []
        for item in value:
            normalized.extend(_normalize_string_list(item))
        return normalized
    text = _decode_unicode_escapes(value).strip()
    return [text] if text else []


def _image_lookup_key(value: Any) -> str:
    key = str(value or "")
    for prefix in ("url:", "face:"):
        if key.startswith(prefix):
            return key[len(prefix) :]
    return key


def discover_local_images(image_root: str | Path) -> dict[str, Path]:
    """Return STARK image keys mapped to locally downloaded image files."""

    root = Path(image_root)
    if not root.exists():
        return {}
    return {
        path.stem: path.resolve()
        for path in sorted(root.iterdir())
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    }


def _select_image(
    candidates: list[str],
    local_images: Mapping[str, Path],
    *,
    max_image_rank: int = 4,
) -> tuple[str, int | None, Path | None]:
    for rank, candidate in enumerate(candidates[: max_image_rank + 1]):
        path = local_images.get(_image_lookup_key(candidate))
        if path is not None:
            return candidate, rank, path
    if candidates:
        return candidates[0], 0, None
    return "", None, None


def _speaker_role(speaker: str) -> str:
    normalized = speaker.strip().casefold()
    if normalized in {"ai", "ai assistant", "assistant", "system"}:
        return "assistant"
    return "user"


def _turn_content(*, role: str, utterance: str, has_image: bool) -> str:
    label = "Assistant" if role == "assistant" else "User"
    if has_image:
        chunks = [f"{label}: {utterance}"] if utterance else []
        chunks.append(f"{label} shared an image.")
        return "\n".join(chunks)
    if utterance:
        return f"{label}: {utterance}"
    return ""


def _searchable_summary(*, content: str, image_description: str) -> str:
    """Keep private image semantics searchable without leaking them publicly."""

    chunks = [value for value in (content, image_description) if value]
    return _summary("\n".join(chunks))


def _image_public_content(role: str) -> str:
    """Represent an image record without cloning its linked dialogue record."""

    label = "Assistant" if role == "assistant" else "User"
    return f"{label} shared an image."


def stark_row_has_local_image(
    row: Mapping[str, Any],
    local_images: Mapping[str, Path],
    *,
    max_image_rank: int = 4,
) -> bool:
    """Whether an episode references at least one locally available image."""

    if not local_images:
        return False
    session_count = min(int(row.get("number_of_session") or 0), SESSION_COUNT)
    for session_index in range(1, session_count + 1):
        for turn_value in _json_list(row.get(f"session{session_index}:images_key")):
            for candidate in _flatten_image_keys(turn_value)[: max_image_rank + 1]:
                if _image_lookup_key(candidate) in local_images:
                    return True
    return False


def _group_dialogue_rounds(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group STARK messages into Mem-Gallery-style dialogue rounds.

    STARK serializes text and image-only messages in one chronological list.
    Mem-Gallery instead indexes a user contribution together with the assistant
    messages that answer it. A session may begin with an assistant prompt, so
    leading assistant messages belong to the first user-anchored round.
    """

    rounds: list[list[dict[str, Any]]] = []
    leading_assistant: list[dict[str, Any]] = []
    current: list[dict[str, Any]] = []
    current_has_user = False
    for message in messages:
        if message["role"] == "user":
            if current_has_user:
                rounds.append(current)
                current = []
            if not current:
                current.extend(leading_assistant)
                leading_assistant = []
            current.append(message)
            current_has_user = True
        elif current_has_user:
            current.append(message)
        else:
            leading_assistant.append(message)
    if current:
        rounds.append(current)
    elif leading_assistant:
        rounds.append(leading_assistant)
    return rounds


def iter_stark_row_records(
    row: Mapping[str, Any],
    *,
    source_row_index: int,
    source_path: str | Path,
    local_images: Mapping[str, Path] | None = None,
    max_image_rank: int = 4,
) -> Iterator[MemoryRecord]:
    """Convert one STARK episode into dialogue rounds with linked images."""

    local_images = local_images or {}
    conversation_id = _decode_unicode_escapes(row.get("index"))
    scenario = _safe_id(conversation_id)
    persona = {
        field: _decode_unicode_escapes(row.get(field))
        for field in PERSONA_FIELDS
        if row.get(field) not in (None, "")
    }
    session_count = min(int(row.get("number_of_session") or 0), SESSION_COUNT)

    for session_index in range(1, session_count + 1):
        prefix = f"session{session_index}:"
        values = {
            field: _json_list(row.get(prefix + field))
            for field in SESSION_LIST_FIELDS
        }
        turn_count = max((len(items) for items in values.values()), default=0)
        session_id = f"S{session_index:02d}"
        session_date = _normalize_date(row.get(prefix + "date"))

        messages: list[dict[str, Any]] = []
        for offset in range(turn_count):
            turn_index = offset + 1
            speaker = _decode_unicode_escapes(_value_at(values["speakers"], offset)).strip()
            utterance = _decode_unicode_escapes(_value_at(values["utterances"], offset)).strip()
            image_description = _decode_unicode_escapes(
                _value_at(values["image_descriptions"], offset)
            ).strip()
            candidates = _flatten_image_keys(_value_at(values["images_key"], offset))
            image_key, image_rank, image_path = _select_image(
                candidates,
                local_images,
                max_image_rank=max_image_rank,
            )
            has_image = bool(image_description or candidates)
            modality = "image" if has_image else "text"
            role = _speaker_role(speaker)
            content = _turn_content(
                role=role,
                utterance=utterance,
                has_image=has_image,
            )
            local_turn_id = f"{session_id}:T{turn_index:04d}"
            image_id = image_key or (f"{local_turn_id}:IMG_001" if has_image else "")
            image_source = _decode_unicode_escapes(
                _value_at(values["image_sources"], offset)
            ).strip()
            image_module = _decode_unicode_escapes(
                _value_at(values["images_module_name"], offset)
            ).strip()
            rationale = _decode_unicode_escapes(_value_at(values["rationales"], offset)).strip()
            mobile_image_id = _decode_unicode_escapes(
                _value_at(values["image_id_from_mobile"], offset)
            ).strip()
            keywords = _normalize_string_list(_value_at(values["keywords"], offset))
            source_memory_id = (
                f"{STARK_DATASET}:{scenario}:{session_id}_T{turn_index:04d}:{modality}"
            )
            messages.append(
                {
                    "source_turn_index": turn_index,
                    "source_local_turn_id": local_turn_id,
                    "source_memory_id": source_memory_id,
                    "speaker": speaker,
                    "role": role,
                    "utterance": utterance,
                    "content": content,
                    "has_image": has_image,
                    "image_id": image_id,
                    "image_key": image_key,
                    "image_key_rank": image_rank,
                    "image_path": image_path,
                    "image_description": image_description,
                    "image_source": image_source,
                    "image_module_name": image_module,
                    "image_id_from_mobile": mobile_image_id,
                    "image_keywords": keywords,
                    "image_rationale": rationale,
                }
            )

        for round_index, messages_in_round in enumerate(
            _group_dialogue_rounds(messages), start=1
        ):
            local_round_id = f"{session_id}:R{round_index:04d}"
            turn_id = f"{scenario}:{local_round_id}"
            timestamp = (
                f"{session_date}R{round_index:04d}" if session_date else local_round_id
            )
            round_content = "\n".join(
                message["content"] for message in messages_in_round if message["content"]
            )
            image_messages = [message for message in messages_in_round if message["has_image"]]
            user_messages = [message for message in messages_in_round if message["role"] == "user"]
            common_metadata: dict[str, Any] = {
                "dataset": STARK_DATASET,
                "scenario": conversation_id,
                "scenario_file": str(source_path),
                "source_row_index": source_row_index,
                "character_profile": persona,
                "session_id": session_id,
                "session_date": session_date,
                "session_event": _decode_unicode_escapes(row.get(prefix + "event")),
                "session_experience": _decode_unicode_escapes(row.get(prefix + "experience")),
                "round_id": local_round_id,
                "local_turn_id": local_round_id,
                "turn_index": round_index,
                "source_turn_indices": [message["source_turn_index"] for message in messages_in_round],
                "source_local_turn_ids": [message["source_local_turn_id"] for message in messages_in_round],
                "source_memory_ids": [message["source_memory_id"] for message in messages_in_round],
                "speakers": [message["speaker"] for message in messages_in_round],
                "roles": [message["role"] for message in messages_in_round],
                "utterances": [message["utterance"] for message in messages_in_round],
                "speaker": user_messages[0]["speaker"] if user_messages else messages_in_round[0]["speaker"],
                "role": "user" if user_messages else "assistant",
                "utterance": "\n".join(
                    message["utterance"] for message in user_messages if message["utterance"]
                ),
                "image_ids": [message["image_id"] for message in image_messages],
                "input_images": [
                    str(message["image_path"])
                    for message in image_messages
                    if message["image_path"] is not None
                ],
                "image_captions": [message["image_description"] for message in image_messages],
            }
            yield MemoryRecord(
                memory_id=f"{STARK_DATASET}:{scenario}:{session_id}_R{round_index:04d}:text",
                turn_id=turn_id,
                timestamp=timestamp,
                author="dialogue",
                modality="text",
                source_type="dialogue_turn",
                summary=_summary(round_content),
                content=round_content,
                metadata=dict(common_metadata),
            )

            for image_index, message in enumerate(image_messages, start=1):
                metadata = dict(common_metadata)
                metadata.update(
                    {
                        "source_memory_id": message["source_memory_id"],
                        "source_local_turn_id": message["source_local_turn_id"],
                        "source_turn_index": message["source_turn_index"],
                        "speaker": message["speaker"],
                        "role": message["role"],
                        "utterance": message["utterance"],
                        "image_index": image_index - 1,
                        "image_id": message["image_id"],
                        "image_key": message["image_key"],
                        "image_key_rank": message["image_key_rank"],
                        "image_available": message["image_path"] is not None,
                        "image_description": message["image_description"],
                        "image_source": message["image_source"],
                        "image_module_name": message["image_module_name"],
                        "image_id_from_mobile": message["image_id_from_mobile"],
                        "image_keywords": message["image_keywords"],
                        "image_rationale": message["image_rationale"],
                        "mixed_text_image_turn": bool(message["utterance"]),
                    }
                )
                yield MemoryRecord(
                    memory_id=(
                        f"{STARK_DATASET}:{scenario}:{session_id}_R{round_index:04d}"
                        f"_IMG_{image_index:03d}:image"
                    ),
                    turn_id=turn_id,
                    timestamp=timestamp,
                    author=message["role"],
                    modality="image",
                    source_type="dialogue_image",
                    summary=_searchable_summary(
                        content=round_content,
                        image_description=message["image_description"],
                    ),
                    content=_image_public_content(message["role"]),
                    raw_pointer=(
                        str(message["image_path"])
                        if message["image_path"] is not None
                        else None
                    ),
                    metadata=metadata,
                )


def _parquet_columns() -> list[str]:
    columns = ["index", "number_of_session", *PERSONA_FIELDS]
    for session_index in range(1, SESSION_COUNT + 1):
        prefix = f"session{session_index}:"
        columns.extend(prefix + field for field in ("date", "event", "experience"))
        columns.extend(prefix + field for field in SESSION_LIST_FIELDS)
    return list(dict.fromkeys(columns))


def _iter_parquet_rows(path: Path, *, batch_size: int) -> Iterator[tuple[int, dict[str, Any]]]:
    import pyarrow.parquet as pq

    parquet = pq.ParquetFile(path)
    row_index = 0
    for batch in parquet.iter_batches(columns=_parquet_columns(), batch_size=batch_size):
        columns = batch.to_pydict()
        for offset in range(batch.num_rows):
            yield row_index, {name: values[offset] for name, values in columns.items()}
            row_index += 1


def build_stark_memory_store(
    *,
    dialogue_parquet: str | Path,
    image_root: str | Path,
    output_dir: str | Path,
    selection: str = "local_image_overlap",
    max_episodes: int | None = None,
    batch_size: int = 128,
    max_image_rank: int = 4,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a streaming STARK ``records.jsonl`` compatible with Mem-Gallery."""

    if selection not in {"local_image_overlap", "all"}:
        raise ValueError(f"Unsupported selection: {selection}")
    source = Path(dialogue_parquet)
    images = Path(image_root)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    records_path = output / "records.jsonl"
    episode_ids_path = output / "episode_ids.txt"
    manifest_path = output / "manifest.json"
    existing = [path for path in (records_path, episode_ids_path, manifest_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"Output already exists ({', '.join(str(path) for path in existing)}); pass --overwrite to replace it."
        )

    local_images = discover_local_images(images)
    counts: Counter[str] = Counter()
    selected_ids: list[str] = []
    records_tmp = records_path.with_suffix(".jsonl.tmp")
    try:
        with records_tmp.open("w", encoding="utf-8") as handle:
            for source_row_index, row in _iter_parquet_rows(source, batch_size=batch_size):
                counts["source_episode_count"] += 1
                if selection == "local_image_overlap" and not stark_row_has_local_image(
                    row,
                    local_images,
                    max_image_rank=max_image_rank,
                ):
                    continue
                conversation_id = _decode_unicode_escapes(row.get("index"))
                selected_ids.append(conversation_id)
                counts["episode_count"] += 1
                for record in iter_stark_row_records(
                    row,
                    source_row_index=source_row_index,
                    source_path=source,
                    local_images=local_images,
                    max_image_rank=max_image_rank,
                ):
                    handle.write(json.dumps(record.to_dict(include_internal_id=True), ensure_ascii=False) + "\n")
                    counts["record_count"] += 1
                    counts[f"{record.modality}_record_count"] += 1
                    if not record.content:
                        counts["empty_content_record_count"] += 1
                    if record.modality == "image":
                        if record.raw_pointer:
                            counts["local_image_record_count"] += 1
                        else:
                            counts["missing_image_record_count"] += 1
                        if record.metadata.get("mixed_text_image_turn"):
                            counts["mixed_text_image_record_count"] += 1
                if max_episodes is not None and counts["episode_count"] >= max_episodes:
                    break
        records_tmp.replace(records_path)
    except BaseException:
        records_tmp.unlink(missing_ok=True)
        raise

    episode_ids_path.write_text("".join(f"{value}\n" for value in selected_ids), encoding="utf-8")
    manifest = {
        "dataset": STARK_DATASET,
        "dialogue_parquet": str(source.resolve()),
        "image_root": str(images.resolve()),
        "output_dir": str(output.resolve()),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "conversion": "one_dialogue_turn_with_linked_image_stubs",
        "record_schema_version": 2,
        "image_record_public_content": "share_marker",
        "image_description_visibility": "private_search_summary",
        "selection": selection,
        "max_episodes": max_episodes,
        "max_image_rank": max_image_rank,
        "local_image_file_count": len(local_images),
        **dict(counts),
        "records_path": str(records_path.resolve()),
        "episode_ids_path": str(episode_ids_path.resolve()),
        "indexes": {},
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dialogue-parquet", default="dataset/Stark/dialogue/stark.parquet")
    parser.add_argument("--image-root", default="dataset/Stark/images_sample")
    parser.add_argument("--output-dir", default="dataset/Stark/opd_mm_store")
    parser.add_argument(
        "--selection",
        choices=("local_image_overlap", "all"),
        default="local_image_overlap",
        help="By default, retain complete episodes that reference at least one locally downloaded image.",
    )
    parser.add_argument("--max-episodes", type=int)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--max-image-rank",
        type=int,
        default=4,
        help="Only materialize a local image when it is within the top-ranked candidates (zero based).",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Iterable[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    manifest = build_stark_memory_store(
        dialogue_parquet=args.dialogue_parquet,
        image_root=args.image_root,
        output_dir=args.output_dir,
        selection=args.selection,
        max_episodes=args.max_episodes,
        batch_size=args.batch_size,
        max_image_rank=args.max_image_rank,
        overwrite=args.overwrite,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
