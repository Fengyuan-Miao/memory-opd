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

"""Build grounded Mem-Gallery-style QA data from normalized STARK memories."""

from __future__ import annotations

import hashlib
import json
import random
import re
import threading
import time
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import requests

from verl.experimental.opd_mm.models import MemoryRecord, OPDSample
from verl.experimental.opd_mm.retrieval import HiddenMemoryStore
from verl.experimental.opd_mm.tools import memory_record_from_dict

POINT_ORDER = {"FR": 0, "TR": 1, "VS": 2}
DIRECT_POINT_ORDER = {
    "FR": 0,
    "VS": 1,
    "TTL": 2,
    "TR": 3,
    "VR": 4,
    "MR": 5,
    "KR": 6,
    "CD": 7,
    "AR": 8,
}
DIRECT_POINTS = frozenset(DIRECT_POINT_ORDER)
DIRECT_ANSWER_WORD_LIMITS = {
    "FR": 40,
    "TTL": 8,
    "TR": 16,
    "VR": 20,
    "MR": 40,
    "KR": 30,
}
SPLIT_ORDER = ("train", "validation", "test")
_WORD = re.compile(r"[A-Za-z0-9][A-Za-z0-9'-]*")
_SPACE = re.compile(r"\s+")
_CODE_FENCE = re.compile(r"^```(?:json)?\s*|\s*```$", re.IGNORECASE)
_AVAILABLE_WORD = re.compile(r"\bavailable\b\s*", re.IGNORECASE)
_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "had",
    "has",
    "have",
    "he",
    "her",
    "his",
    "i",
    "in",
    "is",
    "it",
    "my",
    "of",
    "on",
    "she",
    "that",
    "the",
    "their",
    "they",
    "to",
    "was",
    "we",
    "were",
    "with",
    "you",
    "your",
}


@dataclass(frozen=True)
class GeneratedBundle:
    conversation_id: str
    qas: list[dict[str, Any]]
    usage: dict[str, int]
    raw_response: str


def _normalize_text(value: Any) -> str:
    return _SPACE.sub(" ", str(value or "")).strip()


def _answer_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").casefold()).strip()


def _safe_id(value: Any) -> str:
    return re.sub(r"[^0-9A-Za-z_.-]+", "_", str(value or "")).strip("_") or "unknown"


def _tokens(value: Any) -> set[str]:
    return {
        token.casefold()
        for token in _WORD.findall(str(value or ""))
        if token.casefold() not in _STOPWORDS and len(token) > 2
    }


def _stable_noise(*values: Any) -> float:
    payload = "\x1f".join(str(value) for value in values).encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:12], 16) / float(16**12)


def load_records_by_episode(path: str | Path) -> dict[str, list[MemoryRecord]]:
    """Load normalized STARK records grouped by original episode ID."""

    grouped: dict[str, list[MemoryRecord]] = defaultdict(list)
    with Path(path).open(encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            value = json.loads(line)
            record = memory_record_from_dict(value, index)
            scenario = str(record.metadata.get("scenario") or "")
            if not scenario:
                raise ValueError(f"STARK memory {record.memory_id} has no scenario")
            grouped[scenario].append(record)
    for records in grouped.values():
        records.sort(key=lambda record: (record.timestamp, record.turn_id, record.memory_id))
    return dict(grouped)


def split_episode_ids(
    episode_ids: Iterable[str],
    *,
    seed: int = 20260722,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> dict[str, list[str]]:
    """Split before QA generation so no persona or image crosses splits."""

    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("invalid split ratios")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below one")
    values = sorted(set(str(value) for value in episode_ids))
    random.Random(seed).shuffle(values)
    train_end = int(len(values) * train_ratio)
    validation_end = train_end + int(len(values) * validation_ratio)
    return {
        "train": sorted(values[:train_end]),
        "validation": sorted(values[train_end:validation_end]),
        "test": sorted(values[validation_end:]),
    }


def split_episode_records(
    records_by_episode: Mapping[str, Sequence[MemoryRecord]],
    *,
    seed: int = 20260722,
    train_ratio: float = 0.8,
    validation_ratio: float = 0.1,
) -> dict[str, list[str]]:
    """Group-split episodes so the same locally available image cannot leak."""

    if not 0 < train_ratio < 1 or not 0 <= validation_ratio < 1:
        raise ValueError("invalid split ratios")
    if train_ratio + validation_ratio >= 1:
        raise ValueError("train_ratio + validation_ratio must be below one")
    episode_ids = sorted(str(value) for value in records_by_episode)
    parent = {episode_id: episode_id for episode_id in episode_ids}

    def find(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    def union(left: str, right: str) -> None:
        left_root, right_root = find(left), find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    episodes_by_image: dict[str, list[str]] = defaultdict(list)
    for episode_id, records in records_by_episode.items():
        image_ids = {
            str(record.metadata.get("image_id"))
            for record in records
            if record.raw_pointer and record.metadata.get("image_id")
        }
        for image_id in image_ids:
            episodes_by_image[image_id].append(episode_id)
    for image_episodes in episodes_by_image.values():
        first = image_episodes[0]
        for episode_id in image_episodes[1:]:
            union(first, episode_id)

    components: dict[str, list[str]] = defaultdict(list)
    for episode_id in episode_ids:
        components[find(episode_id)].append(episode_id)
    rng = random.Random(seed)
    groups = [sorted(group) for group in components.values()]
    rng.shuffle(groups)
    groups.sort(key=len, reverse=True)

    total = len(episode_ids)
    targets = {
        "train": int(total * train_ratio),
        "validation": int(total * validation_ratio),
    }
    targets["test"] = total - targets["train"] - targets["validation"]
    assigned = {split: [] for split in SPLIT_ORDER}
    for group in groups:
        remaining = {
            split: targets[split] - len(assigned[split])
            for split in SPLIT_ORDER
        }
        fitting = [split for split in SPLIT_ORDER if remaining[split] >= len(group)]
        choices = fitting or list(SPLIT_ORDER)
        split = max(
            choices,
            key=lambda name: (remaining[name], targets[name], -SPLIT_ORDER.index(name)),
        )
        assigned[split].extend(group)
    return {split: sorted(assigned[split]) for split in SPLIT_ORDER}


def _record_payload(record: MemoryRecord, *, private_image_description: bool = False) -> dict[str, Any]:
    value = {
        "memory_id": record.memory_id,
        "turn_id": record.turn_id,
        "session_id": record.metadata.get("session_id"),
        "date": record.metadata.get("session_date"),
        "speaker": record.metadata.get("speaker"),
        "role": record.metadata.get("role"),
        "content": record.content,
    }
    if private_image_description:
        value["image_id"] = record.metadata.get("image_id")
        value["image_description_private"] = record.metadata.get("image_description")
    return value


def build_episode_graph(conversation_id: str, records: Sequence[MemoryRecord]) -> dict[str, Any]:
    """Create a compact persona/session/turn graph with provenance."""

    if not records:
        raise ValueError(f"empty STARK episode: {conversation_id}")
    profile = dict(records[0].metadata.get("character_profile") or {})
    sessions: dict[str, dict[str, Any]] = {}
    for record in records:
        session_id = str(record.metadata.get("session_id") or "")
        session = sessions.setdefault(
            session_id,
            {
                "session_id": session_id,
                "date": record.metadata.get("session_date"),
                "event": record.metadata.get("session_event"),
                "experience": record.metadata.get("session_experience"),
                "turns": [],
            },
        )
        session["turns"].append(
            {
                **_record_payload(record, private_image_description=record.modality == "image"),
                "modality": record.modality,
                "image_available": bool(record.raw_pointer),
            }
        )
    return {
        "conversation_id": conversation_id,
        "profile": profile,
        "sessions": [sessions[key] for key in sorted(sessions)],
    }


def _fr_candidates(conversation_id: str, records: Sequence[MemoryRecord], limit: int = 8) -> list[MemoryRecord]:
    by_session: dict[str, list[tuple[float, MemoryRecord]]] = defaultdict(list)
    for record in records:
        if record.modality != "text" or record.metadata.get("role") != "user":
            continue
        utterance = _normalize_text(record.metadata.get("utterance") or record.content)
        words = _WORD.findall(utterance)
        if len(words) < 8 or len(words) > 90:
            continue
        score = min(len(words), 45) / 45
        score += 0.5 if "?" not in utterance else 0.0
        score += 0.35 if re.search(r"\b(I|my|we|our)\b", utterance, re.IGNORECASE) else 0.0
        score += 0.1 * _stable_noise(conversation_id, record.memory_id)
        by_session[str(record.metadata.get("session_id") or "")].append((score, record))

    selected = []
    for session_id in sorted(by_session):
        ranked = sorted(by_session[session_id], key=lambda item: (-item[0], item[1].memory_id))
        selected.extend(record for _, record in ranked[:2])
    return sorted(
        selected,
        key=lambda record: (
            str(record.metadata.get("session_id") or ""),
            int(record.metadata.get("turn_index") or 0),
        ),
    )[:limit]


def _event_support(event: str, records: Sequence[MemoryRecord], limit: int = 3) -> list[MemoryRecord]:
    event_tokens = _tokens(event)
    ranked = []
    for record in records:
        if record.modality != "text":
            continue
        content_tokens = _tokens(record.content)
        overlap = len(event_tokens & content_tokens)
        ratio = overlap / max(1, len(event_tokens))
        role_bonus = 0.05 if record.metadata.get("role") == "user" else 0.0
        ranked.append((ratio + role_bonus, overlap, record))
    ranked.sort(key=lambda item: (-item[0], -item[1], item[2].memory_id))
    return [record for score, overlap, record in ranked[:limit] if overlap >= 2 or score >= 0.25]


def _tr_target(conversation_id: str, records: Sequence[MemoryRecord]) -> dict[str, Any] | None:
    by_session: dict[str, list[MemoryRecord]] = defaultdict(list)
    for record in records:
        by_session[str(record.metadata.get("session_id") or "")].append(record)
    candidates = []
    for session_id, session_records in sorted(by_session.items()):
        first = session_records[0]
        date = str(first.metadata.get("session_date") or "")
        event = _normalize_text(first.metadata.get("session_event"))
        support = _event_support(event, session_records)
        if date and event and support:
            candidates.append((session_id, date, event, support))
    if len(candidates) < 2:
        return None
    left, right = candidates[0], candidates[-1]
    if left[1] == right[1]:
        return None
    if _stable_noise(conversation_id, "tr_order") >= 0.5:
        left, right = right, left
    relation = "before" if left[1] < right[1] else "after"
    return {
        "target_id": "tr",
        "point": "TR",
        "event_a": left[2],
        "date_a": left[1],
        "event_b": right[2],
        "date_b": right[1],
        "fixed_answer": relation,
        "support_a": [_record_payload(record) for record in left[3]],
        "support_b": [_record_payload(record) for record in right[3]],
    }


def _vs_target(records: Sequence[MemoryRecord]) -> dict[str, Any] | None:
    images = [
        record
        for record in records
        if record.modality == "image" and record.raw_pointer and record.metadata.get("image_id")
    ]
    if not images:
        return None
    images.sort(key=lambda record: (_stable_noise(record.turn_id), record.memory_id))
    image = images[0]
    session_id = str(image.metadata.get("session_id") or "")
    turn_index = int(image.metadata.get("turn_index") or 0)
    context = [
        record
        for record in records
        if record.modality == "text"
        and str(record.metadata.get("session_id") or "") == session_id
        and abs(int(record.metadata.get("turn_index") or 0) - turn_index) <= 2
    ]
    context.sort(key=lambda record: abs(int(record.metadata.get("turn_index") or 0) - turn_index))
    return {
        "target_id": "vs",
        "point": "VS",
        "fixed_answer": str(image.metadata["image_id"]),
        "session_event": image.metadata.get("session_event"),
        "image": _record_payload(image, private_image_description=True),
        "context": [_record_payload(record) for record in context[:3]],
    }


def build_episode_targets(conversation_id: str, records: Sequence[MemoryRecord]) -> dict[str, Any]:
    """Choose gold evidence before asking an LLM to verbalize questions."""

    profile = dict(records[0].metadata.get("character_profile") or {}) if records else {}
    targets: list[dict[str, Any]] = []
    fr = _fr_candidates(conversation_id, records)
    if fr:
        targets.append(
            {
                "target_id": "fr",
                "point": "FR",
                "requested_count": min(2, len(fr)),
                "candidates": [_record_payload(record) for record in fr],
            }
        )
    tr = _tr_target(conversation_id, records)
    if tr is not None:
        targets.append(tr)
    vs = _vs_target(records)
    if vs is not None:
        targets.append(vs)
    return {
        "conversation_id": conversation_id,
        "profile_name": profile.get("name") or "the user",
        "targets": targets,
    }


def write_prepared_expansion(
    *,
    records_path: str | Path,
    output_dir: str | Path,
    seed: int = 20260722,
) -> dict[str, Any]:
    """Write canonical graphs, episode-level splits, and answer-first targets."""

    records_by_episode = load_records_by_episode(records_path)
    splits = split_episode_records(records_by_episode, seed=seed)
    output = Path(output_dir)
    graph_dir = output / "memory_graph"
    target_dir = output / "targets"
    split_dir = output / "splits"
    for directory in (graph_dir, target_dir, split_dir):
        directory.mkdir(parents=True, exist_ok=True)

    split_by_episode = {
        episode_id: split
        for split, episode_ids in splits.items()
        for episode_id in episode_ids
    }
    target_counts: Counter[str] = Counter()
    with (graph_dir / "episodes.jsonl").open("w", encoding="utf-8") as graph_handle:
        target_handles = {
            split: (target_dir / f"{split}.jsonl").open("w", encoding="utf-8")
            for split in SPLIT_ORDER
        }
        try:
            for conversation_id in sorted(records_by_episode):
                records = records_by_episode[conversation_id]
                graph_handle.write(
                    json.dumps(build_episode_graph(conversation_id, records), ensure_ascii=False) + "\n"
                )
                bundle = build_episode_targets(conversation_id, records)
                split = split_by_episode[conversation_id]
                target_handles[split].write(json.dumps(bundle, ensure_ascii=False) + "\n")
                target_counts[f"{split}_episode_count"] += 1
                for target in bundle["targets"]:
                    count = int(target.get("requested_count") or 1)
                    target_counts[f"{split}_{target['point']}_target_count"] += count
        finally:
            for handle in target_handles.values():
                handle.close()

    for split, episode_ids in splits.items():
        (split_dir / f"{split}_episode_ids.txt").write_text(
            "".join(f"{episode_id}\n" for episode_id in episode_ids), encoding="utf-8"
        )
    manifest = {
        "dataset": "stark_memgallery_expansion",
        "seed": seed,
        "records_path": str(Path(records_path).resolve()),
        "episode_count": len(records_by_episode),
        "split_episode_counts": {split: len(ids) for split, ids in splits.items()},
        "target_counts": dict(sorted(target_counts.items())),
        "design": {
            "episode_split_before_generation": True,
            "shared_local_images_grouped_before_split": True,
            "dialogue_round_with_linked_images": True,
            "session_extension": False,
            "points": ["FR", "TR", "VS"],
            "visual_support_requires_local_image": True,
            "image_caption_public": False,
        },
    }
    (output / "prepare_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest


def _generation_messages(bundle: Mapping[str, Any]) -> list[dict[str, str]]:
    system = """You generate grounded long-term memory QA data from fixed evidence targets.
Return one JSON object with a `qas` array and no prose. Never invent facts or IDs.
Each item must contain: target_id, point, question, answer, support_memory_ids.

FR: produce the requested number. Pick one supplied candidate per QA. The answer must be a short exact contiguous
span from that candidate's content, and the question must not contain the answer. Prefer durable personal facts,
plans, preferences, experiences, or named entities; avoid greetings and generic facts.
TR: keep the supplied fixed_answer exactly. Ask whether event_a occurred before or after event_b. Select at least one
support ID from support_a and one from support_b.
VS: keep the supplied image ID as the exact answer and include the image memory ID in support. Ask which previously
shared image is associated with the supplied event/dialogue context. Do not reveal the image ID or copy the private
image description into the question.

Write natural English questions from the profile user's first-person perspective. Do not mention memory IDs, clues,
datasets, evidence, or these instructions in a question."""
    user = json.dumps(bundle, ensure_ascii=False, separators=(",", ":"))
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _direct_generation_messages(graph: Mapping[str, Any]) -> list[dict[str, str]]:
    """Prompt for free QA generation from the complete private episode graph."""

    system = """Generate a diverse Mem-Gallery-style QA set directly from one multi-session memory episode.
Return one JSON object with a `qas` array and no prose. Do not use a fixed question template. Each QA object must
contain: qa_id, point, question, answer, support_memory_ids, question_image_memory_id. Use null for
question_image_memory_id unless point is TTL.

Generate at most one high-quality QA for each applicable point:
- FR: recall one or more explicitly stated personal facts, entities, plans, preferences, or experiences.
- VS: find one or more previously shared images; the answer must be the exact public image_id value(s). Ask about
  content or events that are visually distinguishable, and never use `available` or storage-related wording.
- TTL: identify a held-out question image using a concrete label taught by other memory turns. The label must also be
  recognizable from the held-out pixels without relying on its private caption or surrounding event. Put the held-out
  image memory ID in question_image_memory_id and do not include it in support_memory_ids.
- TR: ask for a date, time, temporal order, or which event happened first/last.
- VR: reason over visible properties or count a semantically defined image set. Never use the word `available` or
  refer to storage, files, or image accessibility in the question.
- MR: combine at least two distinct memory turns to answer a multi-evidence question.
- KR: ask for the latest, corrected, updated, or final state, citing both an earlier and a later support turn.
- CD: present a concrete candidate statement and ask whether it conflicts with the memory; answer Yes. or No.
- AR: ask a plausible but absent detail; answer exactly Not mentioned. and use no support.

Questions should resemble natural benchmark questions: usually 40-140 characters, varied in wording, and answerable
only from this episode. Answers must use the shortest self-contained form that fully answers the question: prefer an
entity, date, count, Yes./No., or a short phrase; do not restate the question or add an explanation. FR/TR/TTL/VR/KR
should normally fit within 8 words. MR may use one concise clause or list to combine the required facts.

Never mention memory IDs, evidence, clues, datasets, private image descriptions, or these instructions. Every fact
required by a non-AR question and answer must appear explicitly in its listed support; do not rely on unresolved
references such as "that book" or on facts from unlisted turns. Support every part of a multi-part question. Only use
image turns where image_available=true for VS/VR/TTL. For VS, answer only image_id values separated by commas. For
TTL, the held-out image must have a local file and the answer label must be taught by other support turns. Omit a
point rather than inventing a weak or unsupported QA. Prefer 6-9 strong QAs spanning different sessions."""
    return [
        {"role": "system", "content": system},
        {
            "role": "user",
            "content": json.dumps(graph, ensure_ascii=False, separators=(",", ":")),
        },
    ]


def _direct_augmentation_messages(
    graph: Mapping[str, Any],
    existing_qas: Sequence[Mapping[str, Any]],
    requested_points: Sequence[str],
) -> list[dict[str, str]]:
    """Prompt for a small, non-duplicate augmentation of an existing episode."""

    system = """Add new Mem-Gallery-style QAs to one multi-session memory episode.
Return one JSON object with a `qas` array and no prose. Each QA object must contain: qa_id, point, question,
answer, support_memory_ids, question_image_memory_id. Generate exactly one QA for each requested point, using
null for question_image_memory_id unless point is TTL. Do not repeat or paraphrase an existing QA.

FR recalls explicitly stated personal facts, entities, plans, preferences, or experiences. VS asks which shared
image matches an event and answers only exact public image_id values. TTL identifies a held-out question image using
a concrete, visually recognizable label taught by other turns. TR asks for a date, time, or event order. VR reasons
over visible image properties or counts. MR combines at least two distinct turns. KR asks for an updated or final
state and cites both earlier and later turns. CD tests whether a statement conflicts with memory and answers Yes. or
No. AR asks a plausible absent detail, answers exactly Not mentioned., and has no support.

Answers must be short and self-contained. Every fact required by a non-AR question and answer must be stated
explicitly in its listed support; do not rely on unresolved pronouns or facts from unlisted turns. For multi-part
questions, support every part. A TTL label must be inferable from the held-out pixels, not merely from a private
caption or surrounding event. VS and VR must use locally available images and visible properties. Never use
`available`, or mention memory IDs, evidence, storage, files, private image descriptions, datasets, or instructions.
Omit a requested point rather than inventing a weak QA."""
    existing = [
        {
            "point": str(qa.get("point") or ""),
            "question": str(qa.get("question") or ""),
            "answer": str(qa.get("answer") or ""),
        }
        for qa in existing_qas
    ]
    user = {
        "requested_points": list(requested_points),
        "existing_qas": existing,
        "episode": graph,
    }
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": json.dumps(user, ensure_ascii=False, separators=(",", ":"))},
    ]


class GPTQAClient:
    """Small resilient client for OpenAI-compatible reasoning-model gateways."""

    def __init__(
        self,
        *,
        api_key: str,
        base_url: str,
        model: str,
        reasoning_effort: str = "low",
        timeout: float = 180,
        max_retries: int = 8,
    ) -> None:
        self.api_key = api_key
        self.url = base_url.rstrip("/") + "/chat/completions"
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.timeout = timeout
        self.max_retries = max_retries
        self._thread_local = threading.local()

    def _session(self) -> requests.Session:
        session = getattr(self._thread_local, "session", None)
        if session is None:
            session = requests.Session()
            session.headers.update(
                {
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "User-Agent": "curl/8.0",
                }
            )
            self._thread_local.session = session
        return session

    def _generate_from_messages(
        self,
        *,
        conversation_id: str,
        messages: Sequence[Mapping[str, str]],
        max_completion_tokens: int,
    ) -> GeneratedBundle:
        payload = {
            "model": self.model,
            "messages": list(messages),
            "reasoning_effort": self.reasoning_effort,
            "max_completion_tokens": max_completion_tokens,
            "response_format": {"type": "json_object"},
        }
        last_error = "request did not run"
        for attempt in range(self.max_retries):
            try:
                response = self._session().post(self.url, json=payload, timeout=self.timeout)
                if response.status_code != 200:
                    last_error = f"HTTP {response.status_code}: {response.text[:240]}"
                else:
                    value = response.json()
                    message = ((value.get("choices") or [{}])[0].get("message") or {})
                    content = message.get("content") or message.get("output_text")
                    if isinstance(content, list):
                        content = "".join(
                            str(item.get("text") or "") if isinstance(item, dict) else str(item)
                            for item in content
                        )
                    content = str(content or "").strip()
                    if not content:
                        last_error = "successful response contained no message content"
                    else:
                        parsed = json.loads(_CODE_FENCE.sub("", content).strip())
                        qas = parsed.get("qas") if isinstance(parsed, dict) else None
                        if not isinstance(qas, list):
                            raise ValueError("response JSON has no qas array")
                        usage = value.get("usage") or {}
                        return GeneratedBundle(
                            conversation_id=conversation_id,
                            qas=qas,
                            usage={
                                "prompt_tokens": int(usage.get("prompt_tokens") or 0),
                                "completion_tokens": int(usage.get("completion_tokens") or 0),
                                "total_tokens": int(usage.get("total_tokens") or 0),
                            },
                            raw_response=content,
                        )
            except (requests.RequestException, ValueError, TypeError, json.JSONDecodeError) as exc:
                last_error = f"{type(exc).__name__}: {exc}"
            if attempt + 1 < self.max_retries:
                time.sleep(min(30.0, 1.5 * (2**attempt)) + random.random())
        raise RuntimeError(
            f"GPT QA generation failed for {conversation_id} after {self.max_retries} attempts: "
            f"{last_error}"
        )

    def generate(self, bundle: Mapping[str, Any]) -> GeneratedBundle:
        return self._generate_from_messages(
            conversation_id=str(bundle["conversation_id"]),
            messages=_generation_messages(bundle),
            max_completion_tokens=4096,
        )


class DirectGPTQAClient(GPTQAClient):
    """Generate questions, answers, and supporting memories directly from an episode."""

    def generate(self, graph: Mapping[str, Any]) -> GeneratedBundle:
        return self._generate_from_messages(
            conversation_id=str(graph["conversation_id"]),
            messages=_direct_generation_messages(graph),
            max_completion_tokens=8192,
        )

    def generate_augmentation(
        self,
        graph: Mapping[str, Any],
        existing_qas: Sequence[Mapping[str, Any]],
        requested_points: Sequence[str],
    ) -> GeneratedBundle:
        return self._generate_from_messages(
            conversation_id=str(graph["conversation_id"]),
            messages=_direct_augmentation_messages(graph, existing_qas, requested_points),
            max_completion_tokens=4096,
        )


class DeterministicQAClient:
    """Generate conservative TR/VS rows when the language-model gateway is unavailable.

    The client deliberately skips FR: selecting a natural short answer span and
    writing a non-leading question requires language understanding. TR and VS
    already have fixed answers and fixed support in the answer-first targets, so
    they can be verbalized without changing dataset truth.
    """

    model = "stark-answer-first-templates-v1"
    reasoning_effort = "none"

    @staticmethod
    def _variant(conversation_id: str, point: str, count: int) -> int:
        payload = f"{conversation_id}\x1f{point}".encode("utf-8")
        return int(hashlib.sha256(payload).hexdigest()[:8], 16) % count

    @staticmethod
    def _shorten(value: Any, limit: int) -> str:
        text = _normalize_text(value).rstrip(" .!?;:")
        if len(text) <= limit:
            return text
        shortened = text[:limit].rsplit(" ", 1)[0].rstrip(" ,;:-")
        return shortened + "…"

    def generate(self, bundle: Mapping[str, Any]) -> GeneratedBundle:
        conversation_id = str(bundle["conversation_id"])
        qas: list[dict[str, Any]] = []
        for target in bundle.get("targets") or []:
            point = str(target.get("point") or "")
            if point == "TR":
                event_a = self._shorten(target.get("event_a"), 135)
                event_b = self._shorten(target.get("event_b"), 135)
                variants = (
                    f"Was “{event_a}” before or after “{event_b}” in the timeline?",
                    f"In the timeline, did “{event_a}” occur before or after “{event_b}”?",
                    f"Relative to “{event_b}”, was “{event_a}” before or after it?",
                    f"Did the event “{event_a}” take place before or after “{event_b}”?",
                )
                support_a = target.get("support_a") or []
                support_b = target.get("support_b") or []
                if event_a and event_b and support_a and support_b:
                    qas.append(
                        {
                            "target_id": str(target["target_id"]),
                            "point": point,
                            "question": variants[self._variant(conversation_id, point, len(variants))],
                            "answer": str(target["fixed_answer"]),
                            "support_memory_ids": [
                                str(support_a[0]["memory_id"]),
                                str(support_b[0]["memory_id"]),
                            ],
                        }
                    )
            elif point == "VS":
                event = self._shorten(target.get("session_event"), 220)
                image = target.get("image") or {}
                variants = (
                    f"Which previously shared image is associated with “{event}”?",
                    f"Which image from our earlier conversation is connected to “{event}”?",
                    f"What is the image ID of the picture shared during “{event}”?",
                    f"Which earlier image belongs to the conversation about “{event}”?",
                )
                if event and image.get("memory_id") and target.get("fixed_answer"):
                    support_ids = [str(image["memory_id"])]
                    support_ids.extend(
                        str(item["memory_id"])
                        for item in (target.get("context") or [])
                        if item.get("memory_id")
                    )
                    qas.append(
                        {
                            "target_id": str(target["target_id"]),
                            "point": point,
                            "question": variants[self._variant(conversation_id, point, len(variants))],
                            "answer": str(target["fixed_answer"]),
                            "support_memory_ids": list(dict.fromkeys(support_ids)),
                        }
                    )
        raw_response = json.dumps({"qas": qas}, ensure_ascii=False)
        return GeneratedBundle(
            conversation_id=conversation_id,
            qas=qas,
            usage={},
            raw_response=raw_response,
        )


def _allowed_support(target: Mapping[str, Any]) -> set[str]:
    point = target.get("point")
    if point == "FR":
        return {str(item["memory_id"]) for item in target.get("candidates") or []}
    if point == "TR":
        return {
            str(item["memory_id"])
            for key in ("support_a", "support_b")
            for item in target.get(key) or []
        }
    if point == "VS":
        return {
            str(item["memory_id"])
            for key in ("image",)
            for item in [target.get(key) or {}]
            if item.get("memory_id")
        } | {str(item["memory_id"]) for item in target.get("context") or []}
    return set()


def validate_generated_bundle(
    target_bundle: Mapping[str, Any],
    generated: GeneratedBundle,
    records_by_id: Mapping[str, MemoryRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fail closed: only retain grounded QA rows with valid clue IDs."""

    targets = {str(target["target_id"]): target for target in target_bundle.get("targets") or []}
    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    for raw in generated.qas:
        reasons = []
        if not isinstance(raw, dict):
            rejected.append({"qa": raw, "reasons": ["not_an_object"]})
            continue
        target_id = str(raw.get("target_id") or "")
        target = targets.get(target_id)
        point = str(raw.get("point") or "").upper()
        question = _normalize_text(raw.get("question"))
        answer = _normalize_text(raw.get("answer"))
        support_ids = list(dict.fromkeys(str(value) for value in raw.get("support_memory_ids") or []))
        if target is None:
            reasons.append("unknown_target")
        elif point != target.get("point"):
            reasons.append("point_mismatch")
        if not 12 <= len(question) <= 400:
            reasons.append("bad_question_length")
        if not answer:
            reasons.append("empty_answer")
        if not support_ids:
            reasons.append("empty_support")
        if target is not None:
            allowed = _allowed_support(target)
            if any(memory_id not in allowed for memory_id in support_ids):
                reasons.append("support_outside_target")
            if any(memory_id not in records_by_id for memory_id in support_ids):
                reasons.append("unknown_support")
            limit = int(target.get("requested_count") or 1)
            if counts[target_id] >= limit:
                reasons.append("too_many_for_target")
            if point == "FR" and len(support_ids) != 1:
                reasons.append("fr_requires_one_support")
            if point == "FR" and support_ids:
                support = records_by_id.get(support_ids[0])
                if support is None or _answer_key(answer) not in _answer_key(support.content):
                    reasons.append("fr_answer_not_exact_support_span")
                if len(_WORD.findall(answer)) > 15:
                    reasons.append("fr_answer_too_long")
            if point == "TR":
                if _answer_key(answer) != _answer_key(target.get("fixed_answer")):
                    reasons.append("tr_answer_changed")
                a_ids = {str(item["memory_id"]) for item in target.get("support_a") or []}
                b_ids = {str(item["memory_id"]) for item in target.get("support_b") or []}
                if not (a_ids & set(support_ids)) or not (b_ids & set(support_ids)):
                    reasons.append("tr_missing_event_support")
                lowered = question.casefold()
                if "before" not in lowered or "after" not in lowered:
                    reasons.append("tr_question_not_before_after")
            if point == "VS":
                if answer != str(target.get("fixed_answer") or ""):
                    reasons.append("vs_answer_changed")
                image_id = str((target.get("image") or {}).get("memory_id") or "")
                if image_id not in support_ids:
                    reasons.append("vs_missing_image_support")
                if answer and answer.casefold() in question.casefold():
                    reasons.append("vs_answer_leaked_in_question")
        if point == "FR" and answer and _answer_key(answer) in _answer_key(question):
            reasons.append("fr_answer_leaked_in_question")
        if any(memory_id in question for memory_id in support_ids):
            reasons.append("memory_id_leaked_in_question")

        if reasons:
            rejected.append({"qa": raw, "reasons": sorted(set(reasons))})
            continue
        counts[target_id] += 1
        accepted.append(
            {
                "target_id": target_id,
                "point": point,
                "question": question,
                "answer": answer,
                "support_memory_ids": support_ids,
            }
        )
    return accepted, rejected


def validate_direct_generated_bundle(
    generated: GeneratedBundle,
    records_by_id: Mapping[str, MemoryRecord],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Validate free API output without supplying the model with fixed QA targets."""

    accepted: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    seen_points: set[str] = set()
    seen_questions: set[str] = set()
    for index, raw in enumerate(generated.qas):
        if not isinstance(raw, dict):
            rejected.append({"qa": raw, "reasons": ["not_an_object"]})
            continue
        reasons: list[str] = []
        point = str(raw.get("point") or "").upper()
        qa_id = _safe_id(raw.get("qa_id") or f"qa-{index:02d}")
        question = _normalize_text(raw.get("question"))
        if point in {"VS", "VR"}:
            question = _normalize_text(_AVAILABLE_WORD.sub("", question))
        answer = _normalize_text(raw.get("answer"))
        raw_support = raw.get("support_memory_ids") or []
        if not isinstance(raw_support, list):
            raw_support = []
            reasons.append("support_not_a_list")
        support_ids = list(dict.fromkeys(str(value) for value in raw_support if str(value)))
        question_image_id = str(raw.get("question_image_memory_id") or "") or None

        if point not in DIRECT_POINTS:
            reasons.append("unknown_point")
        if point in seen_points:
            reasons.append("duplicate_point")
        question_key = _answer_key(question)
        if not 20 <= len(question) <= 320:
            reasons.append("bad_question_length")
        if not question_key or question_key in seen_questions:
            reasons.append("empty_or_duplicate_question")
        if not answer or len(answer) > 1000:
            reasons.append("bad_answer_length")
        answer_word_limit = DIRECT_ANSWER_WORD_LIMITS.get(point)
        if answer_word_limit is not None and len(answer.split()) > answer_word_limit:
            reasons.append("answer_too_long_for_point")
        if any(memory_id not in records_by_id for memory_id in support_ids):
            reasons.append("unknown_support")
        if question_image_id and question_image_id not in records_by_id:
            reasons.append("unknown_question_image")
        if any(memory_id in question for memory_id in records_by_id):
            reasons.append("memory_id_leaked_in_question")

        support = [records_by_id[memory_id] for memory_id in support_ids if memory_id in records_by_id]
        local_image_support = [
            record for record in support if record.modality == "image" and record.raw_pointer
        ]
        if point == "AR":
            if answer.casefold().rstrip(".") != "not mentioned":
                reasons.append("ar_answer_not_not_mentioned")
            if support_ids:
                reasons.append("ar_has_support")
        elif not support_ids:
            reasons.append("missing_support")

        if point == "VS":
            if not local_image_support:
                reasons.append("vs_missing_local_image_support")
            answer_image_ids = {
                value.strip().rstrip(".")
                for value in re.split(r"[,;]", answer)
                if value.strip()
            }
            supported_image_ids = {
                str(record.metadata.get("image_id"))
                for record in local_image_support
                if record.metadata.get("image_id")
            }
            if not answer_image_ids or answer_image_ids != supported_image_ids:
                reasons.append("vs_answer_does_not_match_support_images")
        elif point == "TTL":
            if not question_image_id:
                reasons.append("ttl_missing_question_image")
            else:
                image = records_by_id.get(question_image_id)
                if image is None or image.modality != "image" or not image.raw_pointer:
                    reasons.append("ttl_question_image_unavailable")
                if question_image_id in support_ids:
                    reasons.append("ttl_question_image_in_support")
            support_text = " ".join(record.searchable_text() for record in support)
            if answer and _answer_key(answer) not in _answer_key(support_text):
                reasons.append("ttl_label_not_taught_by_support")
        elif question_image_id:
            reasons.append("question_image_outside_ttl")

        if point == "VR" and not local_image_support:
            reasons.append("vr_missing_local_image_support")
        if point == "TR" and not support:
            reasons.append("tr_missing_support")
        if point == "MR" and len(support_ids) < 2:
            reasons.append("mr_requires_multiple_support")
        if point == "KR" and len(support_ids) < 2:
            reasons.append("kr_requires_update_support")
        if point == "CD" and answer.casefold().rstrip(".") not in {"yes", "no"}:
            reasons.append("cd_answer_not_yes_no")

        if reasons:
            rejected.append({"qa": raw, "reasons": sorted(set(reasons))})
            continue
        seen_points.add(point)
        seen_questions.add(question_key)
        accepted.append(
            {
                "qa_id": qa_id,
                "target_id": qa_id,
                "point": point,
                "question": question,
                "answer": answer,
                "support_memory_ids": support_ids,
                "question_image_memory_id": question_image_id,
            }
        )
    return accepted, rejected


def qas_to_mem_gallery_rows(
    conversation_id: str,
    qas: Sequence[Mapping[str, Any]],
    records_by_id: Mapping[str, MemoryRecord],
    *,
    generator_model: str,
    start_index: int = 0,
) -> list[dict[str, Any]]:
    rows = []
    sorted_qas = sorted(
        qas,
        key=lambda qa: (DIRECT_POINT_ORDER.get(str(qa["point"]), 99), str(qa["question"])),
    )
    scenario_id = _safe_id(conversation_id)
    for qa_index, qa in enumerate(sorted_qas, start=start_index):
        support = [records_by_id[str(memory_id)] for memory_id in qa["support_memory_ids"]]
        support_turn_ids = list(dict.fromkeys(record.turn_id for record in support))
        sessions = list(dict.fromkeys(str(record.metadata.get("session_id") or "") for record in support))
        clues = list(dict.fromkeys(str(record.metadata.get("local_turn_id") or "") for record in support))
        image_ids = list(
            dict.fromkeys(
                str(record.metadata.get("image_id") or "")
                for record in support
                if record.metadata.get("image_id")
            )
        )
        answer = str(qa["answer"])
        question_image_memory_id = str(qa.get("question_image_memory_id") or "") or None
        question_image_record = (
            records_by_id.get(question_image_memory_id) if question_image_memory_id else None
        )
        question_image = question_image_record.raw_pointer if question_image_record is not None else None
        rows.append(
            {
                "sample_id": f"stark:{scenario_id}:qa:{qa_index:04d}",
                "dataset": "stark",
                "scenario": conversation_id,
                "scenario_file": "dataset/Stark/dialogue/stark.parquet",
                "qa_index": qa_index,
                "point": qa["point"],
                "question": (
                    _normalize_text(_AVAILABLE_WORD.sub("", str(qa["question"])))
                    if qa["point"] in {"VS", "VR"}
                    else qa["question"]
                ),
                "answer": answer,
                "gold_answer": answer,
                "session_id": sessions,
                "clue": clues,
                "support_turn_ids": support_turn_ids,
                "support_memory_ids": list(qa["support_memory_ids"]),
                "answer_image_ids": image_ids if qa["point"] == "VS" else [],
                "question_image": question_image,
                "question_image_relative": question_image,
                "raw_qa": {
                    "target_id": qa.get("target_id") or qa.get("qa_id"),
                    "generator_model": generator_model,
                    "support_memory_ids": list(qa["support_memory_ids"]),
                    "question_image_memory_id": question_image_memory_id,
                },
            }
        )
    return rows


def qas_to_opd_samples(
    qas: Sequence[Mapping[str, Any]],
    records_by_episode: Mapping[str, Sequence[MemoryRecord]],
) -> list[OPDSample]:
    samples = []
    for index, qa in enumerate(qas):
        scenario = str(qa["scenario"])
        question_image_memory_id = str(
            (qa.get("raw_qa") or {}).get("question_image_memory_id") or ""
        ) or None
        episode_records = [
            record
            for record in records_by_episode[scenario]
            if record.memory_id != question_image_memory_id
        ]
        samples.append(
            OPDSample(
                sample_id=str(qa["sample_id"]),
                query=str(qa["question"]),
                gold_answer=str(qa["gold_answer"]),
                memory_store=HiddenMemoryStore(episode_records),
                metadata={
                    "index": index,
                    "data_source": "opd_mm",
                    "agent_name": "tool_agent",
                    "opd_mm_online_self_distill": True,
                    "scenario": scenario,
                    "point": qa["point"],
                    "qa_index": qa["qa_index"],
                    "question_image": qa.get("question_image"),
                    "extra_info": {
                        "stark_sample_id": qa["sample_id"],
                        "scenario": scenario,
                        "point": qa["point"],
                        "qa_index": qa["qa_index"],
                        "support_turn_ids": qa["support_turn_ids"],
                        "support_memory_ids": qa["support_memory_ids"],
                        "clue": qa["clue"],
                        "question_image": qa.get("question_image"),
                        "question_image_relative": qa.get("question_image_relative"),
                    },
                },
            )
        )
    return samples


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def generate_qa_split(
    *,
    target_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
    split: str,
    client: GPTQAClient | DeterministicQAClient,
    max_episodes: int | None = None,
    resume: bool = True,
    workers: int = 1,
    circuit_breaker_errors: int = 8,
) -> dict[str, Any]:
    """Generate, validate, and checkpoint one split one episode at a time."""

    targets = _read_jsonl(Path(target_path))
    if max_episodes is not None:
        targets = targets[:max_episodes]
    records_by_episode = load_records_by_episode(records_path)
    output = Path(output_dir)
    generation_dir = output / "generation"
    qa_dir = output / "qa"
    validation_dir = output / "validation"
    for directory in (generation_dir, qa_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    bundles_path = generation_dir / f"{split}_bundles.jsonl"
    errors_path = validation_dir / f"{split}_api_errors.jsonl"
    existing_rows = _read_jsonl(bundles_path) if resume else []
    completed = {str(row["conversation_id"]) for row in existing_rows}

    if workers <= 0:
        raise ValueError("workers must be positive")
    if circuit_breaker_errors <= 0:
        raise ValueError("circuit_breaker_errors must be positive")
    pending = [
        (index, target_bundle)
        for index, target_bundle in enumerate(targets, start=1)
        if str(target_bundle["conversation_id"]) not in completed
    ]

    def generate_one(item: tuple[int, dict[str, Any]]) -> tuple[int, str, dict[str, Any]]:
        index, target_bundle = item
        conversation_id = str(target_bundle["conversation_id"])
        episode_records = records_by_episode.get(conversation_id)
        if not episode_records:
            raise ValueError(f"target references unknown episode: {conversation_id}")
        generated = client.generate(target_bundle)
        records_by_id = {record.memory_id: record for record in episode_records}
        accepted, rejected = validate_generated_bundle(target_bundle, generated, records_by_id)
        return (
            index,
            conversation_id,
            {
                "conversation_id": conversation_id,
                "accepted": accepted,
                "rejected": rejected,
                "usage": generated.usage,
                "raw_response": generated.raw_response,
            },
        )

    mode = "a" if resume and bundles_path.exists() else "w"
    consecutive_errors = 0
    stopped_by_circuit_breaker = False
    with bundles_path.open(mode, encoding="utf-8") as bundle_handle, errors_path.open(
        "a" if resume and errors_path.exists() else "w", encoding="utf-8"
    ) as error_handle:
        chunk_size = max(workers, workers * 2)
        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start : chunk_start + chunk_size]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_item = {executor.submit(generate_one, item): item for item in chunk}
                for future in as_completed(future_to_item):
                    index, target_bundle = future_to_item[future]
                    conversation_id = str(target_bundle["conversation_id"])
                    try:
                        _, _, row = future.result()
                        bundle_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        bundle_handle.flush()
                        completed.add(conversation_id)
                        consecutive_errors = 0
                        print(
                            f"[{split} {index}/{len(targets)}] {conversation_id}: "
                            f"accepted={len(row['accepted'])} rejected={len(row['rejected'])}",
                            flush=True,
                        )
                    except Exception as exc:
                        consecutive_errors += 1
                        error_handle.write(
                            json.dumps(
                                {
                                    "conversation_id": conversation_id,
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        error_handle.flush()
                        print(f"[{split} {index}/{len(targets)}] {conversation_id}: ERROR {exc}", flush=True)
            if consecutive_errors >= circuit_breaker_errors:
                stopped_by_circuit_breaker = True
                print(
                    f"[{split}] stopping after {consecutive_errors} consecutive API failures; rerun to resume",
                    flush=True,
                )
                break

    bundle_rows = _read_jsonl(bundles_path)
    target_ids = {str(bundle["conversation_id"]) for bundle in targets}
    bundle_rows = [row for row in bundle_rows if str(row["conversation_id"]) in target_ids]
    qas = []
    rejected_rows = []
    usage: Counter[str] = Counter()
    for row in sorted(bundle_rows, key=lambda item: str(item["conversation_id"])):
        conversation_id = str(row["conversation_id"])
        episode_records = records_by_episode[conversation_id]
        records_by_id = {record.memory_id: record for record in episode_records}
        qas.extend(
            qas_to_mem_gallery_rows(
                conversation_id,
                row.get("accepted") or [],
                records_by_id,
                generator_model=client.model,
            )
        )
        rejected_rows.extend(
            {"conversation_id": conversation_id, **value}
            for value in row.get("rejected") or []
        )
        usage.update({key: int(value) for key, value in (row.get("usage") or {}).items()})

    qa_path = qa_dir / f"{split}_qa.jsonl"
    with qa_path.open("w", encoding="utf-8") as handle:
        for qa in sorted(qas, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")
    rejected_path = validation_dir / f"{split}_rejected.jsonl"
    with rejected_path.open("w", encoding="utf-8") as handle:
        for row in rejected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    counts_by_point = Counter(str(qa["point"]) for qa in qas)
    rejection_reasons = Counter(
        reason for row in rejected_rows for reason in row.get("reasons") or []
    )
    summary = {
        "split": split,
        "target_episode_count": len(targets),
        "completed_episode_count": len(bundle_rows),
        "qa_count": len(qas),
        "counts_by_point": dict(sorted(counts_by_point.items())),
        "rejected_count": len(rejected_rows),
        "rejection_reasons": dict(sorted(rejection_reasons.items())),
        "usage": dict(sorted(usage.items())),
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "stopped_by_circuit_breaker": stopped_by_circuit_breaker,
        "qa_path": str(qa_path.resolve()),
        "bundles_path": str(bundles_path.resolve()),
        "rejected_path": str(rejected_path.resolve()),
    }
    (output / f"{split}_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def generate_direct_qa_split(
    *,
    graph_path: str | Path,
    split_episode_ids_path: str | Path,
    records_path: str | Path,
    output_dir: str | Path,
    split: str,
    client: DirectGPTQAClient,
    max_episodes: int | None = None,
    resume: bool = True,
    workers: int = 1,
    circuit_breaker_errors: int = 8,
    min_accepted_per_episode: int = 5,
) -> dict[str, Any]:
    """Generate free-form multi-category QA directly from complete episode graphs."""

    allowed_ids = {
        line.strip()
        for line in Path(split_episode_ids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    graphs = [
        graph
        for graph in _read_jsonl(Path(graph_path))
        if str(graph.get("conversation_id")) in allowed_ids
    ]
    graphs.sort(key=lambda graph: str(graph["conversation_id"]))
    if max_episodes is not None:
        graphs = graphs[:max_episodes]
    records_by_episode = load_records_by_episode(records_path)
    output = Path(output_dir)
    generation_dir = output / "generation"
    qa_dir = output / "qa"
    validation_dir = output / "validation"
    for directory in (generation_dir, qa_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    bundles_path = generation_dir / f"{split}_bundles.jsonl"
    errors_path = validation_dir / f"{split}_api_errors.jsonl"
    existing_rows = _read_jsonl(bundles_path) if resume else []
    completed = {str(row["conversation_id"]) for row in existing_rows}
    pending = [graph for graph in graphs if str(graph["conversation_id"]) not in completed]
    if workers <= 0:
        raise ValueError("workers must be positive")

    def generate_one(graph: Mapping[str, Any]) -> dict[str, Any]:
        conversation_id = str(graph["conversation_id"])
        episode_records = records_by_episode.get(conversation_id) or []
        if not episode_records:
            raise ValueError(f"graph references unknown episode: {conversation_id}")
        generated = client.generate(graph)
        records_by_id = {record.memory_id: record for record in episode_records}
        accepted, rejected = validate_direct_generated_bundle(generated, records_by_id)
        if len(accepted) < min_accepted_per_episode:
            reasons = Counter(
                reason for row in rejected for reason in row.get("reasons") or []
            )
            raise ValueError(
                f"only {len(accepted)} direct QAs passed validation; rejection_reasons={dict(reasons)}"
            )
        return {
            "conversation_id": conversation_id,
            "accepted": accepted,
            "rejected": rejected,
            "usage": generated.usage,
            "raw_response": generated.raw_response,
        }

    mode = "a" if resume and bundles_path.exists() else "w"
    consecutive_errors = 0
    stopped_by_circuit_breaker = False
    with bundles_path.open(mode, encoding="utf-8") as bundle_handle, errors_path.open(
        "a" if resume and errors_path.exists() else "w", encoding="utf-8"
    ) as error_handle:
        chunk_size = max(workers, workers * 2)
        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start : chunk_start + chunk_size]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_graph = {executor.submit(generate_one, graph): graph for graph in chunk}
                for future in as_completed(future_to_graph):
                    graph = future_to_graph[future]
                    conversation_id = str(graph["conversation_id"])
                    try:
                        row = future.result()
                        bundle_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        bundle_handle.flush()
                        completed.add(conversation_id)
                        consecutive_errors = 0
                        counts = Counter(qa["point"] for qa in row["accepted"])
                        print(
                            f"[{split} direct {len(completed)}/{len(graphs)}] {conversation_id}: "
                            f"accepted={len(row['accepted'])} points={dict(counts)} "
                            f"rejected={len(row['rejected'])}",
                            flush=True,
                        )
                    except Exception as exc:
                        consecutive_errors += 1
                        error_handle.write(
                            json.dumps(
                                {
                                    "conversation_id": conversation_id,
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        error_handle.flush()
                        print(f"[{split} direct] {conversation_id}: ERROR {exc}", flush=True)
            if consecutive_errors >= circuit_breaker_errors:
                stopped_by_circuit_breaker = True
                break

    bundle_rows = [
        row
        for row in _read_jsonl(bundles_path)
        if str(row["conversation_id"]) in {str(graph["conversation_id"]) for graph in graphs}
    ]
    qas: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    usage: Counter[str] = Counter()
    for row in sorted(bundle_rows, key=lambda item: str(item["conversation_id"])):
        conversation_id = str(row["conversation_id"])
        episode_records = records_by_episode[conversation_id]
        records_by_id = {record.memory_id: record for record in episode_records}
        qas.extend(
            qas_to_mem_gallery_rows(
                conversation_id,
                row.get("accepted") or [],
                records_by_id,
                generator_model=client.model,
            )
        )
        rejected_rows.extend(
            {"conversation_id": conversation_id, **value}
            for value in row.get("rejected") or []
        )
        usage.update({key: int(value) for key, value in (row.get("usage") or {}).items()})

    qa_path = qa_dir / f"{split}_qa.jsonl"
    with qa_path.open("w", encoding="utf-8") as handle:
        for qa in sorted(qas, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")
    rejected_path = validation_dir / f"{split}_rejected.jsonl"
    with rejected_path.open("w", encoding="utf-8") as handle:
        for row in rejected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "split": split,
        "target_episode_count": len(graphs),
        "completed_episode_count": len(bundle_rows),
        "qa_count": len(qas),
        "counts_by_point": dict(sorted(Counter(qa["point"] for qa in qas).items())),
        "rejected_count": len(rejected_rows),
        "rejection_reasons": dict(
            sorted(
                Counter(
                    reason for row in rejected_rows for reason in row.get("reasons") or []
                ).items()
            )
        ),
        "usage": dict(sorted(usage.items())),
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "stopped_by_circuit_breaker": stopped_by_circuit_breaker,
        "qa_path": str(qa_path.resolve()),
        "bundles_path": str(bundles_path.resolve()),
        "rejected_path": str(rejected_path.resolve()),
    }
    (output / f"{split}_generation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def _allocate_proportional_counts(
    counts: Mapping[str, int],
    total: int,
) -> dict[str, int]:
    """Allocate an integer total while preserving the observed point mixture."""

    if total < 0:
        raise ValueError("total must be non-negative")
    weights = {point: max(0, int(counts.get(point, 0))) for point in DIRECT_POINTS}
    weight_sum = sum(weights.values())
    if weight_sum <= 0:
        weights = {point: 1 for point in DIRECT_POINTS}
        weight_sum = len(weights)
    exact = {point: total * weight / weight_sum for point, weight in weights.items()}
    allocated = {point: int(value) for point, value in exact.items()}
    remainder = total - sum(allocated.values())
    for point in sorted(
        DIRECT_POINTS,
        key=lambda value: (exact[value] - allocated[value], -DIRECT_POINT_ORDER[value]),
        reverse=True,
    )[:remainder]:
        allocated[point] += 1
    return allocated


def _augmentation_point_plan(
    conversation_ids: Sequence[str],
    existing_qas: Sequence[Mapping[str, Any]],
    additional_per_episode: int,
) -> tuple[dict[str, list[str]], dict[str, int]]:
    """Assign distinct requested points to episodes with a balanced global quota."""

    if not 1 <= additional_per_episode <= len(DIRECT_POINTS):
        raise ValueError(f"additional_per_episode must be between 1 and {len(DIRECT_POINTS)}")
    total = len(conversation_ids) * additional_per_episode
    existing_counts = Counter(str(qa.get("point") or "") for qa in existing_qas)
    quotas = _allocate_proportional_counts(existing_counts, total)
    if quotas and max(quotas.values()) > len(conversation_ids):
        raise ValueError("point quota cannot be assigned at most once per episode")

    remaining = dict(quotas)
    order = sorted(DIRECT_POINTS, key=DIRECT_POINT_ORDER.get)
    plan: dict[str, list[str]] = {}
    for episode_index, conversation_id in enumerate(conversation_ids):
        rotated = order[episode_index % len(order) :] + order[: episode_index % len(order)]
        tie_rank = {point: index for index, point in enumerate(rotated)}
        selected = sorted(
            (point for point in order if remaining[point] > 0),
            key=lambda point: (remaining[point], -tie_rank[point]),
            reverse=True,
        )[:additional_per_episode]
        if len(selected) != additional_per_episode:
            raise ValueError("unable to allocate distinct augmentation points")
        for point in selected:
            remaining[point] -= 1
        plan[conversation_id] = sorted(selected, key=DIRECT_POINT_ORDER.get)
    if any(remaining.values()):
        raise ValueError(f"augmentation point allocation left a remainder: {remaining}")
    return plan, quotas


def generate_direct_qa_augmentation_split(
    *,
    graph_path: str | Path,
    split_episode_ids_path: str | Path,
    records_path: str | Path,
    existing_qa_dir: str | Path,
    output_dir: str | Path,
    split: str,
    client: DirectGPTQAClient,
    additional_per_episode: int = 4,
    max_episodes: int | None = None,
    resume: bool = True,
    workers: int = 1,
    circuit_breaker_errors: int = 8,
    min_accepted_per_episode: int | None = None,
) -> dict[str, Any]:
    """Add grounded, non-duplicate QAs while retaining the original generated rows."""

    allowed_ids = {
        line.strip()
        for line in Path(split_episode_ids_path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    graphs = [
        graph
        for graph in _read_jsonl(Path(graph_path))
        if str(graph.get("conversation_id")) in allowed_ids
    ]
    graphs.sort(key=lambda graph: str(graph["conversation_id"]))
    if max_episodes is not None:
        graphs = graphs[:max_episodes]
    graph_ids = [str(graph["conversation_id"]) for graph in graphs]
    graph_id_set = set(graph_ids)
    records_by_episode = load_records_by_episode(records_path)
    existing_rows = [
        qa
        for qa in _read_jsonl(Path(existing_qa_dir) / f"{split}_qa.jsonl")
        if str(qa.get("scenario")) in graph_id_set
    ]
    existing_by_episode: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for qa in existing_rows:
        existing_by_episode[str(qa["scenario"])].append(qa)
    missing_existing = sorted(graph_id_set - set(existing_by_episode))
    if missing_existing:
        raise ValueError(f"episodes missing existing QAs: {missing_existing[:3]}")

    point_plan, point_quotas = _augmentation_point_plan(
        graph_ids,
        existing_rows,
        additional_per_episode,
    )
    minimum = (
        additional_per_episode
        if min_accepted_per_episode is None
        else min_accepted_per_episode
    )
    if not 1 <= minimum <= additional_per_episode:
        raise ValueError("min_accepted_per_episode must be within the requested count")
    if workers <= 0:
        raise ValueError("workers must be positive")

    output = Path(output_dir)
    generation_dir = output / "generation"
    qa_dir = output / "qa"
    validation_dir = output / "validation"
    for directory in (generation_dir, qa_dir, validation_dir):
        directory.mkdir(parents=True, exist_ok=True)
    bundles_path = generation_dir / f"{split}_augmentation_bundles.jsonl"
    errors_path = validation_dir / f"{split}_augmentation_api_errors.jsonl"
    prior_bundle_rows = _read_jsonl(bundles_path) if resume else []
    completed = {str(row["conversation_id"]) for row in prior_bundle_rows}
    pending = [graph for graph in graphs if str(graph["conversation_id"]) not in completed]

    def generate_one(graph: Mapping[str, Any]) -> dict[str, Any]:
        conversation_id = str(graph["conversation_id"])
        episode_records = records_by_episode.get(conversation_id) or []
        if not episode_records:
            raise ValueError(f"graph references unknown episode: {conversation_id}")
        generated = client.generate_augmentation(
            graph,
            existing_by_episode[conversation_id],
            point_plan[conversation_id],
        )
        records_by_id = {record.memory_id: record for record in episode_records}
        accepted, rejected = validate_direct_generated_bundle(generated, records_by_id)
        requested = set(point_plan[conversation_id])
        existing_question_keys = {
            _answer_key(qa.get("question")) for qa in existing_by_episode[conversation_id]
        }
        filtered: list[dict[str, Any]] = []
        for qa in accepted:
            reasons = []
            if str(qa["point"]) not in requested:
                reasons.append("point_not_requested")
            if _answer_key(qa["question"]) in existing_question_keys:
                reasons.append("duplicates_existing_question")
            if reasons:
                rejected.append({"qa": qa, "reasons": reasons})
            else:
                filtered.append(qa)
        accepted = filtered
        if len(accepted) < minimum:
            reasons = Counter(reason for row in rejected for reason in row.get("reasons") or [])
            raise ValueError(
                f"only {len(accepted)}/{additional_per_episode} augmentation QAs passed validation; "
                f"rejection_reasons={dict(reasons)}"
            )
        return {
            "conversation_id": conversation_id,
            "requested_points": point_plan[conversation_id],
            "accepted": accepted,
            "rejected": rejected,
            "usage": generated.usage,
            "raw_response": generated.raw_response,
        }

    mode = "a" if resume and bundles_path.exists() else "w"
    consecutive_errors = 0
    stopped_by_circuit_breaker = False
    with bundles_path.open(mode, encoding="utf-8") as bundle_handle, errors_path.open(
        "a" if resume and errors_path.exists() else "w", encoding="utf-8"
    ) as error_handle:
        chunk_size = max(workers, workers * 2)
        for chunk_start in range(0, len(pending), chunk_size):
            chunk = pending[chunk_start : chunk_start + chunk_size]
            with ThreadPoolExecutor(max_workers=workers) as executor:
                future_to_graph = {executor.submit(generate_one, graph): graph for graph in chunk}
                for future in as_completed(future_to_graph):
                    graph = future_to_graph[future]
                    conversation_id = str(graph["conversation_id"])
                    try:
                        row = future.result()
                        bundle_handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                        bundle_handle.flush()
                        completed.add(conversation_id)
                        consecutive_errors = 0
                        counts = Counter(qa["point"] for qa in row["accepted"])
                        print(
                            f"[{split} augment {len(completed)}/{len(graphs)}] {conversation_id}: "
                            f"accepted={len(row['accepted'])} points={dict(counts)} "
                            f"rejected={len(row['rejected'])}",
                            flush=True,
                        )
                    except Exception as exc:
                        consecutive_errors += 1
                        error_handle.write(
                            json.dumps(
                                {
                                    "conversation_id": conversation_id,
                                    "requested_points": point_plan[conversation_id],
                                    "error": f"{type(exc).__name__}: {exc}",
                                },
                                ensure_ascii=False,
                            )
                            + "\n"
                        )
                        error_handle.flush()
                        print(f"[{split} augment] {conversation_id}: ERROR {exc}", flush=True)
            if consecutive_errors >= circuit_breaker_errors:
                stopped_by_circuit_breaker = True
                break

    bundle_rows = [
        row
        for row in _read_jsonl(bundles_path)
        if str(row["conversation_id"]) in graph_id_set
    ]
    bundle_by_episode = {str(row["conversation_id"]): row for row in bundle_rows}
    merged_qas: list[dict[str, Any]] = []
    rejected_rows: list[dict[str, Any]] = []
    usage: Counter[str] = Counter()
    added_counts: Counter[str] = Counter()
    for conversation_id in graph_ids:
        original = sorted(
            existing_by_episode[conversation_id],
            key=lambda qa: int(qa.get("qa_index") or 0),
        )
        for qa in original:
            copied = dict(qa)
            if copied.get("point") in {"VS", "VR"}:
                copied["question"] = _normalize_text(
                    _AVAILABLE_WORD.sub("", str(copied.get("question") or ""))
                )
            merged_qas.append(copied)
        bundle = bundle_by_episode.get(conversation_id)
        if bundle is None:
            continue
        records_by_id = {
            record.memory_id: record for record in records_by_episode[conversation_id]
        }
        start_index = max((int(qa.get("qa_index") or 0) for qa in original), default=-1) + 1
        new_rows = qas_to_mem_gallery_rows(
            conversation_id,
            bundle.get("accepted") or [],
            records_by_id,
            generator_model=client.model,
            start_index=start_index,
        )
        merged_qas.extend(new_rows)
        added_counts.update(qa["point"] for qa in new_rows)
        rejected_rows.extend(
            {"conversation_id": conversation_id, **value}
            for value in bundle.get("rejected") or []
        )
        usage.update({key: int(value) for key, value in (bundle.get("usage") or {}).items()})

    qa_path = qa_dir / f"{split}_qa.jsonl"
    with qa_path.open("w", encoding="utf-8") as handle:
        for qa in sorted(merged_qas, key=lambda item: item["sample_id"]):
            handle.write(json.dumps(qa, ensure_ascii=False) + "\n")
    rejected_path = validation_dir / f"{split}_augmentation_rejected.jsonl"
    with rejected_path.open("w", encoding="utf-8") as handle:
        for row in rejected_rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "split": split,
        "target_episode_count": len(graphs),
        "completed_episode_count": len(bundle_rows),
        "existing_qa_count": len(existing_rows),
        "additional_qa_count": sum(added_counts.values()),
        "total_qa_count": len(merged_qas),
        "additional_per_episode": additional_per_episode,
        "requested_counts_by_point": dict(sorted(point_quotas.items())),
        "added_counts_by_point": dict(sorted(added_counts.items())),
        "rejected_count": len(rejected_rows),
        "usage": dict(sorted(usage.items())),
        "model": client.model,
        "reasoning_effort": client.reasoning_effort,
        "stopped_by_circuit_breaker": stopped_by_circuit_breaker,
        "qa_path": str(qa_path.resolve()),
        "bundles_path": str(bundles_path.resolve()),
        "rejected_path": str(rejected_path.resolve()),
    }
    (output / f"{split}_augmentation_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return summary


def finalize_expansion_dataset(
    *,
    expansion_dir: str | Path,
    records_path: str | Path,
    qa_dir: str | Path | None = None,
    output_dir: str | Path | None = None,
) -> dict[str, Any]:
    """Validate generated QA splits and write verl-ready JSONL/Parquet rows."""

    from verl.experimental.opd_mm.dataset import write_opd_rlhf_jsonl, write_opd_rlhf_parquet

    expansion = Path(expansion_dir)
    qa_root = Path(qa_dir) if qa_dir is not None else expansion / "qa"
    output = Path(output_dir) if output_dir is not None else expansion
    rlhf_dir = output / "rlhf"
    rlhf_dir.mkdir(parents=True, exist_ok=True)
    records_by_episode = load_records_by_episode(records_path)
    all_sample_ids: set[str] = set()
    episode_question_keys: set[tuple[str, str]] = set()
    global_question_counts: Counter[str] = Counter()
    split_scenarios: dict[str, set[str]] = {}
    split_image_ids: dict[str, set[str]] = {}
    summaries: dict[str, Any] = {}

    for split in SPLIT_ORDER:
        split_ids_path = expansion / "splits" / f"{split}_episode_ids.txt"
        allowed_scenarios = {
            line.strip() for line in split_ids_path.read_text(encoding="utf-8").splitlines() if line.strip()
        }
        split_scenarios[split] = allowed_scenarios
        split_image_ids[split] = {
            str(record.metadata.get("image_id"))
            for scenario in allowed_scenarios
            for record in records_by_episode.get(scenario, [])
            if record.raw_pointer and record.metadata.get("image_id")
        }
        qas = _read_jsonl(qa_root / f"{split}_qa.jsonl")
        point_counts: Counter[str] = Counter()
        support_counts = []
        for qa in qas:
            sample_id = str(qa.get("sample_id") or "")
            scenario = str(qa.get("scenario") or "")
            question_key = _answer_key(qa.get("question"))
            if not sample_id or sample_id in all_sample_ids:
                raise ValueError(f"duplicate or empty sample_id: {sample_id}")
            episode_question_key = (scenario, question_key)
            if not question_key or episode_question_key in episode_question_keys:
                raise ValueError(f"duplicate or empty question: {qa.get('question')}")
            if scenario not in allowed_scenarios:
                raise ValueError(f"{sample_id} belongs to the wrong split: {scenario}")
            episode_records = records_by_episode.get(scenario)
            if not episode_records:
                raise ValueError(f"{sample_id} references unknown episode: {scenario}")
            records_by_id = {record.memory_id: record for record in episode_records}
            point = str(qa.get("point") or "")
            support_ids = [str(value) for value in qa.get("support_memory_ids") or []]
            if any(memory_id not in records_by_id for memory_id in support_ids):
                raise ValueError(f"{sample_id} has invalid support memory IDs")
            if point == "AR":
                if support_ids:
                    raise ValueError(f"{sample_id} AR sample unexpectedly has support memories")
            elif not support_ids:
                raise ValueError(f"{sample_id} has no support memories")
            expected_turn_ids = list(
                dict.fromkeys(records_by_id[memory_id].turn_id for memory_id in support_ids)
            )
            if expected_turn_ids != list(qa.get("support_turn_ids") or []):
                raise ValueError(f"{sample_id} support turn IDs do not match support memories")
            if point == "VS":
                image_support = [
                    records_by_id[memory_id]
                    for memory_id in support_ids
                    if records_by_id[memory_id].modality == "image"
                ]
                if not image_support or not all(record.raw_pointer for record in image_support):
                    raise ValueError(f"{sample_id} has unavailable VS image support")
            if point == "TTL":
                question_image_id = str(
                    (qa.get("raw_qa") or {}).get("question_image_memory_id") or ""
                )
                question_image = records_by_id.get(question_image_id)
                if (
                    question_image is None
                    or question_image.modality != "image"
                    or not question_image.raw_pointer
                ):
                    raise ValueError(f"{sample_id} has unavailable TTL question image")
                if question_image_id in support_ids:
                    raise ValueError(f"{sample_id} leaks its TTL question image into memory support")
                if qa.get("question_image") != question_image.raw_pointer:
                    raise ValueError(f"{sample_id} TTL question image does not match its memory ID")
            all_sample_ids.add(sample_id)
            episode_question_keys.add(episode_question_key)
            global_question_counts[question_key] += 1
            point_counts[point] += 1
            support_counts.append(len(support_ids))

        samples = qas_to_opd_samples(qas, records_by_episode)
        jsonl_path = write_opd_rlhf_jsonl(samples, rlhf_dir / f"{split}.jsonl")
        parquet_path = write_opd_rlhf_parquet(samples, rlhf_dir / f"{split}.parquet")
        summaries[split] = {
            "episode_count": len({str(qa["scenario"]) for qa in qas}),
            "qa_count": len(qas),
            "counts_by_point": dict(sorted(point_counts.items())),
            "avg_support_memories": (
                sum(support_counts) / len(support_counts) if support_counts else 0.0
            ),
            "jsonl": str(jsonl_path.resolve()),
            "parquet": str(parquet_path.resolve()),
        }

    for left_index, left in enumerate(SPLIT_ORDER):
        for right in SPLIT_ORDER[left_index + 1 :]:
            overlap = split_scenarios[left] & split_scenarios[right]
            if overlap:
                raise ValueError(f"episode leakage between {left} and {right}: {sorted(overlap)[:3]}")
            image_overlap = split_image_ids[left] & split_image_ids[right]
            if image_overlap:
                raise ValueError(
                    f"local-image leakage between {left} and {right}: {sorted(image_overlap)[:3]}"
                )
    manifest = {
        "dataset": "stark_memgallery_expansion",
        "records_path": str(Path(records_path).resolve()),
        "qa_dir": str(qa_root.resolve()),
        "qa_count": sum(summary["qa_count"] for summary in summaries.values()),
        "cross_episode_duplicate_question_rows": sum(
            count - 1 for count in global_question_counts.values() if count > 1
        ),
        "splits": summaries,
        "validation": {
            "episode_disjoint": True,
            "local_images_disjoint": True,
            "sample_ids_unique": True,
            "questions_unique_within_episode": True,
            "support_ids_resolved": True,
            "vs_images_available": True,
            "ttl_question_images_held_out": True,
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return manifest
