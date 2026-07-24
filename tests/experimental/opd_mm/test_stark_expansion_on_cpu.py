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

from pathlib import Path

from verl.experimental.opd_mm.models import MemoryRecord
from verl.experimental.opd_mm.stark_expansion import (
    DeterministicQAClient,
    GeneratedBundle,
    _augmentation_point_plan,
    build_episode_targets,
    split_episode_ids,
    split_episode_records,
    validate_direct_generated_bundle,
    validate_generated_bundle,
)


def _record(
    memory_id: str,
    *,
    session: str,
    turn: int,
    date: str,
    content: str,
    event: str,
    role: str = "user",
    modality: str = "text",
    image_id: str = "",
    raw_pointer: str | None = None,
    scenario: str = "episode",
) -> MemoryRecord:
    return MemoryRecord(
        memory_id=memory_id,
        turn_id=f"episode:{session}:T{turn:04d}",
        timestamp=f"{date}T{turn:04d}",
        author=role,
        modality=modality,
        source_type="dialogue_image" if modality == "image" else "dialogue_turn",
        content=content,
        raw_pointer=raw_pointer,
        metadata={
            "scenario": scenario,
            "session_id": session,
            "session_date": date,
            "session_event": event,
            "session_experience": "",
            "turn_index": turn,
            "local_turn_id": f"{session}:T{turn:04d}",
            "speaker": "Alice" if role == "user" else "AI Assistant",
            "role": role,
            "utterance": content,
            "character_profile": {"name": "Alice"},
            "image_id": image_id or None,
            "image_description": "A finish-line photograph" if image_id else "",
        },
    )


def test_episode_split_is_deterministic_and_disjoint():
    first = split_episode_ids([f"e{i}" for i in range(20)], seed=7)
    second = split_episode_ids([f"e{i}" for i in reversed(range(20))], seed=7)
    assert first == second
    assert not (set(first["train"]) & set(first["validation"]))
    assert not (set(first["train"]) & set(first["test"]))
    assert sum(len(values) for values in first.values()) == 20


def test_episode_split_keeps_shared_local_images_together(tmp_path: Path):
    image = tmp_path / "shared.jpg"
    image.write_bytes(b"image")
    records_by_episode = {
        f"episode-{index}": [
            _record(
                f"m{index}",
                session="S01",
                turn=1,
                date="2023-01-01",
                content="User shared an image.",
                event="An event",
                modality="image",
                image_id="shared" if index < 2 else f"image-{index}",
                raw_pointer=str(image),
                scenario=f"episode-{index}",
            )
        ]
        for index in range(12)
    }
    splits = split_episode_records(records_by_episode, seed=7)
    locations = {
        episode_id: split
        for split, episode_ids in splits.items()
        for episode_id in episode_ids
    }
    assert locations["episode-0"] == locations["episode-1"]
    assert sum(len(values) for values in splits.values()) == 12


def test_targets_and_generated_qas_are_grounded(tmp_path: Path):
    image = tmp_path / "finish.jpg"
    image.write_bytes(b"image")
    records = [
        _record(
            "m1",
            session="S01",
            turn=1,
            date="2023-01-01",
            content="User: I started training for my first marathon in the city park every morning.",
            event="Alice starts training for her first marathon in the city park",
        ),
        _record(
            "m2",
            session="S02",
            turn=1,
            date="2023-06-01",
            content="User: I completed my first marathon and celebrated with my family at the finish line.",
            event="Alice completes her first marathon and celebrates with family",
        ),
        _record(
            "m3",
            session="S02",
            turn=2,
            date="2023-06-01",
            content="User shared an image.",
            event="Alice completes her first marathon and celebrates with family",
            modality="image",
            image_id="finish-image",
            raw_pointer=str(image),
        ),
    ]
    targets = build_episode_targets("episode", records)
    assert {target["point"] for target in targets["targets"]} == {"FR", "TR", "VS"}

    generated = GeneratedBundle(
        conversation_id="episode",
        qas=[
            {
                "target_id": "fr",
                "point": "FR",
                "question": "Where did I say I trained for my first marathon each morning?",
                "answer": "city park",
                "support_memory_ids": ["m1"],
            },
            {
                "target_id": "tr",
                "point": "TR",
                "question": "Did I start marathon training before or after I completed my first marathon?",
                "answer": "before",
                "support_memory_ids": ["m1", "m2"],
            },
            {
                "target_id": "vs",
                "point": "VS",
                "question": "Which image did I share when celebrating the completion of my first marathon?",
                "answer": "finish-image",
                "support_memory_ids": ["m3"],
            },
        ],
        usage={},
        raw_response="",
    )
    accepted, rejected = validate_generated_bundle(
        targets,
        generated,
        {record.memory_id: record for record in records},
    )
    assert len(accepted) == 3
    assert rejected == []


def test_deterministic_fallback_only_emits_fixed_answer_tasks(tmp_path: Path):
    image = tmp_path / "finish.jpg"
    image.write_bytes(b"image")
    records = [
        _record(
            "m1",
            session="S01",
            turn=1,
            date="2023-01-01",
            content="User: I started training for my first marathon in the city park every morning.",
            event="Alice starts training for her first marathon in the city park",
        ),
        _record(
            "m2",
            session="S02",
            turn=1,
            date="2023-06-01",
            content="User: I completed my first marathon and celebrated with my family at the finish line.",
            event="Alice completes her first marathon and celebrates with family",
        ),
        _record(
            "m3",
            session="S02",
            turn=2,
            date="2023-06-01",
            content="User shared an image.",
            event="Alice completes her first marathon and celebrates with family",
            modality="image",
            image_id="finish-image",
            raw_pointer=str(image),
        ),
    ]
    targets = build_episode_targets("episode", records)
    generated = DeterministicQAClient().generate(targets)
    assert {qa["point"] for qa in generated.qas} == {"TR", "VS"}
    accepted, rejected = validate_generated_bundle(
        targets,
        generated,
        {record.memory_id: record for record in records},
    )
    assert len(accepted) == 2
    assert rejected == []


def test_temporal_target_order_is_not_always_before():
    records = [
        _record(
            "m1",
            session="S01",
            turn=1,
            date="2023-01-01",
            content="User: I started training for my first marathon in the city park every morning.",
            event="Alice starts training for her first marathon in the city park",
        ),
        _record(
            "m2",
            session="S02",
            turn=1,
            date="2023-06-01",
            content="User: I completed my first marathon and celebrated with my family at the finish line.",
            event="Alice completes her first marathon and celebrates with family",
        ),
    ]
    answers = {
        next(
            target["fixed_answer"]
            for target in build_episode_targets(f"episode-{index}", records)["targets"]
            if target["point"] == "TR"
        )
        for index in range(20)
    }
    assert answers == {"before", "after"}


def test_direct_generated_qas_support_all_mem_gallery_points(tmp_path: Path):
    image = tmp_path / "finish.jpg"
    image.write_bytes(b"image")
    records = [
        _record(
            "m1",
            session="S01",
            turn=1,
            date="2023-01-01",
            content="User: I started training for my first marathon in the city park every morning.",
            event="Alice starts training for her first marathon in the city park",
        ),
        _record(
            "m2",
            session="S02",
            turn=1,
            date="2023-06-01",
            content="User: I completed my first marathon and celebrated with my family at the finish line.",
            event="Alice completes her first marathon and celebrates with family",
        ),
        _record(
            "m3",
            session="S02",
            turn=2,
            date="2023-06-01",
            content="User shared an image.",
            event="Alice completes her first marathon and celebrates with family",
            modality="image",
            image_id="finish-image",
            raw_pointer=str(image),
        ),
    ]
    qas = [
        ("FR", "Where did I train for my first marathon each morning?", "city park", ["m1"], None),
        ("VS", "Which available image was shared after I completed my marathon?", "finish-image", ["m3"], None),
        ("TTL", "What activity is represented in this image?", "marathon", ["m2"], "m3"),
        ("TR", "Which happened first, training or completing my marathon?", "training", ["m1", "m2"], None),
        ("VR", "How many available finish-line images did I share?", "1", ["m3"], None),
        ("MR", "Where did I train and what milestone did I later complete?", "I trained in the city park and completed my first marathon.", ["m1", "m2"], None),
        ("KR", "What was the latest state of my marathon journey?", "I completed my first marathon.", ["m1", "m2"], None),
        ("CD", "Would it be accurate to say I never completed a marathon?", "No.", ["m2"], None),
        ("AR", "Did I mention the brand of my running shoes?", "Not mentioned.", [], None),
    ]
    generated = GeneratedBundle(
        conversation_id="episode",
        qas=[
            {
                "qa_id": point.lower(),
                "point": point,
                "question": question,
                "answer": answer,
                "support_memory_ids": support,
                "question_image_memory_id": question_image,
            }
            for point, question, answer, support, question_image in qas
        ],
        usage={},
        raw_response="",
    )
    accepted, rejected = validate_direct_generated_bundle(
        generated,
        {record.memory_id: record for record in records},
    )
    assert {qa["point"] for qa in accepted} == {qa[0] for qa in qas}
    assert next(qa["question"] for qa in accepted if qa["point"] == "VR") == (
        "How many finish-line images did I share?"
    )
    assert next(qa["question"] for qa in accepted if qa["point"] == "VS") == (
        "Which image was shared after I completed my marathon?"
    )
    assert rejected == []


def test_augmentation_point_plan_is_distinct_and_preserves_mix():
    episodes = [f"episode-{index}" for index in range(20)]
    existing = [
        {"point": point}
        for point, count in {"FR": 20, "VS": 20, "TTL": 10, "TR": 20, "VR": 20,
                             "MR": 20, "KR": 20, "CD": 20, "AR": 20}.items()
        for _ in range(count)
    ]
    plan, quotas = _augmentation_point_plan(episodes, existing, additional_per_episode=4)
    assert set(plan) == set(episodes)
    assert all(len(points) == len(set(points)) == 4 for points in plan.values())
    planned_counts = {
        point: sum(point in points for points in plan.values()) for point in quotas
    }
    assert planned_counts == quotas
    assert sum(quotas.values()) == 80
