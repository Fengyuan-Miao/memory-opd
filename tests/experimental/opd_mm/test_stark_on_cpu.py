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

import json

from verl.experimental.opd_mm.stark import (
    discover_local_images,
    iter_stark_row_records,
    stark_row_has_local_image,
)


def _row() -> dict:
    return {
        "index": "0:test-episode-experience",
        "name": "友理",
        "age": "28",
        "number_of_session": 1,
        "session1:date": "2023.05.10",
        "session1:event": "A test event",
        "session1:experience": "",
        "session1:speakers": json.dumps(["AI Assistant", r"\u53cb\u7406", r"\u53cb\u7406"]),
        "session1:utterances": json.dumps(["Hello!", "", "Look at this."]),
        "session1:rationales": json.dumps(["", "Sharing a bicycle", "A second view"]),
        "session1:image_descriptions": json.dumps(["", "A red bicycle", "The bicycle from the side"]),
        "session1:image_sources": json.dumps(["", "internet", "mobile"]),
        "session1:keywords": json.dumps(["", "bicycle", ["bicycle", "side"]]),
        "session1:image_id_from_mobile": json.dumps(["", "", "new added image"]),
        "session1:images_key": json.dumps(["", "local-image", ["missing-image", "local-image"]]),
        "session1:images_module_name": json.dumps(["", "p-t2i", "retrieval"]),
    }


def test_stark_messages_become_mem_gallery_dialogue_rounds_with_linked_images(tmp_path):
    image = tmp_path / "local-image.jpg"
    image.write_bytes(b"image")
    local_images = discover_local_images(tmp_path)
    row = _row()

    assert stark_row_has_local_image(row, local_images)
    records = list(
        iter_stark_row_records(
            row,
            source_row_index=7,
            source_path="dataset/Stark/dialogue/stark.parquet",
            local_images=local_images,
        )
    )

    assert len(records) == 4
    assert [record.modality for record in records] == ["text", "image", "text", "image"]
    assert [record.source_type for record in records] == [
        "dialogue_turn",
        "dialogue_image",
        "dialogue_turn",
        "dialogue_image",
    ]
    assert records[0].content == "Assistant: Hello!\nUser shared an image."
    assert records[1].content == "User shared an image."
    assert records[2].content == "User: Look at this.\nUser shared an image."
    assert records[3].content == "User shared an image."
    assert records[1].content != records[0].content
    assert records[3].content != records[2].content
    assert "A red bicycle" not in records[1].content
    assert "A red bicycle" in records[1].summary
    assert records[1].raw_pointer == str(image.resolve())
    assert records[3].metadata["image_key"] == "local-image"
    assert records[3].metadata["image_key_rank"] == 1
    assert records[3].metadata["speaker"] == "友理"
    assert records[1].metadata["image_keywords"] == ["bicycle"]
    assert records[3].metadata["image_keywords"] == ["bicycle", "side"]
    assert records[0].timestamp == "2023-05-10R0001"
    assert records[0].turn_id == records[1].turn_id
    assert records[2].turn_id == records[3].turn_id
    assert len({record.turn_id for record in records}) == 2
    assert len({record.memory_id for record in records}) == len(records)
    assert all(record.status == "active" for record in records)


def test_stark_does_not_materialize_low_rank_local_candidate(tmp_path):
    image = tmp_path / "local-image.jpg"
    image.write_bytes(b"image")
    row = _row()
    row["session1:images_key"] = json.dumps(
        ["", ["missing-0", "missing-1", "local-image"], "missing-image"]
    )
    local_images = discover_local_images(tmp_path)
    assert not stark_row_has_local_image(row, local_images, max_image_rank=1)
    records = list(
        iter_stark_row_records(
            row,
            source_row_index=7,
            source_path="dataset/Stark/dialogue/stark.parquet",
            local_images=local_images,
            max_image_rank=1,
        )
    )
    assert records[1].raw_pointer is None


def test_stark_assistant_image_stays_in_the_preceding_user_round(tmp_path):
    image = tmp_path / "assistant-image.jpg"
    image.write_bytes(b"image")
    row = _row()
    row["session1:speakers"] = json.dumps(["AI Assistant", "友理", "AI Assistant", "友理"])
    row["session1:utterances"] = json.dumps(
        ["What did you enjoy?", "The tea ceremony.", "", "I also visited a temple."]
    )
    row["session1:images_key"] = json.dumps(["", "", "assistant-image", ""])
    row["session1:image_descriptions"] = json.dumps(["", "", "A tea ceremony", ""])
    records = list(
        iter_stark_row_records(
            row,
            source_row_index=7,
            source_path="dataset/Stark/dialogue/stark.parquet",
            local_images=discover_local_images(tmp_path),
        )
    )

    first_round = [record for record in records if record.metadata["round_id"] == "S01:R0001"]
    assert [record.modality for record in first_round] == ["text", "image"]
    assert first_round[0].turn_id == first_round[1].turn_id
    assert "User: The tea ceremony." in first_round[0].content
    assert "Assistant shared an image." in first_round[0].content
    assert first_round[1].content == "Assistant shared an image."
    assert first_round[1].content != first_round[0].content
    assert first_round[1].raw_pointer == str(image.resolve())
