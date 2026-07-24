#!/usr/bin/env python3
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

"""Build OPD-MM dense, vision, or hybrid indexes from an existing records JSONL."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.vector_index import (
    GMEQwen2VLUnifiedEncoder,
    MiniLMTextEncoder,
    SigLIPVisionEncoder,
    _record_item,
    _records_from_jsonl,
    _write_failed_index,
    _write_index,
)


def _build_dense(args: argparse.Namespace, records: list) -> dict:
    encoder = MiniLMTextEncoder(args.dense_model_path, device=args.device)
    selected = [record for record in records if record.searchable_text()]
    texts = [record.searchable_text() for record in selected]
    embeddings = encoder.encode_batch(texts, batch_size=args.dense_batch_size)
    return _write_index(
        index_dir=args.output_dir / "indexes" / "dense",
        name="dense",
        embeddings=embeddings,
        items=[
            _record_item(record, text, "dense_text")
            for record, text in zip(selected, texts, strict=True)
        ],
        model_path=args.dense_model_path,
        encoder=type(encoder).__name__,
    )


def _build_vision(args: argparse.Namespace, records: list) -> dict:
    encoder = SigLIPVisionEncoder(args.vision_model_path, device=args.device)
    selected = [
        record
        for record in records
        if record.raw_pointer and Path(record.raw_pointer).exists()
    ]
    embeddings = encoder.encode_images(
        [str(record.raw_pointer) for record in selected],
        batch_size=args.vision_batch_size,
    )
    return _write_index(
        index_dir=args.output_dir / "indexes" / "vision",
        name="vision",
        embeddings=embeddings,
        items=[
            _record_item(record, record.searchable_text(), "vision_image")
            for record in selected
        ],
        model_path=args.vision_model_path,
        encoder=type(encoder).__name__,
    )


def _build_hybrid(args: argparse.Namespace, records: list) -> dict:
    encoder = GMEQwen2VLUnifiedEncoder(args.hybrid_model_path, device=args.device)
    text_records = [
        record
        for record in records
        if record.searchable_text() and not record.raw_pointer
    ]
    image_records = [
        record
        for record in records
        if record.raw_pointer and Path(record.raw_pointer).exists()
    ]
    text_embeddings = encoder.encode_texts(
        [record.searchable_text() for record in text_records],
        batch_size=args.hybrid_text_batch_size,
    )
    image_embeddings = encoder.encode_text_images(
        [record.searchable_text() for record in image_records],
        [str(record.raw_pointer) for record in image_records],
        batch_size=args.hybrid_image_batch_size,
    )
    selected = text_records + image_records
    embeddings = (
        np.concatenate([text_embeddings, image_embeddings], axis=0)
        if image_records
        else text_embeddings
    )
    return _write_index(
        index_dir=args.output_dir / "indexes" / "hybrid",
        name="hybrid",
        embeddings=embeddings,
        items=[
            _record_item(record, record.searchable_text(), "hybrid_unified")
            for record in selected
        ],
        model_path=args.hybrid_model_path,
        encoder=type(encoder).__name__,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--index", choices=("dense", "vision", "hybrid"), required=True)
    parser.add_argument(
        "--dense-model-path",
        default="/home/miaofy/data/pretrained_models/all-MiniLM-L6-v2",
    )
    parser.add_argument(
        "--vision-model-path",
        default="/home/miaofy/data/pretrained_models/SigLIP-Base-Patch16-384",
    )
    parser.add_argument(
        "--hybrid-model-path",
        default="/home/miaofy/data/pretrained_models/gme-Qwen2-VL-2B-Instruct",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dense-batch-size", type=int, default=128)
    parser.add_argument("--vision-batch-size", type=int, default=32)
    parser.add_argument("--hybrid-text-batch-size", type=int, default=16)
    parser.add_argument("--hybrid-image-batch-size", type=int, default=4)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    records = _records_from_jsonl(args.records)
    builders = {
        "dense": _build_dense,
        "vision": _build_vision,
        "hybrid": _build_hybrid,
    }
    try:
        manifest = builders[args.index](args, records)
    except Exception as exc:
        manifest = _write_failed_index(
            args.output_dir / "indexes" / args.index,
            args.index,
            exc,
        )
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        raise
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
