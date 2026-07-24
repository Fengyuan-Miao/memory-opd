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

"""Build Mem-Gallery-compatible dense, vision, and hybrid indexes for STARK."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.vector_index import build_jsonl_vector_store


DEFAULT_DATASET = (
    "dataset/Stark/opd_mm_store_rounds_3000/expansion_v3/"
    "direct_api_3000_rounds_clean_qwen35_9b_support_ablation_"
    "cd_recleaned_full365_ttl_reviewed"
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--records",
        default="dataset/Stark/opd_mm_store_rounds_3000/records.jsonl",
    )
    parser.add_argument("--qa-dir", default=f"{DEFAULT_DATASET}/qa")
    parser.add_argument(
        "--output-dir",
        default="dataset/Stark/opd_mm_store_rounds_3000/indexed_store_ttl_reviewed",
    )
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
    parser.add_argument("--device", default="cuda:4")
    parser.add_argument("--dense-batch-size", type=int, default=128)
    parser.add_argument("--vision-batch-size", type=int, default=32)
    parser.add_argument("--hybrid-text-batch-size", type=int, default=16)
    parser.add_argument("--hybrid-image-batch-size", type=int, default=4)
    parser.add_argument("--skip-dense", action="store_true")
    parser.add_argument("--skip-vision", action="store_true")
    parser.add_argument("--skip-hybrid", action="store_true")
    args = parser.parse_args()

    qa_dir = Path(args.qa_dir)
    qa_paths = [qa_dir / f"{split}_qa.jsonl" for split in ("train", "validation", "test")]
    missing = [str(path) for path in qa_paths if not path.exists()]
    if missing:
        parser.error(f"missing QA files: {missing}")

    manifest = build_jsonl_vector_store(
        records_path=args.records,
        qa_paths=qa_paths,
        output_dir=args.output_dir,
        dataset_name="stark_memgallery_expansion",
        dense_model_path=args.dense_model_path,
        vision_model_path=args.vision_model_path,
        hybrid_model_path=args.hybrid_model_path,
        device=args.device,
        dense_batch_size=args.dense_batch_size,
        vision_batch_size=args.vision_batch_size,
        hybrid_text_batch_size=args.hybrid_text_batch_size,
        hybrid_image_batch_size=args.hybrid_image_batch_size,
        build_dense=not args.skip_dense,
        build_vision=not args.skip_vision,
        build_hybrid=not args.skip_hybrid,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
