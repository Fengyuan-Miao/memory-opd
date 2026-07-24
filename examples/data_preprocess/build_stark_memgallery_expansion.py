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

"""Prepare answer-first STARK targets for Mem-Gallery-style data generation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.stark_expansion import write_prepared_expansion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", default="dataset/Stark/opd_mm_store/records.jsonl")
    parser.add_argument("--output-dir", default="dataset/Stark/opd_mm_store/expansion_v1")
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()
    manifest = write_prepared_expansion(records_path=args.records, output_dir=args.output_dir, seed=args.seed)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
