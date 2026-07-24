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

"""Validate STARK QA splits and write verl-ready expansion data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from verl.experimental.opd_mm.stark_expansion import finalize_expansion_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expansion-dir", default="dataset/Stark/opd_mm_store/expansion_v1")
    parser.add_argument("--records", default="dataset/Stark/opd_mm_store/records.jsonl")
    parser.add_argument("--qa-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    manifest = finalize_expansion_dataset(
        expansion_dir=args.expansion_dir,
        records_path=args.records,
        qa_dir=args.qa_dir,
        output_dir=args.output_dir,
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
