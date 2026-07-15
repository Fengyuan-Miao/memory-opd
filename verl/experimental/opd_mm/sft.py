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

"""SFT data conversion helpers for OPD-MM.

The original OPD-MM baseline emits JSONL rows with input and target fields,
where target is a corrected JSON tool trajectory. verl's multiturn SFT trainer
consumes parquet rows with a messages column. These helpers bridge that format
gap without moving the standalone OmniMem training loop into verl.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

from verl.experimental.opd_mm.models import SFTExample, ToolAction

OPD_POLICY_SYSTEM_PROMPT = (
    "You are an OPD-MM memory retrieval policy. Use only the provided tools to "
    "plan evidence retrieval over the hidden memory store. Do not mention or "
    "invent hidden memory IDs."
)


def parse_action_target(target: str | list[dict[str, Any]]) -> list[ToolAction]:
    """Parse an OPD-MM trajectory target into ToolAction objects."""
    raw = json.loads(target) if isinstance(target, str) else target
    if not isinstance(raw, list):
        raise ValueError("OPD-MM target must be a JSON array of tool calls")
    return [ToolAction.from_dict(item) for item in raw]


def action_to_openai_tool_call(action: ToolAction, index: int = 0) -> dict[str, Any]:
    """Convert one OPD action into an OpenAI function-call fragment."""
    return {
        "id": f"opd_call_{index}",
        "type": "function",
        "function": {
            "name": action.tool.lower(),
            "arguments": json.dumps(action.arguments, ensure_ascii=False),
        },
    }


def opd_sft_row_to_verl_messages(
    row: dict[str, Any] | SFTExample,
    *,
    system_prompt: str = OPD_POLICY_SYSTEM_PROMPT,
    native_tool_calls: bool = False,
) -> list[dict[str, Any]]:
    """Convert one OPD-MM SFT row to verl multiturn-SFT messages."""
    data = row.to_dict(include_metadata=True) if isinstance(row, SFTExample) else row
    prompt = str(data["input"])
    target = str(data["target"]).strip()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if native_tool_calls:
        actions = parse_action_target(target)
        messages.append(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [action_to_openai_tool_call(action, i) for i, action in enumerate(actions)],
            }
        )
    else:
        messages.append({"role": "assistant", "content": target})
    return messages


def opd_sft_row_to_verl_record(
    row: dict[str, Any] | SFTExample,
    *,
    include_tools: bool = False,
    native_tool_calls: bool = False,
) -> dict[str, Any]:
    """Convert an OPD-MM row into one parquet-ready verl SFT record."""
    data = row.to_dict(include_metadata=True) if isinstance(row, SFTExample) else row
    record = {
        "messages": opd_sft_row_to_verl_messages(data, native_tool_calls=native_tool_calls),
        "sample_id": data.get("sample_id", ""),
        "round_index": data.get("round_index", 0),
    }
    if "metadata" in data:
        record["metadata"] = data["metadata"]
    if include_tools:
        from verl.experimental.opd_mm.tools import openai_tool_schemas

        record["tools"] = openai_tool_schemas()
    return record


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read newline-delimited JSON rows."""
    rows: list[dict[str, Any]] = []
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def convert_opd_sft_jsonl(
    input_path: str | Path,
    output_path: str | Path,
    *,
    include_tools: bool = False,
    native_tool_calls: bool = False,
) -> Path:
    """Convert OPD-MM JSONL SFT data to verl multiturn-SFT parquet."""
    rows = [
        opd_sft_row_to_verl_record(row, include_tools=include_tools, native_tool_calls=native_tool_calls)
        for row in read_jsonl(input_path)
    ]
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    import pandas as pd

    pd.DataFrame(rows).to_parquet(output, index=False)
    return output


def iter_verl_sft_records(
    rows: Iterable[dict[str, Any] | SFTExample],
    *,
    include_tools: bool = False,
    native_tool_calls: bool = False,
) -> Iterable[dict[str, Any]]:
    """Yield parquet-ready records from in-memory OPD-MM examples."""
    for row in rows:
        yield opd_sft_row_to_verl_record(row, include_tools=include_tools, native_tool_calls=native_tool_calls)
