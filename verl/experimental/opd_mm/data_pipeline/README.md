# MMem Base-v0 data pipeline

This package implements the deterministic core of the Mem-Gallery-style data
construction design:

- canonical event/image/fact/QA artifact models;
- an incremental observed fact ledger with event/span/visual provenance;
- cutoff-aware FR/VS/TTL/TR/VR/MR/KR/CD/AR task oracles;
- alternative minimal event-evidence mining;
- public-caption/private-visual-fact, answer-leak, cutoff, and evidence checks;
- per-QA quality certificates and rejection reports;
- deterministic Mem-Gallery layout and OPD-MM record/QA store export;
- optional GPT Image 2 image materialization with deterministic 512x512 normalization.

It also implements the model-backed construction stages described in
`docs/mem_gallery_style_data_pipeline.md`: private episode/event planning,
semantic image contracts, GPT Image prompt compilation and visual verification,
role-isolated dialogue generation, incremental observed-state extraction,
QA specification/realization/judging, and final deterministic acceptance.

The generative model is not allowed to write directly to release data. It
produces one episode artifact, and this package accepts or rejects every QA
against the observed artifact. Planning-only facts must live in a different
file; `planned_facts`, `planned_evidence`, and `planned_provenance` are rejected.

## Generate from a request

Create a construction-only request such as:

```json
{
  "episode_id": "EP000001",
  "language": "en",
  "session_count": 4,
  "time_span_days": 30,
  "rounds_per_session_min": 4,
  "rounds_per_session_max": 7,
  "images_per_session_min": 1,
  "images_per_session_max": 2,
  "qa_count": 12,
  "scenario_constraints": ["natural longitudinal project", "student life"],
  "seed": 20260901
}
```

Then run:

```bash
export MMEM_MULTIMODAL_API_KEY=...
export MMEM_IMAGE_API_KEY=...
python examples/data_preprocess/generate_mmem_v2_dataset.py \
  --request artifacts/EP000001.request.json \
  --output dataset/mmem_v2_generated
```

Private plans, stage inputs/outputs, rejected QA candidates, image candidates,
and state-audit reports are written under `OUTPUT/.construction/EPISODE_ID`.
Only the canonical observed episode is passed to the deterministic builder.
The exact stage prompts live in `data_pipeline/prompts.py`.

The incremental observed-state extractor is the default truth path. The
additional full-state model audit is disabled by default to avoid a redundant
model pass; enable it only for diagnostics with `--run-full-state-audit`.

## Input

The canonical input is one JSON object containing `sessions[].events`,
`images`, `observed_facts`, and `qa_candidates`. Generate the machine-readable
contract with:

```bash
python examples/data_preprocess/build_mmem_v2_dataset.py \
  --write-schema /tmp/mmem-v2-episode.schema.json
```

One event is one complete user/assistant round and all images attached to that
round. Every observed fact must cite an event plus exact text spans and/or
verified visual fact IDs.

Images may already exist in `images`, or be requested construction-privately:

```json
{
  "image_requests": [
    {
      "image_id": "S04_IMG03",
      "prompt": "Documentary-style photo ...",
      "role": "memory",
      "public_retrieval_description": "A person working beside a telescope.",
      "private_verified_visual_facts": []
    }
  ]
}
```

The prompt is removed after materialization and never reaches public captions
or OPD-MM records.

## Build

```bash
python examples/data_preprocess/build_mmem_v2_dataset.py \
  --input artifacts/ \
  --output dataset/mmem_v2_smoke \
  --strict
```

The fixed image model is `gpt-image-2` at `https://xiaohondou.com/v1`. The
client requests the supported `1024x1024` source size, then normalizes every
candidate to a 512x512 RGB PNG before visual verification or dataset export.
Set the credential through `MMEM_IMAGE_API_KEY`; it is never written to an
artifact. Generation uses one candidate by default; a failed candidate may
trigger one contract-repair round and one replacement. `--image-workers`
controls concurrent requests in the artifact materializer.

The output contains:

```text
data/dialog/*.json
data/image/<episode>/*
opd_mm_store/records.jsonl
opd_mm_store/qas.jsonl
opd_mm_store/{records,qas}.parquet  # when parquet dependencies exist
reports/<episode>/quality_certificates.json
reports/<episode>/rejected_qas.json
manifest.json
```

Vector indexes remain a separate hardware-dependent step and can be built
with the existing OPD-MM index scripts from `opd_mm_store/records.jsonl`.

## Multimodal model-backed stages

Model-backed visual extraction and generation helpers use the Responses API:

```text
base URL: https://1pkapi.com/v1
model:    gpt-5.6-terra
key env:  MMEM_MULTIMODAL_API_KEY
```

The client sends each stage-specific task contract directly as the Responses
API `instructions` value, without a provider-override preamble. The key is
read only from the environment (or an explicit in-memory argument) and is
never written to an artifact, report, or release file.
