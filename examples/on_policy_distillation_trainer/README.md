# On-Policy Distillation

This trainer jointly trains a student model with policy-gradient on-policy rollouts and a distillation loss against a frozen teacher model served by a separate Ray cluster. Compared to pure SFT from teacher generations, on-policy distillation typically closes more of the teacher/student gap at the same compute budget.

## Canonical Scripts

| Script                          | Teachers | Modality   | Infer | Train    | Platform |
|---------------------------------|----------|------------|-------|----------|----------|
| `run_qwen3_8b_fsdp.sh`          | single   | text       | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_megatron.sh`      | single   | text       | vLLM  | Megatron | NVIDIA   |
| `run_qwen3_vl_8b_fsdp.sh`       | single   | VL         | vLLM  | FSDP     | NVIDIA   |
| `run_qwen3_8b_mopd_fsdp.sh`     | multi    | text + VL  | vLLM  | FSDP     | NVIDIA   |
| `run_opd_mm_tool_agent_fsdp.sh` | single   | OPD-MM     | vLLM  | FSDP     | NVIDIA   |
| `run_opd_mm_grpo_fsdp.sh`       | outcome  | OPD-MM     | vLLM  | FSDP     | NVIDIA   |

Override `STUDENT_MODEL` and `TEACHER_MODEL` via env vars to swap model pairs in
the single-teacher scripts. The MOPD script exposes per-teacher overrides.

## Key Flags

- `distillation.enabled=True`
- `distillation.teacher_models.teacher_model.model_path=<HF path>` (single-teacher)
- `+distillation.teacher_models.<name>.{key,model_path,num_replicas,inference.*}` (multi-teacher)
- `distillation.distillation_loss.loss_mode={k1, k3, forward_kl_topk, ...}`
- `distillation.distillation_loss.use_policy_gradient=True|False`
- `distillation.distillation_loss.topk=64`

## OPD-MM Warm Start Then GRPO

`run_opd_mm_grpo_fsdp.sh` starts a fresh GRPO optimizer from a merged OPD actor
checkpoint. Set `OPD_MODEL_PATH` to a prepared HF directory, or
`OPD_CHECKPOINT_DIR` to a verl `global_step_*` directory that should be merged
and validated first. When neither is set, the newest prepared OPD-MM checkpoint
under `checkpoints/verl_distill_opd_mm` is selected.

The default GRPO dataset contains 630 QAs: all 338 OPD warm-start QAs plus
292 additional examples sampled up to four per `(scenario, point)` cell. The
fixed 100-example evaluation set is explicitly excluded and reused after
training.

```bash
python3 examples/data_preprocess/build_mem_gallery_opd_mm_train_subset.py \
  --output-dir dataset/mem_gallery/opd_mm_store/subsets/balanced_grpo_cap4_holdout100 \
  --per-cell-cap 4 --seed 20260713 \
  --base-sample-ids dataset/mem_gallery/opd_mm_store/subsets/balanced_train_cap2/train_sample_ids.txt \
  --reserve-eval-samples 100 --reserve-eval-seed 20260705
```

The script reserves GPUs 0-5 for actor training and starts a fixed outcome VLM
on GPUs 6-7. Each query receives four rollouts by default. After a real `STOP`,
the outcome VLM first answers from the final public evidence without seeing the
gold answer, then judges that generated answer against the private gold answer.
Only the terminal correctness and small trajectory penalties become the GRPO
reward; the judge output is never added to the student context.

Because OPD-MM refreshes rather than accumulates observations, the rollout also
stores the exact prompt, sampled action, and rollout log-probability at every
visited state. The PPO update expands these state-action pairs and applies their
trajectory's terminal GRPO advantage, instead of recomputing later actions from
an observation-free concatenated history.

```bash
bash examples/on_policy_distillation_trainer/run_opd_mm_grpo_fsdp.sh
```

For an already running OpenAI-compatible outcome service, set
`START_OUTCOME_SERVER=0`, `OUTCOME_SERVER_BASE_URL`, and
`OUTCOME_SERVED_MODEL`. Outcome audit dumps are written under
`logs/opd_mm_grpo_outcomes_*`.
