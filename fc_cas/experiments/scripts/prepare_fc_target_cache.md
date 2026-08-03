# Prepare FC target cache & domain draft (T8)

## Prerequisites

- GPU free (≥20 GiB)
- Conda env `lf` (transformers with `qwen3_5` + `gemma4`)
- Force `CUDA_HOME=/home/caden/anaconda3/envs/lf` (parent shells may set a broken `/usr/local/cuda`)

## Steps

```bash
cd /home/caden/workspace/DeepSpec

# 1) Convert processed_FC_dataset → DeepSpec jsonl (arguments as dict)
python fc_cas/experiments/scripts/convert_fc_to_deepspec_jsonl.py \
  --limit-train 200 --limit-val 50 --limit-test 100

# 2) Target cache
bash fc_cas/experiments/scripts/prepare_fc_target_cache.sh

# 3) Short train (60 steps)
bash fc_cas/experiments/scripts/train_fc_draft.sh

# 4) Eval smoke (user-only turns; tools-aware eval still TODO)
LIMIT=20 CONF_TH=0.0 bash fc_cas/experiments/scripts/eval_fc_draft.sh
```

## Outputs

| Artifact | Path |
|----------|------|
| jsonl | `fc_cas/data/deepspec_jsonl/` |
| cache | `/storage/caden/deepspec/fc_cas_qwen3_5_4b_target_cache` |
| draft ckpt | `~/checkpoints/deepspec/dspark_fc_cas_qwen3_5_4b/step_60` |
| summary | `fc_cas/experiments/results/gpu_pipeline_summary.json` |

## Notes

- DeepSpec `parser.py` now forwards `tools` into `apply_chat_template` and sets `enable_thinking=False`.
- DSpark draft training/eval is a **prior-art carrier**; FC-CAS claims are segment scheduling + schema constraints.
