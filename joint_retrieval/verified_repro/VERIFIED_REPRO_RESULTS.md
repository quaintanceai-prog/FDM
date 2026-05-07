# Joint Retrieval Verified Reproduction (April 24, 2026)

Purpose: Confirm that pod 2 local weights (which produced prior reproduction numbers in `second_run/`) reproduce consistently under fresh evaluation.

## Method

For each model, starting from the local checkpoint:
1. Load weights from `/workspace/verified_repro/{model}/`
2. Run joint accuracy eval on pod 2 local test data
3. Compare to prior reproduction number in `second_run/results/`

## Results

| Model | Prior Reproduction | Verified Reproduction | Delta |
|---|---|---|---|
| GPT-2 | 93.7% | 93.7% (1405/1500) | 0.00 |
| Qwen3 | 99.20% | 99.20% | 0.00 |
| LFM2.5 | 2.07% | 2.07% | 0.00 |
| Hermes3 | 99.93% | 99.93% (1499/1500) | 0.00 |

All four models reproduce exactly. Weights are confirmed as the ones that produced the Table 4 (joint retrieval) paper results.

## Weight md5s (pod 2 local)

| Model | md5 | HF md5 | HF match? |
|---|---|---|---|
| GPT-2 | 596e3ad22878b012c4a6d197e892e4a0 | 36e058b1... | NO |
| Qwen3 | 873b6ffa... | b795061a... | NO |
| LFM2.5 | da251c78... | da251c78... | YES |
| Hermes3 | 696db498... | 696db498... | YES |

**Action needed**: Upload pod 2 local GPT-2 and Qwen3 weights to HuggingFace to replace the mismatched ones currently there.

## Environment

- RunPod RTX 5090 (32 GB VRAM)
- Ubuntu 24, PyTorch 2.8.0+cu128
- transformers 4.57.x

## Date

April 24, 2026 02:30-04:10 (4.5 hours elapsed, sequential evals on single GPU)
