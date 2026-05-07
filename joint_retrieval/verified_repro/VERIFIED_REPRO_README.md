# Joint Retrieval Verified Reproduction (April 24, 2026)

Purpose: Confirm the pod 2 local weights that produced prior reproduction numbers in `joint_retrieval/second_run/` reproduce consistently under fresh evaluation.

## Method

For each of 4 models:
1. Load weights from `/workspace/verified_repro/{model}/`
2. Load test data from same folder  
3. Run eval.py with deterministic seed
4. Compare joint accuracy to prior reproduction

## Results

| Model | Prior Repro (second_run/) | Verified Repro | Delta |
|---|---|---|---|
| GPT-2 | 93.7% | 93.7% (1405/1500) | 0.00 |
| Qwen3 | 99.20% | 99.20% | 0.00 |
| LFM2.5 | 2.07% | 2.07% | 0.00 |
| Hermes3 | 99.93% | 99.93% (1499/1500) | 0.00 |

**All 4 models reproduce exactly.** Weights are confirmed as the ones that produced the Table 4 (joint retrieval) paper results.

## LFM2.5 high-frequency collapse (per-channel detail)

LFM2.5 shows a clean frequency-dependent accuracy rolloff:

| Channel | Frequency | Accuracy |
|---|---|---|
| ch 8-34 | 9-35 Hz | ~100% |
| ch 35 | 36 Hz | 98.93% |
| ch 36 | 37 Hz | 86.13% |
| ch 37 | 38 Hz | 47.53% |
| ch 38 | 39 Hz | 27.27% |
| ch 39 | 40 Hz | 15.93% (near random for 4-option channel) |

The joint accuracy of 2.07% is entirely explained by the high-frequency collapse:

P(all 32 correct) ≈ 0.8613 × 0.4753 × 0.2727 × 0.1593 × (others at ~100%)
                  = 1.78% (observed: 2.07%, within statistical noise)

This is a mechanistic finding: LFM2.5's hybrid conv-attention architecture acts as a low-pass filter with cutoff around 35-36 Hz. Hermes3 and Qwen3 don't show this pattern.

## Weight md5s (pod 2 local)

These md5s identify the weights that reproduce the paper:

| Model | pod 2 local md5 |
|---|---|
| GPT-2 | 596e3ad22878b012c4a6d197e892e4a0 |
| Qwen3 | 873b6ffa... |
| LFM2.5 | da251c78... |
| Hermes3 | 696db498... |

## HuggingFace upload status

| Model | HF repo | Match? | Action |
|---|---|---|---|
| GPT-2 | <ANON>/eidetic-fdm-40ch-turbo-v3 | NO | **needs upload** |
| Qwen3 | <ANON>/fdm-40ch-turbo-v3-qwen3 | NO | **needs upload** |
| LFM2.5 | <ANON>/fdm-40ch-turbo-v3-lfm2 | YES | ok |
| Hermes3 | <ANON>/fdm-40ch-hermes3-3b | YES | ok |

## Environment

- RunPod RTX 5090 (32 GB VRAM)
- Ubuntu 24, PyTorch 2.8.0+cu128, transformers 4.57.x

## Date

April 24, 2026 02:30-04:10 (sequential evals, one GPU)
