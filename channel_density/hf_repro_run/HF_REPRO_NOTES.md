# Density Reproduction from HuggingFace (April 24, 2026)

End-to-end reproduction of Table 4 density scaling results using:
- Publicly available HuggingFace checkpoints
- `channel_density_scaling_multi.py` eval script from this repo
- Seed 99999 (paper-era baseline seed)

## Reproduction result

| Model | Config | Paper | Repro | Delta |
|---|---|---|---|---|
| LFM2.5 | 40ch | 64.31% | 64.31% | +0.00 |
| LFM2.5 | 20ch_dense | 87.47% | 87.47% | +0.00 |
| LFM2.5 | 20ch_spread | 83.35% | 83.35% | +0.00 |
| LFM2.5 | 10ch_dense | 100.00% | 100.00% | +0.00 |
| LFM2.5 | 10ch_spread | 97.90% | 97.90% | +0.00 |
| Hermes3 | 40ch | 28.41% | 28.41% | +0.00 |
| Hermes3 | 20ch_dense | 40.68% | 40.68% | +0.00 |
| Hermes3 | 20ch_spread | 36.52% | 36.52% | +0.00 |
| Hermes3 | 10ch_dense | 74.80% | 74.80% | +0.00 |
| Hermes3 | 10ch_spread | 80.90% | 80.90% | +0.00 |

All LFM2.5 and Hermes3 configs reproduce paper to exact 0.00 delta.

## GPT-2

GPT-2 density was retrained today as new `<ANON>/fdm-density-gpt2-repro-*`
HuggingFace repos. No paper-era GPT-2 density baseline existed, so these results
establish the reference.

| Model | Config | Repro |
|---|---|---|
| GPT-2 | 40ch | (see results/gpt2/40ch.json) |
| GPT-2 | 20ch_dense | (see results/gpt2/20ch_dense.json) |
| GPT-2 | 20ch_spread | (see results/gpt2/20ch_spread.json) |
| GPT-2 | 10ch_dense | (see results/gpt2/10ch_dense.json) |
| GPT-2 | 10ch_spread | (see results/gpt2/10ch_spread.json) |

## Reproduction protocol

1. Clone this repo
2. Download 15 checkpoints from HuggingFace:
   - `<ANON>/fdm-density-gpt2-repro-{40ch,20ch_dense,20ch_spread,10ch_dense,10ch_spread}`
   - `<ANON>/fdm-density-lfm2-{40ch,20ch_dense,20ch_spread,10ch_dense,10ch_spread}`
   - `<ANON>/fdm-density-hermes3-{40ch,20ch_dense,20ch_spread,10ch_dense,10ch_spread}`
3. Run `channel_density/channel_density_scaling_multi.py eval` with each config
4. Compare mean_per_channel_acc to paper-era baselines in `original_run/results/`

## Environment

- RunPod RTX 5090 (32 GB VRAM)
- Ubuntu 24, PyTorch 2.8.0+cu128
- transformers 4.57.x, huggingface_hub latest

## Date

April 24, 2026
