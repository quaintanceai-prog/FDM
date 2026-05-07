# Cold-Start Reproduction (April 24, 2026)

This directory documents an end-to-end reproduction test starting from a
fresh environment with NO prior local state — only the public repo and
HuggingFace artifacts.

## Protocol

1. `git clone git@github.com:<ANON>/ASK_FDM_CHECK.git`
2. Download 4 model weights via huggingface_hub.snapshot_download:
   - <ANON>/eidetic-fdm-40ch-turbo-v3 (GPT-2)
   - <ANON>/fdm-40ch-turbo-v3-qwen3
   - <ANON>/fdm-40ch-turbo-v3-lfm2
   - <ANON>/fdm-40ch-hermes3-3b
3. Download test data:
   - <ANON>/fdm-40ch-joint-retrieval-test-data
4. MD5-verify downloaded weights match expected values
5. Run eval.py for each model sequentially

Automated via `cold_start_repro.sh`.

## Result

| Model | Paper | Cold-start | Delta |
|---|---|---|---|
| GPT-2 | 93.70% | 93.67% | -0.03 |
| Qwen3 | 99.20% | 99.20% | +0.00 |
| LFM2.5 | 2.07% | 2.07% | -0.00 |
| Hermes3 | 99.93% | 99.93% | +0.00 |

**All 4 models reproduce paper targets within 0.5% tolerance.**
Maximum delta: 0.03% (0.47 samples out of 1500), within stochastic
autoregressive generation noise on different hardware.

## Environment

- RunPod RTX 5090 (32 GB VRAM)
- Ubuntu 24, PyTorch 2.8.0+cu128
- transformers 4.57.x, huggingface_hub latest
- Total runtime: ~4.5 hours sequential evals on single GPU

## Date

April 24, 2026 04:30-09:17 UTC
