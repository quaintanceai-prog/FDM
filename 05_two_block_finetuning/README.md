# Section 5.1: Two-Block Fine-Tuning

A brief additional fine-tuning stage (60K samples, ~10K steps) teaches
the host model to apply frequency decomposition at two designated
position ranges. This recovers split-substrate composition that the
naive single-block configuration fails to support (Section 4).

## Scripts

- `fdm_two_block_training.py` — canonical two-block trainer (Qwen3-0.6B).
- `fdm_two_block_training_hermes3.py` — Hermes3-3B variant
  (lr 1e-4 for stability).
- `fdm_two_block_training_lfm2.py` — LFM2.5-1.2B variant.
- `eval_split_source_n200.py` — split-source evaluation (Qwen3).
- `eval_split_source_n200_hermes3.py` — split-source evaluation (Hermes3).
- `eval_two_block_n200_lfm2.py` — two-block evaluation (LFM2).

## Results (n=400, two seed rounds combined)

| Architecture | Slot A | Slot B | Joint (32) |
|--------------|-------:|-------:|-----------:|
| Qwen3-0.6B   | 99.58%% | 99.72%% | 91.25%% |
| Hermes3-3B   | 99.98%% | 100.00%% | 99.75%% |
