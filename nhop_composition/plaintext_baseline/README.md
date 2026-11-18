# Plaintext N-Hop Baseline

Matched plaintext baseline for the N-hop compositional reasoning experiment.
Isolates what FDM encoding contributes versus plaintext retrieval under
identical compositional training.

## Experiment design

Following the docstring of `nhop_plaintext_baseline.py`:

- **Condition A (FDM)**: Base model → FDM training → N-hop training on FDM memory
- **Condition B (Plaintext)**: Base model → N-hop training on plaintext memory

Both conditions use identical:
- MEMORY_SCHEMAS (40 channels)
- NHOP_TEMPLATES (10 templates covering 1-5 hops)
- Question/answer format (decision + Context block)
- Train/val/test split (seed=77777)

Only the `[MEMORY]` block content differs:
- FDM: quantized sinusoidal signal tokenized via pseudorandom bijection
- Plaintext: `NAME=VALUE` pairs newline-separated

## Results summary

| Model | Hop 1 | Hop 2 | Hop 3 | Hop 4 | Hop 5 |
|---|---|---|---|---|---|
| LFM2.5-1.2B  | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Qwen3-0.6B   | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| Hermes3-3B   | pending | pending | pending | pending | pending |

Both plaintext N-hop baselines achieve 100% action accuracy across all hop
counts on clean memory, matching the FDM N-hop results (Table 6 in paper).

## Implications for the paper

Under clean memory conditions, plaintext and FDM achieve comparable
compositional accuracy when given matching compositional training. This
result does NOT invalidate FDM's contribution — it confirms that the
relevant comparison between the two encodings is not clean accuracy,
but behavior under stress (memory corruption, reduced sample rate,
cross-length transfer).

See the main paper for:
- Signal corruption experiments (Table 7) — distinguishes FDM's
  signal-dependent retrieval from parametric fallback
- Channel swap experiments (Table 8) — confirms value-level read from
  signal tokens
- Cross-length transfer (Table 3) — tests generalization without retraining

## Status

- ✅ LFM2.5-1.2B: trained, evaluated
- ✅ Qwen3-0.6B: trained, evaluated
- ⏳ Hermes3-3B: training pending

Model weights to be uploaded to HuggingFace:
- `<ANON>/nhop-plaintext-lfm2` (2.2 GB)
- `<ANON>/nhop-plaintext-qwen3` (1.2 GB)
- Hermes3 pending training completion

## Reproduction

```bash
