# Section 4: Split-Substrate Architecture

Split-substrate FDM memory composes a parametric KV cache region with
an in-context FDM token region. Both are read by a single attention
mechanism in one forward pass.

## Scripts

- `fdm_split_source_unified.py` — main split-substrate training and
  evaluation (model-agnostic; uses `_get_vocab(model_path)` to select
  the correct tokenizer vocab size for Qwen3, Hermes3, LFM2.5, GPT-2).
- `fdm_split_source_hybrid_hermes3.py` — Hermes3-specific split-source
  training driver.
- `fdm_in_weights_kv.py` — in-weights KV cache injection infrastructure
  (frozen-injection baseline used for the naive composition test).

## Results

- `default_memory_probe.json` — substrate isolation probe.
- `naive_composition_baseline.json` — 0%% joint accuracy when a
  single-block-trained host attempts split-substrate composition
  without two-block fine-tuning (Table: Naive composition).
