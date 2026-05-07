# ASK_FDM_CHECK

Evaluation and benchmarking suite for FDM (Frequency-Domain Memory) experiments.

## Directory Structure

Each experiment category has its own directory with a `results/` subdirectory for output files:

```
├── channel_density/
│   └── results/
├── cross_length_transfer/
│   └── results/
├── frequency_gradient/
│   └── results/
├── full_paper_results/
│   └── results/
├── joint_retrieval/
│   └── results/
├── nhop_composition/
│   └── results/
├── scaling_replication/
│   └── results/
├── signal_corruption/
│   └── results/
└── README.md
```

## Experiment Categories

| Directory | Description |
|-----------|-------------|
| `channel_density` | Channel density scaling experiments (Paper Table 4) |
| `cross_length_transfer` | Cross-length transfer evaluation |
| `frequency_gradient` | Frequency gradient analysis |
| `full_paper_results` | Consolidated paper results |
| `joint_retrieval` | Joint retrieval benchmarks (including corruption variants & cross-length transfer) |
| `nhop_composition` | N-hop composition experiments |
| `scaling_replication` | Scaling law replication experiments |
| `signal_corruption` | Signal corruption robustness tests |

## Usage

Result files (`.json`, `.csv`, etc.) should be placed in the `results/` subdirectory of each category. Scripts and configs go in the category root.


---

## Supplement: Sections 4–5

# FDM: Split-Substrate Composition (Sections 4–5)

Reproducibility supplement for the paper *Frequency-Division Multiplexed
Memory: A Composable Substrate for Structured Facts in Transformer Inputs
and Caches*. This supplement covers Sections 4 and 5: split-substrate
architecture, two-block fine-tuning, position-range ablations, and the
partition-ratio plateau.

Sections 1–3 (single-block FDM encoding, mechanistic validation, and
corruption asymmetry) are released separately in the parent repository.

## Layout

| Directory | Contents |
|-----------|----------|
| `04_split_substrate/` | Section 4 — Split-substrate architecture (parametric KV + in-context FDM) |
| `05_two_block_finetuning/` | Section 5.1 — Two-block fine-tuning recovers split-substrate composition |
| `06_position_range/` | Section 5.2 — Position-range dependence (ablations and carrier head) |
| `07_partition_ratio_sweep/` | Section 5.3 — Partition-ratio plateau (5-seed Qwen3 + 3-seed Hermes3) |

Inside each section:

- `README.md` — section-specific notes and result summaries
- `scripts/` — Python training and evaluation scripts, plus shell drivers
- `results/` — JSON result files used to populate the corresponding tables

## Models and datasets

Model checkpoints and training datasets are hosted on HuggingFace under
anonymized identifiers (referred to as `<ANON>/...` in the code). Specific
URLs will be released after the review period.

## Host architectures

Experiments span four transformer architectures: GPT-2 Medium,
Qwen3-0.6B-Base, LFM2.5-1.2B-Base, and Hermes-3-Llama-3.2-3B. Code is
model-agnostic and selects the correct vocabulary size via a
`_get_vocab(model_path)` helper; see
`05_two_block_finetuning/scripts/fdm_two_block_training.py`.
