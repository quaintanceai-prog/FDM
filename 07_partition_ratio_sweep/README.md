# Section 5.3: Partition-Ratio Plateau

A sweep across 9 partition ratios (0/32 through 32/0) on Qwen3-0.6B
(5 seeds) and Hermes3-3B (3 seeds), each at n=200 evaluation samples.
Demonstrates that the parametric/in-context boundary can move from
all-context to 88%% parametric without systematic slot-level
degradation — a continuous plateau.

## Scripts

- `fdm_split_ratio_sweep.py` — main sweep driver: trains write head
  + two-block evaluation per (seed, partition) cell.
- `aggregate_5seed.py` — Wilson 95%% CI aggregation across seeds.
- `run_5seed_sweep.sh` — Qwen3 5-seed sweep runner.
- `run_hermes3_3seed_sweep.sh` — Hermes3 3-seed sweep runner.

## Results

The plateau spans partitions 0/32 through 28/4 on both architectures.
Slot-level retrieval stays within 99.27%–99.63%% on Qwen3 and
≥99.98%% on Hermes3. Sample-level joint retrieval (all 32 channels
correct simultaneously) ranges 83.0%–91.6%% on Qwen3 and
99.50%–100%% on Hermes3.

The 32/0 fully-parametric setting is the only consistent failure:
no seed converges within the 4K-step training budget.
