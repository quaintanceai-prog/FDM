# Section 5.2: Position-Range Dependence

Two-block fine-tuning establishes a position-range policy: the host
learns FDM decomposition specifically at the trained position ranges.
Shifting or swapping these ranges collapses joint retrieval to 0%%,
despite identical FDM content.

## Scripts

- `carrier_head_position_independence.py` — sliding-window evaluation
  across position offsets.
- `carrier_head_position_shift_eval.py` — fixed-offset and swapped-range
  ablations.

## Results

| Position config | Slot A | Slot B | Joint (32) |
|-----------------|-------:|-------:|-----------:|
| Baseline (trained) | 99.58%% | 99.72%% | 91.25%% |
| Shifted            | 52.08%% | 52.11%% | 0.00%%  |
| Swapped            | 24.45%% | 15.58%% | 0.00%%  |

The carrier head experiments use 16 channels × 16 levels and run
across three seeds to confirm position-range specificity.
