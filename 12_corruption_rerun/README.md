# Corruption conditions, re-run on the current LFM2.5 checkpoints

Re-run of `corruption_experiments/` on LFM2.5-1.2B with the retained n-hop FDM and n-hop plaintext
checkpoints: 150 held-out n-hop test samples per condition, `max_new_tokens=400`, the [MEMORY] block
padded in its second half (block) or at every other token (stride). Context = fraction of the 32
Context-channel NAME=VALUE pairs reproduced verbatim.

| Condition | Plaintext Decision | FDM Decision | Plaintext Context | FDM Context |
|---|---|---|---|---|
| clean  | 100.0 | 100.0 | 91.0 | 99.6 |
| block  | 100.0 |  90.7 | 32.3 | 40.1 |
| stride |  94.7 |  94.7 | 23.2 | 49.8 |

Commands:

    python nhop_composition/nhop_training_multi.py generate lfm2
    python nhop_composition/nhop_plaintext_baseline.py generate lfm2
    python FDM_eval_stress_nhop.py <nhop-fdm-lfm2 checkpoint>   nhop_training_data_lfm2/nhop_test.jsonl  results/lfm2_fdm.json
    python eval_stress_nhop.py     <nhop-plaintext-lfm2 checkpoint> nhop_plaintext_data_lfm2/nhop_test.jsonl results/lfm2_plain.json

Qwen3 was not re-run: its n-hop FDM checkpoint was not retained.
