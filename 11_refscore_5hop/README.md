# 5-hop FACT re-score on reference-present values

`nhop_refscore.py` regenerates the 965 five-hop test samples with the n-hop checkpoint and scores
FACT two ways: on all consulted channel values (the paper's check) and only on values present in
the reference answer. Run on LFM2.5-1.2B, the only n-hop FDM checkpoint retained.
Result (`results/lfm2_5hop_refscore.json`): standard 82.9%, ceiling of the standard check 83.2%
(810 of 4,825 consulted values are absent from the reference), reference-only 99.3%.

## Commands used

    python nhop_composition/nhop_training_multi.py generate lfm2
    # -> nhop_training_data_lfm2/nhop_test.jsonl (seed 77777; 5,000 samples, 965 with nhops=5)

    python nhop_refscore.py lfm2 --model_path <nhop-fdm-lfm2 checkpoint> \
           --test_data nhop_training_data_lfm2/nhop_test.jsonl 2>&1 | tee results/refscore.log

The checkpoint is the LFM2.5 n-hop FDM model produced by `nhop_training_multi.py train lfm2`.
Generation is greedy with `max_new_tokens=250`, matching the n-hop evaluation.
