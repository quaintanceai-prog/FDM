# 5-hop FACT re-score on reference-present values

`nhop_refscore.py` regenerates the 965 five-hop test samples with the n-hop checkpoint and scores
FACT two ways: on all consulted channel values (the paper's check) and only on values present in
the reference answer. Run on LFM2.5-1.2B, the only n-hop FDM checkpoint retained.
Result (`results/lfm2_5hop_refscore.json`): standard 82.9%, ceiling of the standard check 83.2%
(810 of 4,825 consulted values are absent from the reference), reference-only 99.3%.
