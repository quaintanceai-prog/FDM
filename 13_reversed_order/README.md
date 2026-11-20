# Reversed emission order, LFM2.5-1.2B

Reviewer dteT (Q3) asked for the Context block emitted in a shuffled channel order, to separate
emission position from carrier frequency. We ran the most informative fixed permutation: the
exact reverse (ch39 emitted first, ch8 last), so the highest carriers are written first and the
lowest last. A random fixed permutation is a straightforward follow-up with the same script
(`--order random --seed N`) and has not been run.

`shuffled_order.py` finetunes the Table 2 LFM2.5 checkpoint to emit the 32 Context channels in
the chosen order, then evaluates on the stage-4 test split at a 512-token cap and re-scores the
same responses cut at 250 tokens, per channel and per emission position (strict NAME=VALUE).

Run here: `--order reversed`, 1 epoch then 3 more (4 total), lr 1e-5, batch 4, training set =
the generator's stage-4 train split (25,500 samples).

Results (`results/`): `_lfm2_reversed.json` after 1 epoch (1,500 test samples);
`_ep4_lfm2_reversed.json` after 4 epochs (500 samples); `*_samples.json` hold every response.

- At the 250-token cap the errors move to the channels emitted last, now the lowest carriers
  (TEAM 0.0, REGION 1.8, PHASE 23.0, COMM 70.6%); all earlier-emitted channels are identical
  between the 250 and 512 caps. The LFM2.5 tail errors in Table 2 were emission position, not
  carrier frequency.
- At the 512-token cap the reversed-order model reads held-out memories at 77-97% per channel
  (all 32 correct in 6.2% of answers) after 4 epochs, versus 100% in the trained order; training
  loss is at zero, so more training does not help.
- A control finetune with the order left unchanged (`--order identity`, same settings) is in
  progress to separate the effect of the order from the effect of the finetuning procedure.

Commands:

    python joint_retrieval/verified_repro/lfm2/generator.py generate
    python shuffled_order.py lfm2 --model_path <table-2 lfm2 checkpoint> \
        --train_data fdm_40ch_turbo_v3_lfm2_stage4_train.jsonl \
        --test_data fdm_40ch_turbo_v3_lfm2_stage4_test.jsonl --order reversed --epochs 1
    python shuffled_order.py lfm2 --model_path results_shuffled/lfm2_reversed_ckpt \
        --train_data fdm_40ch_turbo_v3_lfm2_stage4_train.jsonl \
        --test_data fdm_40ch_turbo_v3_lfm2_stage4_test.jsonl --order reversed --epochs 3 --n_eval 500
