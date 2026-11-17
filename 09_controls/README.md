# Generation cap, strict scoring and controls (Appendix F)

`baseline_controls.py` re-runs the first 500 of the 1,500 stage-4 test samples per host
on the Table 2 checkpoints with `max_new_tokens=512`, scores each response at the full
length and cut at 250 tokens, under the bare-value check (Tables 2-4) and a strict
`NAME=VALUE` check, and runs two controls on the same samples: the finetuned host with an
empty `[MEMORY]` block (`none`) and the base model on the plaintext block with no
finetuning (`plain_base`). `chance` computes chance/ceiling levels and `tokens` the
FDM vs plaintext token cost (Table 10).

`run_controls.sh` runs everything; set the checkpoint and test-file paths at the top.
`results/<model>_<condition>.json` are the summaries, `*_samples.json` the per-sample
prompts and responses, `*.log` the run logs; `controls_2026-11-17.log` is the full run.
