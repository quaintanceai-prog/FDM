# Re-tokenization check

`check_retokenization.py` re-tokenizes 500 stage-4 FDM blocks per host exactly as the prompt
pipeline does, reports the resulting lengths, and tests whether the 64 decoded level strings form
a uniquely decodable code (Sardinas-Patterson). Output in `results/retok.log`: lossless for
GPT-2, Qwen3 and Hermes3; not guaranteed for LFM2.5.

## Commands used

    for m in gpt2 qwen3 lfm2 hermes3; do
      python check_retokenization.py --model $m --test_data stage4_test/${m}_test.jsonl --n 500
    done > results/retok.log 2>&1

`stage4_test/<host>_test.jsonl` are the released single-block test files (field `fdm_text`),
the same ones used for Table 2. Tokenizers are loaded from the public base models named in
`CFG`; no finetuned checkpoint is involved.
