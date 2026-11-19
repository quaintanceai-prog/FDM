# Re-tokenization check

`check_retokenization.py` re-tokenizes 500 stage-4 FDM blocks per host exactly as the prompt
pipeline does, reports the resulting lengths, and tests whether the 64 decoded level strings form
a uniquely decodable code (Sardinas-Patterson). Output in `results/retok.log`: lossless for
GPT-2, Qwen3 and Hermes3; not guaranteed for LFM2.5.
