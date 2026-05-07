# Channel Swap Experiment — Signal-Dependent Retrieval Test

Tests whether FDM-trained models read channel values from the signal (non-parametric)
or recall from weights (parametric retrieval).

## Method

Two FDM encodings differing in exactly one channel value are generated. The model is
run on both and the response checked for the expected change. Each model uses its own
encoder with the correct vocabulary size matching training.

| Model   | Vocab size |
|---------|-----------|
| GPT-2   | 50,257    |
| Qwen3   | 151,936   |
| LFM2.5  | 65,536    |
| Hermes3 | 128,256   |

## Results (n=100 per model per experiment)

| Model   | STATUS_swap | META_emergency | META_lockdown |
|---------|------------|----------------|---------------|
| GPT-2   | 100%       | 100%           | 100%          |
| Qwen3   | 100%       | 100%           | 100%          |
| LFM2.5  | 100%       | 100%           | 100%          |
| Hermes3 | 100%       | 100%           | 100%          |

100% correct response change across all models and experiments rules out parametric retrieval.

## Usage
```bash
python channel_swap_eval_v2.py --model all --n_pairs 100
```
