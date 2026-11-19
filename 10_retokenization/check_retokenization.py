import argparse, json
import numpy as np
from transformers import AutoTokenizer

CFG = {
    "gpt2":    ("openai-community/gpt2-medium",       50257),
    "qwen3":   ("Qwen/Qwen3-0.6B-Base",               151936),
    "lfm2":    ("LiquidAI/LFM2.5-1.2B-Base",          65536),
    "hermes3": ("NousResearch/Hermes-3-Llama-3.2-3B", 128256),
}

def uniquely_decodable(words):
    words = set(words)
    if "" in words: return False
    def dangling(a, b):
        out = set()
        for x in a:
            for y in b:
                if x != y and y.startswith(x): out.add(y[len(x):])
        return out
    S = dangling(words, words)
    seen = set()
    while S:
        if S & words: return False
        if S <= seen: return True
        seen |= S
        S = dangling(S, words) | dangling(words, S)
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--test_data", default=None)
    a = ap.parse_args()
    rs = np.random.RandomState(7)
    for name, (mid, V) in CFG.items():
        if a.model and name != a.model: continue
        tok = AutoTokenizer.from_pretrained(mid, trust_remote_code=True)
        token_map = np.random.RandomState(42).choice(V, size=64, replace=False)
        strings = [tok.decode([int(t)]) for t in token_map]
        ud = uniquely_decodable(strings)
        changed = 0; lens = []; n_eq_len = 0
        if a.test_data:
            samples = [json.loads(l) for l in open(a.test_data)][: a.n]
            seqs = [(s.get("token_ids"), s["fdm_text"]) for s in samples]
        else:
            seqs = []
            for _ in range(a.n):
                ids = [int(token_map[x]) for x in rs.randint(0, 64, size=512)]
                seqs.append((ids, tok.decode(ids)))
        for ids, text in seqs:
            re_ids = tok.encode(text, add_special_tokens=False)
            lens.append(len(re_ids))
            if ids is not None and re_ids != ids: changed += 1
            if len(re_ids) == 512: n_eq_len += 1
        print(f"{name:8s} vocab={V:6d}  changed={changed}/{len(seqs)}  len==512: {n_eq_len}/{len(seqs)}  "
              f"re-tokenized length min/mean/max={min(lens)}/{np.mean(lens):.1f}/{max(lens)}  "
              f"uniquely decodable alphabet: {ud}")
        sp = sum(1 for s in strings if s.startswith(" ") or s.startswith("Ġ"))
        print(f"         {sp}/64 mapped tokens begin with a space; the rest are word-internal pieces that can merge")

if __name__ == "__main__":
    main()
