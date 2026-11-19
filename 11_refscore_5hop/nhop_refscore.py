"""
5-hop FACT re-score: standard (all consulted channels) vs reference-present only.

Reviewer dteT (Q4): the META-chain 5-hop reference answer omits some consulted
values, so a perfect output caps at ~83% under the standard FACT check. This
re-generates the 5-hop test samples with the n-hop checkpoint and scores FACT
both ways, saving every response.

  python nhop_refscore.py qwen3 --model_path <nhop ckpt dir or HF id> \
         --test_data nhop_training_data_qwen3/nhop_test.jsonl [--n 965] [--hops 5]
"""
import argparse, json, os, re, torch
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--hops", type=int, default=5)
    ap.add_argument("--n", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=250)
    ap.add_argument("--out", default="results_refscore")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    samples = [json.loads(l) for l in open(a.test_data)]
    samples = [s for s in samples if s["nhops"] == a.hops]
    if a.n: samples = samples[: a.n]
    print(f"{len(samples)} samples with nhops={a.hops}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = torch.float32 if a.model == "gpt2" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=dtype, trust_remote_code=True).to(dev).eval()

    std_hit = std_tot = ref_hit = ref_tot = 0
    n_omitted = 0
    rows = []
    for s in tqdm(samples, desc=f"{a.model} {a.hops}-hop"):
        prompt = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        ids = tok.encode(prompt, return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=a.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id)
        resp = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
        ref = s["answer"]; mem = s["memory"]
        per = []
        for ch in s["channels"]:
            val = mem[str(ch)]
            in_ref = val in ref
            hit = val in resp
            std_tot += 1; std_hit += hit
            if in_ref: ref_tot += 1; ref_hit += hit
            else: n_omitted += 1
            per.append({"ch": ch, "val": val, "in_reference": in_ref, "in_response": hit})
        rows.append({"question": s["question"], "reference": ref, "response": resp, "channels": per})

    res = {
        "model": a.model, "hops": a.hops, "n_samples": len(samples),
        "fact_standard_pct": 100 * std_hit / std_tot,
        "fact_reference_only_pct": 100 * ref_hit / ref_tot,
        "consulted_values_scored": std_tot, "consulted_values_in_reference": ref_tot,
        "consulted_values_omitted_from_reference": n_omitted,
        "ceiling_standard_pct": 100 * ref_tot / std_tot,
    }
    print(json.dumps(res, indent=1))
    json.dump(res, open(f"{a.out}/{a.model}_{a.hops}hop_refscore.json", "w"), indent=1)
    json.dump(rows, open(f"{a.out}/{a.model}_{a.hops}hop_refscore_samples.json", "w"), indent=1)

if __name__ == "__main__":
    main()
