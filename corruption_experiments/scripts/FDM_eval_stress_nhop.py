"""
Corruption eval for FDM N-hop compositionally-trained models.

Tests FDM model under clean/block/stride memory corruption using the
trained N-hop question format. Companion script to eval_stress_nhop.py
(which handles plaintext).
"""

import json, os, sys, random
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
import re


def corrupt_none(input_ids, start, end, pad_id):
    return input_ids.clone()

def corrupt_block(input_ids, start, end, pad_id):
    out = input_ids.clone()
    half = start + (end - start) // 2
    out[0, half:end] = pad_id
    return out

def corrupt_stride(input_ids, start, end, pad_id):
    out = input_ids.clone()
    for i in range(start + 1, end, 2):
        out[0, i] = pad_id
    return out

CORRUPTIONS = {"clean": corrupt_none, "block": corrupt_block, "stride": corrupt_stride}


def find_mem_fdm(tokenizer, fdm_text, input_ids):
    """FDM memory = the fdm_text portion at start of prompt."""
    fdm_tokens = tokenizer.encode(fdm_text, add_special_tokens=False)
    return 0, min(len(fdm_tokens), input_ids.shape[1])


def score_decision(expected, generated):
    """Template prefix match (e.g. FULL_5HOP, TACTICAL_4HOP, THREAT_CHAIN)."""
    exp_first = expected.split(":")[0].split()[0]
    return exp_first in generated[:100]


def score_context_block(expected, generated):
    """Count NAME=VALUE matches in generated Context block."""
    if "Context:" not in expected:
        return 0, 0
    exp_ctx = expected.split("Context:", 1)[1]
    exp_pairs = re.findall(r"([A-Z][A-Z0-9_]*)=([A-Z][A-Z0-9_]+)", exp_ctx)
    matches = 0
    for k, v in exp_pairs:
        if f"{k}={v}" in generated:
            matches += 1
    return matches, len(exp_pairs)


def eval_fdm_nhop(model_path, test_file, output_json, n_samples=150, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n{'='*60}\nFDM N-HOP CORRUPTION EVAL")
    print(f"Model: {model_path}")
    print(f"Data: {test_file}\n{'='*60}\n")
    
    device = torch.device('cuda')
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, trust_remote_code=True
    ).to(device)
    model.eval()
    pad_id = tokenizer.pad_token_id
    
    with open(test_file) as f:
        samples = [json.loads(line) for line in f]
    random.shuffle(samples)
    samples = samples[:n_samples]
    print(f"Using {len(samples)} samples\n")
    
    results = {"encoding": "fdm_nhop", "model_path": model_path, "conditions": {}}
    
    for cond in ["clean", "block", "stride"]:
        print(f"--- {cond} ---")
        dec_correct = 0
        ctx_matched = 0
        ctx_total = 0
        records = []
        fn = CORRUPTIONS[cond]
        
        for s in tqdm(samples):
            fdm_text = s['fdm_text']
            prompt = fdm_text + "\nQuestion: " + s['question'] + "\nAnswer:"
            expected = s['answer']
            
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True,
                             max_length=4096).to(device)
            mem_start, mem_end = find_mem_fdm(tokenizer, fdm_text, inputs.input_ids)
            corrupted = fn(inputs.input_ids, mem_start, mem_end, pad_id)
            attn = torch.ones_like(corrupted)
            
            with torch.no_grad():
                out = model.generate(
                    input_ids=corrupted, attention_mask=attn,
                    max_new_tokens=400, do_sample=False,
                    pad_token_id=pad_id,
                )
            gen = tokenizer.decode(out[0, corrupted.shape[1]:], skip_special_tokens=True).strip()
            
            dec_ok = score_decision(expected, gen)
            ctx_m, ctx_t = score_context_block(expected, gen)
            
            if dec_ok: dec_correct += 1
            ctx_matched += ctx_m
            ctx_total += ctx_t
            
            records.append({
                "expected": expected[:250], "generated": gen[:250],
                "decision_ok": dec_ok, "ctx_matched": ctx_m, "ctx_total": ctx_t,
                "nhops": s.get('nhops'),
            })
        
        dec_acc = dec_correct / len(samples)
        ctx_acc = ctx_matched / max(ctx_total, 1)
        print(f"  Decision: {dec_acc:.1%}  Context channels: {ctx_acc:.1%} ({ctx_matched}/{ctx_total})\n")
        
        results["conditions"][cond] = {
            "decision_accuracy": dec_acc,
            "context_accuracy": ctx_acc,
            "ctx_matched": ctx_matched, "ctx_total": ctx_total,
            "samples": records[:10],
        }
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*60}\nFDM SUMMARY\n{'='*60}")
    print(f"{'Cond':<10} {'Decision':>12} {'Context':>12}")
    for c in ["clean", "block", "stride"]:
        d = results["conditions"][c]
        print(f"{c:<10} {d['decision_accuracy']:>11.1%} {d['context_accuracy']:>11.1%}")
    print(f"\nSaved: {output_json}")


if __name__ == "__main__":
    # Usage: python FDM_eval_stress_nhop.py <model_dir> <test_jsonl> <output_json>
    eval_fdm_nhop(sys.argv[1], sys.argv[2], sys.argv[3])
