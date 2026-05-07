"""
STRICT POSITIONAL CORRUPTION EVAL - FDM Edition

Tests N-hop trained FDM models under corruption with strict scoring:
- Context block parsed as ordered list
- Each slot must match KEY=VALUE at correct position
- No partial credit for correct value at wrong position
- No credit for substring presence anywhere

Also stores ALL 150 samples (not just 10) for downstream analysis.

Usage:
  python eval_strict_fdm.py <model_dir> <test_jsonl> <output_json>
"""

import json, os, sys, random, re
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm


# ============================================================
# Corruption patterns
# ============================================================

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
    """FDM memory = fdm_text portion at start of prompt."""
    fdm_tokens = tokenizer.encode(fdm_text, add_special_tokens=False)
    return 0, min(len(fdm_tokens), input_ids.shape[1])


# ============================================================
# STRICT SCORING (same as plaintext version)
# ============================================================

def parse_context_block(text):
    if "Context:" not in text:
        return None
    ctx = text.split("Context:", 1)[1]
    if "Context:" in ctx:
        ctx = ctx.split("Context:", 1)[0]
    pairs = re.findall(r"([A-Z][A-Z0-9_]*)\s*=\s*([A-Z][A-Z0-9_]*)", ctx)
    return pairs


def score_strict_positional(expected, generated):
    exp_pairs = parse_context_block(expected)
    gen_pairs = parse_context_block(generated)
    
    if exp_pairs is None:
        return 0, 0, False
    total = len(exp_pairs)
    
    if gen_pairs is None:
        return 0, total, False
    
    matched = 0
    for i, (ek, ev) in enumerate(exp_pairs):
        if i < len(gen_pairs):
            gk, gv = gen_pairs[i]
            if ek == gk and ev == gv:
                matched += 1
    
    exact_sample = (matched == total)
    return matched, total, exact_sample


def score_template_strict(expected, generated):
    if "Context:" not in expected:
        return False
    exp_prefix = expected.split("Context:", 1)[0].strip()
    if "Context:" not in generated:
        return False
    gen_prefix = generated.split("Context:", 1)[0].strip()
    return exp_prefix == gen_prefix


def score_template_loose(expected, generated):
    exp_first = expected.split(":")[0].split()[0]
    return exp_first in generated[:100]


# ============================================================
# Main eval
# ============================================================

def eval_fdm_strict(model_path, test_file, output_json, n_samples=150, seed=42):
    random.seed(seed)
    torch.manual_seed(seed)
    
    print(f"\n{'='*60}")
    print(f"STRICT POSITIONAL FDM CORRUPTION EVAL")
    print(f"Model: {model_path}")
    print(f"Data: {test_file}")
    print(f"{'='*60}\n")
    
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
    
    results = {
        "encoding": "fdm_strict",
        "model_path": model_path,
        "scoring": "strict_positional",
        "n_samples": len(samples),
        "conditions": {},
    }
    
    for cond in ["clean", "block", "stride"]:
        print(f"--- {cond} ---")
        
        template_strict_correct = 0
        template_loose_correct = 0
        positional_matched = 0
        positional_total = 0
        exact_samples = 0
        
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
            
            tpl_strict = score_template_strict(expected, gen)
            tpl_loose = score_template_loose(expected, gen)
            pos_matched, pos_total, is_exact = score_strict_positional(expected, gen)
            
            if tpl_strict: template_strict_correct += 1
            if tpl_loose: template_loose_correct += 1
            positional_matched += pos_matched
            positional_total += pos_total
            if is_exact: exact_samples += 1
            
            records.append({
                "nhops": s.get('nhops'),
                "expected": expected[:250],
                "generated": gen[:250],
                "template_strict": tpl_strict,
                "template_loose": tpl_loose,
                "positional_matched": pos_matched,
                "positional_total": pos_total,
                "exact_match": is_exact,
            })
        
        N = len(samples)
        tpl_strict_acc = template_strict_correct / N
        tpl_loose_acc = template_loose_correct / N
        pos_acc = positional_matched / max(positional_total, 1)
        exact_acc = exact_samples / N
        
        print(f"  Template strict (prefix ==):    {tpl_strict_acc:.1%}")
        print(f"  Template loose (keyword):       {tpl_loose_acc:.1%}")
        print(f"  Positional context (per-slot):  {pos_acc:.1%} ({positional_matched}/{positional_total})")
        print(f"  Exact sample (all slots right): {exact_acc:.1%} ({exact_samples}/{N})\n")
        
        results["conditions"][cond] = {
            "template_strict_accuracy": tpl_strict_acc,
            "template_loose_accuracy": tpl_loose_acc,
            "positional_context_accuracy": pos_acc,
            "positional_matched": positional_matched,
            "positional_total": positional_total,
            "exact_sample_accuracy": exact_acc,
            "exact_samples": exact_samples,
            "n_samples": N,
            "samples": records,
        }
    
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*70}")
    print(f"STRICT FDM SUMMARY")
    print(f"{'='*70}")
    print(f"{'Cond':<10} {'TplStrict':>10} {'TplLoose':>10} {'Positional':>12} {'Exact':>10}")
    for c in ["clean", "block", "stride"]:
        d = results["conditions"][c]
        print(f"{c:<10} "
              f"{d['template_strict_accuracy']:>9.1%}  "
              f"{d['template_loose_accuracy']:>9.1%}  "
              f"{d['positional_context_accuracy']:>11.1%}  "
              f"{d['exact_sample_accuracy']:>9.1%}")
    print(f"\nSaved: {output_json}")


if __name__ == "__main__":
    eval_fdm_strict(sys.argv[1], sys.argv[2], sys.argv[3])
