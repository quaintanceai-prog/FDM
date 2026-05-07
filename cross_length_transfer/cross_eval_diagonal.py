"""
Cross-length diagonal evaluation.
Loads any trained Qwen3 FDM model and evaluates on any test data file.
Usage:
  python cross_eval_diagonal.py --model MODEL_DIR --test TEST_JSONL --label LABEL
"""
import numpy as np
import torch
import json
import sys
import argparse
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import constants from main script
sys.path.insert(0, '.')
from eidetic_qwen3_fdm_e2e import MEMORY_SCHEMAS, NUM_CHANNELS, DTYPE

def evaluate(model_path, test_path, label=""):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Cross-eval: {label}")
    print(f"  Model: {model_path}")
    print(f"  Test:  {test_path}")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_path, torch_dtype=DTYPE, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    test_samples = [json.loads(l) for l in open(test_path)]
    print(f"Evaluating on {len(test_samples)} samples...\n")

    action_correct = 0
    ch_correct = {}
    ch_total = {}
    extra_correct = 0
    extra_total = 0
    total = 0

    for i, s in enumerate(tqdm(test_samples, desc=f"Eval [{label}]")):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=250, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        expected = s['answer']

        action_ok = False
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                   'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_ok = True
                break
        if action_ok: action_correct += 1

        memory = s['memory']
        for ch in range(8, NUM_CHANNELS):
            expected_val = memory[str(ch)]
            extra_total += 1
            ch_total[ch] = ch_total.get(ch, 0) + 1
            ch_correct[ch] = ch_correct.get(ch, 0)
            if expected_val in response:
                extra_correct += 1
                ch_correct[ch] += 1

        total += 1

    print(f"\nRESULTS [{label}]")
    print(f"  Action: {100*action_correct/total:.1f}%")
    print(f"  Extra:  {100*extra_correct/max(extra_total,1):.1f}%")
    print("\nPer-channel (ch8-39):")
    for ch in range(8, NUM_CHANNELS):
        t = ch_total.get(ch, 0)
        c = ch_correct.get(ch, 0)
        pct = 100 * c / t if t else 0
        flag = " <<<" if pct < 90 else ""
        print(f"  ch{ch:2d} ({MEMORY_SCHEMAS[ch][0]:<12}): {pct:.1f}%{flag}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True)
    parser.add_argument('--test', required=True)
    parser.add_argument('--label', default='')
    args = parser.parse_args()
    evaluate(args.model, args.test, args.label)
