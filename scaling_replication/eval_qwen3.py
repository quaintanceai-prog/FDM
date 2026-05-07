"""
FDM Evaluation — Qwen3-0.6B-Base
Reproduces Table 1 results for Qwen3 (115K samples, 512-token budget).

Usage:
  python eval_qwen3.py
  python eval_qwen3.py --model_path /path/to/model --test_data /path/to/test.jsonl
"""

import json
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_PATH = "<ANON>/fdm-40ch-turbo-v3-qwen3"
TEST_DATA  = "hf://datasets/<ANON>/fdm-40ch-scaling-test-data/qwen3/fdm_40ch_turbo_v3_qwen3_stage4_test.jsonl"
DTYPE      = torch.bfloat16
MODEL_SHORT = "qwen3"

MEMORY_SCHEMAS = {
    0: ("SECRET",   ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT",    ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS",   ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP",   ["AVAILABLE", "UNAVAILABLE"]),
    6: ("RULE",     ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7: ("META",     ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",     ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",   ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",    ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",     ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",    ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",   ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",    ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",  ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",   ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",  ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",  ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",  ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",   ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",     ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",  ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",    ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION", ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",  ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",     ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE", ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",    ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION", ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",    ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",   ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",   ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",   ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",    ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",     ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2", ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE", ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",    ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}
NUM_CHANNELS = 40

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--test_data", default=TEST_DATA)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading Qwen3 from {args.model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=DTYPE, trust_remote_code=True
    )
    model.to(device)
    model.eval()

    test_samples = [json.loads(l) for l in open(args.test_data)]
    print(f"Evaluating on {len(test_samples)} samples...\n")

    action_correct = rule_correct = meta_correct = fact_correct = 0
    rule_correct_meta_none = rule_total_meta_none = 0
    extra_correct = extra_total = total = 0
    ch_correct = {}
    ch_total = {}

    for i, s in enumerate(tqdm(test_samples, desc=f"Evaluating [{MODEL_SHORT}]")):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=250, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        expected = s['answer']
        rule = s['rule']
        meta = s['meta']

        action_ok = False
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                   'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_ok = True
                break
        if action_ok: action_correct += 1

        if rule and rule in response: rule_correct += 1

        if meta == "NONE":
            rule_total_meta_none += 1
            if rule and rule in response: rule_correct_meta_none += 1

        meta_ok = False
        if meta == "EMERGENCY" and "EMERGENCY" in response: meta_ok = True
        elif meta == "LOCKDOWN" and "LOCKDOWN" in response: meta_ok = True
        elif meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY"]: meta_ok = True
        if meta_ok: meta_correct += 1

        facts = s['facts']
        fact_ok = False
        if s['question'].startswith("Should"):
            fact_ok = facts['AGENT'] in response
        elif s['question'].startswith("What is the risk"):
            fact_ok = facts['LOCATION'] in response
        elif s['question'].startswith("Is it safe"):
            fact_ok = facts['SECRET'] in response or facts['LOCATION'] in response
        if fact_ok: fact_correct += 1

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

        if i < 5:
            print(f"\n  [{MODEL_SHORT}] Sample {i+1} | Rule: {rule} | Meta: {meta}")
            print(f"  Expected: {expected[:100]}...")
            print(f"  Got:      {response[:100]}...")

    print(f"\n{'='*60}")
    print(f"QWEN3 40-CHANNEL FDM EVALUATION")
    print(f"{'='*60}")
    print(f"  Action:  {100*action_correct/total:.1f}%")
    print(f"  Rule:    {100*rule_correct/total:.1f}%  (aggregate)")
    if rule_total_meta_none > 0:
        print(f"  Rule (META=NONE): {100*rule_correct_meta_none/rule_total_meta_none:.1f}%  ({rule_correct_meta_none}/{rule_total_meta_none})")
    print(f"  Meta:    {100*meta_correct/total:.1f}%")
    print(f"  Fact:    {100*fact_correct/total:.1f}%")
    print(f"  Extra:   {100*extra_correct/max(extra_total,1):.1f}%")
    print(f"\nPer-Channel Extra Accuracy (ch8-39):")
    for ch in range(8, NUM_CHANNELS):
        t = ch_total.get(ch, 0)
        c = ch_correct.get(ch, 0)
        pct = 100 * c / t if t else 0
        flag = " <<<" if pct < 99 else ""
        print(f"  ch{ch:2d} ({MEMORY_SCHEMAS[ch][0]:<12}): {pct:.1f}%{flag}")

if __name__ == "__main__":
    main()
