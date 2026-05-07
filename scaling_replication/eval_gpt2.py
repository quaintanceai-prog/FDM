"""
FDM Evaluation — GPT-2 Medium (345M)
Reproduces Table 1 results for GPT-2 v3 (115K samples).

Usage:
  python eval_gpt2.py
  python eval_gpt2.py --model_path /path/to/model --test_data /path/to/test.jsonl
"""

import json
import argparse
import torch
from tqdm import tqdm
from transformers import GPT2LMHeadModel, GPT2Tokenizer

MODEL_PATH = "<ANON>/fdm_40ch_turbo_v3_model_final"
TEST_DATA  = "hf://datasets/<ANON>/fdm-40ch-scaling-test-data/gpt2/fdm_40ch_turbo_v3_stage4_test.jsonl"

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

REASONING_RULES = {
    "SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"
}
META_INSTRUCTIONS = {
    "NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"
}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", default=MODEL_PATH)
    parser.add_argument("--test_data", default=TEST_DATA)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading GPT-2 from {args.model_path}...")

    tokenizer = GPT2Tokenizer.from_pretrained(args.model_path)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    test_samples = [json.loads(l) for l in open(args.test_data)]
    print(f"Evaluating on {len(test_samples)} samples...\n")

    action_correct = rule_correct = meta_correct = fact_correct = 0
    extra_correct = extra_total = total = 0
    rule_stats = {r: {"correct": 0, "total": 0} for r in REASONING_RULES}
    meta_stats = {m: {"correct": 0, "total": 0} for m in META_INSTRUCTIONS}

    for i, s in enumerate(tqdm(test_samples, desc="Evaluating [gpt2]")):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=250, do_sample=False,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0])[len(input_text):].strip()
        response = response.split('<|endoftext|>')[0].strip()
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

        rule_ok = rule in response
        if rule_ok: rule_correct += 1
        rule_stats[rule]["total"] += 1
        if rule_ok: rule_stats[rule]["correct"] += 1

        meta_ok = False
        if meta == "EMERGENCY" and "EMERGENCY" in response: meta_ok = True
        elif meta == "LOCKDOWN" and "LOCKDOWN" in response: meta_ok = True
        elif meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY"]: meta_ok = True
        if meta_ok: meta_correct += 1
        meta_stats[meta]["total"] += 1
        if meta_ok: meta_stats[meta]["correct"] += 1

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
            if expected_val in response:
                extra_correct += 1

        total += 1

        if i < 5:
            print(f"\n  [gpt2] Sample {i+1} | Rule: {rule} | Meta: {meta}")
            print(f"  Expected: {expected[:100]}...")
            print(f"  Got:      {response[:100]}...")

    print(f"\n{'='*60}")
    print(f"GPT-2 MEDIUM 40-CHANNEL FDM EVALUATION")
    print(f"{'='*60}")
    print(f"  Action:  {100*action_correct/total:.1f}%")
    print(f"  Rule:    {100*rule_correct/total:.1f}%")
    print(f"  Meta:    {100*meta_correct/total:.1f}%")
    print(f"  Fact:    {100*fact_correct/total:.1f}%")
    print(f"  Extra:   {100*extra_correct/max(extra_total,1):.1f}%")
    print(f"\n--- Per Rule ---")
    for r, stats in rule_stats.items():
        if stats['total'] > 0:
            print(f"  {r:15s}: {100*stats['correct']/stats['total']:.1f}% ({stats['correct']}/{stats['total']})")
    print(f"\n--- Per Meta ---")
    for m, stats in meta_stats.items():
        if stats['total'] > 0:
            print(f"  {m:20s}: {100*stats['correct']/stats['total']:.1f}% ({stats['correct']}/{stats['total']})")

if __name__ == "__main__":
    main()
