"""
Joint Accuracy Computation — Model-Agnostic

Runs inference on the base FDM model (NOT nhop-trained) and computes:
  1. Per-channel accuracy for all 40 channels
  2. Actual per-sample joint accuracy (not independence estimate)
  3. Error correlation analysis
  4. Frequency gradient (carrier freq vs accuracy)

Uses the stage 4 test data from each model's FDM training.

Usage:
  python joint_accuracy_multi.py gpt2 [model_path]
  python joint_accuracy_multi.py qwen3 [model_path]
  python joint_accuracy_multi.py lfm2 [model_path]
"""

import json, re, sys, os
import torch
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


# ============================================================
# Model configs
# ============================================================

MODEL_CONFIGS = {
    "gpt2": {
        "model_id": "openai-community/gpt2-medium",
        "dtype": torch.float32,
        "trust_remote_code": False,
        "model_class": "gpt2",
        "default_model_dir": "fdm_40ch_turbo_v3_model_final",
        "test_data": "fdm_40ch_turbo_v3_stage4_test.jsonl",
        "output_dir": "nhop_v3_data",
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "model_class": "auto",
        "default_model_dir": "/workspace/fdm-40ch-qwen3",
        "test_data": "fdm_40ch_turbo_v3_qwen3_stage4_test.jsonl",
        "output_dir": "joint_accuracy_qwen3",
    },
    "hermes3": {
        "model_id": "NousResearch/Hermes-3-Llama-3.2-3B",
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "model_class": "auto",
        "default_model_dir": "/workspace/fdm-40ch-hermes3-3b",
        "test_data": "fdm_40ch_turbo_v3_hermes3_stage4_test.jsonl",
        "output_dir": "joint_accuracy_hermes3",
    },
    "lfm2": {
        "model_id": "LiquidAI/LFM2.5-1.2B-Base",
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "model_class": "auto",
        "default_model_dir": "/workspace/fdm-40ch-lfm2",
        "test_data": "fdm_40ch_turbo_v3_lfm2_stage4_test.jsonl",
        "output_dir": "joint_accuracy_lfm2",
    },
}


# Channel definitions
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
CARRIER_FREQS = [1.0 + i * 1.0 for i in range(NUM_CHANNELS)]


def main():
    if len(sys.argv) < 2:
        print("Usage: python joint_accuracy_multi.py <model> [model_path]")
        print("Models: gpt2, qwen3, lfm2")
        sys.exit(0)

    model_name = sys.argv[1]
    if model_name not in MODEL_CONFIGS:
        print(f"Unknown model: {model_name}")
        sys.exit(1)

    cfg = MODEL_CONFIGS[model_name]
    model_path = sys.argv[2] if len(sys.argv) > 2 else cfg["default_model_dir"]
    test_data = cfg["test_data"]
    output_dir = cfg["output_dir"]

    if not os.path.exists(model_path):
        print(f"ERROR: Model not found at {model_path}/")
        return

    if not os.path.exists(test_data):
        print(f"ERROR: Test data not found at {test_data}")
        print(f"  Expected from stage 4 of FDM training.")
        return

    os.makedirs(output_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Model: {model_name} | Path: {model_path} | Device: {device}")

    # Load model
    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=cfg["trust_remote_code"]
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if cfg["model_class"] == "gpt2":
        from transformers import GPT2LMHeadModel
        model = GPT2LMHeadModel.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=cfg["dtype"], trust_remote_code=True
        )
    model.to(device)
    model.eval()

    # Load test data
    with open(test_data) as f:
        samples = [json.loads(line) for line in f]
    print(f"Loaded {len(samples)} test samples\n")

    extra_channels = list(range(8, NUM_CHANNELS))
    num_extra = len(extra_channels)

    # Track per-sample, per-channel correctness
    per_sample_wrong = []
    ch_correct = defaultdict(int)
    ch_total = defaultdict(int)

    # Also track action/fact/rule/meta accuracy
    action_correct = 0
    fact_correct = 0
    rule_correct = 0
    meta_correct = 0
    total = 0

    for i, s in enumerate(tqdm(samples, desc=f"Inference [{model_name}]")):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        input_ids = tokenizer.encode(input_text, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids, max_new_tokens=250, do_sample=False,
                pad_token_id=tokenizer.pad_token_id or tokenizer.eos_token_id
            )

        response = tokenizer.decode(output[0][input_ids.shape[1]:], skip_special_tokens=True).strip()
        expected = s['answer']
        memory = s['memory']
        rule = s.get('rule', '')
        meta = s.get('meta', '')

        # Action
        action_ok = False
        for kw in ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN',
                    'share', 'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']:
            if kw.lower() in expected.lower() and kw.lower() in response.lower():
                action_ok = True
                break
        if action_ok: action_correct += 1

        # Rule
        if rule and rule in response: rule_correct += 1

        # Meta
        meta_ok = False
        if meta == "EMERGENCY" and "EMERGENCY" in response: meta_ok = True
        elif meta == "LOCKDOWN" and "LOCKDOWN" in response: meta_ok = True
        elif meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", ""]: meta_ok = True
        if meta_ok: meta_correct += 1

        # Fact
        facts = s.get('facts', {})
        fact_ok = False
        if s['question'].startswith("Should"):
            fact_ok = facts.get('AGENT', '') in response
        elif s['question'].startswith("What is the risk"):
            fact_ok = facts.get('LOCATION', '') in response
        elif s['question'].startswith("Is it safe"):
            fact_ok = (facts.get('SECRET', '') in response or
                       facts.get('LOCATION', '') in response)
        if fact_ok: fact_correct += 1

        # Extra channels (per-channel + per-sample)
        sample_wrongs = []
        for ch in extra_channels:
            name = MEMORY_SCHEMAS[ch][0]
            expected_val = memory[str(ch)]
            ch_total[ch] += 1
            if expected_val in response:
                ch_correct[ch] += 1
            else:
                sample_wrongs.append((ch, name, expected_val))

        per_sample_wrong.append(sample_wrongs)
        total += 1

        if i < 5:
            print(f"\n  [{model_name}] Sample {i+1} | Rule: {rule} | Meta: {meta}")
            print(f"  Expected: {expected[:100]}...")
            print(f"  Got:      {response[:100]}...")

    # ── Results ──

    print(f"\n{'='*70}")
    print(f"{model_name.upper()} — 40-CHANNEL FDM EVALUATION + JOINT ACCURACY")
    print(f"{'='*70}")

    print(f"\n  Action:  {100*action_correct/total:.1f}%")
    print(f"  Rule:    {100*rule_correct/total:.1f}%")
    print(f"  Meta:    {100*meta_correct/total:.1f}%")
    print(f"  Fact:    {100*fact_correct/total:.1f}%")

    # Per-channel with frequency
    print(f"\n  Per-Channel Accuracy (with carrier frequency):")
    print(f"  {'Ch':<5} {'Name':<14} {'Freq (Hz)':<10} {'Accuracy':<10}")
    print(f"  {'-'*40}")

    ch_accuracies = {}
    for ch in range(NUM_CHANNELS):
        if ch_total[ch] > 0:
            acc = 100 * ch_correct[ch] / ch_total[ch]
        else:
            acc = -1  # channels 0-7 won't have data here
        ch_accuracies[ch] = acc
        if ch >= 8:
            freq = CARRIER_FREQS[ch]
            flag = " <<<" if acc < 95 else ""
            print(f"  ch{ch:<3} {MEMORY_SCHEMAS[ch][0]:<14} {freq:<10.1f} {acc:.1f}%{flag}")

    # Frequency gradient analysis
    print(f"\n  Frequency Gradient (grouped by frequency band):")
    bands = [
        ("Low (9-18 Hz)",    range(8, 18)),
        ("Mid (19-28 Hz)",   range(18, 28)),
        ("High (29-40 Hz)",  range(28, 40)),
    ]
    for band_name, ch_range in bands:
        accs = [100 * ch_correct[ch] / ch_total[ch]
                for ch in ch_range if ch_total[ch] > 0]
        if accs:
            print(f"  {band_name:<20}: mean={np.mean(accs):.2f}%, "
                  f"min={np.min(accs):.1f}%, max={np.max(accs):.1f}%")

    # Joint accuracy
    print(f"\n  Joint Accuracy (per-sample, all N channels correct):")
    print(f"  {'N':<6} {'Actual':<12} {'Independence Est.':<18}")
    print(f"  {'-'*36}")

    for n in [1, 2, 4, 8, 16, 24, 32]:
        if n > num_extra:
            break
        target_chs = extra_channels[:n]
        count = sum(1 for wrongs in per_sample_wrong
                    if all(ch not in [w[0] for w in wrongs] for ch in target_chs))
        actual_pct = 100 * count / total

        # Independence estimate
        indep = 1.0
        for ch in target_chs:
            if ch_total[ch] > 0:
                indep *= ch_correct[ch] / ch_total[ch]
        indep *= 100

        print(f"  {n:<6} {actual_pct:.1f}%{'':<7} {indep:.1f}%")

    all_correct = sum(1 for wrongs in per_sample_wrong if len(wrongs) == 0)
    print(f"\n  ALL {num_extra} correct: {all_correct}/{total} = {100*all_correct/total:.1f}%")

    # Error correlation
    error_samples = [(i, w) for i, w in enumerate(per_sample_wrong) if w]
    multi_error = sum(1 for _, w in error_samples if len(w) > 1)
    single_error = sum(1 for _, w in error_samples if len(w) == 1)
    print(f"\n  Error Analysis:")
    print(f"    Samples with errors: {len(error_samples)}/{total}")
    print(f"    Single-channel: {single_error}")
    print(f"    Multi-channel:  {multi_error}")
    if multi_error + single_error > 0:
        corr = "correlated" if multi_error > single_error else "mostly independent"
        print(f"    Pattern: {corr}")

    ch_error_count = defaultdict(int)
    for _, wrongs in error_samples:
        for ch, name, exp in wrongs:
            ch_error_count[ch] += 1
    if ch_error_count:
        print(f"    Most error-prone channels:")
        for ch, count in sorted(ch_error_count.items(), key=lambda x: -x[1])[:5]:
            print(f"      ch{ch} ({MEMORY_SCHEMAS[ch][0]}): {count} errors (freq={CARRIER_FREQS[ch]:.1f} Hz)")

    # Save
    results = {
        "model": model_name,
        "model_path": model_path,
        "total_samples": total,
        "action_accuracy": 100 * action_correct / total,
        "rule_accuracy": 100 * rule_correct / total,
        "meta_accuracy": 100 * meta_correct / total,
        "fact_accuracy": 100 * fact_correct / total,
        "per_channel": {str(ch): 100 * ch_correct[ch] / ch_total[ch]
                        for ch in extra_channels if ch_total[ch] > 0},
        "frequency_gradient": {
            str(ch): {"freq_hz": CARRIER_FREQS[ch],
                      "accuracy": 100 * ch_correct[ch] / ch_total[ch]}
            for ch in extra_channels if ch_total[ch] > 0
        },
        "all_32_correct_pct": 100 * all_correct / total,
        "error_samples": len(error_samples),
        "single_errors": single_error,
        "multi_errors": multi_error,
    }

    out_path = os.path.join(output_dir, "joint_accuracy.json")
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main()
