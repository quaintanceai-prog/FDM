"""
Baseline controls for the FDM single-block evaluation (Table 2 / Table 4).

Uses the SAME stage-4 test samples and the SAME scoring as
joint_retrieval/verified_repro/<model>/eval.py. Only the [MEMORY] block and
the host checkpoint change.

Conditions
  fdm         finetuned FDM host, FDM block            (should reproduce Table 2)
  none        finetuned FDM host, block removed        (chance floor)
  random      finetuned FDM host, 512 random FDM-alphabet tokens (shuffled-signal floor)
  plain_base  BASE model (no finetuning), plaintext NAME=VALUE block, zero-shot
  plain_ft    plaintext-finetuned checkpoint, plaintext block (needs --model_path)
  tokens      no inference; token counts of FDM block vs plaintext block

Usage (on the pod):
  python baseline_controls.py qwen3 none       --model_path /workspace/fdm-40ch-qwen3
  python baseline_controls.py qwen3 plain_base
  python baseline_controls.py qwen3 tokens
  python baseline_controls.py qwen3 fdm        --model_path /workspace/fdm-40ch-qwen3   # sanity check
Optional: --test_data <path> --n 1500 --out results_controls
"""
import argparse, json, os, random, re, sys
from collections import defaultdict
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList


class StopAfterLastChannel(StoppingCriteria):
    """Stop once the response contains 'COMMS=<VALUE>' followed by at least one more character.
    The hosts were trained without EOS and otherwise re-emit the list until the cap; everything the
    scorer needs is present once the 32nd channel is out. Checked every `every` steps for speed."""
    def __init__(self, tok, prompt_len, every=8):
        self.tok, self.prompt_len, self.every, self.n = tok, prompt_len, every, 0
        self.pat = re.compile(r"COMMS=[A-Z_]+[\s.,]")
    def __call__(self, input_ids, scores, **kw):
        self.n += 1
        if self.n % self.every: return False
        text = self.tok.decode(input_ids[0, self.prompt_len:][-120:], skip_special_tokens=True)
        return bool(self.pat.search(text))

MODEL_CONFIGS = {
    "gpt2":    {"model_id": "openai-community/gpt2-medium",          "dtype": torch.float32,  "trust": False,
                "test_data": "fdm_40ch_turbo_v3_stage4_test.jsonl"},
    "qwen3":   {"model_id": "Qwen/Qwen3-0.6B-Base",                  "dtype": torch.bfloat16, "trust": True,
                "test_data": "fdm_40ch_turbo_v3_qwen3_stage4_test.jsonl"},
    "lfm2":    {"model_id": "LiquidAI/LFM2.5-1.2B-Base",             "dtype": torch.bfloat16, "trust": True,
                "test_data": "fdm_40ch_turbo_v3_lfm2_stage4_test.jsonl"},
    "hermes3": {"model_id": "NousResearch/Hermes-3-Llama-3.2-3B",    "dtype": torch.bfloat16, "trust": True,
                "test_data": "fdm_40ch_turbo_v3_hermes3_stage4_test.jsonl"},
}

MEMORY_SCHEMAS = {
    0: ("SECRET", ["RED", "BLUE", "GREEN", "GOLD"]), 1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT", ["ALICE", "BOB", "CAROL", "DAVE"]), 3: ("STATUS", ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]), 5: ("BACKUP", ["AVAILABLE", "UNAVAILABLE"]),
    6: ("RULE", ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7: ("META", ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"]),
    8: ("TEAM", ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]), 9: ("REGION", ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE", ["ALPHA", "BETA", "GAMMA", "DELTA"]), 11: ("COMM", ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET", ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]), 13: ("WINDOW", ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER", ["DEEP", "SHALLOW", "NONE"]), 15: ("SUPPORT", ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT", ["LOW", "MEDIUM", "HIGH", "CRITICAL"]), 17: ("WEATHER", ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN", ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]), 19: ("EXTRACT", ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER", ["AES", "RSA", "BLOWFISH", "TWOFISH"]), 21: ("FREQ", ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD", ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]), 23: ("ROUTE", ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION", ["SHORT", "MEDIUM", "LONG", "EXTENDED"]), 25: ("CONTACT", ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL", ["FULL", "HALF", "LOW", "CRITICAL"]), 27: ("ALTITUDE", ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]), 29: ("NOISE", ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION", ["SINGLE", "PAIR", "SQUAD", "PLATOON"]), 31: ("ARMOR", ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL", ["STRONG", "WEAK", "JAMMED", "LOST"]), 33: ("MORALE", ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY", ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]), 35: ("INTEL", ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC", ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]), 37: ("WEATHER2", ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE", ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]), 39: ("COMMS", ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}
NUM_CHANNELS = 40
EXTRA = list(range(8, NUM_CHANNELS))
ACTION_KW = ['abort', 'proceed', 'hold', 'wait', 'EMERGENCY', 'LOCKDOWN', 'share',
             'Do NOT share', 'Do not share', 'HIGH RISK', 'LOW RISK', 'MODERATE']


def plaintext_block(memory):
    """memory: dict with str or int channel keys -> value."""
    return "\n" + "\n".join(f"{MEMORY_SCHEMAS[ch][0]}={memory[str(ch)] if str(ch) in memory else memory[ch]}"
                            for ch in range(NUM_CHANNELS)) + "\n"


def build_prompt(sample, condition, rng, fdm_alphabet=None):
    q = sample["question"]
    if condition in ("fdm",):
        mem = sample["fdm_text"]
    elif condition == "none":
        mem = ""
    elif condition == "random":
        # same length as the real block, tokens drawn from the FDM alphabet seen in this test file
        toks = sample["fdm_text"].split()
        mem = " ".join(rng.choice(fdm_alphabet) for _ in range(len(toks)))
    elif condition in ("plain_base", "plain_ft"):
        mem = plaintext_block(sample["memory"])
    else:
        raise ValueError(condition)
    return f"[MEMORY]{mem}[/MEMORY]\nQuestion: {q}\nAnswer:"


def score(sample, response, acc):
    expected, memory = sample["answer"], sample["memory"]
    rule, meta, facts = sample.get("rule", ""), sample.get("meta", ""), sample.get("facts", {})
    # Action
    if any(kw.lower() in expected.lower() and kw.lower() in response.lower() for kw in ACTION_KW):
        acc["action"] += 1
    # Rule
    if rule and rule in response:
        acc["rule"] += 1
    # Meta
    if (meta == "EMERGENCY" and "EMERGENCY" in response) or (meta == "LOCKDOWN" and "LOCKDOWN" in response) \
            or meta in ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", ""]:
        acc["meta"] += 1
    # Fact
    qtext = sample["question"]
    if qtext.startswith("Should"):
        ok = facts.get("AGENT", "") in response
    elif qtext.startswith("What is the risk"):
        ok = facts.get("LOCATION", "") in response
    elif qtext.startswith("Is it safe"):
        ok = facts.get("SECRET", "") in response or facts.get("LOCATION", "") in response
    else:
        ok = False
    if ok:
        acc["fact"] += 1
    # Extra (per channel, NAME=VALUE substring not required here: eval.py checks value only)
    wrongs, wrongs_strict = [], []
    for ch in EXTRA:
        val = memory[str(ch)] if str(ch) in memory else memory[ch]
        name = MEMORY_SCHEMAS[ch][0]
        acc["ch_total"][ch] += 1
        if val in response:                       # lenient: bare value anywhere (verified_repro eval.py)
            acc["ch_correct"][ch] += 1
        else:
            wrongs.append(ch)
        if re.search(rf"\b{re.escape(name)}\s*=\s*{re.escape(val)}\b", response):   # strict: NAME=VALUE
            acc["ch_correct_strict"][ch] += 1
        else:
            wrongs_strict.append(ch)
    acc["per_sample_wrong"].append(wrongs)
    acc["per_sample_wrong_strict"].append(wrongs_strict)
    acc["total"] += 1


def summarize(acc, label):
    t = acc["total"]
    out = {"condition": label, "n": t,
           "action": 100 * acc["action"] / t, "rule": 100 * acc["rule"] / t,
           "meta": 100 * acc["meta"] / t, "fact": 100 * acc["fact"] / t}
    for tag, key, wkey in [("", "ch_correct", "per_sample_wrong"), ("_strict", "ch_correct_strict", "per_sample_wrong_strict")]:
        chs = {ch: 100 * acc[key][ch] / acc["ch_total"][ch] for ch in EXTRA}
        out["extra_mean" + tag] = float(np.mean(list(chs.values())))
        out["per_channel" + tag] = {MEMORY_SCHEMAS[ch][0]: round(v, 2) for ch, v in chs.items()}
        bands = {"low ch8-13": range(8, 14), "mid-low ch14-20": range(14, 21), "mid-high ch21-30": range(21, 31),
                 "high ch31-35": range(31, 36), "very high ch36-39": range(36, 40)}
        out["bands" + tag] = {b: (round(min(chs[c] for c in r), 1), round(max(chs[c] for c in r), 1)) for b, r in bands.items()}
        joint = {}
        for n in [1, 2, 4, 8, 16, 24, 32]:
            tgt = EXTRA[:n]
            joint[n] = 100 * sum(1 for w in acc[wkey] if not any(c in w for c in tgt)) / t
        out["joint" + tag] = joint
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model", choices=MODEL_CONFIGS)
    ap.add_argument("condition", choices=["fdm", "none", "random", "plain_base", "plain_ft", "tokens", "chance"])
    ap.add_argument("--model_path", default=None, help="checkpoint dir / HF id (finetuned host for fdm|none|random|plain_ft)")
    ap.add_argument("--test_data", default=None)
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--max_new_tokens", type=int, default=512, help="generation cap (paper used 250)")
    ap.add_argument("--also_truncate_at", type=int, default=250, help="re-score the same responses cut to this many tokens (0 = off)")
    ap.add_argument("--out", default="results_controls")
    ap.add_argument("--no_stop", dest="stop_after_last", action="store_false",
                    help="disable early stop after the 32nd channel (default: stop, ~2x faster)")
    args = ap.parse_args()

    cfg = MODEL_CONFIGS[args.model]
    test_data = args.test_data or cfg["test_data"]
    samples = [json.loads(l) for l in open(test_data)]
    if args.n:
        samples = samples[:args.n]
    os.makedirs(args.out, exist_ok=True)
    rng = random.Random(0)

    # ---- chance and ceiling: no inference ----
    if args.condition == "chance":
        vocab = {ch: len(MEMORY_SCHEMAS[ch][1]) for ch in range(NUM_CHANNELS)}
        extra_chance = float(np.mean([100.0 / vocab[ch] for ch in EXTRA]))
        # ceiling of the Fact check: fraction of consulted values that actually appear in the reference answer
        # (the scorer can only credit a value the reference contains). Also the question-copy floor for Fact:
        # fraction of samples whose Fact value already appears in the question text.
        n_fact = n_fact_in_ref = n_fact_in_q = 0
        for smp in samples:
            facts, q, ans = smp.get("facts", {}), smp["question"], smp["answer"]
            if q.startswith("Should"):            tgt = [facts.get("AGENT", "")]
            elif q.startswith("What is the risk"): tgt = [facts.get("LOCATION", "")]
            elif q.startswith("Is it safe"):      tgt = [facts.get("SECRET", ""), facts.get("LOCATION", "")]
            else: continue
            n_fact += 1
            if any(v and v in ans for v in tgt): n_fact_in_ref += 1
            if any(v and v in q for v in tgt):   n_fact_in_q += 1
        res = {"model": args.model, "n": len(samples),
               "extra_chance_mean_pct": extra_chance,
               "extra_chance_per_channel": {MEMORY_SCHEMAS[ch][0]: round(100.0 / vocab[ch], 1) for ch in EXTRA},
               "joint32_chance_pct": 100.0 * float(np.prod([1.0 / vocab[ch] for ch in EXTRA])),
               "rule_chance_pct": 100.0 / vocab[6], "meta_chance_pct": 100.0 / vocab[7],
               "fact_ceiling_pct": 100.0 * n_fact_in_ref / max(n_fact, 1),
               "fact_value_in_question_pct": 100.0 * n_fact_in_q / max(n_fact, 1),
               "note": "fact_value_in_question_pct is the share of samples where the Fact target is copyable from the question (the 'Should {AGENT}' template)."}
        print(json.dumps(res, indent=2))
        json.dump(res, open(f"{args.out}/{args.model}_chance.json", "w"), indent=2)
        return

    # ---- token counts: no inference ----
    if args.condition == "tokens":
        tok = AutoTokenizer.from_pretrained(cfg["model_id"], trust_remote_code=cfg["trust"])
        fdm = [len(tok.encode(s["fdm_text"], add_special_tokens=False)) for s in samples]
        pl = [len(tok.encode(plaintext_block(s["memory"]), add_special_tokens=False)) for s in samples]
        res = {"model": args.model, "tokenizer": cfg["model_id"], "n": len(samples),
               "fdm_block_tokens_mean": float(np.mean(fdm)), "fdm_block_tokens_min": int(min(fdm)), "fdm_block_tokens_max": int(max(fdm)),
               "plaintext_block_tokens_mean": float(np.mean(pl)), "plaintext_block_tokens_min": int(min(pl)), "plaintext_block_tokens_max": int(max(pl))}
        print(json.dumps(res, indent=2))
        json.dump(res, open(f"{args.out}/{args.model}_tokens.json", "w"), indent=2)
        return

    # ---- which checkpoint ----
    if args.condition == "plain_base":
        model_path = cfg["model_id"]
    else:
        if not args.model_path:
            sys.exit("--model_path is required for this condition")
        model_path = args.model_path

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_path, trust_remote_code=cfg["trust"])
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=cfg["dtype"], trust_remote_code=cfg["trust"]).to(device).eval()

    fdm_alphabet = None
    if args.condition == "random":
        fdm_alphabet = sorted({t for s in samples for t in s["fdm_text"].split()})
        print(f"FDM alphabet size seen in test file: {len(fdm_alphabet)}")

    acc = {"action": 0, "rule": 0, "meta": 0, "fact": 0, "total": 0,
           "ch_correct": defaultdict(int), "ch_correct_strict": defaultdict(int), "ch_total": defaultdict(int),
           "per_sample_wrong": [], "per_sample_wrong_strict": []}
    acc_t = {"action": 0, "rule": 0, "meta": 0, "fact": 0, "total": 0,
             "ch_correct": defaultdict(int), "ch_correct_strict": defaultdict(int), "ch_total": defaultdict(int),
             "per_sample_wrong": [], "per_sample_wrong_strict": []}
    gen_lens = []
    dumps = []
    for i, s in enumerate(tqdm(samples, desc=f"{args.model}/{args.condition}")):
        prompt = build_prompt(s, args.condition, rng, fdm_alphabet)
        ids = tok(prompt, return_tensors="pt").input_ids.to(device)
        with torch.no_grad():
            stopper = StoppingCriteriaList([StopAfterLastChannel(tok, ids.shape[1])]) if args.stop_after_last else None
            out = model.generate(ids, max_new_tokens=args.max_new_tokens, do_sample=False,
                                 pad_token_id=tok.pad_token_id or tok.eos_token_id,
                                 stopping_criteria=stopper)
        gen = out[0][ids.shape[1]:]
        resp = tok.decode(gen, skip_special_tokens=True).strip()
        score(s, resp, acc)
        if args.also_truncate_at:
            resp_t = tok.decode(gen[: args.also_truncate_at], skip_special_tokens=True).strip()
            score(s, resp_t, acc_t)
            gen_lens.append(int(len(gen)))
        if i < 5:
            dumps.append({"i": i, "prompt_tail": prompt[-300:]})
            print(f"\n--- sample {i} ---\nPROMPT tail: ...{prompt[-200:]}\nEXPECTED: {s['answer'][:150]}\nGOT:      {resp[:150]}")
        dumps.append({"i": i, "expected": s["answer"], "response": resp})   # all responses kept for re-scoring

    res = summarize(acc, args.condition)
    res["model"], res["model_path"], res["test_data"] = args.model, model_path, test_data
    res["max_new_tokens"] = args.max_new_tokens
    if args.also_truncate_at:
        rt = summarize(acc_t, f"{args.condition}_trunc{args.also_truncate_at}")
        res["truncated"] = {k: v for k, v in rt.items() if not k.startswith("per_channel")}
        res["truncated"]["per_channel_strict"] = rt["per_channel_strict"]
        res["generated_tokens_mean"] = float(np.mean(gen_lens)); res["generated_tokens_max"] = int(max(gen_lens))
        res["generated_over_250_pct"] = 100.0 * sum(1 for g in gen_lens if g > 250) / len(gen_lens)
        print(f"\n[truncation check] generated tokens mean/max = {res['generated_tokens_mean']:.0f}/{res['generated_tokens_max']}, "
              f"{res['generated_over_250_pct']:.1f}% of responses exceed 250 tokens")
        print(f"[truncation check] Extra strict: full cap {res['extra_mean_strict']:.2f}  vs cut at {args.also_truncate_at}: {rt['extra_mean_strict']:.2f}; "
              f"joint-32 strict {res['joint_strict'][32]:.1f} vs {rt['joint_strict'][32]:.1f}")
        print("[truncation check] very-high band strict, full vs cut:", res["bands_strict"]["very high ch36-39"], rt["bands_strict"]["very high ch36-39"])
    print("\n" + json.dumps({k: v for k, v in res.items() if not k.startswith("per_channel")}, indent=2))
    print("per-channel (lenient):", res["per_channel"])
    print("per-channel (strict NAME=VALUE):", res["per_channel_strict"])
    json.dump(res, open(f"{args.out}/{args.model}_{args.condition}.json", "w"), indent=2)
    json.dump(dumps, open(f"{args.out}/{args.model}_{args.condition}_samples.json", "w"), indent=2)


if __name__ == "__main__":
    main()
