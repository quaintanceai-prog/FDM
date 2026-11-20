"""
Shuffled-order test (reviewer dteT, Q3). Finetune the single-block FDM host to emit the
Context block in a different fixed channel order, then evaluate on the full stage-4 test file at
a 512-token cap and re-score the same responses cut at 250 tokens, per channel and per emission
position.

Orders:
  reversed        ch39 first ... ch8 last (highest carriers emitted first)
  random          a fixed random permutation of ch8..39 (seed --seed)

  python shuffled_order.py lfm2 --model_path <stage-4 checkpoint> \
      --train_data fdm_40ch_turbo_v3_lfm2_stage4_train.jsonl \
      --test_data  stage4_test/lfm2_test.jsonl --order reversed
"""
import argparse, json, os, re, random, torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

NAMES = ["TEAM","REGION","PHASE","COMM","ASSET","WINDOW","COVER","SUPPORT","THREAT","WEATHER",
         "TERRAIN","EXTRACT","CIPHER","FREQ","PAYLOAD","ROUTE","DURATION","CONTACT","FUEL","ALTITUDE",
         "VISIBILITY","NOISE","FORMATION","ARMOR","SIGNAL","MORALE","SUPPLY","INTEL","EVAC","WEATHER2",
         "DOCTRINE","COMMS"]  # ch8..ch39, original emission order

def prompt_of(s):
    return f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"

def reorder_context(answer, perm):
    """Rewrite the NAME=VALUE list after 'Context:' in the order given by perm (indices into NAMES)."""
    i = answer.find("Context:")
    if i < 0: return None
    head, tail = answer[:i + len("Context:")], answer[i + len("Context:"):].strip()
    trailing = "." if tail.endswith(".") else ""
    pairs = [p.strip() for p in tail.rstrip(".").split(",") if "=" in p]
    if len(pairs) != 32: return None
    by_name = {p.split("=")[0].strip(): p for p in pairs}
    if any(n not in by_name for n in NAMES): return None
    return head + " " + ", ".join(by_name[NAMES[j]] for j in perm) + trailing

def score(resp, mem):
    out = {}
    for i, n in enumerate(NAMES):
        v = str(mem[str(8 + i)])
        out[n] = bool(re.search(rf"\b{re.escape(n)}\s*=\s*{re.escape(v)}\b", resp))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--model_path", required=True)
    ap.add_argument("--train_data", required=True)
    ap.add_argument("--test_data", required=True)
    ap.add_argument("--order", choices=["reversed", "random", "identity"], default="reversed")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_train", type=int, default=0, help="0 = all")
    ap.add_argument("--n_eval", type=int, default=0, help="0 = all")
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-5)
    ap.add_argument("--bs", type=int, default=4)
    ap.add_argument("--out", default="results_shuffled")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    random.seed(a.seed); torch.manual_seed(a.seed)

    perm = list(range(32)) if a.order == "identity" else list(range(32))[::-1] if a.order == "reversed" else random.Random(a.seed).sample(range(32), 32)
    emit_order = [NAMES[j] for j in perm]
    print("emission order:", emit_order)

    def load(path, n):
        rows = [json.loads(l) for l in open(path)]
        out = []
        for s in rows:
            r = reorder_context(s["answer"], perm)
            if r: out.append({**s, "answer_new": r})
        return out[:n] if n else out
    train, evals = load(a.train_data, a.n_train), load(a.test_data, a.n_eval)
    print(f"train {len(train)}  eval {len(evals)}")
    print("example target:", train[0]["answer_new"][:160])

    dev = "cuda"
    tok = AutoTokenizer.from_pretrained(a.model_path, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    dtype = torch.float32 if a.model == "gpt2" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(a.model_path, torch_dtype=dtype, trust_remote_code=True).to(dev)
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=0.0)

    model.train(); step = 0
    for ep in range(a.epochs):
        random.shuffle(train)
        for b in tqdm(range(0, len(train), a.bs), desc=f"train ep{ep}"):
            batch = train[b: b + a.bs]
            ids_l, lab_l = [], []
            for s in batch:
                p = tok.encode(prompt_of(s), add_special_tokens=False)
                t = tok.encode(" " + s["answer_new"], add_special_tokens=False) + [tok.eos_token_id]
                ids_l.append(p + t); lab_l.append([-100] * len(p) + t)
            L = max(len(x) for x in ids_l)
            ids = torch.tensor([x + [tok.pad_token_id] * (L - len(x)) for x in ids_l], device=dev)
            lab = torch.tensor([x + [-100] * (L - len(x)) for x in lab_l], device=dev)
            att = torch.zeros_like(ids)
            for k, x in enumerate(ids_l): att[k, : len(x)] = 1
            loss = model(input_ids=ids, attention_mask=att, labels=lab).loss
            loss.backward(); opt.step(); opt.zero_grad(); step += 1
            if step % 100 == 0: print(f"step {step} loss {loss.item():.4f}", flush=True)
    model.eval()
    model.save_pretrained(f"{a.out}/{a.model}_{a.order}_ckpt"); tok.save_pretrained(f"{a.out}/{a.model}_{a.order}_ckpt")

    per = {c: {n: 0 for n in NAMES} for c in ("512", "250")}
    joint = {"512": 0, "250": 0}; gen_len = []; dump = []
    for s in tqdm(evals, desc="eval"):
        ids = tok.encode(prompt_of(s), return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=512, do_sample=False, pad_token_id=tok.pad_token_id)
        new = out[0][ids.shape[1]:]
        gen_len.append(int(new.shape[0]))
        for c, cut in (("512", None), ("250", 250)):
            resp = tok.decode(new if cut is None else new[:cut], skip_special_tokens=True)
            sc = score(resp, s["memory"])
            for n in NAMES: per[c][n] += sc[n]
            joint[c] += all(sc.values())
        dump.append({"question": s["question"], "target": s["answer_new"],
                     "response": tok.decode(new, skip_special_tokens=True)})
    N = len(evals)
    pc = {c: {n: 100 * per[c][n] / N for n in NAMES} for c in per}
    res = {"model": a.model, "order": a.order, "seed": a.seed, "emission_order": emit_order,
           "n_train": len(train), "epochs": a.epochs, "lr": a.lr, "n_eval": N,
           "generated_tokens_mean": sum(gen_len) / N, "generated_over_250_pct": 100 * sum(g > 250 for g in gen_len) / N,
           "per_channel_strict_pct": pc,
           "per_emission_position_strict_pct": {c: [pc[c][n] for n in emit_order] for c in pc},
           "joint32_strict_pct": {c: 100 * joint[c] / N for c in joint}}
    for c in ("512", "250"):
        print(f"[cap {c}] by channel ch36-39: {[round(pc[c][n],1) for n in NAMES[-4:]]}  ch8-13: {[round(pc[c][n],1) for n in NAMES[:6]]}")
        pos = res["per_emission_position_strict_pct"][c]
        print(f"[cap {c}] by emission position first4: {[round(x,1) for x in pos[:4]]}  last6: {[round(x,1) for x in pos[-6:]]}  joint-32 {res['joint32_strict_pct'][c]:.1f}")
    print(f"generated tokens mean {res['generated_tokens_mean']:.0f}, over 250: {res['generated_over_250_pct']:.1f}%")
    json.dump(res, open(f"{a.out}/{a.model}_{a.order}.json", "w"), indent=1)
    json.dump(dump, open(f"{a.out}/{a.model}_{a.order}_samples.json", "w"), indent=1)

if __name__ == "__main__":
    main()
