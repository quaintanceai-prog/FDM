import json, torch, numpy as np, random
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import eidetic_hermes3_fdm_e2e as m

print("Generating 1024-token test data for Hermes3...")
encoder = m.TurboFDMSignalEncoder(num_tokens_per_encoder=512, a_low=0.25, a_high=1.0)
random.seed(99999)
np.random.seed(99999)

samples = []
for _ in tqdm(range(1500), desc="Generating"):
    memory = {ch: random.choice(m.MEMORY_SCHEMAS[ch][1]) for ch in range(m.NUM_CHANNELS)}
    fdm_text, _ = encoder.encode_memory(memory)
    facts = {m.MEMORY_SCHEMAS[ch][0]: memory[ch] for ch in range(6)}
    rule, meta = memory[6], memory[7]
    qt = random.choice(m.QUESTION_TEMPLATES)
    question = qt['question'].format(**facts)
    base_answer = qt['fn'](facts, rule, meta)
    extra = [f"{m.MEMORY_SCHEMAS[ch][0]}={memory[ch]}" for ch in range(8, m.NUM_CHANNELS)]
    answer = f"{base_answer} Context: {', '.join(extra)}."
    samples.append({'fdm_text': fdm_text, 'question': question, 'answer': answer,
                    'memory': {str(k): v for k, v in memory.items()}, 'rule': rule, 'meta': meta,
                    'facts': facts})

with open('fdm_40ch_turbo_v3_hermes3_1024tok_test.jsonl', 'w') as f:
    for s in samples:
        f.write(json.dumps(s) + '\n')
print("Saved 1500 samples.")

print("\nEvaluating 512-trained Hermes3 on 1024-tok test data...")
tok = AutoTokenizer.from_pretrained('/workspace/fdm-40ch-hermes3-3b', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('/workspace/fdm-40ch-hermes3-3b', dtype=torch.bfloat16, trust_remote_code=True).cuda()

ec, et = 0, 0
for s in tqdm(samples, desc="512->1024"):
    input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
    ids = tok.encode(input_text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=250, do_sample=False, pad_token_id=tok.eos_token_id)
    got = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    for ch in range(8, 40):
        et += 1
        if f"{m.MEMORY_SCHEMAS[ch][0]}={s['memory'][str(ch)]}" in got:
            ec += 1

print(f"\nHermes3 512-trained -> 1024-tok test: extra={100*ec/et:.1f}%")
