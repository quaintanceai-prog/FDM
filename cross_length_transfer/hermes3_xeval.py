import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from eidetic_hermes3_fdm_e2e import MEMORY_SCHEMAS

tok = AutoTokenizer.from_pretrained('fdm_40ch_turbo_v3_hermes3_1024tok_model_final', trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained('fdm_40ch_turbo_v3_hermes3_1024tok_model_final', dtype=torch.bfloat16, trust_remote_code=True).cuda()

samples = [json.loads(l) for l in open('fdm_40ch_turbo_v3_hermes3_512tok_test.jsonl')]
ec, et = 0, 0
for s in tqdm(samples, desc='1024-trained -> 512-tok'):
    input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
    ids = tok.encode(input_text, return_tensors='pt').to('cuda')
    with torch.no_grad():
        out = model.generate(ids, max_new_tokens=250, do_sample=False, pad_token_id=tok.eos_token_id)
    got = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
    for ch in range(8, 40):
        et += 1
        if f"{MEMORY_SCHEMAS[ch][0]}={s['memory'][str(ch)]}" in got:
            ec += 1

print(f'Hermes3 1024-trained -> 512-tok: extra={100*ec/et:.1f}%')
