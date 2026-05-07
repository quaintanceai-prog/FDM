import json, torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
from eidetic_lfm2_fdm_e2e import MEMORY_SCHEMAS

def eval_cross(model_path, test_file, label, token=None):
    kwargs = {"trust_remote_code": True, "dtype": torch.bfloat16}
    tok_kwargs = {"trust_remote_code": True}
    if token:
        kwargs["token"] = token
        tok_kwargs["token"] = token
    tok = AutoTokenizer.from_pretrained(model_path, **tok_kwargs)
    model = AutoModelForCausalLM.from_pretrained(model_path, **kwargs).cuda()
    samples = [json.loads(l) for l in open(test_file)]
    ec, et = 0, 0
    for s in tqdm(samples, desc=label):
        input_text = f"[MEMORY]{s['fdm_text']}[/MEMORY]\nQuestion: {s['question']}\nAnswer:"
        ids = tok.encode(input_text, return_tensors='pt').to('cuda')
        with torch.no_grad():
            out = model.generate(ids, max_new_tokens=250, do_sample=False, pad_token_id=tok.eos_token_id)
        got = tok.decode(out[0][ids.shape[1]:], skip_special_tokens=True)
        for ch in range(8, 40):
            et += 1
            if f"{MEMORY_SCHEMAS[ch][0]}={s['memory'][str(ch)]}" in got:
                ec += 1
    print(f"{label}: extra={100*ec/et:.1f}%")
    del model
    torch.cuda.empty_cache()

import os
HF_TOKEN = os.environ.get('HF_TOKEN')

eval_cross('<ANON>/fdm-40ch-turbo-v3-lfm2',
           'fdm_40ch_turbo_v3_lfm2_1024tok_test.jsonl',
           '512-trained -> 1024-tok test', token=HF_TOKEN)

eval_cross('/workspace/fdm-40ch-lfm2-1024tok',
           'fdm_40ch_turbo_v3_lfm2_stage4_test.jsonl',
           '1024-trained -> 512-tok test')
