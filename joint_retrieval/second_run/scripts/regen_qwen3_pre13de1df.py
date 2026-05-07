"""
Regenerate Qwen3 test data using the pre-13de1df encoder.
Saves to /workspace/triple_eval/fdm_40ch_turbo_v3_qwen3_stage4_test.jsonl
"""
import sys, os, shutil, importlib.util, random
import numpy as np

# Change to edeidic dir so relative paths in the module work
os.chdir('/workspace/edeidic_memory_14_percent')

# Load the pre-13de1df script as a module
spec = importlib.util.spec_from_file_location("qwen3_old", "/tmp/qwen3_pre_13de1df.py")
mod = importlib.util.module_from_spec(spec)
sys.modules['qwen3_old'] = mod
spec.loader.exec_module(mod)

print(f"Loaded encoder module. VOCAB_SIZE={mod.VOCAB_SIZE}, NUM_CHANNELS={mod.NUM_CHANNELS}")

# Seed for reproducibility
random.seed(99999)
np.random.seed(99999)

# Use module's function with stage 4 config
stage4 = mod.STAGES[4]
print(f"Stage 4 config: {stage4}")

# Generate to a temp prefix, then move just the test file to our target
os.makedirs('/tmp/qwen3_regen', exist_ok=True)
output_prefix = '/tmp/qwen3_regen/stage4'

mod.generate_stage_data(stage4, num_samples=30000, output_prefix=output_prefix)

# Move just the test split to our target location
src_test = f"{output_prefix}_test.jsonl"
dst_test = '/workspace/triple_eval/fdm_40ch_turbo_v3_qwen3_stage4_test.jsonl'

# Sanity check: don't overwrite the 1024tok file we preserved
if os.path.exists(dst_test):
    print(f"WARN: {dst_test} already exists. Checking what it is...")
    import json
    with open(dst_test) as f:
        s = json.loads(f.readline())
    print(f"  Existing file: {len(s['fdm_text'])} chars, first sample rule={s.get('rule')}")

shutil.copy(src_test, dst_test)
print(f"\nCopied {src_test} -> {dst_test}")

# Verify
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained('/workspace/fdm-40ch-qwen3')
import json
with open(dst_test) as f:
    s = json.loads(f.readline())
ids = tok.encode(s['fdm_text'], add_special_tokens=False)
print(f"\nVerification: test file has {sum(1 for _ in open(dst_test))} samples")
print(f"First sample: {len(s['fdm_text'])} chars → {len(ids)} tokens")
print(f"First sample rule={s.get('rule')}, meta={s.get('meta')}")

# Clean up the temp train/val files (we only need test)
for split in ('train', 'val'):
    f = f"{output_prefix}_{split}.jsonl"
    if os.path.exists(f):
        os.remove(f)
        print(f"Removed temp: {f}")

