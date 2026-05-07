#!/bin/bash
set -e

WORKDIR=/workspace/density_repro
cd $WORKDIR

echo "============================================================"
echo "Density Reproduction (3 models × 5 configs = 15 evals)"
echo "Seed: 99999 (paper-era baseline)"
echo "============================================================"

# Step 1: Clone ASK_FDM_CHECK via SSH
echo ""
echo "[1/4] Cloning ASK_FDM_CHECK..."
if [ -d ASK_FDM_CHECK ]; then
    cd ASK_FDM_CHECK && git pull && cd ..
else
    git clone git@github.com:<ANON>/ASK_FDM_CHECK.git
fi

# Step 2: Download 15 density checkpoints from HF
echo ""
echo "[2/4] Downloading 15 density checkpoints (~51 GB)..."
python3 << 'PYEOF'
from huggingface_hub import snapshot_download
import os

MODELS = {
    'gpt2':    '<ANON>/fdm-density-gpt2-repro-{cfg}',
    'lfm2':    '<ANON>/fdm-density-lfm2-{cfg}',
    'hermes3': '<ANON>/fdm-density-hermes3-{cfg}',
}
CONFIGS = ['40ch', '20ch_dense', '20ch_spread', '10ch_dense', '10ch_spread']

BASE = "/workspace/density_repro/models"
os.makedirs(BASE, exist_ok=True)

for model, template in MODELS.items():
    for cfg in CONFIGS:
        repo = template.format(cfg=cfg)
        local_dir = f"{BASE}/{model}/model_{cfg}_final"
        if os.path.exists(os.path.join(local_dir, "model.safetensors")):
            print(f"  {model}/{cfg}: already present")
            continue
        print(f"  Downloading {repo}...")
        try:
            snapshot_download(repo_id=repo, local_dir=local_dir)
            print(f"    -> {local_dir}")
        except Exception as e:
            print(f"    ERROR: {e}")
PYEOF

# Step 3: Run density eval for all 15 configs
echo ""
echo "[3/4] Running density evaluations..."
echo "   (Results will be written to /workspace/density_repro/channel_scaling_results/)"

EVAL_SCRIPT=$WORKDIR/ASK_FDM_CHECK/channel_density/channel_density_scaling_multi.py
MODEL_DIR=$WORKDIR/models

# IMPORTANT: cd to WORKDIR so results go under /workspace/density_repro/
cd $WORKDIR

for model in gpt2 lfm2 hermes3; do
    for cfg in 40ch 20ch_dense 20ch_spread 10ch_dense 10ch_spread; do
        echo ""
        echo "  === $model/$cfg ==="
        python3 $EVAL_SCRIPT eval --model $model --config $cfg --model_dir $MODEL_DIR 2>&1 | tail -15
    done
done

# Step 4: Compare to paper-era baseline (from cloned repo)
echo ""
echo "[4/4] Results summary:"
python3 << 'PYEOF'
import json, os

paper_dir = "/workspace/density_repro/ASK_FDM_CHECK/channel_density/original_run/results"
repro_dir = "/workspace/density_repro/channel_scaling_results"

print(f"\n{'Model':<10} {'Config':<14} {'Paper':>10} {'Repro':>10} {'Delta':>10}")
print("-" * 62)

for model in ['gpt2', 'lfm2', 'hermes3']:
    for cfg in ['40ch', '20ch_dense', '20ch_spread', '10ch_dense', '10ch_spread']:
        paper_f = f"{paper_dir}/{model}/{cfg}.json"
        repro_f = f"{repro_dir}/{model}/{cfg}.json"
        
        if not os.path.exists(paper_f) or not os.path.exists(repro_f):
            print(f"{model:<10} {cfg:<14} MISSING ({paper_f if not os.path.exists(paper_f) else repro_f})")
            continue
        
        paper_v = json.load(open(paper_f)).get('mean_per_channel_acc', 0) * 100
        repro_v = json.load(open(repro_f)).get('mean_per_channel_acc', 0) * 100
        delta = repro_v - paper_v
        print(f"{model:<10} {cfg:<14} {paper_v:>9.2f}% {repro_v:>9.2f}% {delta:>+9.2f}")

print(f"\nDone. Results in {repro_dir}/")
PYEOF

echo ""
echo "============================================================"
echo "Reproduction complete."
echo "============================================================"
