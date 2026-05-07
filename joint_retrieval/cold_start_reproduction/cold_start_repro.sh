#!/bin/bash
# ============================================================
# Cold-Start Reproduction Script
# ============================================================
# Purpose: Verify that the GitHub repo + HuggingFace weights together
# can reproduce the paper's joint retrieval numbers from scratch.
#
# Steps:
#   1. Clone ASK_FDM_CHECK fresh
#   2. Download 4 model weights from HuggingFace
#   3. Download test data from HuggingFace
#   4. Verify md5s match expected values
#   5. Run eval.py for each model sequentially
#   6. Compare results to paper targets
#
# Targets (from paper Table 4):
#   GPT-2:   93.7% joint accuracy (1405/1500)
#   Qwen3:   99.20% joint accuracy
#   LFM2.5:  2.07% joint accuracy
#   Hermes3: 99.93% joint accuracy (1499/1500)
#
# Runtime: ~4.5 hours total (4 sequential evals on single GPU)
#
# Usage:
#   chmod +x cold_start_repro.sh
#   ./cold_start_repro.sh [--skip-downloads] [--smoke-test]
#
# Flags:
#   --skip-downloads   Skip clone/download steps (resume from existing files)
#   --smoke-test       Only run GPT-2 (fastest ~65 min) for quick confirmation
# ============================================================

set -e  # Exit on any error

# ----- CONFIG -----
WORKDIR="/workspace/cold_start_repro"
GITHUB_REPO="git@github.com:<ANON>/ASK_FDM_CHECK.git"

declare -A MODELS_HF=(
    ["gpt2"]="<ANON>/eidetic-fdm-40ch-turbo-v3"
    ["qwen3"]="<ANON>/fdm-40ch-turbo-v3-qwen3"
    ["lfm2"]="<ANON>/fdm-40ch-turbo-v3-lfm2"
    ["hermes3"]="<ANON>/fdm-40ch-hermes3-3b"
)

declare -A MODELS_MD5=(
    ["gpt2"]="596e3ad22878b012c4a6d197e892e4a0"
    ["qwen3"]="873b6ffa4cb0967192abf13c3d0dde89"
    ["lfm2"]="da251c78b0837efc0ca3026e09982078"
    ["hermes3"]="696db498e62557dc6973e6f6a278eefc"
)

declare -A TARGET_JOINT=(
    ["gpt2"]="93.7"
    ["qwen3"]="99.20"
    ["lfm2"]="2.07"
    ["hermes3"]="99.93"
)

TEST_DATA_REPO="<ANON>/fdm-40ch-joint-retrieval-test-data"

# Parse flags
SKIP_DOWNLOADS=0
SMOKE_TEST=0
for arg in "$@"; do
    case $arg in
        --skip-downloads) SKIP_DOWNLOADS=1 ;;
        --smoke-test)     SMOKE_TEST=1 ;;
    esac
done

# Which models to evaluate
if [ $SMOKE_TEST -eq 1 ]; then
    EVAL_MODELS=("gpt2")
    echo "Smoke test mode: evaluating GPT-2 only"
else
    EVAL_MODELS=("gpt2" "qwen3" "lfm2" "hermes3")
fi

# ============================================================
# Step 1: Clone repo
# ============================================================

if [ $SKIP_DOWNLOADS -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "STEP 1: Clone ASK_FDM_CHECK fresh"
    echo "============================================================"
    
    mkdir -p "$WORKDIR"
    cd "$WORKDIR"
    
    if [ -d "ASK_FDM_CHECK" ]; then
        echo "Removing existing clone..."
        rm -rf ASK_FDM_CHECK
    fi
    
    git clone "$GITHUB_REPO"
    echo ""
    echo "Contents of joint_retrieval/verified_repro/:"
    ls ASK_FDM_CHECK/joint_retrieval/verified_repro/
fi

# ============================================================
# Step 2: Download model weights from HuggingFace
# ============================================================

if [ $SKIP_DOWNLOADS -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "STEP 2: Download weights from HuggingFace (~10.8 GB total)"
    echo "============================================================"
    
    mkdir -p "$WORKDIR/models"
    
    python3 << PYEOF
from huggingface_hub import snapshot_download
import os

MODELS = {
    "gpt2":    "${MODELS_HF[gpt2]}",
    "qwen3":   "${MODELS_HF[qwen3]}",
    "lfm2":    "${MODELS_HF[lfm2]}",
    "hermes3": "${MODELS_HF[hermes3]}",
}

for name, repo in MODELS.items():
    local_dir = f"$WORKDIR/models/{name}"
    if os.path.exists(os.path.join(local_dir, "model.safetensors")):
        print(f"  {name}: already downloaded, skipping")
        continue
    print(f"\n=== Downloading {name} from {repo} ===")
    os.makedirs(local_dir, exist_ok=True)
    snapshot_download(repo_id=repo, local_dir=local_dir)
    for f in sorted(os.listdir(local_dir)):
        if not f.startswith('.'):
            size = os.path.getsize(os.path.join(local_dir, f)) / 1e6
            print(f"  {f}: {size:.1f} MB")
PYEOF
fi

# ============================================================
# Step 3: Download test data from HuggingFace
# ============================================================

if [ $SKIP_DOWNLOADS -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "STEP 3: Download test data from HuggingFace"
    echo "============================================================"
    
    python3 << PYEOF
from huggingface_hub import snapshot_download
import os

local_dir = "$WORKDIR/test_data"
os.makedirs(local_dir, exist_ok=True)

print("Downloading test data...")
snapshot_download(
    repo_id="$TEST_DATA_REPO",
    repo_type="dataset",
    local_dir=local_dir
)

for f in sorted(os.listdir(local_dir)):
    if not f.startswith('.'):
        size = os.path.getsize(os.path.join(local_dir, f)) / 1e6
        print(f"  {f}: {size:.1f} MB")
PYEOF
fi

# ============================================================
# Step 4: Verify md5s
# ============================================================

echo ""
echo "============================================================"
echo "STEP 4: Verify weight md5s"
echo "============================================================"

python3 << PYEOF
import hashlib, os, sys

MODELS = {
    "gpt2":    "${MODELS_MD5[gpt2]}",
    "qwen3":   "${MODELS_MD5[qwen3]}",
    "lfm2":    "${MODELS_MD5[lfm2]}",
    "hermes3": "${MODELS_MD5[hermes3]}",
}

print(f"{'Model':<10} {'Expected':<35} {'Actual':<35} Match")
print("-" * 95)

all_match = True
for model, expected in MODELS.items():
    path = f"$WORKDIR/models/{model}/model.safetensors"
    if not os.path.exists(path):
        print(f"{model:<10} MISSING")
        all_match = False
        continue
    md5 = hashlib.md5()
    with open(path, 'rb') as f:
        while chunk := f.read(8*1024*1024):
            md5.update(chunk)
    actual = md5.hexdigest()
    match = "MATCH" if actual == expected else "MISMATCH"
    if actual != expected:
        all_match = False
    print(f"{model:<10} {expected:<35} {actual:<35} {match}")

if not all_match:
    print("\nERROR: md5 mismatch detected. Downloaded weights are corrupt or not the verified versions.")
    sys.exit(1)
print("\nAll md5s match expected values.")
PYEOF

if [ $? -ne 0 ]; then
    echo "ABORTING: md5 check failed"
    exit 1
fi

# ============================================================
# Step 5: Set up eval environments per model
# ============================================================

echo ""
echo "============================================================"
echo "STEP 5: Set up per-model eval directories"
echo "============================================================"

# The eval.py expects test data in the same directory as the model.
# Copy test data + eval.py into each model folder.
for model in "${EVAL_MODELS[@]}"; do
    MODEL_DIR="$WORKDIR/models/$model"
    
    # Copy eval.py from the cloned repo
    REPO_EVAL="$WORKDIR/ASK_FDM_CHECK/joint_retrieval/verified_repro/$model/eval.py"
    if [ -f "$REPO_EVAL" ]; then
        cp "$REPO_EVAL" "$MODEL_DIR/eval.py"
        echo "  $model: copied eval.py"
    else
        echo "  $model: MISSING eval.py at $REPO_EVAL"
        exit 1
    fi
    
    # Also copy generator.py for completeness (may or may not be needed)
    REPO_GEN="$WORKDIR/ASK_FDM_CHECK/joint_retrieval/verified_repro/$model/generator.py"
    if [ -f "$REPO_GEN" ]; then
        cp "$REPO_GEN" "$MODEL_DIR/generator.py"
    fi
    
    # Copy test data (naming differs per model)
    case $model in
        gpt2)    TEST_FILE="fdm_40ch_turbo_v3_gpt2_stage4_test.jsonl" ;;
        qwen3)   TEST_FILE="fdm_40ch_turbo_v3_qwen3_stage4_test.jsonl" ;;
        lfm2)    TEST_FILE="fdm_40ch_turbo_v3_lfm2_stage4_test.jsonl" ;;
        hermes3) TEST_FILE="fdm_40ch_turbo_v3_hermes3_stage4_test.jsonl" ;;
    esac
    
    # GPT-2's test data is called just 'fdm_40ch_turbo_v3_stage4_test.jsonl' 
    # on HF (without the _gpt2 suffix), but the HF dataset has it WITH suffix.
    # Try both.
    SRC_TEST="$WORKDIR/test_data/$TEST_FILE"
    if [ ! -f "$SRC_TEST" ]; then
        echo "  $model: test data MISSING at $SRC_TEST"
        exit 1
    fi
    
    # Copy with the name eval.py expects (from MODEL_CONFIGS)
    if [ "$model" = "gpt2" ]; then
        # eval.py expects 'fdm_40ch_turbo_v3_stage4_test.jsonl' (no model suffix)
        cp "$SRC_TEST" "$MODEL_DIR/fdm_40ch_turbo_v3_stage4_test.jsonl"
    else
        cp "$SRC_TEST" "$MODEL_DIR/$TEST_FILE"
    fi
    echo "  $model: copied test data"
done

# ============================================================
# Step 6: Run evals sequentially
# ============================================================

echo ""
echo "============================================================"
echo "STEP 6: Run joint accuracy evaluations"
echo "============================================================"

for model in "${EVAL_MODELS[@]}"; do
    MODEL_DIR="$WORKDIR/models/$model"
    
    echo ""
    echo "--- Evaluating $model ---"
    echo "Directory: $MODEL_DIR"
    echo "Start: $(date)"
    
    cd "$MODEL_DIR"
    
    # Run eval.py <model_name> <model_path>
    # Using "." because we cd'd into the model dir
    python3 eval.py "$model" . 2>&1 | tee "${model}_eval.log"
    
    echo "End: $(date)"
done

# ============================================================
# Step 7: Compare results to paper
# ============================================================

echo ""
echo "============================================================"
echo "STEP 7: Compare results to paper targets"
echo "============================================================"

python3 << PYEOF
import json, os, glob

TARGETS = {
    "gpt2":    ${TARGET_JOINT[gpt2]},
    "qwen3":   ${TARGET_JOINT[qwen3]},
    "lfm2":    ${TARGET_JOINT[lfm2]},
    "hermes3": ${TARGET_JOINT[hermes3]},
}

EVAL_MODELS = [$(printf '"%s",' "${EVAL_MODELS[@]}")]

# Find result JSONs (output location varies per model)
RESULT_PATHS = {
    "gpt2":    "$WORKDIR/models/gpt2/nhop_v3_data/joint_accuracy.json",
    "qwen3":   "$WORKDIR/models/qwen3/joint_accuracy_qwen3/joint_accuracy.json",
    "lfm2":    "$WORKDIR/models/lfm2/joint_accuracy_lfm2/joint_accuracy.json",
    "hermes3": "$WORKDIR/models/hermes3/joint_accuracy_hermes3/joint_accuracy.json",
}

print(f"{'Model':<10} {'Paper':>10} {'Repro':>10} {'Delta':>10} {'Status':<10}")
print("-" * 55)

all_ok = True
for model in EVAL_MODELS:
    path = RESULT_PATHS.get(model)
    target = TARGETS[model]
    if not path or not os.path.exists(path):
        print(f"{model:<10} {target:>9.2f}% {'MISSING':>10}")
        all_ok = False
        continue
    d = json.load(open(path))
    joint = d.get('all_32_correct_pct', 0)
    delta = joint - target
    status = "OK" if abs(delta) < 0.5 else "DIVERGED"
    print(f"{model:<10} {target:>9.2f}% {joint:>9.2f}% {delta:>+9.2f} {status:<10}")
    if abs(delta) >= 0.5:
        all_ok = False

print()
if all_ok:
    print("COLD-START REPRODUCTION: SUCCESS")
    print("All models reproduce paper targets within 0.5% tolerance.")
else:
    print("COLD-START REPRODUCTION: DIVERGED")
    print("One or more models did not reproduce expected accuracy.")
PYEOF

echo ""
echo "============================================================"
echo "Cold-start reproduction complete."
echo "Results under: $WORKDIR/models/<model>/"
echo "============================================================"
