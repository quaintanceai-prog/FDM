#!/usr/bin/env bash
# Baseline controls for the rebuttal. Run on the pod from /workspace.
# Needs: the four stage-4 test JSONLs and the four finetuned FDM checkpoints
# (same ones used for Table 2). Set the paths below before running.
set -u
cd /workspace

# ---- EDIT THESE: finetuned single-block FDM checkpoints (local dir or HF id) ----
CKPT_gpt2=<ANON>/fdm_40ch_turbo_v3_model_final
CKPT_qwen3=<ANON>/fdm-40ch-turbo-v3-qwen3
CKPT_lfm2=<ANON>/fdm-40ch-turbo-v3-lfm2
CKPT_hermes3=<ANON>/fdm-40ch-hermes3-3b
# ---- EDIT THESE: stage-4 test files (fields: fdm_text, memory, question, answer, facts, rule, meta) ----
TEST_gpt2=/workspace/tests/gpt2_test.jsonl
TEST_qwen3=/workspace/tests/qwen3_test.jsonl
TEST_lfm2=/workspace/tests/lfm2_test.jsonl
TEST_hermes3=/workspace/tests/hermes3_test.jsonl

OUT=/workspace/results_controls
N=${N:-1500}         # samples per condition; e.g. N=500 ./run_controls.sh
mkdir -p "$OUT"

for m in qwen3 hermes3 lfm2 gpt2; do
  ckpt_var=CKPT_$m; test_var=TEST_$m
  CKPT=${!ckpt_var}; TEST=${!test_var}
  echo "=============== $m ==============="
  # chance / ceiling and token counts (no GPU)
  python baseline_controls.py $m chance     --test_data "$TEST" --out "$OUT" 2>&1 | tee "$OUT/${m}_chance.log"
  python baseline_controls.py $m tokens     --test_data "$TEST" --out "$OUT" 2>&1 | tee "$OUT/${m}_tokens.log"
  # 1. FDM host on FDM block: lenient reproduces Table 2; strict NAME=VALUE and 250-cap re-score are the corrections
  [ -f "$OUT/${m}_fdm.json" ] && echo "SKIP $m fdm" || \
  python baseline_controls.py $m fdm        --model_path "$CKPT" --test_data "$TEST" --out "$OUT" --n $N 2>&1 | tee "$OUT/${m}_fdm.log"
  # 2. chance floor: finetuned host, block removed
  [ -f "$OUT/${m}_none.json" ] && echo "SKIP $m none" || \
  python baseline_controls.py $m none       --model_path "$CKPT" --test_data "$TEST" --out "$OUT" --n $N 2>&1 | tee "$OUT/${m}_none.log"
  # 3. base model, plaintext block, zero-shot
  [ -f "$OUT/${m}_plain_base.json" ] && echo "SKIP $m plain_base" || \
  python baseline_controls.py $m plain_base --test_data "$TEST" --out "$OUT" --n $N 2>&1 | tee "$OUT/${m}_plain_base.log"
done

echo; echo "SUMMARY"
python - <<'EOF'
import json, glob, os
rows = []
for f in sorted(glob.glob('/workspace/results_controls/*_*.json')):
    if f.endswith('_samples.json'): continue
    d = json.load(open(f))
    if 'condition' in d:
        rows.append((d['model'], d['condition'], d['n'], d['action'], d['fact'], d['extra_mean'], d['joint']['32'], d['extra_mean_strict'], d['joint_strict']['32']))
        if 'truncated' in d:
            t = d['truncated']
            rows.append((d['model'], d['condition'] + '@250', d['n'], t['action'], t['fact'], t['extra_mean'], t['joint']['32'], t['extra_mean_strict'], t['joint_strict']['32']))
    elif 'extra_chance_mean_pct' in d:
        print(f"{d['model']:8s} chance  Extra={d['extra_chance_mean_pct']:.1f}%  joint32={d['joint32_chance_pct']:.2e}%  Fact ceiling={d['fact_ceiling_pct']:.1f}%  Fact copyable from question={d['fact_value_in_question_pct']:.1f}%")
    elif 'fdm_block_tokens_mean' in d:
        print(f"{d['model']:8s} tokens  FDM={d['fdm_block_tokens_mean']:.0f}  plaintext={d['plaintext_block_tokens_mean']:.0f} (min {d['plaintext_block_tokens_min']}, max {d['plaintext_block_tokens_max']})")
print(f"{'model':8s} {'cond':11s} {'n':>5s} {'Action':>7s} {'Fact':>7s} {'Extra':>7s} {'Joint32':>8s} | {'Extra*':>7s} {'Joint32*':>8s}   (* = strict NAME=VALUE)")
for r in rows:
    print(f"{r[0]:8s} {r[1]:11s} {r[2]:5d} {r[3]:7.1f} {r[4]:7.1f} {r[5]:7.1f} {r[6]:8.1f} | {r[7]:7.1f} {r[8]:8.1f}")
EOF
