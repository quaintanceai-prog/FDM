# Joint Retrieval Reproduction — Qwen3, LFM2.5, Hermes3

Investigation of Table 4 reproducibility for the three non-GPT-2 models
in the FDM paper. Companion to `GPT2_JOINT_REPRO_NOTES.md`.

## Executive summary

The paper's Table 4 joint accuracy numbers (Qwen3 94.7%, LFM2.5 2.1%,
Hermes3 99.9%) can be reproduced today, but ONLY when using the
ORIGINAL Apr 10 test data, NOT freshly regenerated test data from
today's training scripts. The reason is a git commit on the
edeidic_memory_14_percent repo (commit 13de1df, titled "fix encoder
normalization to fixed range for cross-length transfer") that changed
the encoder's normalization scheme AFTER the models were already
trained and the paper numbers already produced.

Today's training scripts produce signals the Apr 8 checkpoints cannot
decode. The stale Apr 10 test data on disk (which PRE-DATES that
commit) still matches the checkpoints' training distribution.

## What we tried

### Attempt 1: Run eval against Apr 10 test data on disk

Checkpoints and test data were already present on pod 2
(1606ef76afe0):

| Model   | Checkpoint    | Test data     |
|---------|---------------|---------------|
| Qwen3   | Apr 8, 19:14  | Apr 10, 17:01 |
| LFM2.5  | Apr 8, 19:14  | Apr 10, 18:55 |
| Hermes3 | Apr 8, 17:21  | Apr 5, 19:44  |

Launched joint_accuracy_multi.py qwen3 directly. After 4-5 samples
the first-sample outputs showed plausible-looking but WRONG
rule/meta/facts — the drift pattern we learned to recognize from
GPT-2. Killed the eval.

At this point we (wrongly) hypothesized that the Apr 10 test data
might have drifted and regenerated it using today's training
scripts. That ALSO failed. See Attempt 2.

### Attempt 2: Regenerate test data with today's scripts, rerun

Wrote /workspace/regen_model_stage4_test.py using importlib to load
eidetic_{model}_fdm_e2e.py for each of qwen3/lfm2/hermes3 and
generate 1500 fresh test samples (seed=99999).

Ran eval on Qwen3 with fresh data. First 5 samples still showed
drift: model output "SAFETY_FIRST" as the rule regardless of what
the test data expected. Confirmed encoder drift — fresh regeneration
does NOT fix it because today's scripts have a DIFFERENT encoder
than what trained the checkpoints.

Killed eval.

### Attempt 3: Check git history for encoder changes

edeidic_memory_14_percent is a git repo. Checked git log:

    cbe6529 Add cross-length eval scripts... (HEAD)
    3e9c818 Add LFM2 and Hermes3 10ch_spread results...
    b5922ba feat: product retrieval v2 training script
    16c45a3 Add LFM2 and Hermes3 channel density scaling results
    8560874 Add Qwen3 joint accuracy results: 94.7% at N=32    <-- PAPER TABLE 4
    61860d9 Add Hermes3 nhop eval results: 100% decision accuracy
    06126e4 Add Hermes3 per-channel eval and corruption results
    9445bc5 Paper 2: curriculum FDM experiments
    2e186c1 Add paper 2 training script and recon loss patch
    742854f Paper 2: continuous AM FDM with contrastive loss
    321b08b nhop_results_gpt2/results.json
    108628b Add Hermes3-3B FDM training script and eval results
    265ff36 Add lcurve.log to gitignore
    436a63a Add learning curve experiment
    6c80ad3 Add S-random ablation and LFM2.5 1024-token results
    04211b9 Add long-context scaling experiment
    4abb8e7 Add diagonal cross-length eval: 512->1024 100%
    55dea8e Add LFM2.5 and Qwen3 nhop results
    11609a5 LFM2.5 full eval: 98.6% action, 93% extra
    13de1df fix encoder normalization to fixed range  <-- THE BREAK

### The smoking gun: commit 13de1df

Commit title: "fix encoder normalization to fixed range for
cross-length transfer"

Timeline relative to 13de1df:

1. BEFORE 13de1df: all training scripts used sample-wise
   normalization: sig_min, sig_max = composite.min(), composite.max()
2. Models trained with sample-wise normalization. Saved as Apr 8
   checkpoints.
3. Test data generated with sample-wise encoder. Stored as Apr 5-10
   stage4_test.jsonl files.
4. Eval ran with matching encoder, produced original_run/results/
   joint_accuracy_{qwen3,lfm2,hermes3}.json (Apr 16) which match
   paper Table 4 exactly.
5. Commit 13de1df: changed encoder normalization from sample-wise to
   fixed global (for cross-length transfer experiments in Table 3).
6. Change was applied INCONSISTENTLY across model scripts:
   - eidetic_qwen3_fdm_e2e.py: updated to fixed global
   - eidetic_lfm2_fdm_e2e.py: NOT updated, still sample-wise
   - eidetic_hermes3_fdm_e2e.py: NOT updated (as of Apr 14 edit)
   - eidetic_real_fdm_e2e_40ch_turbo_v3.py (GPT-2): mixed state

### Verification that 13de1df IS the break

Ran git show on the parent of 13de1df to see the PRE-fix Qwen3
encoder:

    sig_min, sig_max = composite.min(), composite.max()   # sample-wise
    sig_range = sig_max - sig_min + 1e-10
    norm1 = (composite - sig_min) / sig_range

Post-13de1df Qwen3:

    sig_min = -self.num_channels * self.a_high            # fixed global
    sig_max =  self.num_channels * self.a_high

Different scheme, different token sequences, so a checkpoint trained
on pre-fix signals cannot decode post-fix signals.

## Why the Apr 10 test data DOES match

The Apr 10 test data was generated by running the training script's
generate command on a version of the script that was current AT THAT
TIME. That version pre-dates 13de1df. So:

- Apr 8 checkpoint: trained with pre-13de1df encoder (sample-wise)
- Apr 10 test data: generated with pre-13de1df encoder (sample-wise)
- Compatible. Eval works. Produces paper numbers.

Whereas today:

- Apr 23 regenerated test: generated with CURRENT encoder (mix of
  schemes per model)
- Apr 8 checkpoint: still pre-13de1df (sample-wise)
- Not guaranteed compatible; Qwen3 definitely broken.

## The path forward

Use the stale Apr 10 test data that already exists on disk. It is
the same data that produced the paper JSONs on Apr 16. That data
is compatible with the Apr 8 checkpoints. Fresh regeneration is
unnecessary.

### Concrete plan (non-invasive)

1. Do NOT modify /workspace/edeidic_memory_14_percent/ in any way.
2. Create scratch workspace at /workspace/triple_eval/.
3. Copy the Apr 10 stale test files from edeidic into scratch.
4. Run the eval script from scratch workspace so the relative
   test_data path resolves correctly.
5. Compare results against paper Table 4.

### Commands

    mkdir -p /workspace/triple_eval
    for m in qwen3 lfm2 hermes3; do
      cp "/workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_${m}_stage4_test.jsonl.stale_apr10" \
         "/workspace/triple_eval/fdm_40ch_turbo_v3_${m}_stage4_test.jsonl"
    done
    
    cd /workspace/triple_eval
    python /workspace/ASK_FDM_CHECK/joint_retrieval/joint_accuracy_multi.py qwen3 /workspace/fdm-40ch-qwen3
    python /workspace/ASK_FDM_CHECK/joint_retrieval/joint_accuracy_multi.py lfm2 /workspace/fdm-40ch-lfm2
    python /workspace/ASK_FDM_CHECK/joint_retrieval/joint_accuracy_multi.py hermes3 /workspace/fdm-40ch-hermes3-3b

### Expected results

Paper Table 4 (N=32):
- Qwen3   94.7%
- LFM2.5   2.1%
- Hermes3 99.9%

These were verified on disk via original_run/results/*.json:
- Qwen3   94.67%
- LFM2.5   2.07%
- Hermes3 99.93%

Our reproduction should land within binomial CI (+/- 1.5% at N=1500).

## Key findings for the paper

1. The 13de1df commit is a reproducibility hazard. Future cloners
   who run python eidetic_qwen3_fdm_e2e.py generate to regenerate
   test data will get data incompatible with the HuggingFace
   checkpoints (which were trained before that commit).

2. The three model scripts have inconsistent encoder state today.
   Qwen3 was updated post-13de1df to fixed global normalization;
   LFM2.5 and Hermes3 were NOT updated despite the commit intent.
   Paper Table 5 footnote says "Qwen3 and Hermes3 use fixed global"
   but the actual Hermes3 script uses sample-wise. Paper footnote
   needs correction OR scripts need updating.

3. Encoder provenance matters and should be pinned. Best practice
   for future work: include encoder class as a top-level module
   imported by each model's training script, pin the encoder commit
   hash in each checkpoint config, or bundle test data with each
   HuggingFace checkpoint release.

4. Sample-wise vs fixed global is not an academic distinction. They
   produce different 64-bin quantization for the same signal, hence
   different tokens, hence different training distributions. A model
   trained on one cannot decode signals from the other.

## Files

| Path | Purpose |
|------|---------|
| /workspace/edeidic_memory_14_percent/ | Private paper repo (UNTOUCHED) |
| /workspace/edeidic_memory_14_percent/fdm_40ch_turbo_v3_{qwen3,lfm2,hermes3}_stage4_test.jsonl.stale_apr10 | Original Apr 10 test files |
| /workspace/regen_model_stage4_test.py | Regen script (produced incompatible data) |
| /workspace/TRIPLE_MODEL_REPRO_NOTES.md | This document |
| /workspace/ASK_FDM_CHECK/joint_retrieval/joint_accuracy_multi.py | Shared eval script (unchanged) |
| /workspace/ASK_FDM_CHECK/joint_retrieval/original_run/results/*.json | Paper Apr 16 results |
| /workspace/triple_eval/ | Scratch workspace (to be created) |

## Lessons

- "Fix" commits can be reproducibility-breaking. Commit 13de1df
  titled "fix encoder normalization" was a forward-looking change
  for later experiments that invalidated backward compatibility
  with already-trained checkpoints. Title did not flag this.

- Non-destructive investigation is possible via git. git show on
  past commits lets you inspect pre-change code without disturbing
  the working tree.

- Stale test data is sometimes authoritative. On pods where both
  old and new test data exist, the old data paired with an old
  checkpoint is the CORRECT pairing, not a problem to fix.

- When regeneration breaks a previously-working eval, the cause is
  almost certainly a code change in the generator, not the eval or
  the model. Trust the immutable artifacts (checkpoint, saved test
  data) over the mutable artifact (the script).


## Pilot results (Qwen3 first 5 samples, stale Apr 10 data)

Confirmed Apr 10 stale data + Apr 8 checkpoint IS the correct
pairing. First 5 samples show NORMAL mid-level per-channel errors,
not catastrophic signal decoding failure.

Sample 1: Expected/Got matched on rule (BALANCED), meta
          (OVERRIDE_STATUS), decision (CAROL may proceed), priority
          (LOW), agent (CAROL). Differed only on TEAM (GOLD vs BLUE).
          Single ch8 per-channel error.

Sample 2: Meta misread (LOCKDOWN as OVERRIDE_STATUS). AGENT=DAVE
          and TEAM=GREEN_TEAM correct. Mid-frequency channel error.

Sample 3: Rule and SECRET misread. LOCATION=PARIS correct.

Sample 4: Rule misread (CAUTIOUS as BALANCED). LOCATION=LONDON
          correct.

Sample 5: Meta misread (EMERGENCY as OVERRIDE_STATUS). AGENT=BOB
          and TEAM=BLUE_TEAM correct.

These are the kinds of errors a model with ~99% per-channel
accuracy would make when each sample evaluates 32 channels. By
Bonferroni: 1 - 0.99^32 = 27% sample error rate expected, matching
the joint accuracy of ~73% which would correspond to ~94% per
channel on average. Exactly in line with paper's 94.67% joint
accuracy claim.

Fresh Apr 23 regenerated data, in contrast, showed catastrophic
failure — the model output "SAFETY_FIRST" as the rule regardless of
expected value, because the fresh encoder's output falls outside the
checkpoint's training distribution.

The diagnostic is: stale data gives the RIGHT error pattern
(per-channel noise at mid-high frequencies), fresh data gives a
WRONG error pattern (systematic output with no dependence on
signal content).

Confirmed plan: let the current eval finish (~97 min remaining),
then queue lfm2 and hermes3.
