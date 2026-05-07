# Memory Corruption Experiments

This folder contains the token-level memory corruption experiments
comparing Frequency-Division Multiplexed (FDM) in-context memory
encoding against a matched plaintext baseline under structural damage
to the memory region of the prompt.

The experiment tests whether the two encodings differ in **how** they
fail under corruption, not only whether they fail. Inspection of
generated outputs reveals two fundamentally different failure modes:
plaintext produces tokenization-level fragmentation, FDM produces
amplitude-selection errors with valid channel vocabulary. This
distinction is invisible to aggregate accuracy statistics and holds
across two architectures tested (LFM2.5-1.2B, Qwen3-0.6B).

---

## Experimental design

### Matched plaintext baseline (Condition A vs Condition B)

Both encodings receive identical compositional training:

- **Condition A (FDM)**: base pretrained model → FDM fine-tuning on 115K
  samples → N-hop compositional training on 100K samples with FDM memory
- **Condition B (Plaintext)**: base pretrained model → N-hop compositional
  training on 100K samples with plaintext `NAME=VALUE` memory (no FDM stage)

Both conditions share:
- 40-channel MEMORY_SCHEMAS
- 1-5 hop reasoning templates
- Answer format (decision keyword + 32-channel Context block)
- Train/val/test split (seed=77777)
- 100K N-hop compositional samples (20K per hop count)

Only the `[MEMORY]` block content differs:
- FDM: 512 tokens of amplitude-quantized composite sinusoidal signal
- Plaintext: newline-separated `NAME=VALUE` pairs

### Corruption conditions

All three apply to the memory region of the prompt; question and answer
format held constant using the TRAINED N-hop question templates.

1. **Clean** — no corruption; establishes baseline accuracy
2. **Block deletion** — pad the contiguous second half of memory tokens
3. **Stride deletion** — pad every other token in memory region

For FDM specifically, stride deletion halves the effective sampling rate
(from 100 Hz to 50 Hz), reducing the maximum resolvable carrier frequency
from 50 Hz to 25 Hz by Nyquist-Shannon. Carriers below 25 Hz remain
theoretically recoverable from surviving samples; carriers above 25 Hz
fall below the new resolution limit. In practice under-Nyquist retrieval
also degrades because surviving samples carry noisy amplitude estimates
rather than clean ones.

### Metrics

Two metrics per sample, separating reasoning from retrieval:

- **Decision accuracy**: does the generated response contain the correct
  template prefix keyword (e.g., `FULL_5HOP`, `TACTICAL_4HOP`,
  `THREAT_CHAIN`, `RULE_STATUS`)? Tests whether the reasoning chain
  produced the correct template branch.

- **Context accuracy**: what fraction of the 32 extra-channel
  `NAME=VALUE` pairs in the expected Context block appear verbatim in
  the generated output? Uses exact substring match. Tests whether
  specific fact values are retrievable from memory.

Decision accuracy can remain high even when retrieval fails because
N-hop training stores some reasoning templates parametrically; context
accuracy isolates the actual fact-lookup behavior.

---

## Results: LFM2.5-1.2B

Run on 150 held-out test samples per condition. ~25 min per eval.

| Condition | Plaintext Decision | FDM Decision | Plaintext Context | FDM Context | Context Δ |
|---|---|---|---|---|---|
| Clean                         | 100.0% | 100.0% | 91.0% | 99.6% | +8.6 pp |
| Block deletion (2nd half)     | 100.0% | 90.0%  | 32.8% | 40.2% | +7.4 pp |
| Stride deletion (every other) | 94.7%  | 94.7%  | 23.6% | **49.7%** | **+26.1 pp** |

## Results: Qwen3-0.6B

Same protocol, 150 samples per condition. ~17 min per eval.

| Condition | Plaintext Decision | FDM Decision | Plaintext Context | FDM Context | Context Δ |
|---|---|---|---|---|---|
| Clean                         | 100.0% | 88.7% | 100.0% | 82.0% | -18.0 pp |
| Block deletion (2nd half)     | 100.0% | 5.3%  | 39.7% | 39.1% | -0.6 pp |
| Stride deletion (every other) | 95.3%  | 8.0%  | 23.0% | **39.0%** | **+16.0 pp** |

The Qwen3 context accuracy advantage under stride (+16.0 pp) is smaller
than LFM2.5's (+26.1 pp) but same-signed. The decision accuracy collapse
on Qwen3 FDM under corruption (from 88.7% clean to 5-8% under corruption)
is a separate architecture-specific finding explained below.

---

## Two distinct failure modes

Inspecting per-sample generations reveals the two encodings fail in
**qualitatively different ways**, not just at different rates. This
is the central mechanistic finding.

### Plaintext failure: tokenization fragmentation

Plaintext encodes each fact at specific token positions as a
`NAME=VALUE` string. Stride deletion destroys the character-level
readability of these strings. The model cannot parse the damaged
memory at the word level and falls back to emitting
training-distribution fragments that happen to form superficially
plausible text but do not correspond to any valid channel value.

**Example from LFM2.5 plaintext under stride**:
Expected: `TEAM=GREEN_TEAM, REGION=NORTH, PHASE=ALPHA, ...`
Generated:
```
TEAM=JAFETY_FIRST/COMPROMISED/UNAVAILABLE/LOW. Assess and HOLD.
Context: TEAM=JAFETY_FIRST/COMPROMISED/LOW. Standard ABORT.
```

`JAFETY_FIRST` is not a valid TEAM value. It is a fragmented
concatenation of characters from corrupted `NAME=VALUE` strings, glued
together with training-distribution phrases. The tokenizer has broken
on the corrupted input and the model produces lexical garbage.

### FDM failure: amplitude-selection errors

FDM encodes every fact spectrally across all 512 tokens via
frequency-multiplexed amplitude-shift keying. Corruption degrades the
signal but does not destroy the token stream. The model's trained
behavior — read amplitudes per carrier frequency, emit the
corresponding vocabulary value for each channel — can still execute
to completion; it just reads noisy amplitudes and therefore emits
some wrong values. Critically, the emitted values are still **valid
vocabulary per channel**.

**Example from LFM2.5 FDM under stride**:
Expected: `TEAM=GREEN_TEAM, REGION=NORTH, PHASE=ALPHA, COMM=CLOSED,
ASSET=DRONE, WINDOW=DAWN, COVER=NONE, SUPPORT=OFFLINE, THREAT=MEDIUM,
WEATHER=CLEAR, TERRAIN=MOUNTAIN, EXTRACT=DELAYED`
Generated:
```
TEAM=GREEN_TEAM, REGION=NORTH, PHASE=GAMMA, COMM=CLOSED, ASSET=BOAT,
WINDOW=MIDDAY, COVER=NONE, SUPPORT=STANDBY, THREAT=MEDIUM, WEATHER=CLEAR,
TERRAIN=MOUNTAIN, EXTRACT=DELAYED
```

Every emitted value is valid for its channel. TEAM returns a team
name, ASSET returns an asset type, WINDOW returns a time window.
Some values match the expected values (TEAM, REGION, COMM, COVER,
THREAT, WEATHER, TERRAIN, EXTRACT — 8 of 12 shown); others are wrong
but plausible (PHASE=GAMMA instead of ALPHA, ASSET=BOAT instead of
DRONE).

The Context block **structure** is perfectly preserved. Only the
specific amplitude values selected by the model for certain channels
are wrong.

### Why Qwen3 FDM decision collapses but context survives

On Qwen3 FDM under corruption, decision accuracy collapses to ~5-8%
while context accuracy stays at ~39%. Inspecting generations reveals
this is **not a catastrophic output collapse** but a
**template-selection error caused by amplitude noise on reasoning
channels (0-7)**.

Qwen3 FDM under block:
Expected: `RULE_STATUS: SAFETY_FIRST with CLEAR status. Decision:
PROCEED.`
Generated:
```
LOCKDOWN: Status cannot be judged. All operations in PARIS halted.
Context: TEAM=GOLD_TEAM, REGION=SOUTH, PHASE=ALPHA, COMM=RESTRICTED,
ASSET=DRONE, WINDOW=NIGHT, COVER=SHALLOW, SUPPORT=OFFLINE, THREAT=MEDIUM, ...
```

The model produced a valid, well-formed `LOCKDOWN` template. This is a
completely different reasoning branch from the expected `RULE_STATUS`.
The Context block is well-formed with valid per-channel values, some
matching and some not.

The mechanism: amplitude errors on channel 7 (META, 7 Hz) cause Qwen3
to read META=LOCKDOWN when the true value was NONE. The
template-selection logic then correctly applies: "META=LOCKDOWN →
output LOCKDOWN template". The LOGIC is intact; the INPUT reading is
wrong. One amplitude error at channel 7 produces a completely wrong
template selection.

For context channels (8-39), each amplitude error only corrupts one
channel's output, giving ~60% context error rate but no cascading
failure.

**This is the same amplitude-selection failure mode as LFM2.5 FDM.**
The difference is that LFM2.5 has sufficient FDM-reading capacity to
read reasoning channels (0-7) correctly under corruption, while
Qwen3's weaker FDM-reading capacity causes reasoning-channel errors
that cascade into wrong template selection.

---

## Interpretation

### Cross-architecture consistency

Under stride corruption, FDM retains more context accuracy than
plaintext across both architectures tested:
- LFM2.5: FDM context 49.7% vs plaintext 23.6% → +26.1 pp
- Qwen3: FDM context 39.0% vs plaintext 23.0% → +16.0 pp

The advantage is same-signed across architectures, though the
magnitude depends on the architecture's FDM-reading capacity.

### Architecture-specific decision behavior

LFM2.5 preserves decision accuracy under corruption (94.7% stride).
Qwen3 FDM decision collapses to 8% under corruption, but this reflects
template-switching (model picks wrong but valid template due to
reasoning-channel amplitude errors) rather than output collapse. The
underlying mechanism — amplitude-selection errors from corrupted
signal — is the same.

### The aggregate numbers compress the real story

The +26.1 pp and +16.0 pp context accuracy advantages are real but
compress the mechanistic distinction:

| Encoding | Corruption effect | What model produces |
|---|---|---|
| Plaintext | Tokenization breakdown | Lexical fragments, invalid output |
| FDM | Amplitude noise | Valid structured output with noisy values |

Both encodings emit "Context: ..." strings under corruption, but:
- Plaintext's values are invalid vocabulary (`JAFETY_FIRST`,
  `TEAM=JAWEAK`, `MOR=TERRAIN`)
- FDM's values are valid vocabulary selected from wrong amplitudes
  (`TEAM=GOLD_TEAM` when expected GREEN_TEAM, `PHASE=GAMMA` when
  expected ALPHA)

Without inspecting per-sample generations, the substring-match metric
cannot distinguish these.

### Methodological caution on decision accuracy

Decision accuracy alone is unreliable as a corruption metric. On
LFM2.5 it stays near 100% for both encodings across all conditions —
if decision accuracy were the only metric, both encodings would look
equally robust. N-hop-trained models store some reasoning templates
parametrically, so they produce the correct template keyword even when
fact retrieval has failed.

**Producing a syntactically correct answer is not the same as
retrieving correct facts.** Context-block substring matching is
necessary to detect the difference between encodings.

---

## Results: Other architectures

- **GPT-2 Medium**: plaintext N-hop training pending. Repository
  includes clean accuracy from a previous training run (100% action
  accuracy across hops) but corruption eval requires model weights
  not currently on this pod.
- **Hermes3-3B**: plaintext N-hop training pending. May require H100
  rather than RTX 5090 due to memory constraints at full training
  length.

All four architectures already have FDM N-hop models trained and
evaluated on clean memory (Table 6 in paper). The pending work is
plaintext baseline training and corruption eval on the remaining
architectures.

---

## File inventory

```
corruption_experiments/
├── README.md                          # this file
├── scripts/
│   ├── eval_stress_nhop.py            # plaintext corruption eval
│   └── FDM_eval_stress_nhop.py        # FDM corruption eval
└── results/
    ├── lfm2/
    │   ├── results_lfm2_plain_corr.json
    │   └── results_lfm2_fdm_corr.json
    └── qwen3/
        ├── results_qwen3_plain_corr.json
        └── results_qwen3_fdm_corr.json
```

Results JSON structure:

```json
{
  "encoding": "plaintext_nhop" | "fdm_nhop",
  "model_path": "/workspace/.../model_dir",
  "conditions": {
    "clean": {
      "decision_accuracy": 1.00,
      "context_accuracy": 0.91,
      "ctx_matched": 4368, "ctx_total": 4800,
      "samples": [
        {
          "expected": "...", "generated": "...",
          "decision_ok": true, "ctx_matched": 30,
          "ctx_total": 32, "nhops": 3
        }
      ]
    },
    "block": { ... },
    "stride": { ... }
  }
}
```

First 10 sample records per condition are preserved for manual
inspection; the remaining 140 samples contribute only to aggregate
statistics.

---

## Reproduction

### Prerequisites

- Trained FDM N-hop model for the target architecture
- Matched plaintext N-hop model for the same architecture
- Matching test data (N-hop test split) for each encoding

### Fine-tuned models on HuggingFace

- Plaintext N-hop LFM2.5-1.2B: https://huggingface.co/<ANON>/nhop-plaintext-lfm2
- Plaintext N-hop Qwen3-0.6B: https://huggingface.co/<ANON>/nhop-plaintext-qwen3
- FDM N-hop LFM2.5-1.2B: https://huggingface.co/<ANON>/nhop-fdm-lfm2

### Running evaluations

```bash
# Plaintext corruption eval
python3 scripts/eval_stress_nhop.py \
  /path/to/nhop_plaintext_model_<arch>_final \
  /path/to/nhop_plaintext_data_<arch>/nhop_test.jsonl \
  results_<arch>_plain_corr.json

# FDM corruption eval
python3 scripts/FDM_eval_stress_nhop.py \
  /path/to/nhop_model_<arch>_final \
  /path/to/nhop_training_data_<arch>/nhop_test.jsonl \
  results_<arch>_fdm_corr.json
```

Each eval runs 150 samples × 3 corruption conditions × ~400 tokens
generated per sample. Runtime is ~25 min on RTX 5090 for a 1.2B model.

### Inspecting generations

Looking at actual model outputs is essential to understand the
qualitative failure-mode distinction:

```bash
python3 << 'PYEOF'
import json
plain = json.load(open('results_lfm2_plain_corr.json'))
fdm   = json.load(open('results_lfm2_fdm_corr.json'))

for cond in ['clean', 'block', 'stride']:
    print(f"\n=== {cond.upper()} ===")
    for i in range(3):
        ps = plain['conditions'][cond]['samples'][i]
        fs = fdm['conditions'][cond]['samples'][i]
        print(f"\nExpected: {ps['expected'][:150]}")
        print(f"PLAIN gen ({ps['ctx_matched']}/{ps['ctx_total']}): {ps['generated'][:150]}")
        print(f"FDM   gen ({fs['ctx_matched']}/{fs['ctx_total']}): {fs['generated'][:150]}")
PYEOF
```

---

## Hardware notes

- RTX 5090 (32GB): sufficient for LFM2.5-1.2B and Qwen3-0.6B
  inference at bfloat16
- H100 (80GB): required for Hermes3-3B at full context length
  (1024 tokens) or for training with gradient checkpointing
- GPT-2 Medium (345M): fits comfortably on any modern GPU
- Per-sample runtime: ~3.3s for LFM2 (1.2B), ~6.6s for Qwen3 (0.6B)
  — Qwen3 is slower despite smaller parameters due to tokenizer
  vocabulary size (151K vs 65K) affecting decode latency

---

## Future work

1. **Per-channel Nyquist verification**: modify eval to query all 32
   channels per sample, test whether errors cluster above the 25 Hz
   Nyquist cutoff under stride corruption. Would give direct evidence
   for frequency-ordered degradation predicted by the spectral model.

2. **Surgical extraction experiment**: separate experiment testing
   whether each encoding supports targeted channel queries
   (e.g., "What is the value of TEAM?") versus bulk Context-block
   emission. Tests random-access capability that bulk N-hop emission
   does not require.

3. **Template-validity metric**: add a third metric counting whether
   the model produces ANY valid template keyword (vs the correct one),
   separating "output coherent" from "output correct". Would make
   the plaintext-vs-FDM distinction under corruption even more visible
   since plaintext produces invalid keywords (`TEAM=JAFETY_FIRST`)
   while FDM produces valid but wrong keywords (`LOCKDOWN` instead of
   `RULE_STATUS`).

4. **Train plaintext N-hop for GPT-2 and Hermes3** to complete
   4-architecture corruption coverage.

5. **Extend corruption patterns**: random deletion (not structured),
   variable deletion rates (25%, 50%, 75%), token shuffling within
   memory region.
