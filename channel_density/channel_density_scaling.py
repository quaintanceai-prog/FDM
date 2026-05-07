"""
==========================================================================
Channel Density Scaling Experiment
==========================================================================

Tests whether per-channel FDM accuracy improves when each carrier gets
more spectral room (fewer channels at the same token budget).

Key design choice: FIXED TRAINING BUDGET (20K samples) across all configs.
At 40 channels with 20K samples, v1 achieved only ~44% extra channel
accuracy — well below the ceiling. This gives room to measure whether
fewer channels (= less inter-carrier interference) improves accuracy
at the same data budget.

If per-channel accuracy rises significantly with fewer channels at
fixed training data, that confirms the signal processing prediction:
spectral separation is a real bottleneck for neural FDM reading, and
more samples-per-carrier (via longer context or fewer channels)
directly improves perceptual retrieval.

Experimental design:
  - Fixed: 256 tokens/view (512 total), 64 quant levels, f_s = 100 Hz,
    turbo structure, 20K training samples, same 5-stage curriculum
  - Variable: number of channels (40, 20, 10) and carrier spacing

       Config        Channels   Carriers         Inter-carrier gap
       40ch          40         1-40 Hz          1 Hz
       20ch_dense    20         1-20 Hz          1 Hz (less crowded upper band)
       20ch_spread   20         spread 1-40 Hz   ~2 Hz
       10ch_dense    10         1-10 Hz          1 Hz (very clean low band)
       10ch_spread   10         spread 1-40 Hz   ~4 Hz

Imports from eidetic_real_fdm_e2e_40ch_turbo_v3.py (same as nhop_training.py).

Usage:
  python channel_density_scaling.py generate [config]
  python channel_density_scaling.py train [config]
  python channel_density_scaling.py eval [config]
  python channel_density_scaling.py fft_baseline [config]

  Configs: 40ch, 20ch_dense, 20ch_spread, 10ch_dense, 10ch_spread
"""

import json, random, os, sys
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import GPT2LMHeadModel, get_linear_schedule_with_warmup

# ============================================================
# Import from v3 (same as nhop_training.py)
# ============================================================

from eidetic_real_fdm_e2e_40ch_turbo_v3 import (
    TurboFDMSignalEncoder, MEMORY_SCHEMAS, NUM_CHANNELS,
    FDMReasoningDataset, get_tokenizer,
)

# ============================================================
# Reduced Channel Schemas
# ============================================================
# Use subsets of the v3 MEMORY_SCHEMAS to keep value alphabets identical.
# Core channels (0-7) always included for reasoning task.
# Extra channels vary by config.

def get_channel_subset(num_channels):
    """Return subset of MEMORY_SCHEMAS for fewer channels.
    
    Uses the first N channels from the original schema (channels 0 to num_channels-1).
    This keeps channel IDs consistent with the encoder's MEMORY_SCHEMAS,
    avoiding value mismatches in value_to_bits().
    """
    if num_channels >= NUM_CHANNELS:
        return dict(MEMORY_SCHEMAS)

    subset = {}
    for ch in range(num_channels):
        subset[ch] = MEMORY_SCHEMAS[ch]

    return subset


# ============================================================
# Modified Encoder (subclass of TurboFDMSignalEncoder)
# ============================================================

class ScaledChannelEncoder(TurboFDMSignalEncoder):
    """
    TurboFDMSignalEncoder with configurable channel count and spacing.

    Inherits all encoding logic from v3. Only overrides:
    - num_channels (fewer carriers)
    - carrier_freqs (optionally wider spacing)
    """

    def __init__(self, num_channels=40, carrier_spacing="dense",
                 num_tokens_per_encoder=256, sample_rate=100.0,
                 a_high=1.0, a_low=0.25, num_levels=64, seed=42):
        # Initialize parent with same params
        # But we need to override num_channels and carrier_freqs AFTER
        # parent init sets them to 40

        # Call parent init (sets up tokenizer, token_map, interleaver)
        super().__init__(
            num_tokens_per_encoder=num_tokens_per_encoder,
            sample_rate=sample_rate,
            a_high=a_high,
            a_low=a_low,
            num_levels=num_levels,
            seed=seed,
        )

        # Override channel count and frequencies
        self.num_channels = num_channels

        if carrier_spacing == "dense":
            # Carriers at 1, 2, 3, ..., num_channels Hz
            self.carrier_freqs = [1.0 + i * 1.0 for i in range(num_channels)]
        elif carrier_spacing == "spread":
            # Spread carriers across the full 1-40 Hz range
            if num_channels == 1:
                self.carrier_freqs = [1.0]
            else:
                step = 39.0 / (num_channels - 1)
                self.carrier_freqs = [1.0 + i * step for i in range(num_channels)]
        else:
            raise ValueError(f"Unknown spacing: {carrier_spacing}")

        print(f"ScaledChannelEncoder: {num_channels} channels, "
              f"spacing={carrier_spacing}")
        print(f"  Carriers: {[f'{f:.1f}' for f in self.carrier_freqs[:5]]}..."
              f" (Hz)")
        print(f"  Tokens/view: {num_tokens_per_encoder}, "
              f"total: {num_tokens_per_encoder * 2}")


# ============================================================
# Experiment Configs
# ============================================================

CONFIGS = {
    "40ch": {
        "num_channels": 40,
        "carrier_spacing": "dense",
        "train_samples": 20000,  # Fixed budget across all configs
        "description": "40 channels, carriers 1-40 Hz (v1-equivalent budget)",
    },
    "20ch_dense": {
        "num_channels": 20,
        "carrier_spacing": "dense",
        "train_samples": 20000,
        "description": "20 channels, carriers 1-20 Hz (dense)",
    },
    "20ch_spread": {
        "num_channels": 20,
        "carrier_spacing": "spread",
        "train_samples": 20000,
        "description": "20 channels, carriers spread 1-40 Hz",
    },
    "10ch_dense": {
        "num_channels": 10,
        "carrier_spacing": "dense",
        "train_samples": 20000,
        "description": "10 channels, carriers 1-10 Hz (dense)",
    },
    "10ch_spread": {
        "num_channels": 10,
        "carrier_spacing": "spread",
        "train_samples": 20000,
        "description": "10 channels, carriers spread 1-40 Hz",
    },
}


# ============================================================
# Reasoning Logic (adapted for variable channel count)
# ============================================================

def generate_reasoning(memory, channel_schemas):
    """
    Same reasoning as v3 but uses only channels present in the schema.
    Core channels 0-7 drive the reasoning. Extra channels are reported.
    """
    num_ch = len(channel_schemas)

    # Core channels (always present if num_channels >= 8)
    secret = memory.get(0, "ALPHA")
    location = memory.get(1, "TOKYO")
    agent = memory.get(2, "ALICE")
    status = memory.get(3, "CLEAR")
    priority = memory.get(4, "MEDIUM")
    backup = memory.get(5, "AVAILABLE")
    rule = memory.get(6, "BALANCED")
    meta = memory.get(7, "NONE")

    # Same reasoning logic as v3's generate_proceed_answer
    if meta == "EMERGENCY":
        action = "ABORT"
        reasoning = f"EMERGENCY override: {agent} must abort immediately."
    elif meta == "LOCKDOWN":
        action = "HOLD"
        reasoning = f"LOCKDOWN: {agent} holds position."
    elif meta == "OVERRIDE_STATUS":
        action = "PROCEED"
        reasoning = f"Status overridden to CLEAR. {agent} proceeds."
    elif meta == "OVERRIDE_PRIORITY":
        action = "PROCEED"
        reasoning = f"Priority overridden to CRITICAL. {agent} proceeds."
    else:
        if rule == "SAFETY_FIRST":
            if status in ["COMPROMISED", "UNKNOWN"]:
                action = "ABORT"
                reasoning = f"SAFETY_FIRST: Status {status}. {agent} aborts."
            else:
                action = "PROCEED"
                reasoning = f"SAFETY_FIRST: Status {status}. {agent} proceeds."
        elif rule == "MISSION_FIRST":
            if priority in ["CRITICAL", "HIGH"]:
                action = "PROCEED"
                reasoning = f"MISSION_FIRST: Priority {priority}. {agent} proceeds."
            elif status == "COMPROMISED":
                action = "ABORT"
                reasoning = f"MISSION_FIRST: Priority {priority}, COMPROMISED. Abort."
            else:
                action = "PROCEED"
                reasoning = f"MISSION_FIRST: {agent} proceeds."
        elif rule == "CAUTIOUS":
            if status != "CLEAR" or priority in ["LOW", "ROUTINE"]:
                action = "HOLD"
                reasoning = f"CAUTIOUS: Status {status}, priority {priority}. Hold."
            else:
                action = "PROCEED"
                reasoning = f"CAUTIOUS: All clear. {agent} proceeds."
        else:  # BALANCED
            if status == "COMPROMISED" and priority in ["LOW", "ROUTINE"]:
                action = "ABORT"
                reasoning = f"BALANCED: {status} + {priority}. Abort."
            elif status == "COMPROMISED" and priority in ["CRITICAL", "HIGH"]:
                action = "PROCEED"
                reasoning = f"BALANCED: {status} but {priority}. Proceed."
            else:
                action = "PROCEED"
                reasoning = f"BALANCED: {agent} proceeds."

        if action == "ABORT" and backup == "AVAILABLE":
            action = "EXTRACT"
            reasoning += f" Backup {backup}, switching to extraction."
        elif action == "HOLD" and backup == "UNAVAILABLE":
            reasoning += f" No backup."

    share_secret = "YES" if (status == "CLEAR" and priority in ["CRITICAL", "HIGH"]) else "NO"

    # Build extra context (only channels present in schema, beyond core 0-7)
    extra_parts = []
    for ch in sorted(channel_schemas.keys()):
        if ch >= 8:
            name = channel_schemas[ch][0]
            val = memory.get(ch, channel_schemas[ch][1][0])
            extra_parts.append(f"{name}={val}")
    extra_str = ", ".join(extra_parts) if extra_parts else "NONE"

    response = (
        f"Analysis: {rule} protocol active.\n"
        f"Secret: {secret} | Location: {location} | Agent: {agent}\n"
        f"Status: {status} | Priority: {priority} | Backup: {backup}\n"
        f"Meta: {meta}\n"
        f"Reasoning: {reasoning}\n"
        f"Decision: {action}\n"
        f"Share secret: {share_secret}\n"
        f"Context: {extra_str}"
    )

    return response, action


# ============================================================
# Data Generation
# ============================================================

def generate_data(config_name, output_dir="channel_scaling_data"):
    """Generate training data for a config."""
    cfg = CONFIGS[config_name]
    num_channels = cfg["num_channels"]
    num_samples = cfg["train_samples"]

    channel_schemas = get_channel_subset(num_channels)

    encoder = ScaledChannelEncoder(
        num_channels=num_channels,
        carrier_spacing=cfg["carrier_spacing"],
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"data_{config_name}.jsonl")

    samples = []
    print(f"\nGenerating {num_samples} samples for {config_name}: "
          f"{cfg['description']}")

    random.seed(12345)
    np.random.seed(12345)

    for i in tqdm(range(num_samples)):
        # Random values for channels in this config
        memory = {}
        for ch in sorted(channel_schemas.keys()):
            _, values = channel_schemas[ch]
            memory[ch] = random.choice(values)

        fdm_text, token_ids = encoder.encode_memory(memory)
        response, action = generate_reasoning(memory, channel_schemas)

        question = (
            "Analyze the encoded operational data. Apply the reasoning rule "
            "and meta-instruction to determine the correct action. Report "
            "all context parameters."
        )

        samples.append({
            "fdm_text": fdm_text,
            "question": question,
            "response": response,
            "action": action,
            "memory": {str(k): v for k, v in memory.items()},
            "config": config_name,
            "num_channels": num_channels,
        })

    with open(output_path, 'w') as f:
        for s in samples:
            f.write(json.dumps(s) + "\n")
    print(f"Saved {len(samples)} samples to {output_path}")

    # Splits
    random.shuffle(samples)
    n = len(samples)
    splits = {
        'train': (0, int(0.85 * n)),
        'val': (int(0.85 * n), int(0.95 * n)),
        'test': (int(0.95 * n), n),
    }
    for name, (start, end) in splits.items():
        split_path = output_path.replace(".jsonl", f"_{name}.jsonl")
        with open(split_path, 'w') as f:
            for s in samples[start:end]:
                f.write(json.dumps(s) + "\n")
        print(f"  {name}: {end - start} samples")


# ============================================================
# Dataset (reuses v3 format)
# ============================================================

class ScalingDataset(Dataset):
    """Dataset that re-encodes FDM on the fly for the configured channel count."""

    def __init__(self, data_path, tokenizer, encoder, max_length=768):
        self.samples = [json.loads(l) for l in open(data_path)]
        self.tokenizer = tokenizer
        self.encoder = encoder
        self.max_length = max_length

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        memory = {int(k): v for k, v in s['memory'].items()}

        # Re-encode FDM with configured encoder
        fdm_text, _ = self.encoder.encode_memory(memory)

        full_text = (
            f"[MEMORY]{fdm_text}[/MEMORY]\n"
            f"Question: {s['question']}\n"
            f"Answer: {s['response']}"
        )

        encoding = self.tokenizer(
            full_text,
            truncation=True,
            max_length=self.max_length,
            padding='max_length',
            return_tensors='pt'
        )

        input_ids = encoding['input_ids'].squeeze()
        attention_mask = encoding['attention_mask'].squeeze()
        labels = input_ids.clone()

        prompt = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {s['question']}\nAnswer: "
        prompt_len = len(self.tokenizer.encode(prompt))
        labels[:prompt_len] = -100

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }


# ============================================================
# Training (5-stage curriculum, same as v3)
# ============================================================

CURRICULUM = [
    # (a_high, a_low, fraction_of_data, epochs)
    (1.0, 0.0,  0.13, 5),   # Stage 0: max contrast
    (1.0, 0.05, 0.17, 5),   # Stage 1
    (1.0, 0.10, 0.22, 7),   # Stage 2
    (1.0, 0.15, 0.22, 7),   # Stage 3
    (1.0, 0.25, 0.26, 10),  # Stage 4: target contrast
]


def train(config_name, data_dir="channel_scaling_data",
          batch_size=4, lr=2e-5, output_dir="channel_scaling_models"):
    """Train with curriculum, same pipeline as v3."""
    cfg = CONFIGS[config_name]
    num_channels = cfg["num_channels"]

    print(f"\n{'='*60}")
    print(f"Training {config_name}: {cfg['description']}")
    print(f"{'='*60}")

    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token

    # Start from fresh GPT-2 (fair comparison — v3 also started fresh)
    model = GPT2LMHeadModel.from_pretrained('gpt2')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    train_path = os.path.join(data_dir, f"data_{config_name}_train.jsonl")
    val_path = os.path.join(data_dir, f"data_{config_name}_val.jsonl")

    os.makedirs(output_dir, exist_ok=True)

    for stage_idx, (a_high, a_low, frac, epochs) in enumerate(CURRICULUM):
        print(f"\n--- Stage {stage_idx}: a_high={a_high}, a_low={a_low}, "
              f"epochs={epochs} ---")

        encoder = ScaledChannelEncoder(
            num_channels=num_channels,
            carrier_spacing=cfg["carrier_spacing"],
            a_high=a_high, a_low=a_low, num_levels=64, seed=42,
        )

        train_dataset = ScalingDataset(train_path, tokenizer, encoder)

        # Use fraction of data for this stage
        n_use = int(len(train_dataset) * frac)
        indices = list(range(n_use))
        subset = torch.utils.data.Subset(train_dataset, indices)

        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=len(train_loader) // 10,
            num_training_steps=len(train_loader) * epochs,
        )

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            for batch in tqdm(train_loader,
                            desc=f"S{stage_idx} E{epoch+1}/{epochs}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                              attention_mask=attention_mask,
                              labels=labels)
                loss = outputs.loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                total_loss += loss.item()

            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

        # Save checkpoint after each stage
        ckpt_path = os.path.join(output_dir, f"model_{config_name}_stage{stage_idx}")
        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

    # Save final
    final_path = os.path.join(output_dir, f"model_{config_name}_final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved final model to {final_path}")

    return model


# ============================================================
# Evaluation
# ============================================================

def evaluate(config_name, model_dir="channel_scaling_models",
             num_eval=500):
    """Evaluate trained model. Reports per-channel and joint accuracy."""
    cfg = CONFIGS[config_name]
    num_channels = cfg["num_channels"]
    channel_schemas = get_channel_subset(num_channels)

    model_path = os.path.join(model_dir, f"model_{config_name}_final")
    if not os.path.exists(model_path):
        print(f"No model found at {model_path}")
        return None

    print(f"\nEvaluating {config_name}: {cfg['description']}")

    tokenizer = get_tokenizer()
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    encoder = ScaledChannelEncoder(
        num_channels=num_channels,
        carrier_spacing=cfg["carrier_spacing"],
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

    random.seed(99999)
    np.random.seed(99999)

    # Metrics
    action_correct = 0
    fact_correct = 0
    rule_correct = 0
    meta_correct = 0
    per_channel_correct = defaultdict(int)
    per_channel_total = defaultdict(int)
    total = 0

    # Joint accuracy (for extra channels)
    joint_counts = []  # how many extra channels correct per sample

    for _ in tqdm(range(num_eval)):
        memory = {}
        for ch in sorted(channel_schemas.keys()):
            _, values = channel_schemas[ch]
            memory[ch] = random.choice(values)

        fdm_text, _ = encoder.encode_memory(memory)
        expected_response, expected_action = generate_reasoning(memory, channel_schemas)

        question = (
            "Analyze the encoded operational data. Apply the reasoning rule "
            "and meta-instruction to determine the correct action. Report "
            "all context parameters."
        )

        prompt = f"[MEMORY]{fdm_text}[/MEMORY]\nQuestion: {question}\nAnswer: "
        input_ids = tokenizer.encode(prompt, return_tensors='pt').to(device)

        with torch.no_grad():
            output = model.generate(
                input_ids,
                max_new_tokens=300,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )

        generated = tokenizer.decode(output[0][input_ids.shape[1]:],
                                     skip_special_tokens=True)

        # Action accuracy
        if f"Decision: {expected_action}" in generated:
            action_correct += 1

        # Fact accuracy (channels 0-5)
        facts_ok = True
        for ch in range(min(6, num_channels)):
            val = memory[ch]
            if val not in generated:
                facts_ok = False
        if facts_ok:
            fact_correct += 1

        # Rule
        if memory.get(6, "") in generated:
            rule_correct += 1

        # Meta
        if f"Meta: {memory.get(7, '')}" in generated:
            meta_correct += 1

        # Per extra channel
        extra_correct_this = 0
        num_extra = 0
        for ch in sorted(channel_schemas.keys()):
            if ch >= 8:
                name = channel_schemas[ch][0]
                val = memory[ch]
                per_channel_total[ch] += 1
                num_extra += 1
                if f"{name}={val}" in generated:
                    per_channel_correct[ch] += 1
                    extra_correct_this += 1

        joint_counts.append(extra_correct_this)
        total += 1

    # Report
    num_extra = max(1, num_channels - 8)
    extra_acc = (sum(per_channel_correct.values()) /
                 max(1, sum(per_channel_total.values())))

    print(f"\n{'='*60}")
    print(f"RESULTS: {config_name} ({num_channels} channels)")
    print(f"{'='*60}")
    print(f"  Action:  {100*action_correct/total:.1f}%")
    print(f"  Rule:    {100*rule_correct/total:.1f}%")
    print(f"  Meta:    {100*meta_correct/total:.1f}%")
    print(f"  Fact:    {100*fact_correct/total:.1f}%")
    print(f"  Extra:   {100*extra_acc:.1f}%")

    # Per-channel detail
    if per_channel_total:
        print(f"\n  Per extra channel accuracy:")
        for ch in sorted(per_channel_total.keys()):
            acc = per_channel_correct[ch] / per_channel_total[ch]
            name = channel_schemas[ch][0]
            freq = encoder.carrier_freqs[ch] if ch < len(encoder.carrier_freqs) else "?"
            print(f"    ch{ch:2d} ({name:10s}) @ {freq:5.1f} Hz: {100*acc:.1f}%")

    # Joint accuracy
    thresholds = [1, 2, 4, 8, 16, 24, 32]
    print(f"\n  Joint accuracy (N extra channels all correct):")
    for n in thresholds:
        if n <= num_extra:
            joint = sum(1 for c in joint_counts if c >= n) / total
            print(f"    {n:2d} channels: {100*joint:.1f}%")

    # Mean per-channel accuracy (key metric for scaling comparison)
    mean_per_ch = np.mean([per_channel_correct[ch] / per_channel_total[ch]
                           for ch in per_channel_total])
    print(f"\n  Mean per-channel accuracy: {100*mean_per_ch:.2f}%")

    results = {
        "config": config_name,
        "num_channels": num_channels,
        "carrier_spacing": cfg["carrier_spacing"],
        "action_acc": action_correct / total,
        "rule_acc": rule_correct / total,
        "meta_acc": meta_correct / total,
        "fact_acc": fact_correct / total,
        "extra_acc": extra_acc,
        "mean_per_channel_acc": mean_per_ch,
        "per_channel": {str(ch): per_channel_correct[ch] / per_channel_total[ch]
                        for ch in per_channel_total},
    }

    os.makedirs("channel_scaling_results", exist_ok=True)
    with open(f"channel_scaling_results/{config_name}.json", 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# FFT Baseline (classical signal processing at each config)
# ============================================================

def fft_baseline(config_name, num_test=200):
    """Classical FFT carrier detection at each channel density."""
    cfg = CONFIGS[config_name]
    num_channels = cfg["num_channels"]
    channel_schemas = get_channel_subset(num_channels)

    encoder = ScaledChannelEncoder(
        num_channels=num_channels,
        carrier_spacing=cfg["carrier_spacing"],
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

    random.seed(55555)
    np.random.seed(55555)

    correct = defaultdict(int)
    total = 0
    tokens_per_view = 256

    print(f"\nFFT baseline: {config_name} ({num_channels} channels, "
          f"{cfg['carrier_spacing']} spacing)")

    for _ in tqdm(range(num_test)):
        memory = {}
        for ch in sorted(channel_schemas.keys()):
            _, values = channel_schemas[ch]
            memory[ch] = random.choice(values)

        # Build raw composite
        t = np.arange(tokens_per_view) / encoder.sample_rate
        composite = np.zeros(tokens_per_view)

        for ch_idx in range(num_channels):
            if ch_idx not in channel_schemas:
                continue
            _, values = channel_schemas[ch_idx]
            val = memory.get(ch_idx, values[0])
            val_idx = values.index(val)
            num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
            bits = format(val_idx, f'0{num_bits}b')
            samples_per_bit = tokens_per_view // num_bits

            for j, bit in enumerate(bits):
                start = j * samples_per_bit
                end = min((j + 1) * samples_per_bit, tokens_per_view)
                amp = 1.0 if bit == '1' else 0.25
                composite[start:end] += amp * np.sin(
                    2 * np.pi * encoder.carrier_freqs[ch_idx] * t[start:end]
                )

        # FFT
        freqs = np.fft.rfftfreq(tokens_per_view, d=1.0 / encoder.sample_rate)
        fft_mag = np.abs(np.fft.rfft(composite))
        median_mag = np.median(fft_mag[1:])

        for ch_idx in range(num_channels):
            if ch_idx not in channel_schemas:
                continue
            carrier = encoder.carrier_freqs[ch_idx]
            bin_idx = np.argmin(np.abs(freqs - carrier))
            if fft_mag[bin_idx] > median_mag * 0.5:
                correct[ch_idx] += 1

        total += 1

    mean_detect = np.mean([correct[ch] / total for ch in range(num_channels)
                           if ch in channel_schemas])
    print(f"  Mean carrier detection: {100*mean_detect:.1f}%")

    return {ch: correct[ch] / total for ch in channel_schemas}


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python channel_density_scaling.py generate [config]")
        print("  python channel_density_scaling.py train [config]")
        print("  python channel_density_scaling.py eval [config]")
        print("  python channel_density_scaling.py fft_baseline [config]")
        print(f"\nConfigs: {list(CONFIGS.keys())}")
        print("\nRecommended run order (after nhop completes):")
        print("  # Generate all (fast, ~minutes each)")
        print("  python channel_density_scaling.py generate 40ch")
        print("  python channel_density_scaling.py generate 20ch_spread")
        print("  python channel_density_scaling.py generate 10ch_spread")
        print("  python channel_density_scaling.py generate 20ch_dense")
        print("  python channel_density_scaling.py generate 10ch_dense")
        print("")
        print("  # Train all (hours each, use tmux)")
        print("  python channel_density_scaling.py train 40ch")
        print("  python channel_density_scaling.py train 20ch_spread")
        print("  python channel_density_scaling.py train 10ch_spread")
        print("  python channel_density_scaling.py train 20ch_dense")
        print("  python channel_density_scaling.py train 10ch_dense")
        print("")
        print("  # Evaluate all")
        print("  python channel_density_scaling.py eval")
        print("")
        print("  All configs use 20K training samples.")
        print("  40ch at 20K ≈ v1 conditions (~44% extra accuracy).")
        print("  Prediction: 20ch and 10ch will be significantly higher.")
        sys.exit(1)

    cmd = sys.argv[1]
    config = sys.argv[2] if len(sys.argv) > 2 else None

    if cmd == "generate":
        if config:
            generate_data(config)
        else:
            for c in CONFIGS:
                generate_data(c)

    elif cmd == "train":
        if config:
            train(config)
        else:
            for c in CONFIGS:
                train(c)

    elif cmd == "eval":
        if config:
            evaluate(config)
        else:
            all_results = {}
            for c in CONFIGS:
                r = evaluate(c)
                if r:
                    all_results[c] = r

            if all_results:
                print(f"\n{'='*70}")
                print("CHANNEL DENSITY SCALING SUMMARY")
                print(f"{'='*70}")
                print(f"{'Config':>14s} {'#Ch':>4s} {'Spacing':>8s} "
                      f"{'Action':>7s} {'Extra':>6s} {'MeanPerCh':>10s}")
                print("-" * 70)
                for c in ["40ch", "20ch_dense", "20ch_spread",
                          "10ch_dense", "10ch_spread"]:
                    if c in all_results:
                        r = all_results[c]
                        print(f"{c:>14s} {r['num_channels']:>4d} "
                              f"{r['carrier_spacing']:>8s} "
                              f"{100*r['action_acc']:>6.1f}% "
                              f"{100*r['extra_acc']:>5.1f}% "
                              f"{100*r['mean_per_channel_acc']:>9.2f}%")

    elif cmd == "fft_baseline":
        if config:
            fft_baseline(config)
        else:
            for c in CONFIGS:
                fft_baseline(c)
