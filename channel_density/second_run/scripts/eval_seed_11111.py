"""
==========================================================================
Channel Density Scaling — Model-Agnostic
==========================================================================

Tests whether per-channel FDM accuracy improves when each carrier gets
more spectral room (fewer channels at the same token budget).

Key design: FIXED TRAINING BUDGET (20K samples) across all configs.
If per-channel accuracy rises with fewer channels, that confirms
spectral separation is a real bottleneck for neural FDM reading.

Supports three architectures:
  - GPT-2 Medium (345M): 24 layers, 50259 vocab, fp32
  - Qwen3-0.6B-Base (600M): 28 layers, 151936 vocab, bf16
  - LFM2.5-1.2B-Base (1.2B): 16 layers (10 conv + 6 attn), 65536 vocab, bf16

Each model has its own vocab-appropriate token mapping.

Configs: 40ch, 20ch_dense, 20ch_spread, 10ch_dense, 10ch_spread

Usage:
  python channel_density_scaling_multi.py generate --model gpt2 [--config 40ch]
  python channel_density_scaling_multi.py train --model gpt2 [--config 40ch]
  python channel_density_scaling_multi.py eval --model gpt2 [--config 40ch]
  python channel_density_scaling_multi.py eval --model gpt2  # evals all configs

  # Full pipeline for one model:
  for cfg in 40ch 20ch_dense 20ch_spread 10ch_dense 10ch_spread; do
    python channel_density_scaling_multi.py generate --model qwen3 --config $cfg
    python channel_density_scaling_multi.py train --model qwen3 --config $cfg
  done
  python channel_density_scaling_multi.py eval --model qwen3
"""

import json, random, os, sys, argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    get_linear_schedule_with_warmup,
)

# ============================================================
# Model Configurations
# ============================================================

MODEL_CONFIGS = {
    "gpt2": {
        "model_id": "openai-community/gpt2-medium",
        "vocab_size": 50257,   # base vocab before special tokens
        "dtype": torch.float32,
        "trust_remote_code": False,
        "batch_size": 4,
        "grad_accum": 1,
        "max_length": 768,
        "data_prefix": "gpt2",
    },
    "qwen3": {
        "model_id": "Qwen/Qwen3-0.6B-Base",
        "vocab_size": 151936,
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "batch_size": 4,
        "grad_accum": 1,
        "max_length": 1024,
        "data_prefix": "qwen3",
    },
    "lfm2": {
        "model_id": "LiquidAI/LFM2.5-1.2B-Base",
        "vocab_size": 65536,
        "dtype": torch.bfloat16,
        "trust_remote_code": True,
        "batch_size": 2,
        "grad_accum": 2,  # effective batch = 4
        "max_length": 1024,
        "data_prefix": "lfm2",
    },
    "hermes3": {
        "model_id": "NousResearch/Hermes-3-Llama-3.2-3B",
        "vocab_size": 128256,
        "dtype": torch.bfloat16,
        "trust_remote_code": False,
        "batch_size": 1,
        "grad_accum": 4,  # effective batch = 4
        "max_length": 1024,
        "data_prefix": "hermes3",
        "gradient_checkpointing": True,
    },
}

# ============================================================
# Channel Schemas (same across all models)
# ============================================================

MEMORY_SCHEMAS = {
    0: ("SECRET",   ["RED", "BLUE", "GREEN", "GOLD"]),
    1: ("LOCATION", ["PARIS", "TOKYO", "LONDON", "BERLIN"]),
    2: ("AGENT",    ["ALICE", "BOB", "CAROL", "DAVE"]),
    3: ("STATUS",   ["CLEAR", "COMPROMISED", "UNKNOWN"]),
    4: ("PRIORITY", ["HIGH", "MEDIUM", "LOW"]),
    5: ("BACKUP",   ["AVAILABLE", "UNAVAILABLE"]),
    6: ("RULE",     ["SAFETY_FIRST", "MISSION_FIRST", "BALANCED", "CAUTIOUS"]),
    7: ("META",     ["NONE", "OVERRIDE_STATUS", "OVERRIDE_PRIORITY", "EMERGENCY", "LOCKDOWN"]),
    8:  ("TEAM",     ["RED_TEAM", "BLUE_TEAM", "GREEN_TEAM", "GOLD_TEAM"]),
    9:  ("REGION",   ["NORTH", "SOUTH", "EAST", "WEST"]),
    10: ("PHASE",    ["ALPHA", "BETA", "GAMMA", "DELTA"]),
    11: ("COMM",     ["OPEN", "CLOSED", "RESTRICTED"]),
    12: ("ASSET",    ["VEHICLE", "AIRCRAFT", "DRONE", "BOAT"]),
    13: ("WINDOW",   ["DAWN", "MIDDAY", "DUSK", "NIGHT"]),
    14: ("COVER",    ["DEEP", "SHALLOW", "NONE"]),
    15: ("SUPPORT",  ["ACTIVE", "STANDBY", "OFFLINE"]),
    16: ("THREAT",   ["LOW", "MEDIUM", "HIGH", "CRITICAL"]),
    17: ("WEATHER",  ["CLEAR", "STORM", "FOG"]),
    18: ("TERRAIN",  ["URBAN", "RURAL", "COASTAL", "MOUNTAIN"]),
    19: ("EXTRACT",  ["READY", "DELAYED", "UNAVAILABLE"]),
    20: ("CIPHER",   ["AES", "RSA", "BLOWFISH", "TWOFISH"]),
    21: ("FREQ",     ["HF", "VHF", "UHF", "SHF"]),
    22: ("PAYLOAD",  ["LIGHT", "MEDIUM", "HEAVY", "CRITICAL"]),
    23: ("ROUTE",    ["ALPHA", "BRAVO", "CHARLIE", "DELTA"]),
    24: ("DURATION", ["SHORT", "MEDIUM", "LONG", "EXTENDED"]),
    25: ("CONTACT",  ["FRIENDLY", "NEUTRAL", "HOSTILE", "UNKNOWN"]),
    26: ("FUEL",     ["FULL", "HALF", "LOW", "CRITICAL"]),
    27: ("ALTITUDE", ["LOW", "MEDIUM", "HIGH"]),
    28: ("VISIBILITY", ["CLEAR", "REDUCED", "ZERO"]),
    29: ("NOISE",    ["SILENT", "QUIET", "MODERATE", "LOUD"]),
    30: ("FORMATION", ["SINGLE", "PAIR", "SQUAD", "PLATOON"]),
    31: ("ARMOR",    ["NONE", "LIGHT", "MEDIUM", "HEAVY"]),
    32: ("SIGNAL",   ["STRONG", "WEAK", "JAMMED", "LOST"]),
    33: ("MORALE",   ["HIGH", "MEDIUM", "LOW"]),
    34: ("SUPPLY",   ["ABUNDANT", "ADEQUATE", "SCARCE", "DEPLETED"]),
    35: ("INTEL",    ["CONFIRMED", "PROBABLE", "UNCERTAIN", "NONE"]),
    36: ("EVAC",     ["STANDING", "PREPPED", "LAUNCHED", "ABORTED"]),
    37: ("WEATHER2", ["SUNNY", "OVERCAST", "RAIN", "SNOW"]),
    38: ("DOCTRINE", ["OFFENSIVE", "DEFENSIVE", "RECON", "SUPPORT"]),
    39: ("COMMS",    ["SECURE", "OPEN", "COMPROMISED", "SILENT"]),
}
NUM_CHANNELS = 40


def get_channel_subset(num_channels):
    """Return first N channels from MEMORY_SCHEMAS."""
    if num_channels >= NUM_CHANNELS:
        return dict(MEMORY_SCHEMAS)
    return {ch: MEMORY_SCHEMAS[ch] for ch in range(num_channels)}


# ============================================================
# Experiment Configs
# ============================================================

CONFIGS = {
    "40ch": {
        "num_channels": 40,
        "carrier_spacing": "dense",
        "train_samples": 20000,
        "description": "40 channels, carriers 1-40 Hz",
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
# FDM Encoder — model-agnostic (takes vocab_size)
# ============================================================

class ScaledChannelEncoder:
    """
    FDM encoder with configurable channel count, spacing, and vocab size.
    Token mapping draws from the target model's vocabulary.
    """

    def __init__(self, vocab_size, tokenizer, num_channels=40, use_srandom=True,
                 carrier_spacing="dense",
                 num_tokens_per_encoder=256, sample_rate=100.0,
                 a_high=1.0, a_low=0.25, num_levels=64, seed=42):
        self.num_tokens_per_encoder = num_tokens_per_encoder
        self.total_tokens = num_tokens_per_encoder * 2
        self.sample_rate = sample_rate
        self.a_high = a_high
        self.a_low = a_low
        self.num_levels = num_levels
        self.num_channels = num_channels
        self.tokenizer = tokenizer

        # Carrier frequencies
        if carrier_spacing == "dense":
            self.carrier_freqs = [1.0 + i * 1.0 for i in range(num_channels)]
        elif carrier_spacing == "spread":
            if num_channels == 1:
                self.carrier_freqs = [1.0]
            else:
                step = 39.0 / (num_channels - 1)
                self.carrier_freqs = [1.0 + i * step for i in range(num_channels)]
        else:
            raise ValueError(f"Unknown spacing: {carrier_spacing}")

        # S-random interleaver
        self.use_srandom = use_srandom
        S = int(np.sqrt(num_tokens_per_encoder / 2))
        self.interleaver = self._generate_s_random_interleaver(
            num_tokens_per_encoder, S)

        # Token mapping from model's vocab
        rng = np.random.RandomState(seed)
        self.token_map = rng.choice(vocab_size, size=num_levels, replace=False)

    def _generate_s_random_interleaver(self, length, S):
        import random as rnd
        rnd.seed(42)
        interleaver = list(range(length))
        for i in range(length):
            for _ in range(100):
                j = rnd.randint(i, length - 1)
                valid = True
                for k in range(max(0, i - S + 1), i):
                    if abs(interleaver[j] - interleaver[k]) < S:
                        valid = False
                        break
                if valid:
                    interleaver[i], interleaver[j] = interleaver[j], interleaver[i]
                    break
        return interleaver

    def value_to_bits(self, channel_id, value):
        _, values = MEMORY_SCHEMAS[channel_id]
        idx = values.index(value)
        num_bits = max(1, int(np.ceil(np.log2(max(len(values), 2)))))
        return format(idx, f'0{num_bits}b'), num_bits

    def encode_memory(self, memory):
        """Encode channel values into FDM token sequence."""
        all_bits = {}
        max_bits = 0
        for ch in range(self.num_channels):
            if ch not in memory:
                continue
            bits, nb = self.value_to_bits(ch, memory[ch])
            all_bits[ch] = bits
            max_bits = max(max_bits, nb)
        for ch in all_bits:
            all_bits[ch] = all_bits[ch].ljust(max_bits, '0')

        t = np.arange(self.num_tokens_per_encoder) / self.sample_rate
        samples_per_bit = self.num_tokens_per_encoder // max_bits

        composite = np.zeros(self.num_tokens_per_encoder)
        for ch in range(self.num_channels):
            if ch not in all_bits:
                continue
            bits = all_bits[ch]
            for bi, bit in enumerate(bits):
                start = bi * samples_per_bit
                end = min((bi + 1) * samples_per_bit, self.num_tokens_per_encoder)
                amp = self.a_high if bit == '1' else self.a_low
                composite[start:end] += amp * np.sin(
                    2 * np.pi * self.carrier_freqs[ch] * t[start:end])

        sig_min, sig_max = composite.min(), composite.max()
        sig_range = sig_max - sig_min + 1e-10
        norm = (composite - sig_min) / sig_range
        q1 = np.clip(np.floor(norm * (self.num_levels - 1) + 0.5).astype(int),
                      0, self.num_levels - 1)
        tokens1 = [int(self.token_map[q]) for q in q1]

        if self.use_srandom:
            interleaved = composite[self.interleaver]
        else:
            interleaved = composite
        norm2 = (interleaved - sig_min) / sig_range
        q2 = np.clip(np.floor(norm2 * (self.num_levels - 1) + 0.5).astype(int),
                      0, self.num_levels - 1)
        tokens2 = [int(self.token_map[q]) for q in q2]

        all_tokens = tokens1 + tokens2
        fdm_text = self.tokenizer.decode(all_tokens)
        return fdm_text, all_tokens


# ============================================================
# Reasoning Logic (same across all models)
# ============================================================

def generate_reasoning(memory, channel_schemas):
    """Generate expected answer from channel values."""
    secret = memory.get(0, "RED")
    location = memory.get(1, "TOKYO")
    agent = memory.get(2, "ALICE")
    status = memory.get(3, "CLEAR")
    priority = memory.get(4, "MEDIUM")
    backup = memory.get(5, "AVAILABLE")
    rule = memory.get(6, "BALANCED")
    meta = memory.get(7, "NONE")

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

def generate_data(model_name, config_name, base_dir="channel_scaling_data"):
    """Generate training data for a model+config combination."""
    mcfg = MODEL_CONFIGS[model_name]
    cfg = CONFIGS[config_name]
    num_channels = cfg["num_channels"]
    num_samples = cfg["train_samples"]
    channel_schemas = get_channel_subset(num_channels)

    output_dir = os.path.join(base_dir, model_name)
    os.makedirs(output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["model_id"], trust_remote_code=mcfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    encoder = ScaledChannelEncoder(
        vocab_size=mcfg["vocab_size"],
        tokenizer=tokenizer,
        num_channels=num_channels,
        carrier_spacing=cfg["carrier_spacing"],
        use_srandom=cfg.get("use_srandom", True),
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

    output_path = os.path.join(output_dir, f"data_{config_name}.jsonl")

    print(f"\nGenerating {num_samples} samples for {model_name}/{config_name}: "
          f"{cfg['description']}")
    print(f"  Vocab: {mcfg['vocab_size']}, Carriers: "
          f"{[f'{f:.1f}' for f in encoder.carrier_freqs[:5]]}... Hz")

    random.seed(12345)
    np.random.seed(12345)

    question = (
        "Analyze the encoded operational data. Apply the reasoning rule "
        "and meta-instruction to determine the correct action. Report "
        "all context parameters."
    )

    samples = []
    for i in tqdm(range(num_samples)):
        memory = {}
        for ch in sorted(channel_schemas.keys()):
            _, values = channel_schemas[ch]
            memory[ch] = random.choice(values)

        fdm_text, token_ids = encoder.encode_memory(memory)
        response, action = generate_reasoning(memory, channel_schemas)

        samples.append({
            "fdm_text": fdm_text,
            "question": question,
            "response": response,
            "answer": response,
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
# Dataset
# ============================================================

class ScalingDataset(Dataset):
    """Dataset that re-encodes FDM on the fly."""

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
# Training — 5-stage curriculum
# ============================================================

CURRICULUM = [
    # (a_high, a_low, fraction_of_data, epochs, lr)
    (1.0, 0.00, 0.13, 5,  5e-5),
    (1.0, 0.10, 0.17, 5,  3e-5),
    (1.0, 0.15, 0.22, 7,  2e-5),
    (1.0, 0.20, 0.22, 7,  1e-5),
    (1.0, 0.25, 0.26, 10, 1e-5),
]


def train(model_name, config_name, base_dir="channel_scaling_data",
          model_dir="channel_scaling_models"):
    """Train with curriculum for a model+config."""
    mcfg = MODEL_CONFIGS[model_name]
    cfg = CONFIGS[config_name]
    num_channels = cfg["num_channels"]

    print(f"\n{'='*60}")
    print(f"Training {model_name}/{config_name}: {cfg['description']}")
    print(f"  Model: {mcfg['model_id']}")
    print(f"  Batch: {mcfg['batch_size']} x {mcfg['grad_accum']} accum "
          f"= {mcfg['batch_size'] * mcfg['grad_accum']} effective")
    print(f"{'='*60}")

    tokenizer = AutoTokenizer.from_pretrained(
        mcfg["model_id"], trust_remote_code=mcfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Add special tokens
    special = {"additional_special_tokens": ["[MEMORY]", "[/MEMORY]"]}
    tokenizer.add_special_tokens(special)

    # Load base model
    if mcfg["dtype"] == torch.float32:
        model = AutoModelForCausalLM.from_pretrained(mcfg["model_id"])
    else:
        model = AutoModelForCausalLM.from_pretrained(
            mcfg["model_id"], torch_dtype=mcfg["dtype"],
            trust_remote_code=mcfg["trust_remote_code"])

    model.resize_token_embeddings(len(tokenizer))
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    if mcfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable()
        model.config.use_cache = False

    data_dir = os.path.join(base_dir, model_name)
    train_path = os.path.join(data_dir, f"data_{config_name}_train.jsonl")
    if not os.path.exists(train_path):
        print(f"ERROR: {train_path} not found. Run 'generate' first.")
        return

    out_dir = os.path.join(model_dir, model_name)
    os.makedirs(out_dir, exist_ok=True)

    batch_size = mcfg["batch_size"]
    grad_accum = mcfg["grad_accum"]
    max_length = mcfg["max_length"]

    for stage_idx, (a_high, a_low, frac, epochs, lr) in enumerate(CURRICULUM):
        ckpt_path = os.path.join(out_dir, f"model_{config_name}_stage{stage_idx}")
        if os.path.exists(ckpt_path):
            print(f"\n  Stage {stage_idx} checkpoint exists, loading and skipping...")
            model = AutoModelForCausalLM.from_pretrained(
                ckpt_path,
                torch_dtype=mcfg["dtype"] if mcfg["dtype"] != torch.float32 else None,
                trust_remote_code=mcfg["trust_remote_code"])
            model.to(device)
            continue

        print(f"\n--- Stage {stage_idx}: a_low={a_low}, epochs={epochs}, lr={lr} ---")

        encoder = ScaledChannelEncoder(
            vocab_size=mcfg["vocab_size"],
            tokenizer=tokenizer,
            num_channels=num_channels,
            carrier_spacing=cfg["carrier_spacing"],
            use_srandom=cfg.get("use_srandom", True),
            a_high=a_high, a_low=a_low, num_levels=64, seed=42,
        )

        train_dataset = ScalingDataset(train_path, tokenizer, encoder, max_length)
        n_use = int(len(train_dataset) * frac)
        subset = torch.utils.data.Subset(train_dataset, list(range(n_use)))
        train_loader = DataLoader(subset, batch_size=batch_size, shuffle=True)

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        total_steps = (len(train_loader) // grad_accum) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=total_steps // 10,
            num_training_steps=total_steps,
        )

        for epoch in range(epochs):
            model.train()
            total_loss = 0
            optimizer.zero_grad()

            for step, batch in enumerate(tqdm(
                    train_loader, desc=f"S{stage_idx} E{epoch+1}/{epochs}")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                labels = batch['labels'].to(device)

                outputs = model(input_ids=input_ids,
                                attention_mask=attention_mask,
                                labels=labels)
                loss = outputs.loss / grad_accum
                loss.backward()
                total_loss += loss.item() * grad_accum

                if (step + 1) % grad_accum == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()

            # Handle leftover steps
            if (step + 1) % grad_accum != 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            avg_loss = total_loss / len(train_loader)
            print(f"    Epoch {epoch+1}: loss={avg_loss:.4f}")

        model.save_pretrained(ckpt_path)
        tokenizer.save_pretrained(ckpt_path)

    # Save final
    final_path = os.path.join(out_dir, f"model_{config_name}_final")
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    print(f"\nSaved final model to {final_path}")


# ============================================================
# Evaluation
# ============================================================

def evaluate(model_name, config_name, model_dir="channel_scaling_models",
             num_eval=500):
    """Evaluate trained model on fresh random samples."""
    mcfg = MODEL_CONFIGS[model_name]
    cfg = CONFIGS[config_name]
    num_channels = cfg["num_channels"]
    channel_schemas = get_channel_subset(num_channels)

    out_dir = os.path.join(model_dir, model_name)
    model_path = os.path.join(out_dir, f"model_{config_name}_final")
    if not os.path.exists(model_path):
        print(f"No model at {model_path}")
        return None

    print(f"\nEvaluating {model_name}/{config_name}: {cfg['description']}")

    tokenizer = AutoTokenizer.from_pretrained(
        model_path, trust_remote_code=mcfg["trust_remote_code"])
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if mcfg["dtype"] == torch.float32:
        model = AutoModelForCausalLM.from_pretrained(model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=mcfg["dtype"],
            trust_remote_code=mcfg["trust_remote_code"])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()

    encoder = ScaledChannelEncoder(
        vocab_size=mcfg["vocab_size"],
        tokenizer=tokenizer,
        num_channels=num_channels,
        carrier_spacing=cfg["carrier_spacing"],
        use_srandom=cfg.get("use_srandom", True),
        a_high=1.0, a_low=0.25, num_levels=64, seed=42,
    )

    random.seed(11111)
    np.random.seed(11111)

    action_correct = 0
    fact_correct = 0
    rule_correct = 0
    meta_correct = 0
    per_channel_correct = defaultdict(int)
    per_channel_total = defaultdict(int)
    joint_counts = []
    total = 0

    question = (
        "Analyze the encoded operational data. Apply the reasoning rule "
        "and meta-instruction to determine the correct action. Report "
        "all context parameters."
    )

    for _ in tqdm(range(num_eval), desc=f"Eval {config_name}"):
        memory = {}
        for ch in sorted(channel_schemas.keys()):
            _, values = channel_schemas[ch]
            memory[ch] = random.choice(values)

        fdm_text, _ = encoder.encode_memory(memory)
        expected_response, expected_action = generate_reasoning(memory, channel_schemas)

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

        if f"Decision: {expected_action}" in generated:
            action_correct += 1

        facts_ok = True
        for ch in range(min(6, num_channels)):
            if memory[ch] not in generated:
                facts_ok = False
        if facts_ok:
            fact_correct += 1

        if memory.get(6, "") in generated:
            rule_correct += 1
        if f"Meta: {memory.get(7, '')}" in generated:
            meta_correct += 1

        extra_correct_this = 0
        for ch in sorted(channel_schemas.keys()):
            if ch >= 8:
                name = channel_schemas[ch][0]
                val = memory[ch]
                per_channel_total[ch] += 1
                if f"{name}={val}" in generated:
                    per_channel_correct[ch] += 1
                    extra_correct_this += 1
        joint_counts.append(extra_correct_this)
        total += 1

    num_extra = max(1, num_channels - 8)
    extra_acc = (sum(per_channel_correct.values()) /
                 max(1, sum(per_channel_total.values())))

    print(f"\n{'='*60}")
    print(f"RESULTS: {model_name}/{config_name} ({num_channels} channels)")
    print(f"{'='*60}")
    print(f"  Action:  {100*action_correct/total:.1f}%")
    print(f"  Rule:    {100*rule_correct/total:.1f}%")
    print(f"  Meta:    {100*meta_correct/total:.1f}%")
    print(f"  Fact:    {100*fact_correct/total:.1f}%")
    print(f"  Extra:   {100*extra_acc:.1f}%")

    if per_channel_total:
        print(f"\n  Per extra channel:")
        for ch in sorted(per_channel_total.keys()):
            acc = per_channel_correct[ch] / per_channel_total[ch]
            name = channel_schemas[ch][0]
            freq = encoder.carrier_freqs[ch] if ch < len(encoder.carrier_freqs) else "?"
            print(f"    ch{ch:2d} ({name:10s}) @ {freq:5.1f} Hz: {100*acc:.1f}%")

    mean_per_ch = np.mean([per_channel_correct[ch] / per_channel_total[ch]
                           for ch in per_channel_total]) if per_channel_total else 0
    print(f"\n  Mean per-channel accuracy: {100*mean_per_ch:.2f}%")

    results = {
        "model": model_name,
        "config": config_name,
        "num_channels": num_channels,
        "carrier_spacing": cfg["carrier_spacing"],
        "action_acc": action_correct / total,
        "rule_acc": rule_correct / total,
        "meta_acc": meta_correct / total,
        "fact_acc": fact_correct / total,
        "extra_acc": extra_acc,
        "mean_per_channel_acc": float(mean_per_ch),
        "per_channel": {str(ch): per_channel_correct[ch] / per_channel_total[ch]
                        for ch in per_channel_total},
    }

    results_dir = os.path.join("channel_scaling_results", model_name)
    os.makedirs(results_dir, exist_ok=True)
    with open(os.path.join(results_dir, f"{config_name}.json"), 'w') as f:
        json.dump(results, f, indent=2)

    return results


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="Channel Density Scaling — Multi-Model")
    parser.add_argument('command', choices=['generate', 'train', 'eval'],
                        help="generate data, train model, or evaluate")
    parser.add_argument('--model', type=str, required=True,
                        choices=['gpt2', 'qwen3', 'lfm2', 'hermes3'])
    parser.add_argument('--config', type=str, default=None,
                        choices=list(CONFIGS.keys()),
                        help="Specific config (default: all)")
    parser.add_argument('--num_eval', type=int, default=500)
    parser.add_argument('--data_dir', type=str, default="channel_scaling_data")
    parser.add_argument('--model_dir', type=str, default="channel_scaling_models")
    args = parser.parse_args()

    configs_to_run = [args.config] if args.config else list(CONFIGS.keys())

    if args.command == "generate":
        for cfg in configs_to_run:
            generate_data(args.model, cfg, args.data_dir)

    elif args.command == "train":
        for cfg in configs_to_run:
            train(args.model, cfg, args.data_dir, args.model_dir)

    elif args.command == "eval":
        all_results = {}
        for cfg in configs_to_run:
            r = evaluate(args.model, cfg, args.model_dir, args.num_eval)
            if r:
                all_results[cfg] = r

        if len(all_results) > 1:
            print(f"\n{'='*70}")
            print(f"CHANNEL DENSITY SCALING SUMMARY — {args.model}")
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



CONFIGS["20ch_dense_single"] = {
    "num_channels": 20,
    "carrier_spacing": "dense",
    "train_samples": 20000,
    "use_srandom": False,
    "description": "20 channels dense, single encoding (no S-random)",
}
CONFIGS["8ch_dense_single"] = {
    "num_channels": 8,
    "carrier_spacing": "dense",
    "train_samples": 20000,
    "use_srandom": False,
    "description": "8 channels dense, single encoding (no S-random)",
}
CONFIGS["20ch_dense_srandom"] = {
    "num_channels": 20,
    "carrier_spacing": "dense",
    "train_samples": 20000,
    "use_srandom": True,
    "description": "20 channels dense, S-random interleaved",
}
if __name__ == "__main__":
    main()

# Single-encoding configs for S-random ablation
