"""
Training script that intentionally induces instabilities to test the
Loss Spike Detector & Autopsy Tool.

Instabilities injected:
    1. Corrupted batch (random token injection) at specific steps
    2. Learning rate spike (10x sudden increase)
    3. Bad data injection (all-same-token sequences)
    4. Gradient scaling attack (manually scale gradients)

Uses MPS backend on Apple Silicon for hardware acceleration.
"""

from __future__ import annotations

import os
import random
import time

import numpy as np
import torch
import tiktoken

from model import MiniGPT
from spike_detector import TrainingMonitor, PostMortemReport


# ─── Configuration ───────────────────────────────────────────────────────────

SEED = 42
BLOCK_SIZE = 128
BATCH_SIZE = 16
N_STEPS = 600
BASE_LR = 3e-4
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
LOG_DIR = "./spike_logs"
REPORT_DIR = "./reports"

# Instability injection schedule
CORRUPT_BATCH_STEPS = {150, 151}       # Inject garbage tokens
LR_SPIKE_STEPS = {300, 301, 302}       # 10x learning rate
BAD_DATA_STEPS = {400, 401, 402}       # All-same-token batches
GRAD_SCALE_STEPS = {500}               # Manually scale gradients 50x


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_dataset(block_size: int, tokenizer) -> torch.Tensor:
    """
    Load a small text dataset. Uses a built-in corpus for simplicity:
    generates synthetic text from Shakespeare-like patterns if no file found.
    """
    data_path = "data/input.txt"

    if os.path.exists(data_path):
        with open(data_path, "r") as f:
            text = f.read()
    else:
        # Generate synthetic training data — repeated patterns that a small
        # transformer can learn from. Mix of structured and natural-ish text.
        print("No data/input.txt found. Generating synthetic training corpus...")
        os.makedirs("data", exist_ok=True)

        paragraphs = [
            "The quick brown fox jumps over the lazy dog. "
            "A stitch in time saves nine. "
            "All that glitters is not gold. "
            "The early bird catches the worm. ",

            "In the beginning was the word, and the word was with code, "
            "and the code was good. The function called itself recursively, "
            "and the stack grew deeper with each invocation. ",

            "To be or not to be, that is the question. "
            "Whether it is nobler in the mind to suffer "
            "the slings and arrows of outrageous fortune, "
            "or to take arms against a sea of troubles. ",

            "The rain in Spain falls mainly on the plain. "
            "She sells seashells by the seashore. "
            "Peter Piper picked a peck of pickled peppers. ",

            "Machine learning is a subset of artificial intelligence. "
            "Neural networks are composed of layers of neurons. "
            "Gradient descent optimizes the loss function. "
            "Backpropagation computes gradients efficiently. ",
        ]
        # Repeat to get enough data (~500KB)
        text = "\n\n".join(paragraphs * 200)
        with open(data_path, "w") as f:
            f.write(text)

    tokens = tokenizer.encode(text)
    print(f"Dataset: {len(tokens):,} tokens")
    return torch.tensor(tokens, dtype=torch.long)


def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a random batch of sequences from the dataset."""
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


def corrupt_batch(x: torch.Tensor, y: torch.Tensor, vocab_size: int):
    """Replace batch with random tokens — simulates data corruption."""
    x = torch.randint(0, vocab_size, x.shape, device=x.device)
    y = torch.randint(0, vocab_size, y.shape, device=y.device)
    return x, y


def bad_data_batch(x: torch.Tensor, y: torch.Tensor):
    """Replace batch with all-same-token sequences — simulates degenerate data."""
    token_id = 0  # padding token
    x = torch.full_like(x, token_id)
    y = torch.full_like(y, token_id)
    return x, y


def train():
    set_seed(SEED)
    print(f"Device: {DEVICE}")

    tokenizer = tiktoken.get_encoding("gpt2")
    data = load_dataset(BLOCK_SIZE, tokenizer)

    model = MiniGPT(
        vocab_size=tokenizer.n_vocab,
        block_size=BLOCK_SIZE,
        n_layer=6,
        n_head=6,
        n_embd=192,
    ).to(DEVICE)

    optimizer = torch.optim.AdamW(model.parameters(), lr=BASE_LR, weight_decay=0.01)

    # Initialize the spike detector
    # Use a longer warmup (100 steps) so detectors calibrate after
    # the initial rapid loss decrease, and higher thresholds to reduce
    # noise while still catching the intentional instabilities.
    monitor = TrainingMonitor(
        model=model,
        optimizer=optimizer,
        log_dir=LOG_DIR,
        cusum_allowance=1.0,
        cusum_threshold=5.0,
        shewhart_sigma=3.5,
        warmup_steps=120,
    )

    print(f"\nStarting training for {N_STEPS} steps...")
    print(f"Instability schedule:")
    print(f"  Steps {CORRUPT_BATCH_STEPS}: Corrupted batches (random tokens)")
    print(f"  Steps {LR_SPIKE_STEPS}: Learning rate spike (10x)")
    print(f"  Steps {BAD_DATA_STEPS}: Bad data injection (constant tokens)")
    print(f"  Steps {GRAD_SCALE_STEPS}: Gradient scaling attack (50x)")
    print()

    model.train()
    start_time = time.time()

    for step in range(N_STEPS):
        # Get batch
        x, y = get_batch(data, BATCH_SIZE, BLOCK_SIZE, DEVICE)

        # ─── Instability Injection ───────────────────────────────────────

        # 1. Corrupt batch
        if step in CORRUPT_BATCH_STEPS:
            print(f"[INJECT] Step {step}: Corrupting batch with random tokens")
            x, y = corrupt_batch(x, y, tokenizer.n_vocab)

        # 2. Learning rate spike
        if step in LR_SPIKE_STEPS:
            for pg in optimizer.param_groups:
                pg["lr"] = BASE_LR * 10
            if step == min(LR_SPIKE_STEPS):
                print(f"[INJECT] Step {step}: LR spiked to {BASE_LR * 10:.2e}")
        elif step == max(LR_SPIKE_STEPS) + 1:
            for pg in optimizer.param_groups:
                pg["lr"] = BASE_LR
            print(f"[INJECT] Step {step}: LR restored to {BASE_LR:.2e}")

        # 3. Bad data
        if step in BAD_DATA_STEPS:
            print(f"[INJECT] Step {step}: Injecting degenerate all-zero batch")
            x, y = bad_data_batch(x, y)

        # ─── Training Step ───────────────────────────────────────────────

        _, loss = model(x, y)
        loss.backward()

        # 4. Gradient scaling attack (after backward, before monitor/step)
        if step in GRAD_SCALE_STEPS:
            print(f"[INJECT] Step {step}: Scaling all gradients by 50x")
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data *= 50.0

        # Monitor MUST be called after backward() but before optimizer.step()
        alerts = monitor.step(step, loss.item(), batch=x)

        # Gradient clipping (a good practice, but won't prevent all spikes)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        optimizer.zero_grad()

        # Progress logging
        if step % 50 == 0 or step == N_STEPS - 1:
            elapsed = time.time() - start_time
            print(f"Step {step:4d} | Loss: {loss.item():.4f} | "
                  f"Time: {elapsed:.1f}s | Alerts so far: {len(monitor.alerts)}")

    # ─── Post-mortem Report ──────────────────────────────────────────────

    elapsed = time.time() - start_time
    print(f"\nTraining complete in {elapsed:.1f}s")
    print(f"Total alerts: {len(monitor.alerts)}")
    print(f"Total snapshots: {len(monitor.snapshots)}")

    report = PostMortemReport(monitor)
    report.generate(REPORT_DIR)

    print(f"\nAll artifacts saved:")
    print(f"  Forensic snapshots: {LOG_DIR}/")
    print(f"  Report & plots:     {REPORT_DIR}/")


if __name__ == "__main__":
    train()
