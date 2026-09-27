"""Benchmark sparse local-attention routing at several routing fractions."""
import json
import statistics
import time
from pathlib import Path

import torch
from models import SeqClassifier

ROOT = Path(__file__).resolve().parent
VOCAB_SIZE = 10000
SEQ_LENS = (128, 256, 512, 1024, 2048, 4096)
ROUTING_FRACTIONS = (0.05, 0.1, 0.25, 0.5, 1.0)
BATCH_SIZE = 8

def run_sparse(model, token_ids, mask, fraction):
    hidden = model.pos(model.embed(token_ids))
    for block in model.blocks:
        normalized = block.ln1(hidden)
        mixed = block.mixer.forward_sparse(normalized, mask, frac=fraction)
        hidden = hidden + mixed
        hidden = hidden + block.ff(block.ln2(hidden))
    hidden = model.ln_f(hidden)
    pooled = (hidden * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True).clamp(min=1.0)
    return model.head(pooled)

def bench(model, seq_len, fraction, device, repeats=5):
    model.eval()
    token_ids = torch.randint(1, VOCAB_SIZE - 1, (BATCH_SIZE, seq_len), device=device)
    mask = torch.ones(BATCH_SIZE, seq_len, device=device)
    with torch.inference_mode():
        run_sparse(model, token_ids, mask, fraction)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            run_sparse(model, token_ids, mask, fraction)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            samples.append(time.perf_counter() - started)
    return BATCH_SIZE / statistics.median(samples)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = SeqClassifier(VOCAB_SIZE, "gated_fusion", max_len=max(SEQ_LENS)).to(device)
    results = {}
    for fraction in ROUTING_FRACTIONS:
        row = {}
        print(f"\n=== routing fraction={fraction} ===")
        for length in SEQ_LENS:
            rate = bench(model, length, fraction, device)
            row[str(length)] = rate
            print(f"L={length:6d} {rate:9.1f} sequences/s")
        results[str(fraction)] = row
    (ROOT / "sparse_throughput.json").write_text(
        json.dumps(results, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
