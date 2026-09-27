"""Benchmark full model inference and stress-test long sequence lengths."""
import json
import statistics
import time
from pathlib import Path

import torch
from models import SeqClassifier

ROOT = Path(__file__).resolve().parent
MIXERS = ("glu", "linear_attn", "softmax_attn", "gated_fusion")
SEQ_LENS = (128, 256, 512, 1024, 2048, 4096)
LONG_CTX_LENS = (8192, 16384, 32768, 65536)
VOCAB_SIZE = 10000

def sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)

def bench(model, seq_len, batch_size, repeats, device):
    model.eval()
    ids = torch.randint(1, VOCAB_SIZE - 1, (batch_size, seq_len), device=device)
    with torch.inference_mode():
        model(ids)
        sync(device)
        measurements = []
        for _ in range(repeats):
            started = time.perf_counter()
            model(ids)
            sync(device)
            measurements.append(time.perf_counter() - started)
    return batch_size / statistics.median(measurements)

def load_model(name, device):
    model = SeqClassifier(VOCAB_SIZE, name, max_len=8192).to(device)
    state = torch.load(ROOT / f"{name}.pt", map_location=device, weights_only=True)
    model.load_state_dict(state)
    return model.eval()

def is_oom(error):
    return "out of memory" in str(error).lower()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}; tokenization and data loading are excluded")
    throughput = {}
    long_context = {}
    for name in MIXERS:
        model = load_model(name, device)
        row = {}
        for length in SEQ_LENS:
            estimated_score_bytes = 8 * 4 * length * length * 4 * 2
            if name == "softmax_attn" and estimated_score_bytes > 3.5e9:
                row[str(length)] = None
                print(f"{name:14s} L={length:6d} skipped (predicted memory limit)")
                break
            try:
                value = bench(model, length, batch_size=8, repeats=5, device=device)
                row[str(length)] = value
                print(f"{name:14s} L={length:6d} {value:9.1f} sequences/s")
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                row[str(length)] = None
                print(f"{name:14s} L={length:6d} OOM")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
        throughput[name] = row
        long_row = {}
        for length in LONG_CTX_LENS:
            batch_size = 1 if name == "softmax_attn" else 2
            estimated_score_bytes = batch_size * 4 * length * length * 4
            if name == "softmax_attn" and estimated_score_bytes > 3.5e9:
                long_row[str(length)] = {
                    "status": "oom_predicted",
                    "note": f"attention scores alone require about {estimated_score_bytes / 1e9:.1f} GB",
                }
                print(f"{name:14s} L={length:6d} skipped (predicted memory limit)")
                break
            try:
                elapsed_tps = bench(model, length, batch_size, repeats=3, device=device)
                long_row[str(length)] = {"status": "ok", "batch": batch_size,
                                         "sequences_per_second": elapsed_tps}
                print(f"{name:14s} L={length:6d} {elapsed_tps:9.2f} sequences/s")
            except RuntimeError as exc:
                if not is_oom(exc):
                    raise
                long_row[str(length)] = {"status": "oom"}
                print(f"{name:14s} L={length:6d} OOM")
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                break
        long_context[name] = long_row
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    (ROOT / "throughput_results.json").write_text(
        json.dumps(throughput, indent=2) + "\n", encoding="utf-8")
    (ROOT / "long_context_results.json").write_text(
        json.dumps(long_context, indent=2) + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
