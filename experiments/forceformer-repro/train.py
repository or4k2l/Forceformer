"""Train the four mixer variants on the NLTK movie_reviews corpus."""
import argparse
import json
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from data import get_splits
from models import SeqClassifier, count_params

ROOT = Path(__file__).resolve().parent
MIXERS = ("glu", "linear_attn", "softmax_attn", "gated_fusion")

def batches(features, labels, batch_size, shuffle, seed, device):
    indices = list(range(len(features)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    for start in range(0, len(indices), batch_size):
        batch = indices[start:start + batch_size]
        x = torch.tensor([features[i] for i in batch], dtype=torch.long, device=device)
        y = torch.tensor([labels[i] for i in batch], dtype=torch.long, device=device)
        yield x, y

def evaluate(model, features, labels, batch_size, device):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for x, y in batches(features, labels, batch_size, False, 0, device):
            correct += (model(x).argmax(-1) == y).sum().item()
            total += y.numel()
    return correct / total

def train_model(mixer, x_train, y_train, x_val, y_val, vocab_size,
                max_len, epochs, batch_size, learning_rate, seed, device, output_dir):
    random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)
    model = SeqClassifier(vocab_size, mixer, max_len=8192).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history = []
    best_acc = -1.0
    best_state = None
    started = time.perf_counter()

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        for x, y in batches(x_train, y_train, batch_size, True, seed + epoch, device):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * y.numel()
        val_acc = evaluate(model, x_val, y_val, batch_size * 2, device)
        history.append({"epoch": epoch + 1,
                        "train_loss": total_loss / len(x_train),
                        "val_acc": val_acc})
        print(f"[{mixer}] epoch {epoch + 1}/{epochs} "
              f"loss={history[-1]['train_loss']:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = {key: value.detach().cpu().clone()
                          for key, value in model.state_dict().items()}

    model.load_state_dict(best_state)
    torch.save(best_state, output_dir / f"{mixer}.pt")
    elapsed = time.perf_counter() - started
    return {"mixer": mixer, "params": count_params(model),
            "best_val_acc": best_acc, "train_time_s": elapsed,
            "epochs": epochs, "max_len": max_len, "seed": seed,
            "history": history}

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MIXERS, default=list(MIXERS))
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output-dir", type=Path, default=ROOT,
                        help="Directory for checkpoints and train_results.json (default: this folder)")
    args = parser.parse_args()
    device = torch.device(args.device)
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    (x_train, y_train), (x_val, y_val), stoi, _ = get_splits(
        max_len=args.max_len, vocab_size=10000, seed=args.seed)
    print(f"device={device} train={len(x_train)} val={len(x_val)} vocab={len(stoi)}")

    results = {}
    for mixer in args.models:
        print(f"\n=== Training {mixer} ===")
        results[mixer] = train_model(
            mixer, x_train, y_train, x_val, y_val, len(stoi), args.max_len,
            args.epochs, args.batch_size, args.learning_rate, args.seed, device, output_dir)
    results_path = output_dir / "train_results.json"
    results_path.write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Saved {results_path}")

if __name__ == "__main__":
    main()
