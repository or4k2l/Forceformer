import json, time
from pathlib import Path
import torch
import torch.nn.functional as F
from data import get_splits
from models import SeqClassifier

ROOT = Path(__file__).resolve().parent

torch.manual_seed(0)
VOCAB = 10000
MAX_LEN = 256

def load(mixer_type):
    m = SeqClassifier(VOCAB, mixer_type, max_len=8192)
    m.load_state_dict(torch.load(ROOT / f"{mixer_type}.pt", map_location="cpu", weights_only=True))
    m.eval()
    return m

(Xtr, ytr), (Xval, yval), stoi, itos = get_splits(max_len=MAX_LEN, vocab_size=VOCAB)
Xval_t = torch.tensor(Xval, dtype=torch.long)
yval_t = torch.tensor(yval, dtype=torch.long)
N = len(yval)

cheap = load("glu")
expensive = load("softmax_attn")

with open(ROOT / "throughput_results.json") as f:
    th = json.load(f)
thr_cheap = th["glu"]["256"]        # seq/s at L=256 (our eval length)
thr_expensive = th["softmax_attn"]["256"]

with torch.no_grad():
    logits_cheap = cheap(Xval_t)
    probs_cheap = F.softmax(logits_cheap, dim=-1)
    conf_cheap, pred_cheap = probs_cheap.max(-1)
    correct_cheap = (pred_cheap == yval_t)

    logits_exp = expensive(Xval_t)
    pred_exp = logits_exp.argmax(-1)
    correct_exp = (pred_exp == yval_t)

acc_cheap_only = correct_cheap.float().mean().item()
acc_exp_only = correct_exp.float().mean().item()
print(f"GLU alone:            acc={acc_cheap_only:.3f}  throughput={thr_cheap:.1f} seq/s")
print(f"StandardAttention alone: acc={acc_exp_only:.3f}  throughput={thr_expensive:.1f} seq/s")
print()

# sort by confidence ascending -> defer the least-confident fraction to the expensive model
order = torch.argsort(conf_cheap)  # ascending: least confident first

results = []
for defer_frac in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]:
    n_defer = int(round(defer_frac * N))
    deferred_idx = order[:n_defer]
    kept_idx = order[n_defer:]

    correct = correct_cheap.clone()
    if n_defer > 0:
        correct[deferred_idx] = correct_exp[deferred_idx]
    overall_acc = correct.float().mean().item()

    # effective throughput: harmonic-style weighted average of per-item time
    # time per item = 1/thr; total time = n_kept/thr_cheap + n_defer/thr_expensive (cheap model runs on ALL items first)
    n_kept = N - n_defer
    total_time = N / thr_cheap + n_defer / thr_expensive  # cheap model always runs on everyone first
    eff_throughput = N / total_time
    adj_acc = overall_acc * eff_throughput

    results.append({
        "defer_frac": defer_frac,
        "n_deferred": n_defer,
        "overall_acc": overall_acc,
        "eff_throughput": eff_throughput,
        "adj_acc": adj_acc,
    })
    print(f"defer {defer_frac*100:4.0f}%  (n={n_defer:3d})  acc={overall_acc:.3f}  "
          f"eff_thr={eff_throughput:7.1f} seq/s  adj_acc={adj_acc:7.1f}")

with open(ROOT / "cascade_results.json", "w") as f:
    json.dump(results, f, indent=2)
