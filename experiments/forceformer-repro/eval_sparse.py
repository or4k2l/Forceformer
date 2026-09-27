import json
from pathlib import Path
import torch
import torch.nn.functional as F
from data import get_splits
from models import SeqClassifier

ROOT = Path(__file__).resolve().parent

torch.manual_seed(0)
VOCAB = 10000
MAX_LEN = 256

(Xtr, ytr), (Xval, yval), stoi, itos = get_splits(max_len=MAX_LEN, vocab_size=VOCAB)
Xval_t = torch.tensor(Xval, dtype=torch.long)
yval_t = torch.tensor(yval, dtype=torch.long)

model = SeqClassifier(VOCAB, "gated_fusion", max_len=8192)
model.load_state_dict(torch.load(ROOT / "gated_fusion.pt", map_location="cpu", weights_only=True))
model.eval()

def forward_with_frac(ids, frac):
    mask = (ids != 0).float()
    h = model.embed(ids)
    h = model.pos(h)
    for blk in model.blocks:
        normed = blk.ln1(h)
        mixed = blk.mixer.forward_sparse(normed, mask, frac=frac) if frac < 1.0 else blk.mixer(normed, mask)
        h = h + mixed
        h = h + blk.ff(blk.ln2(h))
    h = model.ln_f(h)
    m = mask.unsqueeze(-1)
    pooled = (h * m).sum(1) / m.sum(1).clamp(min=1.0)
    return model.head(pooled)

with open(ROOT / "sparse_throughput.json") as f:
    thr_data = json.load(f)

print(f"{'frac':>6}{'val_acc':>10}{'thr@256':>10}{'adj_acc':>10}")
rows = []
for frac in [0.05, 0.1, 0.25, 0.5, 1.0]:
    with torch.no_grad():
        logits = forward_with_frac(Xval_t, frac)
    pred = logits.argmax(-1)
    acc = (pred == yval_t).float().mean().item()
    thr = thr_data[str(frac)]["256"]
    adj = acc * thr
    rows.append({"frac": frac, "acc": acc, "thr256": thr, "adj_acc": adj})
    print(f"{frac:6.2f}{acc*100:9.1f}%{thr:9.1f} {adj:9.1f}")

with open(ROOT / "sparse_accuracy.json", "w") as f:
    json.dump(rows, f, indent=2)
