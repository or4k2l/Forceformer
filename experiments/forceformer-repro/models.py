import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=8192):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe)

    def forward(self, x):
        seq_len = x.size(1)
        if seq_len <= self.pe.size(0):
            return x + self.pe[:seq_len].unsqueeze(0)
        position = torch.arange(seq_len, device=x.device).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, self.pe.size(1), 2, device=x.device).float()
                        * (-math.log(10000.0) / self.pe.size(1)))
        pe = torch.zeros(seq_len, self.pe.size(1), device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(position * div)
        pe[:, 1::2] = torch.cos(position * div)
        return x + pe.unsqueeze(0)


# ---------------------------------------------------------------------------
# Mixer 1: EnformerX_GLU -- purely element-wise gating, NO cross-token mixing.
# ---------------------------------------------------------------------------
class GLUMixer(nn.Module):
    def __init__(self, d_model, d_hidden):
        super().__init__()
        self.proj = nn.Linear(d_model, 2 * d_hidden)
        self.out = nn.Linear(d_hidden, d_model)

    def forward(self, x, mask=None):
        a, b = self.proj(x).chunk(2, dim=-1)
        h = a * torch.sigmoid(b)
        return self.out(h)


# ---------------------------------------------------------------------------
# Mixer 2: EnformerX_MH_BiLinear -- multi-head bidirectional LINEAR attention.
# phi(x) = elu(x) + 1 feature map; global (non-causal) linear attention:
# out = phi(Q) @ (phi(K)^T @ V) / (phi(Q) @ sum(phi(K)))   -> O(L) in seq len.
# ---------------------------------------------------------------------------
class LinearAttentionMixer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    @staticmethod
    def feat(x):
        return F.elu(x) + 1.0

    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, h, L, dh)
        if mask is not None:
            m = mask.view(B, 1, L, 1)
            k = k * m
            v = v * m
        qf, kf = self.feat(q), self.feat(k)
        kv = torch.einsum("bhld,bhle->bhde", kf, v)          # (B,h,dh,dh)  O(L)
        z = 1.0 / (torch.einsum("bhld,bhd->bhl", qf, kf.sum(dim=2)) + 1e-6)
        out = torch.einsum("bhld,bhde,bhl->bhle", qf, kv, z)  # (B,h,L,dh)
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        return self.out(out)


# ---------------------------------------------------------------------------
# Mixer 4 (own design, not from a paper): Gated Local-Linear Fusion.
# Every token gets a LEARNED per-token gate deciding how much to trust:
#   (a) a cheap GLOBAL linear-attention path (O(L))            -- broad context
#   (b) a sharp but LOCAL windowed-softmax path (O(L*w), w const) -- precision
# Both paths are computed for every token (no sample-level routing/cascade);
# the fusion happens inside the mixing operation itself, so it stays O(L)
# overall since the window w is a fixed constant, never O(L^2).
# ---------------------------------------------------------------------------
class LocalWindowAttention(nn.Module):
    def __init__(self, d_model, n_heads, window=8):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.w = window
        self.qkv = nn.Linear(d_model, 3 * d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        w = self.w
        qkv = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B,h,L,dh)

        if mask is None:
            mask = torch.ones(B, L, device=x.device)
        # pad K, V, and the token mask by w on each side along the sequence dim
        k_pad = F.pad(k, (0, 0, w, w))          # (B,h,L+2w,dh)
        v_pad = F.pad(v, (0, 0, w, w))
        m_pad = F.pad(mask, (w, w))             # (B, L+2w)

        win = 2 * w + 1
        k_win = k_pad.unfold(2, win, 1)         # (B,h,L,dh,win)
        v_win = v_pad.unfold(2, win, 1)
        m_win = m_pad.unfold(1, win, 1)         # (B,L,win)
        m_win = m_win.unsqueeze(1)              # (B,1,L,win) -> broadcast over heads

        scores = torch.einsum("bhld,bhldw->bhlw", q, k_win) / math.sqrt(self.dh)
        scores = scores.masked_fill(m_win == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)  # rows that are fully masked (pad query) -> 0
        out = torch.einsum("bhlw,bhldw->bhld", attn, v_win)
        return out.permute(0, 2, 1, 3).reshape(B, L, D)

    def forward_selected(self, x, mask, idx):
        """Only compute local windowed attention at the query positions in idx
        (B,k). K/V windows are still built over the full sequence (cheap,
        mostly striding not copying) but the expensive score/softmax/output
        einsums now scale with k, not L -- this is where the real compute
        saving happens versus the dense v1 fusion."""
        B, L, D = x.shape
        w = self.w
        qkv = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        if mask is None:
            mask = torch.ones(B, L, device=x.device)
        k_pad = F.pad(k, (0, 0, w, w))
        v_pad = F.pad(v, (0, 0, w, w))
        m_pad = F.pad(mask, (w, w))
        win = 2 * w + 1
        k_win_full = k_pad.unfold(2, win, 1)   # (B,h,L,dh,win)
        v_win_full = v_pad.unfold(2, win, 1)
        m_win_full = m_pad.unfold(1, win, 1).unsqueeze(1)  # (B,1,L,win)

        kk = idx.shape[1]
        idx_k = idx.view(B, 1, kk, 1, 1).expand(-1, self.h, -1, self.dh, win)
        k_win = torch.gather(k_win_full, 2, idx_k)
        v_win = torch.gather(v_win_full, 2, idx_k)
        idx_m = idx.view(B, 1, kk, 1).expand(-1, 1, -1, win)
        m_win = torch.gather(m_win_full, 2, idx_m)
        idx_q = idx.view(B, 1, kk, 1).expand(-1, self.h, -1, self.dh)
        q_sel = torch.gather(q, 2, idx_q)  # (B,h,kk,dh)

        scores = torch.einsum("bhkd,bhkdw->bhkw", q_sel, k_win) / math.sqrt(self.dh)
        scores = scores.masked_fill(m_win == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        attn = torch.nan_to_num(attn)
        out = torch.einsum("bhkw,bhkdw->bhkd", attn, v_win)
        return out.permute(0, 2, 1, 3).reshape(B, kk, D)  # (B, kk, D)


class GatedLocalLinearMixer(nn.Module):
    def __init__(self, d_model, n_heads, window=8):
        super().__init__()
        self.local = LocalWindowAttention(d_model, n_heads, window)
        self.linear = LinearAttentionMixer(d_model, n_heads)
        self.gate = nn.Linear(d_model, 1)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        local_out = self.local(x, mask)
        # LinearAttentionMixer already has its own out-proj; call the raw mix
        # by temporarily bypassing it is unnecessary -- reuse full module output,
        # both paths are already in model space (D-dim), so we can blend directly.
        lin_out = self.linear(x, mask)
        g = torch.sigmoid(self.gate(x))  # (B, L, 1) -- per-token learned mix weight
        fused = g * local_out + (1 - g) * lin_out
        return self.out(fused)

    def forward_sparse(self, x, mask=None, frac=0.25):
        """v2: the gate decides BEFORE computation which tokens get the
        expensive local-attention refinement at all. Linear attention still
        runs for everyone (it's O(L) and cheap), but the local windowed
        softmax path -- the expensive part -- only runs for the top-`frac`
        tokens by gate score. Non-selected tokens fall back to linear-only
        (equivalent to hard-clamping their gate to 0). This trades a little
        approximation error (their true trained gate wasn't exactly 0) for
        real, measurable compute savings, unlike the dense v1 forward()."""
        B, L, D = x.shape
        lin_out = self.linear(x, mask)                    # (B,L,D), always computed
        gate_logit = self.gate(x).squeeze(-1)              # (B,L)
        if mask is not None:
            gate_logit = gate_logit.masked_fill(mask == 0, float("-inf"))
        k = max(1, round(frac * L))
        k = min(k, L)
        topk_val, topk_idx = torch.topk(gate_logit, k, dim=1)  # (B,k) each
        g_sel = torch.sigmoid(topk_val).unsqueeze(-1)           # (B,k,1)

        local_out_sel = self.local.forward_selected(x, mask, topk_idx)  # (B,k,D)
        lin_out_sel = torch.gather(lin_out, 1, topk_idx.unsqueeze(-1).expand(-1, -1, D))
        fused_sel = g_sel * local_out_sel + (1 - g_sel) * lin_out_sel   # (B,k,D)

        out = lin_out.clone()  # non-selected tokens: linear-only (gate treated as 0)
        out.scatter_(1, topk_idx.unsqueeze(-1).expand(-1, -1, D), fused_sel)
        return self.out(out)


# ---------------------------------------------------------------------------
# Mixer 3: StandardAttention -- reference softmax multi-head attention, O(L^2)
# ---------------------------------------------------------------------------
class SoftmaxAttentionMixer(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        assert d_model % n_heads == 0
        self.h = n_heads
        self.dh = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        B, L, D = x.shape
        qkv = self.qkv(x).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        scores = torch.einsum("bhld,bhmd->bhlm", q, k) / math.sqrt(self.dh)
        if mask is not None:
            m = mask.view(B, 1, 1, L)
            scores = scores.masked_fill(m == 0, float("-inf"))
        attn = torch.softmax(scores, dim=-1)
        out = torch.einsum("bhlm,bhmd->bhld", attn, v)
        out = out.permute(0, 2, 1, 3).reshape(B, L, D)
        return self.out(out)


class Block(nn.Module):
    """Pre-norm residual block: mixer + feed-forward, shared across all 3 models."""
    def __init__(self, mixer, d_model, d_ff):
        super().__init__()
        self.mixer = mixer
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(nn.Linear(d_model, d_ff), nn.GELU(), nn.Linear(d_ff, d_model))

    def forward(self, x, mask=None):
        x = x + self.mixer(self.ln1(x), mask)
        x = x + self.ff(self.ln2(x))
        return x


class SeqClassifier(nn.Module):
    def __init__(self, vocab_size, mixer_type, d_model=64, n_heads=4, d_ff=128,
                 n_layers=2, n_classes=2, max_len=8192):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.pos = PositionalEncoding(d_model, max_len)
        d_hidden_glu = int(d_ff * 0.75)  # roughly param-match vs attention mixers

        def make_mixer():
            if mixer_type == "glu":
                return GLUMixer(d_model, d_hidden_glu)
            elif mixer_type == "linear_attn":
                return LinearAttentionMixer(d_model, n_heads)
            elif mixer_type == "softmax_attn":
                return SoftmaxAttentionMixer(d_model, n_heads)
            elif mixer_type == "gated_fusion":
                return GatedLocalLinearMixer(d_model, n_heads, window=8)
            raise ValueError(mixer_type)

        self.blocks = nn.ModuleList([Block(make_mixer(), d_model, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, n_classes)
        self.mixer_type = mixer_type

    def forward(self, ids):
        mask = (ids != 0).float()
        x = self.embed(ids)
        x = self.pos(x)
        for blk in self.blocks:
            x = blk(x, mask)
        x = self.ln_f(x)
        # masked mean pool
        m = mask.unsqueeze(-1)
        pooled = (x * m).sum(1) / m.sum(1).clamp(min=1.0)
        return self.head(pooled)


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    for mt in ["glu", "linear_attn", "softmax_attn"]:
        m = SeqClassifier(10000, mt)
        print(mt, count_params(m))
        x = torch.randint(1, 9999, (4, 50))
        print(m(x).shape)
