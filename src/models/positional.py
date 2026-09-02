"""Positional information (C2 ablates sinusoidal vs RoPE).

Sinusoidal: a fixed table added to the embeddings once, before the first layer. Encodes
absolute position.

RoPE: no vector is added. Inside every attention block, q and k are rotated by an angle
proportional to their position, so the logit q_m . k_n depends only on (m - n). Positional
information is therefore relative and refreshed at every layer.

Uses the "rotate-half" layout (LLaMA / GPT-NeoX): element i pairs with element i + d/2.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class SinusoidalPositionalEncoding(nn.Module):
    """PE[pos, 2i] = sin(pos / 10000^(2i/d)), PE[pos, 2i+1] = cos(...)."""

    def __init__(self, d_model: int, max_len: int = 4096, dropout: float = 0.0):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                        * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor, offset: int = 0) -> torch.Tensor:
        """x: (B, T, d_model). `offset` supports incremental decoding."""
        t = x.size(1)
        if offset + t > self.pe.size(0):
            raise ValueError(f"Length {offset + t} exceeds max_len={self.pe.size(0)}")
        return self.dropout(x + self.pe[offset:offset + t].unsqueeze(0))

    def at(self, positions: torch.Tensor) -> torch.Tensor:
        """Rows of the table at arbitrary per-element positions. (..., ) long -> (..., d).

        The contiguous `forward` above assumes element i sits at position offset+i, which is
        false once C5's patches have variable width: slot j of patch n is at whatever byte
        index the entropy segmentation put it. This gives those positions the same absolute
        sinusoidal encoding a contiguous run would have received.
        """
        return self.pe[positions]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """(x1, x2) -> (-x2, x1) over the two halves of the last dim."""
    half = x.shape[-1] // 2
    return torch.cat((-x[..., half:], x[..., :half]), dim=-1)


class RotaryPositionalEmbedding(nn.Module):
    """Builds cos/sin tables and rotates q and k (never v)."""

    def __init__(self, d_head: int, max_len: int = 4096, base: float = 10000.0):
        super().__init__()
        if d_head % 2:
            raise ValueError(f"RoPE needs an even head dim, got {d_head}")
        inv_freq = 1.0 / (base ** (torch.arange(0, d_head, 2, dtype=torch.float32) / d_head))
        freqs = torch.outer(torch.arange(max_len, dtype=torch.float32), inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos(), persistent=False)
        self.register_buffer("sin_cached", emb.sin(), persistent=False)

    def get_tables(self, seq_len: int, offset: int = 0, device=None, dtype=None):
        """cos/sin shaped (1, 1, seq_len, d_head), ready to broadcast over (B, H, T, D)."""
        if offset + seq_len > self.cos_cached.size(0):
            raise ValueError(f"Length {offset + seq_len} exceeds the RoPE table")
        cos = self.cos_cached[offset:offset + seq_len]
        sin = self.sin_cached[offset:offset + seq_len]
        if device is not None:
            cos, sin = cos.to(device), sin.to(device)
        if dtype is not None:
            cos, sin = cos.to(dtype), sin.to(dtype)
        return cos[None, None], sin[None, None]

    def forward(self, q: torch.Tensor, k: torch.Tensor, offset: int = 0):
        """q: (B, Hq, Tq, D), k: (B, Hkv, Tk, D). Head counts may differ (GQA)."""
        cos_q, sin_q = self.get_tables(q.size(-2), offset, q.device, q.dtype)
        cos_k, sin_k = self.get_tables(k.size(-2), 0, k.device, k.dtype)
        return q * cos_q + rotate_half(q) * sin_q, k * cos_k + rotate_half(k) * sin_k


if __name__ == "__main__":
    torch.manual_seed(0)

    pe = SinusoidalPositionalEncoding(d_model=64, max_len=512)
    out = pe(torch.zeros(2, 10, 64))
    assert torch.allclose(out[0, 0, 0::2], torch.zeros(32), atol=1e-6)   # sin(0) = 0
    assert torch.allclose(out[0, 0, 1::2], torch.ones(32), atol=1e-6)    # cos(0) = 1
    assert (pe.pe[:100].norm(dim=-1) - math.sqrt(32)).abs().max() < 1e-4
    assert torch.allclose(pe(torch.zeros(1, 1, 64), offset=5)[0, 0], pe.pe[5])
    print("SinusoidalPositionalEncoding: value / norm / offset checks passed")

    rope = RotaryPositionalEmbedding(d_head=32, max_len=512)
    q, k = torch.randn(2, 4, 16, 32), torch.randn(2, 4, 16, 32)
    q_rot, _ = rope(q, k)
    assert (q_rot.norm(dim=-1) - q.norm(dim=-1)).abs().max() < 1e-4, "RoPE must preserve norms"

    v1, v2 = torch.randn(1, 1, 1, 32), torch.randn(1, 1, 1, 32)

    def logit(m: int, n: int) -> float:
        cq, sq = rope.get_tables(1, m)
        ck, sk = rope.get_tables(1, n)
        return ((v1 * cq + rotate_half(v1) * sq) * (v2 * ck + rotate_half(v2) * sk)).sum().item()

    base = logit(3, 7)
    for shift in (1, 5, 20, 100):
        assert abs(logit(3 + shift, 7 + shift) - base) < 1e-3, "RoPE must be translation invariant"
    assert abs(logit(3, 9) - base) > 1e-4, "RoPE must depend on relative distance"
    print(f"RoPE: norm preserving, q3.k7 == q103.k107 == {base:.6f}, varies with distance")
    print("positional.py self-test passed")
