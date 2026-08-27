"""Normalization layers (C4 ablates LayerNorm vs RMSNorm). `nn.LayerNorm` is not used."""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """y = (x - mean) / sqrt(var + eps) * gamma + beta"""

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(-1, keepdim=True)
        var = x.var(-1, keepdim=True, unbiased=False)
        return (x - mean) * torch.rsqrt(var + self.eps) * self.weight + self.bias


class RMSNorm(nn.Module):
    """y = x / sqrt(mean(x^2) + eps) * gamma -- rescaling only, no centring, no bias."""

    def __init__(self, d_model: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * rms * self.weight


def build_norm(kind: str, d_model: int) -> nn.Module:
    if kind == "layernorm":
        return LayerNorm(d_model)
    if kind == "rmsnorm":
        return RMSNorm(d_model)
    raise ValueError(f"Unknown norm {kind!r}")


if __name__ == "__main__":
    torch.manual_seed(0)
    x = torch.randn(4, 7, 512) * 3.0 + 5.0

    ln, rms = LayerNorm(512), RMSNorm(512)
    out, out_r = ln(x), rms(x)

    ref = nn.functional.layer_norm(x, (512,), ln.weight, ln.bias, eps=1e-5)
    assert torch.allclose(out, ref, atol=1e-5)
    print(f"LayerNorm matches F.layer_norm (max diff {(out - ref).abs().max():.2e})")

    assert (out_r.pow(2).mean(-1) - 1).abs().max() < 1e-3
    assert out_r.mean(-1).abs().max() > 0.1, "RMSNorm must not re-centre"
    assert torch.allclose(rms(x * 7.0), out_r, atol=1e-4), "RMSNorm must be scale invariant"
    print("RMSNorm: unit RMS, mean preserved, scale invariant")
    print(f"params: LayerNorm={sum(p.numel() for p in ln.parameters())} "
          f"RMSNorm={sum(p.numel() for p in rms.parameters())}")
    print("norm.py self-test passed")
