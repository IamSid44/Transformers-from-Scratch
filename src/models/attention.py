"""Attention and the encoder-decoder transformer assembled from it.

`nn.MultiheadAttention`, `nn.Transformer` and `F.scaled_dot_product_attention` are not used
anywhere.

Contents: scaled dot-product attention, MHA, GQA (C3 ablates MHA vs GQA), the position-wise
feed-forward network, the Pre-LN encoder/decoder blocks, and `Seq2SeqTransformer`. The full
model lives here rather than in its own file because the assignment fixes the file list, and
`blt.py` needs these blocks -- putting them in `train.py` instead would make the import cycle
train -> blt -> train.

GQA keeps all `n_heads` query heads but projects only `n_kv_heads` key/value heads, shared
across groups of `n_heads / n_kv_heads` queries -- a smaller KV cache and fewer parameters
for some loss of capacity. MHA is the case n_kv_heads == n_heads.

Mask convention: a boolean tensor broadcastable to (B, H, T_q, T_k) where **True means
attention is allowed**. Blocked logits are filled with the dtype minimum rather than -inf so
a fully-blocked row softmaxes to uniform instead of NaN.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from norm import build_norm
from positional import (RotaryPositionalEmbedding,
                        SinusoidalPositionalEncoding)


def scaled_dot_product_attention(q, k, v, mask=None, dropout=None):
    """softmax(Q K^T / sqrt(d_k)) V.

    q: (B, H, T_q, D), k/v: (B, H, T_k, D). Returns (context, weights).

    The masked_fill and the softmax are done in place: the (B, H, T_q, T_k) score tensor is
    the largest allocation in the model, and rewriting it three times over is what makes a
    long source sequence run out of memory.
    """
    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.size(-1))
    if mask is not None:
        scores.masked_fill_(~mask, torch.finfo(scores.dtype).min)
    weights = torch.softmax(scores, dim=-1)
    if dropout is not None:
        weights = dropout(weights)
    return torch.matmul(weights, v), weights


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads:
            raise ValueError(f"d_model={d_model} not divisible by n_heads={n_heads}")
        self.d_model, self.n_heads = d_model, n_heads
        self.n_kv_heads = n_heads
        self.d_head = d_model // n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x, n_heads):
        """(B, T, n_heads * d_head) -> (B, n_heads, T, d_head)"""
        b, t, _ = x.shape
        return x.view(b, t, n_heads, self.d_head).transpose(1, 2)

    def forward(self, query, key_value=None, mask=None, rope=None):
        """`key_value` defaults to `query` (self-attention). `rope` rotates q and k."""
        kv = query if key_value is None else key_value
        b, t_q, _ = query.shape

        q = self._split(self.w_q(query), self.n_heads)
        k = self._split(self.w_k(kv), self.n_heads)
        v = self._split(self.w_v(kv), self.n_heads)
        if rope is not None:
            q, k = rope(q, k)

        context, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        context = context.transpose(1, 2).reshape(b, t_q, self.d_model)
        return self.w_o(context)


class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_kv_heads: int, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads or n_heads % n_kv_heads:
            raise ValueError(f"bad head config: {d_model}/{n_heads}/{n_kv_heads}")
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_kv_heads
        self.n_groups = n_heads // n_kv_heads
        self.d_head = d_model // n_heads
        # Query is full width; key/value are only n_kv_heads wide -- the whole GQA saving.
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.w_v = nn.Linear(d_model, n_kv_heads * self.d_head)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def _split(self, x, n_heads):
        b, t, _ = x.shape
        return x.view(b, t, n_heads, self.d_head).transpose(1, 2)

    def _expand_kv(self, x):
        """(B, n_kv, T, D) -> (B, n_heads, T, D). repeat_interleave maps query head g to
        KV head g // n_groups, so contiguous query heads share a KV head."""
        return x.repeat_interleave(self.n_groups, dim=1)

    def forward(self, query, key_value=None, mask=None, rope=None):
        kv = query if key_value is None else key_value
        b, t_q, _ = query.shape

        q = self._split(self.w_q(query), self.n_heads)
        k = self._split(self.w_k(kv), self.n_kv_heads)
        v = self._split(self.w_v(kv), self.n_kv_heads)
        if rope is not None:                    # before expansion: rotate each KV head once
            q, k = rope(q, k)
        k, v = self._expand_kv(k), self._expand_kv(v)

        context, _ = scaled_dot_product_attention(q, k, v, mask, self.dropout)
        context = context.transpose(1, 2).reshape(b, t_q, self.d_model)
        return self.w_o(context)


def build_attention(cfg, d_model=None, n_heads=None) -> nn.Module:
    d_model = cfg.d_model if d_model is None else d_model
    n_heads = cfg.n_heads if n_heads is None else n_heads
    if cfg.attention == "mha":
        return MultiHeadAttention(d_model, n_heads, cfg.dropout)
    if cfg.attention == "gqa":
        return GroupedQueryAttention(d_model, n_heads, cfg.n_kv_heads, cfg.dropout)
    raise ValueError(f"Unknown attention {cfg.attention!r}")


def causal_mask(seq_len: int, device=None) -> torch.Tensor:
    """(1, 1, T, T) lower-triangular; True where attention is allowed."""
    return torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()[None, None]


def padding_mask(tokens: torch.Tensor, pad_id: int) -> torch.Tensor:
    """(B, 1, 1, T); True on real tokens, False on padding."""
    return (tokens != pad_id)[:, None, None, :]


class FeedForward(nn.Module):
    """Position-wise: Linear -> GELU -> dropout -> Linear."""

    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    """Pre-LN block: x = x + Sublayer(Norm(x))."""

    def __init__(self, cfg):
        super().__init__()
        self.norm1 = build_norm(cfg.norm, cfg.d_model)
        self.self_attn = build_attention(cfg)
        self.norm2 = build_norm(cfg.norm, cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, src_mask=None, rope=None):
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=src_mask, rope=rope))
        return x + self.dropout(self.ffn(self.norm2(x)))


class DecoderLayer(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.norm1 = build_norm(cfg.norm, cfg.d_model)
        self.self_attn = build_attention(cfg)
        self.norm2 = build_norm(cfg.norm, cfg.d_model)
        self.cross_attn = build_attention(cfg)
        self.norm3 = build_norm(cfg.norm, cfg.d_model)
        self.ffn = FeedForward(cfg.d_model, cfg.d_ff, cfg.dropout)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x, memory, tgt_mask=None, memory_mask=None, rope=None, reshape=None):
        """`reshape`, when given, is a (fold, unfold) pair applied around cross-attention.

        blt.py uses it to run self-attention with patches on the batch axis but cross-attention
        with them on the query axis, so the shared source memory is never copied per patch.
        """
        x = x + self.dropout(self.self_attn(self.norm1(x), mask=tgt_mask, rope=rope))
        if memory is not None:
            fold, unfold = reshape or (lambda t: t, lambda t: t)
            # no rope in cross-attention: query and key index unrelated coordinate systems
            h = self.cross_attn(fold(self.norm2(x)), memory, mask=memory_mask)
            x = x + self.dropout(unfold(h))
        return x + self.dropout(self.ffn(self.norm3(x)))


class Seq2SeqTransformer(nn.Module):
    """Full encoder-decoder stack, Pre-LN throughout, final norm on each stack.

    Two modes:
      * token mode (C1-C4): vocab sizes given, so it owns the embeddings and returns logits.
      * latent mode (C5): both None, so it consumes and returns d_model vectors. blt.py feeds
        it patch representations, which keeps C5's global model literally this same class.

    RoPE is applied in encoder and decoder *self*-attention only, never cross-attention: there
    the query indexes the target and the key indexes the source, two unrelated coordinate
    systems, so a relative offset between them is meaningless.
    """

    def __init__(self, cfg, src_vocab_size=None, tgt_vocab_size=None,
                 pad_id: int = 0, max_len: int = 4096, tie_embeddings: bool = True):
        super().__init__()
        self.cfg = cfg
        self.pad_id = pad_id
        self.d_model = cfg.d_model

        self.src_embed = (nn.Embedding(src_vocab_size, cfg.d_model, padding_idx=pad_id)
                          if src_vocab_size else None)
        if tgt_vocab_size:
            self.tgt_embed = nn.Embedding(tgt_vocab_size, cfg.d_model, padding_idx=pad_id)
            self.output_proj = nn.Linear(cfg.d_model, tgt_vocab_size, bias=False)
            if tie_embeddings:
                self.output_proj.weight = self.tgt_embed.weight
        else:
            self.tgt_embed = self.output_proj = None

        # Exactly one positional mechanism is active -- the C1-vs-C2 switch.
        self.use_rope = cfg.pos_encoding == "rope"
        self.rope = RotaryPositionalEmbedding(cfg.d_head, max_len) if self.use_rope else None
        self.pos_enc = (None if self.use_rope
                        else SinusoidalPositionalEncoding(cfg.d_model, max_len, cfg.dropout))

        self.encoder_layers = nn.ModuleList(EncoderLayer(cfg)
                                            for _ in range(cfg.n_encoder_layers))
        self.decoder_layers = nn.ModuleList(DecoderLayer(cfg)
                                            for _ in range(cfg.n_decoder_layers))
        self.encoder_norm = build_norm(cfg.norm, cfg.d_model)
        self.decoder_norm = build_norm(cfg.norm, cfg.d_model)
        self.dropout = nn.Dropout(cfg.dropout)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].fill_(0)

    def _prepare(self, x, embed, offset: int = 0):
        """Ids -> embeddings (or pass latent vectors through), then absolute PE if active."""
        if embed is not None:
            x = embed(x) * math.sqrt(self.d_model)   # keeps magnitude comparable to the PE
        return self.pos_enc(x, offset=offset) if self.pos_enc is not None else self.dropout(x)

    def encode(self, src, src_mask=None):
        """src: (B, S) ids in token mode, (B, S, d_model) vectors in latent mode."""
        if src_mask is None and self.src_embed is not None:
            src_mask = padding_mask(src, self.pad_id)
        x = self._prepare(src, self.src_embed)
        for layer in self.encoder_layers:
            x = layer(x, src_mask=src_mask, rope=self.rope)
        return self.encoder_norm(x)

    def decode(self, tgt_in, memory, tgt_mask=None, memory_mask=None):
        """Returns hidden states (B, T, d_model)."""
        x = self._prepare(tgt_in, self.tgt_embed)
        for layer in self.decoder_layers:
            x = layer(x, memory, tgt_mask=tgt_mask, memory_mask=memory_mask, rope=self.rope)
        return self.decoder_norm(x)

    def build_tgt_mask(self, tgt_in):
        """Causal, plus blocking padded keys when ids are available."""
        mask = causal_mask(tgt_in.size(1), tgt_in.device)
        if self.tgt_embed is not None and tgt_in.dim() == 2:
            mask = mask & padding_mask(tgt_in, self.pad_id)
        return mask

    def forward(self, src, tgt_in, src_mask=None, tgt_mask=None):
        """Teacher-forced. Logits in token mode, hidden states in latent mode."""
        if src_mask is None and self.src_embed is not None:
            src_mask = padding_mask(src, self.pad_id)
        if tgt_mask is None:
            tgt_mask = self.build_tgt_mask(tgt_in)
        memory = self.encode(src, src_mask)
        hidden = self.decode(tgt_in, memory, tgt_mask=tgt_mask, memory_mask=src_mask)
        return self.output_proj(hidden) if self.output_proj is not None else hidden

    @torch.no_grad()
    def greedy_decode(self, src, max_len: int, bos_id: int, eos_id: int, src_mask=None):
        """Greedy decoding (mandated by the assignment). Returns (B, L) ids starting at <bos>;
        rows are padded after their <eos>. No KV cache -- the prefix is re-run each step."""
        assert self.output_proj is not None, "greedy_decode requires token mode"
        self.eval()
        if src_mask is None:
            src_mask = padding_mask(src, self.pad_id)
        memory = self.encode(src, src_mask)

        ids = torch.full((src.size(0),), bos_id, dtype=torch.long, device=src.device)[:, None]
        finished = torch.zeros(src.size(0), dtype=torch.bool, device=src.device)
        for _ in range(max_len - 1):
            hidden = self.decode(ids, memory, tgt_mask=causal_mask(ids.size(1), src.device),
                                 memory_mask=src_mask)
            nxt = self.output_proj(hidden[:, -1]).argmax(-1)
            nxt = torch.where(finished, torch.full_like(nxt, self.pad_id), nxt)
            ids = torch.cat([ids, nxt[:, None]], dim=1)
            finished = finished | (nxt == eos_id)
            if bool(finished.all()):
                break
        return ids


if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)
    B, T, D, H = 2, 12, 64, 8

    q, k, v = (torch.randn(B, H, T, D // H) for _ in range(3))
    ctx, w = scaled_dot_product_attention(q, k, v)
    assert ctx.shape == (B, H, T, D // H) and w.shape == (B, H, T, T)
    assert (w.sum(-1) - 1).abs().max() < 1e-4, "attention rows must be a distribution"

    _, wc = scaled_dot_product_attention(q, k, v, causal_mask(T))
    assert wc[..., torch.triu(torch.ones(T, T, dtype=torch.bool), 1)].abs().max() < 1e-6
    print("causal mask blocks every future position")

    pm = padding_mask(torch.tensor([[5, 6, 7, 0, 0], [5, 0, 0, 0, 0]]), pad_id=0)
    _, wp = scaled_dot_product_attention(*(torch.randn(2, H, 5, D // H) for _ in range(3)), pm)
    assert wp[0, :, :, 3:].abs().max() < 1e-6 and wp[1, :, :, 1:].abs().max() < 1e-6
    print("padding mask blocks every pad key")

    x, mem = torch.randn(B, T, D), torch.randn(B, 19, D)
    mha = MultiHeadAttention(D, H, dropout=0.0).eval()
    out, out_x = mha(x), mha(x, mem)
    assert out.shape == (B, T, D) and out_x.shape == (B, T, D)
    print(f"MHA: self {tuple(out.shape)}, cross {tuple(out_x.shape)}")

    gqa = GroupedQueryAttention(D, H, n_kv_heads=2, dropout=0.0).eval()
    kv = torch.zeros(1, 2, 3, D // H)
    kv[:, 1] = 1.0
    exp = gqa._expand_kv(kv)
    assert (exp[:, :4] == 0).all() and (exp[:, 4:] == 1).all()
    print("GQA grouping: query heads 0-3 -> KV head 0, 4-7 -> KV head 1")

    gqa_full = GroupedQueryAttention(D, H, n_kv_heads=H, dropout=0.0).eval()
    gqa_full.load_state_dict(mha.state_dict())
    assert torch.allclose(gqa_full(x), out, atol=1e-6)
    print("GQA with n_kv_heads == n_heads reduces exactly to MHA")

    def cfg(**over):
        base = dict(d_model=64, n_heads=8, n_kv_heads=2, d_head=8, d_ff=256, dropout=0.0,
                    attention="mha", norm="layernorm", pos_encoding="sinusoidal",
                    n_encoder_layers=2, n_decoder_layers=2)
        return SimpleNamespace(**{**base, **over})

    SRC_V, TGT_V, PAD, BOS, EOS = 64, 48, 0, 2, 3
    for name, over in [("C1", {}), ("C2", {"pos_encoding": "rope"}),
                       ("C3", {"attention": "gqa"}), ("C4", {"norm": "rmsnorm"})]:
        model = Seq2SeqTransformer(cfg(**over), SRC_V, TGT_V, pad_id=PAD, max_len=256).eval()
        src = torch.randint(4, SRC_V, (3, 17)); src[0, -4:] = PAD
        tgt = torch.randint(4, TGT_V, (3, 11)); tgt[:, 0] = BOS; tgt[1, -3:] = PAD

        logits = model(src, tgt)
        assert logits.shape == (3, 11, TGT_V)
        assert model.output_proj.weight is model.tgt_embed.weight, "embeddings must be tied"

        alt = tgt.clone(); alt[:, 6] = (alt[:, 6] + 7) % TGT_V
        assert torch.allclose(logits[:, :6], model(src, alt)[:, :6], atol=1e-5), \
            f"{name} leaked future information"

        out = model.greedy_decode(src, max_len=9, bos_id=BOS, eos_id=EOS)
        assert out.shape[0] == 3 and out.shape[1] <= 9 and (out[:, 0] == BOS).all()
        print(f"{name}: logits {tuple(logits.shape)}, causal, tied, greedy ok")

    assert Seq2SeqTransformer(cfg(pos_encoding="rope"), SRC_V, TGT_V).rope is not None
    assert Seq2SeqTransformer(cfg(), SRC_V, TGT_V).pos_enc is not None
    print("positional switch verified (RoPE vs sinusoidal)")

    latent = Seq2SeqTransformer(cfg(), None, None, max_len=256).eval()
    sv, tv = torch.randn(3, 9, 64), torch.randn(3, 5, 64)
    hidden = latent(sv, tv)
    assert hidden.shape == (3, 5, 64)
    tv2 = tv.clone(); tv2[:, 3] += 1.0
    assert torch.allclose(hidden[:, :3], latent(sv, tv2)[:, :3], atol=1e-5)
    print(f"latent mode: {tuple(hidden.shape)}, causality verified")
    print("attention.py self-test passed")


