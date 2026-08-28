"""Byte Latent Transformer (C5) -- the token-free pathway.

    source bytes --LocalByteEncoder--> byte states --PatchPooler--> source patches
    target bytes --LocalByteEncoder--> byte states --PatchPooler--> target patches
                          GlobalTransformer (Seq2SeqTransformer in latent mode)
                          patch latents --LocalByteDecoder--> byte logits

The global transformer is the same class as C1 with the same depth, width, sinusoidal
encoding, MHA and LayerNorm, so only the representation layer differs: a learned vocabulary is
replaced by learned pooling over raw bytes.

Simplification vs Meta's BLT paper: patching is fixed-stride rather than entropy-driven, which
the assignment explicitly permits ("a simplified BLT").

Causality: the target-side local encoder is causal. Its output for patch t is pooled into what
the global decoder consumes at step t, and step t predicts patch t+1 -- so a byte in patch t
attending into patch t+1 would be reading its own answer.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import (DecoderLayer, EncoderLayer, MultiHeadAttention,
                       Seq2SeqTransformer, causal_mask)
from norm import build_norm
from positional import SinusoidalPositionalEncoding

# Byte "vocabulary": the 256 real byte values plus three control ids. Control ids sit above
# 255 so a raw byte's id is simply its own numeric value -- no offset arithmetic anywhere.
BYTE_PAD_ID, BYTE_EOS_ID, BYTE_PATCH_START_ID = 256, 257, 258
BYTE_VOCAB_SIZE = 259

# Pooling strides: 16 cipher bits (2 characters) per source patch, 8 plaintext bytes per
# target patch. dataset.py pads byte sequences to whole multiples of these.
SRC_PATCH_SIZE = 16
TGT_PATCH_SIZE = 8


@dataclass
class LocalConfig:
    """The narrow config the byte-level blocks run on; norm and dropout are inherited."""

    d_model: int
    n_heads: int
    n_kv_heads: int
    d_ff: int
    dropout: float
    norm: str
    attention: str = "mha"              # local blocks are not part of the C3 ablation
    pos_encoding: str = "sinusoidal"


def _local_cfg(cfg) -> LocalConfig:
    return LocalConfig(d_model=cfg.d_local, n_heads=cfg.local_n_heads,
                       n_kv_heads=cfg.local_n_heads, d_ff=4 * cfg.d_local,
                       dropout=cfg.dropout, norm=cfg.norm)


class ByteEmbedding(nn.Module):
    """Byte lookup plus hashed byte n-gram embeddings.

    A single byte carries almost nothing ('0'/'1' on the source side), so each position is
    augmented with embeddings of the n-grams *ending* there. Hashing into fixed buckets makes
    260^4 possible 4-grams cost only `ngram_buckets` rows. Windows look strictly backwards, so
    this stays usable in the causal target encoder.
    """

    def __init__(self, d_local: int, ngram_sizes=(3, 4), ngram_buckets: int = 8192):
        super().__init__()
        self.ngram_sizes = tuple(ngram_sizes)
        self.ngram_buckets = ngram_buckets
        self.byte_embed = nn.Embedding(BYTE_VOCAB_SIZE, d_local, padding_idx=BYTE_PAD_ID)
        self.ngram_embed = nn.ModuleList(nn.Embedding(ngram_buckets, d_local)
                                         for _ in self.ngram_sizes)
        for emb in [self.byte_embed, *self.ngram_embed]:
            nn.init.normal_(emb.weight, std=0.02)
        with torch.no_grad():
            self.byte_embed.weight[BYTE_PAD_ID].fill_(0)

    def _hash_ngrams(self, x: torch.Tensor, n: int) -> torch.Tensor:
        """Polynomial rolling hash of the length-n window ending at each position. For n <= 4
        the pre-modulo value stays well below int64 overflow."""
        padded = F.pad(x, (n - 1, 0), value=BYTE_PAD_ID)
        h = torch.zeros_like(x)
        for j in range(n):
            h = h * BYTE_VOCAB_SIZE + padded[:, j:j + x.size(1)]
        return h % self.ngram_buckets

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L) byte ids -> (B, L, d_local)."""
        out = self.byte_embed(x)
        for n, emb in zip(self.ngram_sizes, self.ngram_embed):
            out = out + emb(self._hash_ngrams(x, n))
        return out


def _to_blocks(x: torch.Tensor, window: int):
    """(B, L, d) -> (B * n_blocks, window, d), right-padding L to a multiple of `window`.
    Independent blocks keep the local modules affordable: the cross-block quadrants of the
    attention matrix are never materialised."""
    b, length, d = x.shape
    pad = (-length) % window
    if pad:
        x = F.pad(x, (0, 0, 0, pad))
    n_blocks = x.size(1) // window
    return x.reshape(b * n_blocks, window, d), n_blocks, length


def _from_blocks(x: torch.Tensor, batch: int, n_blocks: int, length: int) -> torch.Tensor:
    return x.reshape(batch, n_blocks * x.size(1), x.size(-1))[:, :length]


def _mask_to_blocks(valid: torch.Tensor, window: int, n_blocks: int) -> torch.Tensor:
    """(B, L) validity -> (B * n_blocks, 1, 1, window) key-padding mask."""
    b, length = valid.shape
    pad = (-length) % window
    if pad:
        valid = F.pad(valid, (0, pad), value=False)
    return valid.reshape(b * n_blocks, window)[:, None, None, :]


class LocalByteEncoder(nn.Module):
    """Shallow, narrow transformer over raw bytes with block-local attention.
    `causal=True` also restricts each byte to the past, which the target side requires."""

    def __init__(self, cfg, causal=False, max_len=4096, ngram_buckets=None):
        super().__init__()
        self.causal = causal
        self.window = cfg.local_attn_window
        lcfg = _local_cfg(cfg)
        self.embed = ByteEmbedding(cfg.d_local, cfg.ngram_sizes,
                                   cfg.ngram_buckets if ngram_buckets is None else ngram_buckets)
        # Absolute position over the whole sequence, added before blocking, so a byte knows
        # where it sits globally even though it only attends locally.
        self.pos = SinusoidalPositionalEncoding(cfg.d_local, max_len, cfg.dropout)
        self.layers = nn.ModuleList(EncoderLayer(lcfg)
                                    for _ in range(cfg.n_local_encoder_layers))
        self.norm = build_norm(cfg.norm, cfg.d_local)

    def forward(self, byte_ids: torch.Tensor) -> torch.Tensor:
        """(B, L) byte ids -> (B, L, d_local) contextualised states."""
        batch = byte_ids.size(0)
        x = self.pos(self.embed(byte_ids))
        x, n_blocks, orig_len = _to_blocks(x, self.window)
        mask = _mask_to_blocks(byte_ids != BYTE_PAD_ID, self.window, n_blocks)
        if self.causal:
            mask = mask & causal_mask(self.window, x.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return _from_blocks(self.norm(x), batch, n_blocks, orig_len)


class PatchPooler(nn.Module):
    """Compress a fixed stride of byte states into one patch vector by cross-attention: a
    learned query attends over its patch's bytes, so the model decides which bytes matter. A
    mean-pool residual keeps the output sensible before attention has learned anything."""

    def __init__(self, cfg, patch_size: int):
        super().__init__()
        self.patch_size = patch_size
        self.query = nn.Parameter(torch.randn(1, 1, cfg.d_local) * 0.02)
        self.attn = MultiHeadAttention(cfg.d_local, cfg.local_n_heads, cfg.dropout)
        self.norm = build_norm(cfg.norm, cfg.d_local)
        self.proj = nn.Linear(cfg.d_local, cfg.d_model)

    def forward(self, byte_states: torch.Tensor, valid: torch.Tensor):
        """byte_states: (B, L, d_local), valid: (B, L). Returns (patches, patch_valid)."""
        b, length, d = byte_states.shape
        assert length % self.patch_size == 0, f"{length} is not a multiple of {self.patch_size}"
        n = length // self.patch_size

        # Fold the patch axis into the batch axis: every patch pools independently.
        flat = byte_states.reshape(b * n, self.patch_size, d)
        flat_valid = valid.reshape(b * n, self.patch_size)

        pooled = self.attn(self.query.expand(b * n, 1, d), flat,
                           mask=flat_valid[:, None, None, :])
        # Mean over valid bytes only, so trailing padding does not dilute the vector.
        mean = ((flat * flat_valid.unsqueeze(-1)).sum(1)
                / flat_valid.sum(-1, keepdim=True).clamp(min=1))
        patches = self.proj(self.norm(pooled.squeeze(1) + mean)).reshape(b, n, -1)
        return patches, flat_valid.any(-1).reshape(b, n)


class LocalByteDecoder(nn.Module):
    """Generates one patch's bytes autoregressively from that patch's global latent h_t.

    Two details matter, both found empirically:

    Latent expansion. h_t is not used as a single vector -- one vector describing 8 characters
    starves the decoder, which then falls back on within-patch English statistics: enough to
    score well under teacher forcing, but it collapses into repetition at inference. h_t is
    instead expanded into `patch_size` conditioning vectors, one per byte slot. Causality is
    safe: every slot is a function of h_t alone, and h_t came from patches 0..t-1.

    Direct source access. The decoder also cross-attends to the global *encoder* memory.
    Without it, teacher forcing lets it see the 7 preceding true bytes of its own patch, which
    is enough to emit plausible English unaided, so the global path gets little gradient and
    h_t collapses into a positional code. The memory depends only on the source, so it cannot
    leak a target byte.

    All patches are decoded in parallel: they fold into the batch axis for self-attention and
    into the query axis for cross-attention, so the source memory is never copied per patch.
    """

    def __init__(self, cfg, patch_size: int, byte_embed: ByteEmbedding):
        super().__init__()
        self.patch_size = patch_size
        self.d_local = cfg.d_local
        lcfg = _local_cfg(cfg)
        self.byte_embed = byte_embed                  # shared with the target local encoder
        self.latent_expand = nn.Linear(cfg.d_model, patch_size * cfg.d_local)
        self.memory_proj = nn.Linear(cfg.d_model, cfg.d_local)
        self.slot_pos = nn.Parameter(torch.randn(1, patch_size, cfg.d_local) * 0.02)
        self.layers = nn.ModuleList(DecoderLayer(lcfg)
                                    for _ in range(cfg.n_local_decoder_layers))
        self.norm = build_norm(cfg.norm, cfg.d_local)
        self.out = nn.Linear(cfg.d_local, BYTE_VOCAB_SIZE)

    def forward(self, latents, prev_bytes, memory=None, memory_mask=None):
        """latents: (B, N, d_model); prev_bytes: (B, N, P) shifted inputs.
        Returns byte logits (B, N, P, BYTE_VOCAB_SIZE)."""
        b, n, p = prev_bytes.shape
        assert p == self.patch_size

        slots = self.latent_expand(latents).reshape(b * n, p, self.d_local)
        x = self.byte_embed(prev_bytes.reshape(b * n, p)) + self.slot_pos + slots

        # Self-attention runs per patch, so patches live on the batch axis: (B*N, P, d).
        # Cross-attention runs against the shared source memory, so there the patch axis is
        # folded into the *query* axis instead: (B, N*P, d) against an unexpanded (B, Ns, d).
        # Copying the memory once per patch is what makes a long line run out of memory --
        # for the longest batch in this corpus that copy is an ~11 GB tensor per decoder layer.
        mem = None if memory is None else self.memory_proj(memory)
        reshape = (lambda t: t.reshape(b, n * p, self.d_local),
                   lambda t: t.reshape(b * n, p, self.d_local))

        mask = causal_mask(p, x.device)
        for layer in self.layers:
            x = layer(x, mem, tgt_mask=mask, memory_mask=memory_mask, reshape=reshape)
        return self.out(self.norm(x)).reshape(b, n, p, -1)


class BLTSeq2Seq(nn.Module):
    """Local encoder -> patch pooling -> global transformer -> local byte decoder."""

    def __init__(self, cfg, max_len: int = 4096):
        super().__init__()
        self.cfg = cfg
        self.src_patch, self.tgt_patch = SRC_PATCH_SIZE, TGT_PATCH_SIZE

        # Source alphabet is only {'0','1'}, so its n-gram space is tiny -- hence the separate
        # (much smaller) bucket count.
        self.src_encoder = LocalByteEncoder(cfg, causal=False, max_len=max_len,
                                            ngram_buckets=cfg.src_ngram_buckets)
        self.src_pooler = PatchPooler(cfg, self.src_patch)
        self.tgt_encoder = LocalByteEncoder(cfg, causal=True, max_len=max_len)
        self.tgt_pooler = PatchPooler(cfg, self.tgt_patch)

        self.global_model = Seq2SeqTransformer(cfg, None, None, max_len=max_len)
        # Plays the role <bos> plays in the tokenized models: there is no previous patch.
        self.bos_patch = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.local_decoder = LocalByteDecoder(cfg, self.tgt_patch, self.tgt_encoder.embed)

    def encode_source(self, src_bytes):
        """(B, Ls) bytes -> (memory (B, Ns, d_model), memory_mask (B, 1, 1, Ns))."""
        patches, patch_valid = self.src_pooler(self.src_encoder(src_bytes),
                                               src_bytes != BYTE_PAD_ID)
        mask = patch_valid[:, None, None, :]
        return self.global_model.encode(patches, mask), mask

    def _target_patches(self, tgt_bytes):
        return self.tgt_pooler(self.tgt_encoder(tgt_bytes), tgt_bytes != BYTE_PAD_ID)

    def _shift_for_local_decoder(self, tgt_bytes):
        """(B, Lt) -> (B, Nt, P), each patch's bytes shifted right by one. Slot 0 gets the
        PATCH_START marker, so byte j conditions on bytes 0..j-1 and never on itself."""
        b, length = tgt_bytes.shape
        shifted = torch.roll(tgt_bytes.reshape(b, length // self.tgt_patch, self.tgt_patch),
                             shifts=1, dims=-1)
        shifted[..., 0] = BYTE_PATCH_START_ID
        return shifted

    def forward(self, src_bytes, tgt_bytes, ctx_bytes=None):
        """Teacher-forced. Both inputs are padded to a multiple of their patch size; tgt_bytes
        serves as the labels. Returns (B, Lt, BYTE_VOCAB_SIZE), aligned with tgt_bytes.

        `ctx_bytes` (default `tgt_bytes`) is what actually conditions the decoder: it drives
        both the patch pooling that feeds the global decoder and the within-patch shift fed to
        the local decoder. Scheduled sampling (train.py) passes a version of it with some
        positions replaced by the model's own prediction, so the labels stay the true target
        while the conditioning gets a taste of the model's own mistakes -- exactly the two
        places (patch-to-patch and byte-within-patch) training would otherwise never expose to
        anything but ground truth, unlike greedy decoding at evaluation time.
        """
        if ctx_bytes is None:
            ctx_bytes = tgt_bytes
        memory, memory_mask = self.encode_source(src_bytes)
        tgt_patches, tgt_patch_valid = self._target_patches(ctx_bytes)
        b, n_patches, _ = tgt_patches.shape

        # Shift the patch stream right: slot t holds patch t-1, so the decoder output at slot
        # t has seen patches 0..t-1 and is the conditioning needed to produce patch t.
        dec_in = torch.cat([self.bos_patch.expand(b, 1, -1), tgt_patches[:, :-1]], dim=1)
        tgt_mask = causal_mask(n_patches, dec_in.device) & tgt_patch_valid[:, None, None, :]
        latents = self.global_model.decode(dec_in, memory, tgt_mask=tgt_mask,
                                           memory_mask=memory_mask)

        logits = self.local_decoder(latents, self._shift_for_local_decoder(ctx_bytes),
                                    memory, memory_mask)
        return logits.reshape(b, n_patches * self.tgt_patch, -1)

    @torch.no_grad()
    def greedy_decode(self, src_bytes, max_bytes: int):
        """Patch by patch: one global step for the next latent, then the local decoder emits
        tgt_patch bytes. Bytes so far are re-encoded and pooled exactly as in training."""
        self.eval()
        device, b, p = src_bytes.device, src_bytes.size(0), self.tgt_patch
        memory, memory_mask = self.encode_source(src_bytes)

        generated = torch.zeros(b, 0, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for step in range(max(1, -(-max_bytes // p))):
            if step == 0:
                dec_in = self.bos_patch.expand(b, 1, -1)
            else:
                patches, _ = self._target_patches(generated)
                dec_in = torch.cat([self.bos_patch.expand(b, 1, -1), patches], dim=1)
            latents = self.global_model.decode(dec_in, memory,
                                               tgt_mask=causal_mask(dec_in.size(1), device),
                                               memory_mask=memory_mask)
            h = latents[:, -1:, :]

            patch_bytes = torch.full((b, p), BYTE_PATCH_START_ID, dtype=torch.long,
                                     device=device)
            emitted = []
            for j in range(p):
                logits = self.local_decoder(h, patch_bytes.unsqueeze(1), memory, memory_mask)
                nxt = logits[:, 0, j].argmax(-1)
                # Never emit a control symbol other than EOS; substitute a space.
                nxt = torch.where(nxt >= BYTE_PAD_ID,
                                  torch.where(nxt == BYTE_EOS_ID, nxt,
                                              torch.zeros_like(nxt) + 32), nxt)
                nxt = torch.where(finished, torch.full_like(nxt, BYTE_PAD_ID), nxt)
                if j + 1 < p:
                    patch_bytes[:, j + 1] = nxt
                emitted.append(nxt)
                finished = finished | (nxt == BYTE_EOS_ID)

            generated = torch.cat([generated, torch.stack(emitted, dim=1)], dim=1)
            if bool(finished.all()):
                break
        return generated


def bytes_to_text(byte_row) -> str:
    """One decoded byte row -> string, stopping at EOS and dropping control ids."""
    out = bytearray()
    for value in byte_row:
        value = int(value)
        if value in (BYTE_EOS_ID, BYTE_PAD_ID):
            break
        if value < 256:
            out.append(value)
    return out.decode("latin-1", errors="replace")


if __name__ == "__main__":
    from types import SimpleNamespace

    torch.manual_seed(0)
    # A small stand-in for train.ModelConfig, so this file tests without importing train.py.
    cfg = SimpleNamespace(
        d_model=256, n_heads=8, n_kv_heads=2, d_head=32, d_ff=1024, dropout=0.1,
        attention="mha", norm="layernorm", pos_encoding="sinusoidal",
        n_encoder_layers=2, n_decoder_layers=2,
        d_local=128, n_local_encoder_layers=2, n_local_decoder_layers=2,
        local_n_heads=4, local_attn_window=128,
        ngram_sizes=(3, 4), ngram_buckets=8192, src_ngram_buckets=512,
    )                                     # smaller depth than ModelConfig: this is a shape test
    B, LS, LT = 3, 128, 24

    model = BLTSeq2Seq(cfg, max_len=1024)
    print(f"BLTSeq2Seq: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters "
          f"(src patch {SRC_PATCH_SIZE}, tgt patch {TGT_PATCH_SIZE})")

    src = torch.randint(48, 50, (B, LS)); src[0, -32:] = BYTE_PAD_ID
    tgt = torch.randint(97, 123, (B, LT)); tgt[:, -1] = BYTE_EOS_ID; tgt[1, -5:] = BYTE_PAD_ID

    logits = model(src, tgt)
    assert logits.shape == (B, LT, BYTE_VOCAB_SIZE)
    memory, _ = model.encode_source(src)
    patches, _ = model._target_patches(tgt)
    assert memory.shape == (B, LS // SRC_PATCH_SIZE, cfg.d_model)
    print(f"forward {tuple(logits.shape)}; pooling {LS} src bytes -> {memory.shape[1]} patches, "
          f"{LT} tgt bytes -> {patches.shape[1]} patches")

    shifted = model._shift_for_local_decoder(tgt)
    assert (shifted[..., 0] == BYTE_PATCH_START_ID).all()
    assert torch.equal(shifted[..., 1:], tgt.reshape(B, -1, TGT_PATCH_SIZE)[..., :-1])

    model.eval()
    with torch.no_grad():
        base = model(src, tgt)
        for k in (3, 8, 9, 17):
            alt = tgt.clone()
            alt[:, k] = (alt[:, k] - 97 + 5) % 26 + 97
            out = model(src, alt)
            assert torch.allclose(base[:, :k + 1], out[:, :k + 1], atol=1e-4), f"byte {k} leaked"
            assert not torch.allclose(base[:, k + 1:], out[:, k + 1:], atol=1e-4)
    print("causality verified: no target byte influences its own or any earlier logit")

    out = model.greedy_decode(src, max_bytes=LT)
    assert out.shape[0] == B and out.shape[1] <= LT

    with torch.no_grad():
        default = model(src, tgt)
        same = model(src, tgt, ctx_bytes=tgt)
        assert torch.equal(default, same), "ctx_bytes=tgt_bytes must match the no-arg default"
        corrupted = tgt.clone()
        corrupted[:, 0] = (corrupted[:, 0] - 97 + 7) % 26 + 97   # first byte of the first patch
        different = model(src, tgt, ctx_bytes=corrupted)
        assert not torch.allclose(default, different, atol=1e-4), \
            "ctx_bytes did not reach the decoder"
    print("ctx_bytes overrides the decoder's conditioning independently of the labels")

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=3e-4)
    losses = []
    for _ in range(120):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(src, tgt).reshape(-1, BYTE_VOCAB_SIZE),
                               tgt.reshape(-1), ignore_index=BYTE_PAD_ID)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        losses.append(loss.item())
    assert losses[-1] < 0.5 * losses[0], "BLT stack failed to fit a single batch"
    print(f"overfit one batch: loss {losses[0]:.3f} -> {losses[-1]:.3f}")

    model.eval()
    keep = tgt != BYTE_PAD_ID
    with torch.no_grad():
        acc = (model(src, tgt).argmax(-1)[keep] == tgt[keep]).float().mean().item()
        # A seq2seq model that ignores its encoder still scores well under teacher forcing by
        # modelling the target language alone, then collapses into repetition at decode time.
        acc_bad = (model(torch.roll(src, 1, 0), tgt).argmax(-1)[keep] == tgt[keep]).float().mean().item()
    print(f"teacher-forced accuracy {100 * acc:.1f}%, mismatched sources {100 * acc_bad:.1f}% "
          f"(drop {100 * (acc - acc_bad):.1f} points)")
    assert acc > 0.9 and acc - acc_bad > 0.2, "the model is not using the source"
    print("blt.py self-test passed")
