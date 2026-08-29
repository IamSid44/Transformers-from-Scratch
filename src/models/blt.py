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

Alignment: the two patch grids cover the same span of text (SRC_PATCH_SIZE = 8 *
TGT_PATCH_SIZE, since the corpus spends 8 cipher characters per plaintext character), so
source patch k and target patch k describe the same characters and receive the same sinusoidal
position in the global transformer. See BLT.md for why that matters and what happened when
they were chosen independently.
"""

from __future__ import annotations

import math
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

# Pooling strides. A patch is the unit the global transformer reasons over, so the two grids
# have to cover the *same* span of text: the corpus encodes every plaintext character as
# exactly BITS_PER_CHAR cipher characters, so one source patch must be that many times longer
# than the target patch it corresponds to. With the grids aligned, source patch k and target
# patch k hold the same characters, both get the same sinusoidal position in the global
# transformer, and cross-attention has a diagonal to find.
#
# They were previously chosen independently (16 cipher bits = 2 characters against 8 plaintext
# bytes = 8 characters), which left target character (patch n, slot j) depending on source
# patch 4n + j//2 -- and on one half of a patch whose pooling had already averaged its two
# characters together. C5 never found that alignment: it collapsed onto modelling English
# unconditionally and scored 1.9525 nats/byte, against 1.9721 for a source-blind character
# n-gram with the same receptive field. See BLT.md.
BITS_PER_CHAR = 8
TGT_PATCH_SIZE = 4                                 # plaintext characters per target patch
SRC_PATCH_SIZE = BITS_PER_CHAR * TGT_PATCH_SIZE    # the same 4 characters, as 32 cipher bits


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
        self.d_local = cfg.d_local
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
        # sqrt(d_local) before the positional table, the same rule Seq2SeqTransformer._prepare
        # applies to the tokenized models' embeddings. A sinusoidal row has norm sqrt(d/2) ~
        # 11.3 by construction while a freshly initialised embedding has norm ~0.55, so without
        # the scaling the byte identity is under 5% of the vector the first layer sees (45% in
        # C1-C4). On the source side that is fatal rather than merely slow: a cipher byte is
        # '0' or '1', so its entire content is one bit, and the measured content share of the
        # trained src_encoder's output variance was 0.01%.
        x = self.pos(self.embed(byte_ids) * math.sqrt(self.d_local))
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

    Three details matter, all found empirically:

    Latent expansion. h_t is not used as a single vector -- one vector describing a whole patch
    starves the decoder, which then falls back on within-patch English statistics: enough to
    score well under teacher forcing, but it collapses into repetition at inference. h_t is
    instead expanded into `patch_size` conditioning vectors, one per byte slot. Causality is
    safe: every slot is a function of h_t alone, and h_t came from patches 0..t-1.

    Direct source access. The decoder also cross-attends to the global *encoder* memory.
    Without it, teacher forcing lets it see the preceding true bytes of its own patch, which
    is enough to emit plausible English unaided, so the global path gets little gradient and
    h_t collapses into a positional code. The memory depends only on the source, so it cannot
    leak a target byte.

    Absolute byte position. The cross-attention query has to name a source patch. A learned
    within-patch slot code tells it *j* but not which patch it sits in, so its only handle on
    the patch index was h_t itself -- and h_t only carries a usable index once the query
    already works, a deadlock the first run never escaped. The query therefore carries the
    sinusoidal absolute position of the byte it is emitting, (patch_offset + i) * P + j, which
    is the same "Sinusoidal Absolute" encoding C1-C4 put on their decoder positions.

    All patches are decoded in parallel: they fold into the batch axis for self-attention and
    into the query axis for cross-attention, so the source memory is never copied per patch.
    """

    def __init__(self, cfg, patch_size: int, byte_embed: ByteEmbedding, max_len: int = 4096):
        super().__init__()
        self.patch_size = patch_size
        self.d_local = cfg.d_local
        lcfg = _local_cfg(cfg)
        self.byte_embed = byte_embed                  # shared with the target local encoder
        self.latent_expand = nn.Linear(cfg.d_model, patch_size * cfg.d_local)
        self.memory_proj = nn.Linear(cfg.d_model, cfg.d_local)
        self.pos = SinusoidalPositionalEncoding(cfg.d_local, max_len, cfg.dropout)
        self.layers = nn.ModuleList(DecoderLayer(lcfg)
                                    for _ in range(cfg.n_local_decoder_layers))
        self.norm = build_norm(cfg.norm, cfg.d_local)
        self.out = nn.Linear(cfg.d_local, BYTE_VOCAB_SIZE)

    def forward(self, latents, prev_bytes, memory=None, memory_mask=None, patch_offset: int = 0):
        """latents: (B, N, d_model); prev_bytes: (B, N, P) shifted inputs. `patch_offset` is
        the index of the first patch in `latents` within the full target, so incremental
        decoding still gets the right absolute byte positions.
        Returns byte logits (B, N, P, BYTE_VOCAB_SIZE)."""
        b, n, p = prev_bytes.shape
        assert p == self.patch_size

        slots = self.latent_expand(latents).reshape(b * n, p, self.d_local)
        # sqrt(d_local) on the byte embedding for the same reason LocalByteEncoder applies it:
        # the sinusoidal row added just below has norm ~11.3 and would otherwise bury it.
        x = self.byte_embed(prev_bytes.reshape(b * n, p)) * math.sqrt(self.d_local) + slots
        # Sinusoidal absolute position over the *byte* index, not the slot within the patch.
        x = self.pos(x.reshape(b, n * p, self.d_local), offset=patch_offset * p)
        x = x.reshape(b * n, p, self.d_local)

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
        self.local_decoder = LocalByteDecoder(cfg, self.tgt_patch, self.tgt_encoder.embed,
                                              max_len=max_len)

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
        # The validity flags have to be shifted with the stream they describe: slot 0 is the
        # always-valid <bos> patch and slot t (t >= 1) carries patch t-1.
        dec_valid = torch.cat([torch.ones_like(tgt_patch_valid[:, :1]),
                               tgt_patch_valid[:, :-1]], dim=1)
        tgt_mask = causal_mask(n_patches, dec_in.device) & dec_valid[:, None, None, :]
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
                logits = self.local_decoder(h, patch_bytes.unsqueeze(1), memory, memory_mask,
                                            patch_offset=step)
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
    assert SRC_PATCH_SIZE == BITS_PER_CHAR * TGT_PATCH_SIZE, (
        "the two patch grids must cover the same span of text: a target patch is "
        "TGT_PATCH_SIZE characters and the corpus spends BITS_PER_CHAR cipher characters on "
        "each of them, so a source patch has to be BITS_PER_CHAR times longer")
    print(f"patch grids aligned: {SRC_PATCH_SIZE} cipher characters and {TGT_PATCH_SIZE} "
          f"plaintext characters both span {TGT_PATCH_SIZE} characters of text")

    # A toy that mirrors the corpus -- every plaintext character expands into BITS_PER_CHAR
    # cipher characters -- rather than two independent random tensors. "Does the model read
    # its source?" is then a question about learning the map, not about memorising unrelated
    # pairs, which is what let the first C5 run pass this file and still collapse.
    # The overfit step below is the only expensive part of this file; run it on the GPU when
    # there is one, so `python src/models/blt.py` stays a quick check rather than a coffee break.
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    B, CHARS = 6, 16
    key = torch.tensor([ord(c) for c in "ANLP2026"], device=dev).repeat(CHARS // 8)
    plain = torch.randint(97, 123, (B, CHARS), device=dev)
    bits = ((plain ^ key)[..., None]
            >> torch.arange(7, -1, -1, device=dev)) & 1                  # (B, CHARS, 8), MSB first
    src = bits.reshape(B, CHARS * BITS_PER_CHAR) + ord("0")              # '0'/'1' characters
    tgt = plain.clone(); tgt[:, -1] = BYTE_EOS_ID
    # Row 0 ends four characters early, on both sides, so the grids stay aligned through the
    # padding and the key-padding masks get exercised.
    src[0, -4 * BITS_PER_CHAR:] = BYTE_PAD_ID
    tgt[0, -4:] = BYTE_PAD_ID
    tgt[0, -5] = BYTE_EOS_ID
    LS, LT = src.size(1), tgt.size(1)

    model = BLTSeq2Seq(cfg, max_len=1024).to(dev)
    print(f"BLTSeq2Seq: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters "
          f"(src patch {SRC_PATCH_SIZE}, tgt patch {TGT_PATCH_SIZE}) on {dev}")

    logits = model(src, tgt)
    assert logits.shape == (B, LT, BYTE_VOCAB_SIZE)
    memory, _ = model.encode_source(src)
    patches, _ = model._target_patches(tgt)
    assert memory.shape == (B, LS // SRC_PATCH_SIZE, cfg.d_model)
    assert memory.shape[1] == patches.shape[1], \
        "aligned grids must yield one source patch per target patch"
    print(f"forward {tuple(logits.shape)}; pooling {LS} src bytes -> {memory.shape[1]} patches, "
          f"{LT} tgt bytes -> {patches.shape[1]} patches (one per source patch)")

    shifted = model._shift_for_local_decoder(tgt)
    assert (shifted[..., 0] == BYTE_PATCH_START_ID).all()
    assert torch.equal(shifted[..., 1:], tgt.reshape(B, -1, TGT_PATCH_SIZE)[..., :-1])

    model.eval()
    with torch.no_grad():
        base = model(src, tgt)
        for k in (2, 4, 5, 9):
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
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(300):
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
        real = model(src, tgt).argmax(-1)
        wrong = model(torch.roll(src, 1, 0), tgt).argmax(-1)
    acc = (real[keep] == tgt[keep]).float().mean().item()
    acc_bad = (wrong[keep] == tgt[keep]).float().mean().item()
    assert acc > 0.9, "BLT stack failed to fit a single batch"
    print(f"teacher-forced accuracy {100 * acc:.1f}%, mismatched sources {100 * acc_bad:.1f}% "
          f"(drop {100 * (acc - acc_bad):.1f} points)")

    # The aggregate drop above is informational only, and on a batch this small it is close to
    # meaningless: with six examples the target's own prefix identifies which example it is, so
    # a source-blind model still recalls most of the rest. Byte 0 is the position that cannot
    # be faked. Its decoder input is <patch-start>, its global input is the <bos> patch, and
    # both are constants -- so its prediction is a pure function of the source. Getting it
    # right with the real source and wrong with somebody else's is exactly the claim.
    with torch.no_grad():
        first_ok = (real[:, 0] == tgt[:, 0]).float().mean().item()
        first_bad = (wrong[:, 0] == tgt[:, 0]).float().mean().item()
    print(f"byte 0 (no target context, source-only): {100 * first_ok:.0f}% correct with its own "
          f"source, {100 * first_bad:.0f}% with a mismatched one")
    assert first_ok > 0.99, "byte 0 is not being predicted from the source at all"
    assert first_bad < 0.5, "byte 0 does not depend on which source it was given"

    # The honest version of this check needs a set too large to memorise, which is what
    # `train.source_dependence` runs on the real validation split every epoch. Passing here is
    # necessary and nowhere near sufficient -- the collapsed C5 run passed the old form of this
    # assertion while scoring +0.25 points on real data.
    print("blt.py self-test passed")
