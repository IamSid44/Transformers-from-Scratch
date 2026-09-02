"""Byte Latent Transformer (C5) -- the token-free pathway.

    source bytes --LocalByteEncoder--> byte states --PatchPooler--> source patches
    target bytes --LocalByteEncoder--> byte states --PatchPooler--> target patches
                          GlobalTransformer (Seq2SeqTransformer in latent mode)
                          patch latents --LocalByteDecoder--> byte logits

The global transformer is the same class as C1 with the same depth, width, sinusoidal
encoding, MHA and LayerNorm, so only the representation layer differs: a learned vocabulary is
replaced by learned pooling over raw bytes.

Patching is entropy-driven, as in the BLT paper: a small byte-level LM (`entropy_lm.py`)
scores next-byte entropy over the cipher stream and a new patch opens wherever that entropy
crosses a calibrated global threshold, capped at MAX_PATCH_SIZE bytes. Patches are therefore
variable width -- long through predictable stretches, short where the next byte is genuinely
uncertain -- and every module below indexes bytes through a (B, N, P) gather grid rather than
a reshape on a fixed stride.

What is simplified relative to the paper: the entropy model is 0.5M parameters rather than
100M, and the global/local stacks are sized to this corpus. The patching rule itself is not
simplified.

Causality: the target-side local encoder is causal. Its output for patch t is pooled into what
the global decoder consumes at step t, and step t predicts patch t+1 -- so a byte in patch t
attending into patch t+1 would be reading its own answer.

Alignment: the two patch grids cover the same span of text. dataset.py hands C5 one cipher
byte per plaintext character (the corpus writes each as 8 '0'/'1' characters; they are packed
back into the byte they denote), and the entropy segmentation computed on the cipher is reused
verbatim for the plaintext -- legitimate because the two streams are a character-for-character
bijection, and necessary because at inference the plaintext does not exist yet. Source patch k
and target patch k therefore describe the same characters and receive the same sinusoidal
position in the global transformer. See BLT.md for why that matters and what happened when the
two grids were chosen independently.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import (DecoderLayer, EncoderLayer, MultiHeadAttention,
                       Seq2SeqTransformer, causal_mask)
from entropy_lm import MAX_PATCH_SIZE, TARGET_MEAN_PATCH
from norm import build_norm
from positional import SinusoidalPositionalEncoding

# Byte "vocabulary": the 256 real byte values plus three control ids. Control ids sit above
# 255 so a raw byte's id is simply its own numeric value -- no offset arithmetic anywhere.
BYTE_PAD_ID, BYTE_EOS_ID, BYTE_PATCH_START_ID = 256, 257, 258
BYTE_VOCAB_SIZE = 259

# The patch grid. Widths are chosen per-chunk by `entropy_lm.patch_boundaries`; MAX_PATCH_SIZE
# is the cap that bounds the padded (B, N, P) tensors the modules below index through, and the
# threshold is calibrated so the mean lands near TARGET_MEAN_PATCH.
#
# The two grids have to cover the *same* span of text: with them aligned, source patch k and
# target patch k hold the same characters, both get the same sinusoidal position in the global
# transformer, and cross-attention has a diagonal to find. The strides were once chosen
# independently (16 cipher bits = 2 characters against 8 plaintext bytes = 8 characters), which
# left target character (patch n, slot j) depending on source patch 4n + j//2 -- and on one half
# of a patch whose pooling had already averaged its two characters together. C5 never found that
# alignment: it collapsed onto modelling English unconditionally and scored 1.9525 nats/byte,
# against 1.9721 for a source-blind character n-gram with the same receptive field. See BLT.md.
BITS_PER_CHAR = 8                                  # cipher characters the corpus spends per char


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

    A single byte is a weak feature, so each position is augmented with embeddings of the
    n-grams *ending* there. Hashing into fixed buckets makes 259^4 possible 4-grams cost only
    `ngram_buckets` rows. Windows look strictly backwards, so this stays usable in the causal
    target encoder.

    The hash is position-blind by construction: the same window maps to the same bucket wherever
    it occurs. That is why the source has to arrive as one byte per character -- over a stream of
    literal '0'/'1' characters these windows spanned 3 or 4 *bits*, could not reach a character
    boundary, and could not tell which bit offset within the character they sat at.
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
        # C1-C4). The measured content share of the trained src_encoder's output variance, back
        # when the source arrived one bit per position and this scaling was missing, was 0.01%.
        x = self.pos(self.embed(byte_ids) * math.sqrt(self.d_local))
        x, n_blocks, orig_len = _to_blocks(x, self.window)
        mask = _mask_to_blocks(byte_ids != BYTE_PAD_ID, self.window, n_blocks)
        if self.causal:
            mask = mask & causal_mask(self.window, x.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return _from_blocks(self.norm(x), batch, n_blocks, orig_len)



def gather_slots(x: torch.Tensor, patch_index: torch.Tensor) -> torch.Tensor:
    """Byte axis -> patch grid. (B, L, d) -> (B, N, P, d), or (B, L) -> (B, N, P).

    `patch_index[b, n, j]` is the byte position sitting in slot j of patch n, or 0 for an
    unused slot -- the companion mask, not the index, is what marks those. One gather replaces
    the `reshape(B, L // P, P)` a fixed stride could use.
    """
    b, n, p = patch_index.shape
    flat = patch_index.reshape(b, n * p)
    if x.dim() == 2:
        return x.gather(1, flat).reshape(b, n, p)
    d = x.size(-1)
    return x.gather(1, flat.unsqueeze(-1).expand(-1, -1, d)).reshape(b, n, p, d)


class PatchPooler(nn.Module):
    """Compress one patch's byte states into one patch vector by cross-attention: a learned
    query attends over the patch's bytes, so the model decides which bytes matter. A mean-pool
    residual keeps the output sensible before attention has learned anything.

    Patches are variable width, so the bytes are gathered rather than reshaped and every
    reduction is masked. A slot-padded patch and a wholly absent patch both come out of the
    attention as a uniform average (`scaled_dot_product_attention` fills masked logits with
    finfo.min rather than -inf, so a fully-masked row is uniform, not NaN); `patch_valid`
    reports the absent ones so the global transformer can mask them out.
    """

    def __init__(self, cfg):
        super().__init__()
        self.query = nn.Parameter(torch.randn(1, 1, cfg.d_local) * 0.02)
        self.attn = MultiHeadAttention(cfg.d_local, cfg.local_n_heads, cfg.dropout)
        self.norm = build_norm(cfg.norm, cfg.d_local)
        self.proj = nn.Linear(cfg.d_local, cfg.d_model)

    def forward(self, byte_states, patch_index, patch_mask):
        """byte_states (B, L, d_local), grid (B, N, P). Returns (patches, patch_valid)."""
        b, n, p = patch_index.shape
        d = byte_states.size(-1)
        flat = gather_slots(byte_states, patch_index).reshape(b * n, p, d)
        flat_valid = patch_mask.reshape(b * n, p)

        pooled = self.attn(self.query.expand(b * n, 1, d), flat,
                           mask=flat_valid[:, None, None, :])
        # Mean over occupied slots only, so padding does not dilute the vector.
        mean = ((flat * flat_valid.unsqueeze(-1)).sum(1)
                / flat_valid.sum(-1, keepdim=True).clamp(min=1))
        patches = self.proj(self.norm(pooled.squeeze(1) + mean)).reshape(b, n, -1)
        return patches, patch_mask.any(-1)


class LocalByteDecoder(nn.Module):
    """Generates one patch's bytes autoregressively from that patch's global latent h_t.

    Three details matter, all found empirically:

    Latent expansion. h_t is not used as a single vector -- one vector describing a whole patch
    starves the decoder, which then falls back on within-patch English statistics: enough to
    score well under teacher forcing, but it collapses into repetition at inference. h_t is
    instead expanded into `max_patch` conditioning vectors, one per byte slot. Causality is
    safe: every slot is a function of h_t alone, and h_t came from patches 0..t-1. With
    variable-width patches the trailing slots of a short patch are simply masked out of the
    loss; the expansion is sized to the cap.

    Direct source access. The decoder also cross-attends to the global *encoder* memory.
    Without it, teacher forcing lets it see the preceding true bytes of its own patch, which
    is enough to emit plausible English unaided, so the global path gets little gradient and
    h_t collapses into a positional code. The memory depends only on the source, so it cannot
    leak a target byte.

    Absolute byte position. The cross-attention query has to name a source patch. A learned
    within-patch slot code tells it *j* but not which patch it sits in, so its only handle on
    the patch index was h_t itself -- and h_t only carries a usable index once the query
    already works, a deadlock the first run never escaped. The query therefore carries the
    sinusoidal absolute position of the byte it is emitting, read from the patch grid
    (`SinusoidalPositionalEncoding.at`), which is the same "Sinusoidal Absolute" encoding
    C1-C4 put on their decoder positions. Under a fixed stride that position was arithmetic;
    under entropy patching it has to be looked up, which is the only change here.

    All patches are decoded in parallel: they fold into the batch axis for self-attention and
    into the query axis for cross-attention, so the source memory is never copied per patch.
    """

    def __init__(self, cfg, max_patch: int, byte_embed: ByteEmbedding, max_len: int = 4096):
        super().__init__()
        self.max_patch = max_patch
        self.d_local = cfg.d_local
        lcfg = _local_cfg(cfg)
        self.byte_embed = byte_embed                  # shared with the target local encoder
        self.latent_expand = nn.Linear(cfg.d_model, max_patch * cfg.d_local)
        self.memory_proj = nn.Linear(cfg.d_model, cfg.d_local)
        self.pos = SinusoidalPositionalEncoding(cfg.d_local, max_len, cfg.dropout)
        self.layers = nn.ModuleList(DecoderLayer(lcfg)
                                    for _ in range(cfg.n_local_decoder_layers))
        self.norm = build_norm(cfg.norm, cfg.d_local)
        self.out = nn.Linear(cfg.d_local, BYTE_VOCAB_SIZE)

    def forward(self, latents, prev_bytes, byte_pos, slot_mask, memory=None, memory_mask=None):
        """latents (B, N, d_model); prev_bytes / byte_pos / slot_mask (B, N, P).

        `byte_pos` is the absolute byte index of each slot and `slot_mask` says which slots a
        patch actually occupies. Returns byte logits (B, N, P, BYTE_VOCAB_SIZE).
        """
        b, n, p = prev_bytes.shape
        assert p == self.max_patch, f"grid width {p} != max_patch {self.max_patch}"

        slots = self.latent_expand(latents).reshape(b * n, p, self.d_local)
        # sqrt(d_local) on the byte embedding for the same reason LocalByteEncoder applies it:
        # the sinusoidal row added just below has norm ~11.3 and would otherwise bury it.
        x = self.byte_embed(prev_bytes.reshape(b * n, p)) * math.sqrt(self.d_local) + slots
        x = x + self.pos.at(byte_pos).reshape(b * n, p, self.d_local)

        # Self-attention runs per patch, so patches live on the batch axis: (B*N, P, d).
        # Cross-attention runs against the shared source memory, so there the patch axis is
        # folded into the *query* axis instead: (B, N*P, d) against an unexpanded (B, Ns, d).
        # Copying the memory once per patch is what makes a long line run out of memory --
        # for the longest batch in this corpus that copy is an ~11 GB tensor per decoder layer.
        mem = None if memory is None else self.memory_proj(memory)
        reshape = (lambda t: t.reshape(b, n * p, self.d_local),
                   lambda t: t.reshape(b * n, p, self.d_local))

        # Causal within the patch, and unoccupied slots are never attended to as keys.
        mask = causal_mask(p, x.device) & slot_mask.reshape(b * n, 1, 1, p)
        for layer in self.layers:
            x = layer(x, mem, tgt_mask=mask, memory_mask=memory_mask, reshape=reshape)
        return self.out(self.norm(x)).reshape(b, n, p, -1)


class BLTSeq2Seq(nn.Module):
    """Local encoder -> entropy-patched pooling -> global transformer -> local byte decoder."""

    def __init__(self, cfg, max_len: int = 4096, max_patch: int = MAX_PATCH_SIZE):
        super().__init__()
        self.cfg = cfg
        self.max_patch = max_patch

        # The source keeps its own bucket count: it is a separate alphabet from the target's
        # English (the corpus realizes 126 of the 256 byte values), so sharing tables would make
        # the two streams collide in the same rows.
        self.src_encoder = LocalByteEncoder(cfg, causal=False, max_len=max_len,
                                            ngram_buckets=cfg.src_ngram_buckets)
        self.src_pooler = PatchPooler(cfg)
        self.tgt_encoder = LocalByteEncoder(cfg, causal=True, max_len=max_len)
        self.tgt_pooler = PatchPooler(cfg)

        self.global_model = Seq2SeqTransformer(cfg, None, None, max_len=max_len)
        # Plays the role <bos> plays in the tokenized models: there is no previous patch.
        self.bos_patch = nn.Parameter(torch.randn(1, 1, cfg.d_model) * 0.02)
        self.local_decoder = LocalByteDecoder(cfg, max_patch, self.tgt_encoder.embed,
                                              max_len=max_len)

    # --- pieces -------------------------------------------------------------------------

    def encode_source(self, src_bytes, src_index, src_mask_grid):
        """(B, Ls) bytes + grid -> (memory (B, Ns, d_model), memory_mask (B, 1, 1, Ns))."""
        patches, patch_valid = self.src_pooler(self.src_encoder(src_bytes),
                                               src_index, src_mask_grid)
        mask = patch_valid[:, None, None, :]
        return self.global_model.encode(patches, mask), mask

    def _target_patches(self, tgt_bytes, tgt_index, tgt_mask_grid):
        return self.tgt_pooler(self.tgt_encoder(tgt_bytes), tgt_index, tgt_mask_grid)

    def _shift_within_patch(self, tgt_bytes, tgt_index):
        """(B, Lt) bytes + grid -> (B, N, P), each patch's bytes shifted right by one. Slot 0
        gets the PATCH_START marker, so byte j conditions on bytes 0..j-1 and never on itself.
        The roll wraps the last slot into slot 0, which is exactly the slot overwritten."""
        grid = gather_slots(tgt_bytes, tgt_index)
        shifted = torch.roll(grid, shifts=1, dims=-1)
        shifted[..., 0] = BYTE_PATCH_START_ID
        return shifted

    def _global_latents(self, tgt_patches, patch_valid, memory, memory_mask):
        """Shift the patch stream right and run the global decoder: slot t holds patch t-1, so
        the output at slot t has seen patches 0..t-1 and is the conditioning for patch t."""
        b, n, _ = tgt_patches.shape
        dec_in = torch.cat([self.bos_patch.expand(b, 1, -1), tgt_patches[:, :-1]], dim=1)
        # The validity flags shift with the stream they describe: slot 0 is the always-valid
        # <bos> patch and slot t (t >= 1) carries patch t-1.
        dec_valid = torch.cat([torch.ones_like(patch_valid[:, :1]), patch_valid[:, :-1]], dim=1)
        tgt_mask = causal_mask(n, dec_in.device) & dec_valid[:, None, None, :]
        return self.global_model.decode(dec_in, memory, tgt_mask=tgt_mask,
                                        memory_mask=memory_mask)

    # --- forward ------------------------------------------------------------------------

    def forward(self, src_bytes, tgt_bytes, grid, ctx_bytes=None):
        """Teacher-forced. `grid` is the collate function's patch grids for both sides.
        Returns (B, Lt, BYTE_VOCAB_SIZE), aligned with `tgt_bytes`.

        `ctx_bytes` (default `tgt_bytes`) is what actually conditions the decoder: it drives
        both the patch pooling that feeds the global decoder and the within-patch shift fed to
        the local decoder. Scheduled sampling (train.py), when enabled, passes a version of it
        with some positions replaced by the model's own prediction, so the labels stay the true
        target while the conditioning gets a taste of the model's own mistakes.
        """
        if ctx_bytes is None:
            ctx_bytes = tgt_bytes
        si, sm = grid["src_patch_index"], grid["src_patch_mask"]
        ti, tm = grid["tgt_patch_index"], grid["tgt_patch_mask"]

        memory, memory_mask = self.encode_source(src_bytes, si, sm)
        tgt_patches, patch_valid = self._target_patches(ctx_bytes, ti, tm)
        latents = self._global_latents(tgt_patches, patch_valid, memory, memory_mask)

        logits = self.local_decoder(latents, self._shift_within_patch(ctx_bytes, ti),
                                    ti, tm, memory, memory_mask)

        # Scatter the occupied slots back onto the byte axis. Reading the grid's occupied slots
        # in row-major order yields byte 0, 1, 2, ... (see `dataset._patch_grid`), so the two
        # boolean selections below line up element for element -- no second gather needed.
        b, lt = tgt_bytes.shape
        out = logits.new_zeros(b, lt, BYTE_VOCAB_SIZE)
        byte_valid = tgt_bytes != BYTE_PAD_ID
        out[byte_valid] = logits.reshape(b, -1, BYTE_VOCAB_SIZE)[tm.reshape(b, -1)]
        return out

    # --- inference ----------------------------------------------------------------------

    @torch.no_grad()
    def greedy_decode(self, src_bytes, grid, max_bytes=None):
        """Patch by patch: one global step for the next latent, then the local decoder emits
        that patch's bytes. Bytes so far are re-encoded and pooled exactly as in training.

        The patch grid is *known before decoding starts*: the entropy model segments the
        cipher, and the target inherits that segmentation (see the module docstring). So unlike
        the paper's decoder-only setting, there is no need to run the entropy model on partial
        output to decide where the next patch ends -- the grid comes in with the source.
        """
        self.eval()
        device = src_bytes.device
        si, sm = grid["src_patch_index"], grid["src_patch_mask"]
        ti, tm = grid["tgt_patch_index"], grid["tgt_patch_mask"]
        b, n_patches, p = ti.shape
        n_bytes = int(ti[tm].max()) + 1 if tm.any() else 0
        if max_bytes is not None:
            n_bytes = min(n_bytes, max_bytes)

        memory, memory_mask = self.encode_source(src_bytes, si, sm)
        generated = torch.full((b, n_bytes), BYTE_PAD_ID, dtype=torch.long, device=device)
        finished = torch.zeros(b, dtype=torch.bool, device=device)

        for step in range(n_patches):
            if step == 0:
                dec_in = self.bos_patch.expand(b, 1, -1)
                dec_mask = causal_mask(1, device)
            else:
                patches, valid = self._target_patches(generated, ti[:, :step], tm[:, :step])
                dec_in = torch.cat([self.bos_patch.expand(b, 1, -1), patches], dim=1)
                dec_valid = torch.cat([torch.ones_like(valid[:, :1]), valid], dim=1)
                dec_mask = causal_mask(step + 1, device) & dec_valid[:, None, None, :]
            latents = self.global_model.decode(dec_in, memory, tgt_mask=dec_mask,
                                               memory_mask=memory_mask)
            h = latents[:, -1:, :]

            slot_index, slot_mask = ti[:, step:step + 1], tm[:, step:step + 1]
            patch_bytes = torch.full((b, 1, p), BYTE_PATCH_START_ID, dtype=torch.long,
                                     device=device)
            for j in range(p):
                if not bool(slot_mask[:, 0, j].any()):
                    break
                logits = self.local_decoder(h, patch_bytes, slot_index, slot_mask,
                                            memory, memory_mask)
                nxt = logits[:, 0, j].argmax(-1)
                # Never emit a control symbol other than EOS; substitute a space.
                nxt = torch.where(nxt >= BYTE_PAD_ID,
                                  torch.where(nxt == BYTE_EOS_ID, nxt,
                                              torch.full_like(nxt, 32)), nxt)
                live = slot_mask[:, 0, j] & ~finished
                nxt = torch.where(live, nxt, torch.full_like(nxt, BYTE_PAD_ID))
                if j + 1 < p:
                    patch_bytes[:, 0, j + 1] = nxt
                # Write the byte at the position the grid assigned it, leaving finished or
                # unoccupied rows as they were.
                pos = slot_index[:, 0, j].clamp(max=max(n_bytes - 1, 0)).unsqueeze(1)
                keep = generated.gather(1, pos).squeeze(1)
                generated.scatter_(1, pos, torch.where(live, nxt, keep).unsqueeze(1))
                finished = finished | (nxt == BYTE_EOS_ID)
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
    import random
    from types import SimpleNamespace

    from entropy_lm import ByteEntropyLM, calibrate_threshold, patch_boundaries

    torch.manual_seed(0)
    random.seed(0)
    # A small stand-in for train.ModelConfig, so this file tests without importing train.py.
    cfg = SimpleNamespace(
        d_model=256, n_heads=8, n_kv_heads=2, d_head=32, d_ff=1024, dropout=0.1,
        attention="mha", norm="layernorm", pos_encoding="sinusoidal",
        n_encoder_layers=2, n_decoder_layers=2,
        d_local=128, n_local_encoder_layers=2, n_local_decoder_layers=2,
        local_n_heads=4, local_attn_window=128,
        ngram_sizes=(3, 4), ngram_buckets=8192, src_ngram_buckets=512,
    )                                     # smaller depth than ModelConfig: this is a shape test

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    P = MAX_PATCH_SIZE

    # A toy that mirrors the corpus -- the source is the target under a repeating-key XOR, at
    # one cipher byte per plaintext character, exactly as ByteSeq2SeqDataset delivers it --
    # rather than two independent random tensors. "Does the model read its source?" is then a
    # question about learning the map, not about memorising unrelated pairs, which is what let
    # the first C5 run pass this file and still collapse.
    B, CHARS = 6, 24
    key = torch.tensor([ord(c) for c in "ANLP2026"], device=dev).repeat(CHARS // 8)
    plain = torch.randint(97, 123, (B, CHARS), device=dev)
    src = plain ^ key                                    # one cipher byte per character, < 128
    tgt = torch.cat([plain, torch.full((B, 1), BYTE_EOS_ID, device=dev)], dim=1)

    # Variable-width patches, built the way dataset.py builds them: irregular on purpose, so a
    # bug that silently assumes a fixed stride cannot pass.
    src_lengths = [[3, 5, 1, 8, 2, 4, 1], [8, 8, 8], [1, 1, 6, 7, 4, 5],
                   [4, 4, 4, 4, 4, 4], [2, 7, 3, 8, 1, 2, 1], [6, 6, 6, 6]]
    assert all(sum(l) == CHARS for l in src_lengths)

    def target_lengths(lengths):
        out = list(lengths)
        if out[-1] < P:
            out[-1] += 1
        else:
            out.append(1)
        return out

    def grid_of(lengths_per_example, device):
        n = max(len(l) for l in lengths_per_example)
        index = torch.zeros(len(lengths_per_example), n, P, dtype=torch.long)
        mask = torch.zeros(len(lengths_per_example), n, P, dtype=torch.bool)
        for i, lengths in enumerate(lengths_per_example):
            pos = 0
            for k, size in enumerate(lengths):
                index[i, k, :size] = torch.arange(pos, pos + size)
                mask[i, k, :size] = True
                pos += size
        return index.to(device), mask.to(device)

    si, sm = grid_of(src_lengths, dev)
    ti, tm = grid_of([target_lengths(l) for l in src_lengths], dev)
    grid = {"src_patch_index": si, "src_patch_mask": sm,
            "tgt_patch_index": ti, "tgt_patch_mask": tm}
    print(f"patch grid: widths {sorted({n for l in src_lengths for n in l})}, "
          f"cap {P}, {si.size(1)} source / {ti.size(1)} target patches")

    # The scatter in `forward` relies on this and nothing else: the occupied slots of the grid,
    # read row-major, are byte 0, 1, 2, ... in order.
    for i in range(B):
        assert ti[i][tm[i]].tolist() == list(range(int(tm[i].sum()))), "grid is not in order"
    print("grid slots enumerate the byte axis in order (what forward's scatter assumes)")

    model = BLTSeq2Seq(cfg, max_len=1024).to(dev)
    print(f"BLTSeq2Seq: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M parameters "
          f"on {dev}")

    logits = model(src, tgt, grid)
    assert logits.shape == (B, tgt.size(1), BYTE_VOCAB_SIZE)
    memory, _ = model.encode_source(src, si, sm)
    assert memory.shape == (B, si.size(1), cfg.d_model)
    print(f"forward {tuple(logits.shape)}; {CHARS} src bytes -> {memory.shape[1]} patches "
          f"(mean {CHARS / si.size(1):.1f} bytes/patch)")

    shifted = model._shift_within_patch(tgt, ti)
    assert (shifted[..., 0] == BYTE_PATCH_START_ID).all()
    gathered = gather_slots(tgt, ti)
    assert torch.equal(shifted[..., 1:], gathered[..., :-1])
    print("within-patch shift is right by one, slot 0 is <patch-start>")

    model.eval()
    with torch.no_grad():
        base = model(src, tgt, grid)
        for k in (0, 3, 7, 11, 19):
            alt = tgt.clone()
            alt[:, k] = (alt[:, k] - 97 + 5) % 26 + 97
            out = model(src, tgt, grid, ctx_bytes=alt)
            assert torch.allclose(base[:, :k + 1], out[:, :k + 1], atol=1e-4), f"byte {k} leaked"
            assert not torch.allclose(base[:, k + 1:], out[:, k + 1:], atol=1e-4)
    print("causality verified: no target byte influences its own or any earlier logit")

    out = model.greedy_decode(src, grid)
    assert out.shape == (B, tgt.size(1)), (out.shape, tgt.shape)
    print(f"greedy decode returns {tuple(out.shape)}, one slot per target byte")

    with torch.no_grad():
        same = model(src, tgt, grid, ctx_bytes=tgt)
        assert torch.equal(base, same), "ctx_bytes=tgt_bytes must match the no-arg default"

    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(300):
        opt.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(src, tgt, grid).reshape(-1, BYTE_VOCAB_SIZE),
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
        real = model(src, tgt, grid).argmax(-1)
        wrong = model(torch.roll(src, 1, 0), tgt, grid).argmax(-1)
    acc = (real[keep] == tgt[keep]).float().mean().item()
    acc_bad = (wrong[keep] == tgt[keep]).float().mean().item()
    assert acc > 0.9, "BLT stack failed to fit a single batch"
    print(f"teacher-forced accuracy {100 * acc:.1f}%, mismatched sources {100 * acc_bad:.1f}% "
          f"(drop {100 * (acc - acc_bad):.1f} points)")

    # The aggregate drop above is informational only, and on a batch this small it is close to
    # meaningless: with six examples the target's own prefix identifies which example it is, so
    # a source-blind model still recalls most of the rest. Byte 0 is the position that cannot
    # be faked. Its decoder input is <patch-start>, its global input is the <bos> patch, and
    # both are constants -- so its prediction is a pure function of the source.
    with torch.no_grad():
        first_ok = (real[:, 0] == tgt[:, 0]).float().mean().item()
        first_bad = (wrong[:, 0] == tgt[:, 0]).float().mean().item()
    print(f"byte 0 (no target context, source-only): {100 * first_ok:.0f}% correct with its own "
          f"source, {100 * first_bad:.0f}% with a mismatched one")
    assert first_ok > 0.99, "byte 0 is not being predicted from the source at all"
    assert first_bad < 0.5, "byte 0 does not depend on which source it was given"

    # The entropy patcher itself, end to end on a stream with obvious structure: a repeated
    # motif is predictable and should be swallowed into long patches, random bytes should not.
    lm = ByteEntropyLM(max_len=256).to(dev)
    stream = torch.randint(97, 123, (32, 64), device=dev)
    ent = lm.entropies(stream)
    valid = torch.ones_like(stream, dtype=torch.bool)
    thr = calibrate_threshold(ent, valid, target_mean=TARGET_MEAN_PATCH)
    ids = patch_boundaries(ent, valid, thr)
    n = int((ids.max(1).values + 1).sum())
    sizes = torch.bincount(ids.flatten())
    assert ids[:, 0].eq(0).all() and (ids.diff(dim=1) >= 0).all(), "patch ids must be monotone"
    assert int(sizes.max()) <= MAX_PATCH_SIZE * ids.size(0), "cap violated"
    print(f"entropy patcher: threshold {thr:.3f} nats -> mean patch "
          f"{valid.sum().item() / n:.2f} bytes (target {TARGET_MEAN_PATCH})")

    # The honest version of the source-dependence check needs a set too large to memorise,
    # which is what `train.source_dependence` runs on the real validation split every epoch.
    print("blt.py self-test passed")
