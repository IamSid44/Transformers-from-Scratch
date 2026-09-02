"""The small byte-level LM that drives C5's entropy-based dynamic patching.

BLT (Pagnoni et al., 2024) does not cut its byte stream on a fixed stride. It runs a small
causal byte-level language model over the stream, reads off the next-byte entropy

    H(x_t) = - sum_v p(x_t = v | x_<t) log p(x_t = v | x_<t)

and opens a new patch wherever that entropy crosses a global threshold. Patches are therefore
long through predictable stretches and short where the next byte is genuinely uncertain --
compute is spent where the information is. This file is that entropy model, and
`patch_boundaries` is that rule.

Which stream is it trained on? The cipher, not the plaintext -- and this matters twice.

  * At inference the plaintext is exactly what we do not have. Boundaries derived from the
    source are available before a single target byte exists, so greedy decoding knows its
    patch grid up front.
  * The corpus cipher is c_i = p_i XOR K[i mod 8] with K = "ANLP2026" (recovered in
    `dataset.py`'s self-test; never supplied to any model). That is a *bijection* between the
    two streams, character for character. So one boundary set is simultaneously the right
    boundary set for both sides -- source patch k and target patch k keep covering the same
    span of text, which is the alignment property BLT.md records C5 collapsing without.
    The model needs positions to see the XOR phase, so it gets the same sinusoidal encoding
    the rest of the stack uses; with that, entropy over cipher bytes is entropy over the
    plaintext characters they stand for.

Deliberately small (2 layers, d=128, ~0.5M parameters). BLT uses a 100M-parameter entropy
model over a whole pretraining corpus; the assignment's clarification allows anything from an
n-gram estimator upwards. This is the middle road: a real learned next-byte model, trained in
a couple of minutes on the training split only.

    python src/models/entropy_lm.py          # train, pick the threshold, save + report
"""

from __future__ import annotations

import math
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from attention import EncoderLayer, causal_mask
from norm import build_norm
from positional import SinusoidalPositionalEncoding

# Kept in sync with blt.py by import there; duplicated here so this file trains standalone.
BYTE_PAD_ID, BYTE_EOS_ID, BYTE_PATCH_START_ID = 256, 257, 258
BYTE_VOCAB_SIZE = 259

# The patch grid the pooler and local decoder are built around.
MAX_PATCH_SIZE = 8            # hard cap: a patch is cut here even if entropy never spikes
TARGET_MEAN_PATCH = 4.0       # the threshold is calibrated to hit this, so C5's mean patch
                              # length matches the fixed stride the first version used and the
                              # dynamic/fixed comparison is not confounded by sequence length

ENTROPY_LM_PATH = Path(__file__).resolve().parents[2] / "outputs" / "entropy_lm.pt"


class _Cfg:
    """The three fields EncoderLayer reads, without importing train.ModelConfig."""

    def __init__(self, d_model, n_heads, d_ff, dropout, norm="layernorm"):
        self.d_model, self.n_heads, self.n_kv_heads = d_model, n_heads, n_heads
        self.d_ff, self.dropout, self.norm = d_ff, dropout, norm
        self.attention, self.pos_encoding = "mha", "sinusoidal"


class ByteEntropyLM(nn.Module):
    """Causal byte-level LM. `entropies(x)` is the only thing C5 uses at run time."""

    def __init__(self, d_model=128, n_layers=2, n_heads=4, d_ff=512, dropout=0.0,
                 max_len=1024):
        super().__init__()
        cfg = _Cfg(d_model, n_heads, d_ff, dropout)
        self.d_model = d_model
        self.embed = nn.Embedding(BYTE_VOCAB_SIZE, d_model, padding_idx=BYTE_PAD_ID)
        nn.init.normal_(self.embed.weight, std=0.02)
        self.pos = SinusoidalPositionalEncoding(d_model, max_len, dropout)
        self.layers = nn.ModuleList(EncoderLayer(cfg) for _ in range(n_layers))
        self.norm = build_norm("layernorm", d_model)
        self.out = nn.Linear(d_model, BYTE_VOCAB_SIZE)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """(B, L) byte ids -> (B, L, V) logits; position t predicts the byte after input t."""
        x = self.pos(self.embed(inputs) * math.sqrt(self.d_model))
        mask = causal_mask(inputs.size(1), inputs.device)
        for layer in self.layers:
            x = layer(x, src_mask=mask)
        return self.out(self.norm(x))

    @staticmethod
    def _shift(x: torch.Tensor) -> torch.Tensor:
        """[x_0..x_{n-1}] -> [<start>, x_0..x_{n-2}], so output t predicts x_t."""
        start = torch.full_like(x[:, :1], BYTE_PATCH_START_ID)
        return torch.cat([start, x[:, :-1]], dim=1)

    @torch.no_grad()
    def entropies(self, x: torch.Tensor) -> torch.Tensor:
        """(B, L) bytes -> (B, L) next-byte entropy in nats: element t is H(x_t | x_<t).

        Position 0 is included and is genuinely defined -- it is the entropy of the model's
        unconditional first-byte distribution -- but `patch_boundaries` opens a patch at 0
        regardless, so its value never decides anything.
        """
        logp = F.log_softmax(self(self._shift(x)).float(), dim=-1)
        return -(logp.exp() * logp).sum(-1)


def patch_boundaries(entropy: torch.Tensor, valid: torch.Tensor, threshold: float,
                     max_patch: int = MAX_PATCH_SIZE) -> torch.Tensor:
    """Entropy -> patch id per byte. (B, L) float, (B, L) bool -> (B, L) long.

    BLT's "global threshold" rule: byte t starts a new patch when H(x_t) > threshold, i.e.
    when the model was surprised by it. A patch is additionally cut at `max_patch` bytes,
    which bounds the padded patch grid the pooler and local decoder are built on -- without a
    cap a long predictable stretch would make one enormous patch and the (B, N, P_max) tensors
    would size to it.

    Padding bytes never open a patch and are left in the last real patch's id; the caller's
    validity mask is what keeps them out of the pooling and the loss.
    """
    b, length = entropy.shape
    device = entropy.device
    ids = torch.zeros(b, length, dtype=torch.long, device=device)
    # Sequential in L (short here: 32-byte chunks), vectorised across the batch.
    cur = torch.zeros(b, dtype=torch.long, device=device)
    run = torch.ones(b, dtype=torch.long, device=device)
    for t in range(1, length):
        spike = (entropy[:, t] > threshold) & valid[:, t]
        full = (run >= max_patch) & valid[:, t]
        new = spike | full
        cur = cur + new.long()
        run = torch.where(new, torch.ones_like(run), run + 1)
        ids[:, t] = cur
    return ids


def calibrate_threshold(entropies: torch.Tensor, valid: torch.Tensor,
                        target_mean: float = TARGET_MEAN_PATCH,
                        max_patch: int = MAX_PATCH_SIZE) -> float:
    """Pick the global threshold whose mean patch length is closest to `target_mean`.

    BLT chooses its threshold to hit a target patch size for exactly this reason: patch count
    sets the global transformer's sequence length, so leaving it to whatever an arbitrary
    entropy cut-off produces makes every downstream cost number incomparable. Bisection on the
    threshold, since mean patch length is monotone in it.
    """
    lo = float(entropies[valid].min())
    hi = float(entropies[valid].max())
    for _ in range(30):
        mid = 0.5 * (lo + hi)
        ids = patch_boundaries(entropies, valid, mid, max_patch)
        n_patches = (ids.max(dim=1).values + 1).sum().item()
        mean = valid.sum().item() / max(n_patches, 1)
        if mean < target_mean:
            lo = mid          # too many patches -> raise the bar for a spike
        else:
            hi = mid
    return 0.5 * (lo + hi)


def load_entropy_lm(device, path: Path = ENTROPY_LM_PATH):
    """Rebuild the trained entropy model and its calibrated threshold."""
    if not path.exists():
        raise FileNotFoundError(
            f"No entropy model at {path}. C5 needs it for dynamic patching; train it with "
            f"`python src/models/entropy_lm.py`.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = ByteEntropyLM(**ckpt["arch"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), ckpt["threshold"], ckpt


# --- training ------------------------------------------------------------------------------


def train_entropy_lm(train_bytes, val_bytes, device, epochs=6, batch_size=512, lr=3e-4):
    """`train_bytes` / `val_bytes`: (N, L) padded byte tensors of cipher chunks."""
    model = ByteEntropyLM(max_len=train_bytes.size(1) + 8).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    n = train_bytes.size(0)
    steps = epochs * -(-n // batch_size)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps,
                                                pct_start=0.1)
    print(f"[entropy-lm] {n_params / 1e6:.2f}M parameters, {n:,} chunks, {steps} steps")

    step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total, count = 0.0, 0
        for i in range(0, n, batch_size):
            x = train_bytes[perm[i:i + batch_size]].to(device)
            logits = model(model._shift(x))
            loss = F.cross_entropy(logits.reshape(-1, BYTE_VOCAB_SIZE), x.reshape(-1),
                                   ignore_index=BYTE_PAD_ID)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
            if step + 1 < steps:
                sched.step()
            step += 1
            k = int((x != BYTE_PAD_ID).sum())
            total += loss.item() * k
            count += k

        model.eval()
        vl, vc = 0.0, 0
        with torch.no_grad():
            for i in range(0, val_bytes.size(0), batch_size):
                x = val_bytes[i:i + batch_size].to(device)
                logits = model(model._shift(x))
                loss = F.cross_entropy(logits.reshape(-1, BYTE_VOCAB_SIZE), x.reshape(-1),
                                       ignore_index=BYTE_PAD_ID)
                k = int((x != BYTE_PAD_ID).sum())
                vl += loss.item() * k
                vc += k
        print(f"[entropy-lm] epoch {epoch}/{epochs}  train {total / count:.4f}  "
              f"val {vl / vc:.4f} nats/byte")
    return model


if __name__ == "__main__":
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import dataset as ds

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(ds.SEED)

    splits, _ = ds.build_splits()
    chunked = {name: ds.chunk_pairs(splits[name]) for name in ("train", "val")}

    def to_tensor(pairs):
        width = max(len(p.plain) for p in pairs)
        out = torch.full((len(pairs), width), BYTE_PAD_ID, dtype=torch.long)
        for i, p in enumerate(pairs):
            row = ds.cipher_to_bytes(p.cipher)
            out[i, :len(row)] = torch.tensor(list(row), dtype=torch.long)
        return out

    train_bytes, val_bytes = to_tensor(chunked["train"]), to_tensor(chunked["val"])
    print(f"[entropy-lm] cipher-byte chunks: {tuple(train_bytes.shape)} train, "
          f"{tuple(val_bytes.shape)} val")

    model = train_entropy_lm(train_bytes, val_bytes, device)

    # Calibrate on a training subsample: the threshold is a property of the model, and reading
    # it off val or test would leak the evaluation distribution into the architecture.
    sample = train_bytes[torch.randperm(train_bytes.size(0))[:8192]].to(device)
    ent = model.entropies(sample)
    valid = sample != BYTE_PAD_ID
    threshold = calibrate_threshold(ent, valid)

    ids = patch_boundaries(ent, valid, threshold)
    sizes = torch.zeros_like(ids, dtype=torch.float).scatter_add_(
        1, ids, valid.float()).flatten()
    sizes = sizes[sizes > 0]
    n_patches = int((ids.max(dim=1).values + 1).sum())
    print(f"[entropy-lm] threshold {threshold:.4f} nats  ->  "
          f"mean patch {valid.sum().item() / n_patches:.2f} bytes, "
          f"sizes {int(sizes.min())}-{int(sizes.max())}, "
          f"std {sizes.std():.2f}")
    hist = torch.bincount(sizes.long(), minlength=MAX_PATCH_SIZE + 1)[1:MAX_PATCH_SIZE + 1]
    print("[entropy-lm] patch-length histogram: " +
          "  ".join(f"{i + 1}:{100 * c / hist.sum():.0f}%" for i, c in enumerate(hist)))

    ENTROPY_LM_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(),
                "arch": {"d_model": 128, "n_layers": 2, "n_heads": 4, "d_ff": 512,
                         "dropout": 0.0, "max_len": train_bytes.size(1) + 8},
                "threshold": threshold,
                "max_patch": MAX_PATCH_SIZE,
                "mean_patch": valid.sum().item() / n_patches,
                "length_histogram": (hist / hist.sum()).tolist()},
               ENTROPY_LM_PATH)
    print(f"[entropy-lm] saved -> {ENTROPY_LM_PATH}")
