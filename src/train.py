"""Configurations, training loop, and evaluation.

    python src/train.py --config C1                 # train, then evaluate, C1
    python src/train.py --config C1 --smoke         # a few steps, wiring check, no evaluation
    python src/train.py --all --push                # train+evaluate C1, then C2, ... then C5
    python src/train.py --evaluate --all            # re-evaluate existing checkpoints only

All five configurations share one TrainConfig, so optimiser, schedule, batch size and epoch
budget are identical by construction -- any difference in the results comes from the one
architectural component that changed. C2-C5 are built from C1 with `dataclasses.replace`, so
it is structurally impossible for more than the named field to differ.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR / "models"))

import dataset as ds                     
import utils                             
from attention import Seq2SeqTransformer 
from blt import BLTSeq2Seq, bytes_to_text

# --- identity / paths --------------------------------------------------------------------
ROLL_NUMBER = "2023102040"
WANDB_PROJECT = f"anlp-a1-{ROLL_NUMBER}"
HF_REPO_TEMPLATE = "{user}/anlp-a1-" + ROLL_NUMBER + "-{config}"

# Checkpoints live under outputs/ so the top level matches the required submission tree; they
# are excluded from the zip and mirrored on HuggingFace instead.
CKPT_DIR = ds.OUTPUT_DIR / "checkpoints"
ENV_FILE = ds.PROJECT_ROOT / ".env"


@dataclass
class ModelConfig:
    """Architecture of one configuration. Only the four ablated axes vary across C1-C5."""

    name: str
    changed_from_base: str

    # the four ablated axes
    pos_encoding: str = "sinusoidal"       # "sinusoidal" | "rope"
    attention: str = "mha"                 # "mha" | "gqa"
    norm: str = "layernorm"                # "layernorm" | "rmsnorm"
    tokenization: str = "bpe"              # "bpe" | "blt"

    # shared geometry
    d_model: int = 256
    n_heads: int = 8                       # d_head = 32
    n_kv_heads: int = 2                    # only used when attention == "gqa"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 4
    d_ff: int = 2048
    dropout: float = 0.1

    # BLT only (ignored unless tokenization == "blt")
    d_local: int = 256
    n_local_encoder_layers: int = 2
    n_local_decoder_layers: int = 2
    local_n_heads: int = 4
    local_attn_window: int = 128
    ngram_sizes: tuple = (3, 4)
    ngram_buckets: int = 8192              # target side (English)
    src_ngram_buckets: int = 512           # source side: only 2^3 3-grams exist in binary

    @property
    def is_blt(self) -> bool:
        return self.tokenization == "blt"

    @property
    def d_head(self) -> int:
        assert self.d_model % self.n_heads == 0
        return self.d_model // self.n_heads

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


@dataclass
class TrainConfig:
    """Shared across all five configurations, so any difference is architectural."""

    epochs: int = 50
    batch_size: int = 16
    lr: float = 6e-4                       # Adam, linear warmup then cosine decay
    warmup_steps: int = 250
    grad_clip: float = 1.0
    label_smoothing: float = 0.1
    scheduled_sampling_floor: float = 0.7  # see teacher_forcing_prob
    early_stopping_patience: int = 5
    group_by_length: bool = True           # batch similar-length lines together
    num_workers: int = 0
    log_every: int = 50

    def to_dict(self) -> dict:
        return dataclasses.asdict(self)


_BASE = ModelConfig(name="C1", changed_from_base="None (base)")

CONFIGS: dict[str, ModelConfig] = {
    "C1": _BASE,
    "C2": dataclasses.replace(_BASE, name="C2", changed_from_base="Positional encoding",
                              pos_encoding="rope"),
    "C3": dataclasses.replace(_BASE, name="C3", changed_from_base="Attention mechanism",
                              attention="gqa"),
    "C4": dataclasses.replace(_BASE, name="C4", changed_from_base="Normalization",
                              norm="rmsnorm"),
    "C5": dataclasses.replace(_BASE, name="C5", changed_from_base="Tokenization",
                              tokenization="blt"),
}


def get_config(name: str) -> ModelConfig:
    key = name.upper()
    if key not in CONFIGS:
        raise KeyError(f"Unknown config {name!r}; expected one of {sorted(CONFIGS)}")
    return CONFIGS[key]


def describe_configs() -> str:
    header = (f"{'Config':<8}{'Changed':<22}{'PosEnc':<14}{'Attention':<14}"
              f"{'Norm':<12}{'Tokenization':<14}")
    rows = [header, "-" * len(header)]
    for cfg in CONFIGS.values():
        attn = "GQA (8q/2kv)" if cfg.attention == "gqa" else "MHA"
        rows.append(f"{cfg.name:<8}{cfg.changed_from_base:<22}{cfg.pos_encoding:<14}"
                    f"{attn:<14}{cfg.norm:<12}{cfg.tokenization:<14}")
    return "\n".join(rows)


# --- model / loss ------------------------------------------------------------------------


def build_model(model_cfg, meta: dict) -> nn.Module:
    """C1-C4 get Seq2SeqTransformer; C5 gets BLTSeq2Seq, whose global transformer is that
    same class in latent mode."""
    max_len = max(meta["max_src_len"], meta["max_tgt_len"]) + 64
    if model_cfg.is_blt:
        return BLTSeq2Seq(model_cfg, max_len=max_len)
    return Seq2SeqTransformer(model_cfg, meta["cipher_vocab_size"], meta["plain_vocab_size"],
                              pad_id=ds.PAD_ID, max_len=max_len)


def teacher_forcing_prob(step: int, warmup_steps: int, total_steps: int, floor: float) -> float:
    """1.0 (pure teacher forcing) through warmup, then linear decay to `floor` by the end of
    training -- the probability that a given decoder-input position uses the true previous
    token rather than the model's own prediction for it (scheduled sampling, Bengio et al.
    2015). Training is otherwise 100% teacher-forced, so the model never practices recovering
    from its own mistakes; every position at evaluation time is one it has never seen. This is
    a training-time fix only -- the assignment mandates greedy decoding for every reported
    metric, so it leaves evaluation untouched.

    Waits for the same warmup the LR schedule uses: early in training the model's own
    predictions are close to random, and mixing that noise into the supervision before it has
    learned anything would corrupt training rather than toughen it.
    """
    if step < warmup_steps:
        return 1.0
    progress = min(1.0, (step - warmup_steps) / max(total_steps - warmup_steps, 1))
    return 1.0 - progress * (1.0 - floor)


def compute_loss(model_cfg, model, batch, label_smoothing: float, tf_prob: float = 1.0):
    """Teacher-forced forward pass. Returns (loss, supervised token count).

    Tokenized: target is <bos> w1..wn <eos>; feed all but the last, supervise all but the
    first, so position t predicts t+1. BLT shifts internally (patch stream by one patch, and
    bytes within each patch by one), so its logits already align with the raw target bytes.

    `tf_prob < 1.0` (only ever passed while `model.training`; `evaluate_loss` leaves it at the
    default) mixes in the model's own predictions: one extra no-grad forward pass gets them,
    then some positions are swapped in at rate `1 - tf_prob` -- the standard parallelizable
    approximation of scheduled sampling for a non-recurrent (Transformer) decoder, since a true
    sequential mix would need one forward pass per position. For BLT this is one mix
    (`ctx_bytes`) that feeds both the patch pooling driving the global decoder and the
    within-patch local decoder, since both are exposure-bias points there; for the tokenized
    models `<bos>` (position 0) is never replaced, since it has no "own prediction" to be
    replaced with.
    """
    src, tgt = batch["src"], batch["tgt"]
    if model_cfg.is_blt:
        ctx = tgt
        if model.training and tf_prob < 1.0:
            with torch.no_grad():
                own = model(src, tgt).argmax(-1)
            replace = (torch.rand(tgt.shape, device=tgt.device) >= tf_prob) \
                & (tgt != ds.BYTE_PAD_ID)
            ctx = torch.where(replace, own, tgt)
        logits, labels, ignore = model(src, tgt, ctx_bytes=ctx), tgt, ds.BYTE_PAD_ID
    else:
        tgt_in = tgt[:, :-1]
        if model.training and tf_prob < 1.0 and tgt_in.size(1) > 1:
            with torch.no_grad():
                own = model(src, tgt_in).argmax(-1)
            replace = torch.rand(tgt_in[:, 1:].shape, device=tgt_in.device) >= tf_prob
            mixed = tgt_in.clone()
            mixed[:, 1:] = torch.where(replace, own[:, :-1], tgt_in[:, 1:])
            tgt_in = mixed
        logits, labels, ignore = model(src, tgt_in), tgt[:, 1:], ds.PAD_ID

    loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)), labels.reshape(-1),
                           ignore_index=ignore, label_smoothing=label_smoothing)
    return loss, int((labels != ignore).sum())


def lr_lambda_factory(warmup_steps: int, total_steps: int):
    """Linear warmup, then cosine decay to 10% of the peak.

    The shallow 75% floor this replaces was chosen when the runs were still ending mid-descent
    and a high late-epoch rate looked like the thing keeping them learning. It was not: the
    losses plateaued high because the tokenization gave the model no aligned units to learn
    from (see `dataset.py`), and a rate that never really came down just kept the late epochs
    noisy. With the source now BPE over byte-boundary-respecting cipher symbols there are
    real, character-aligned units to fit, which is what a decay to 10% is for.
    """

    floor = 0.10

    def fn(step: int) -> float:
        if step < warmup_steps:
            return (step + 1) / max(warmup_steps, 1)
        progress = min(max((step - warmup_steps) / max(total_steps - warmup_steps, 1), 0.0), 1.0)
        return floor + (1.0 - floor) * 0.5 * (1.0 + math.cos(math.pi * progress))

    return fn


@torch.no_grad()
def evaluate_loss(model_cfg, model, loader, device) -> float:
    """Token-weighted mean cross-entropy, no label smoothing, so it is a true likelihood."""
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        loss, n = compute_loss(model_cfg, model, batch, label_smoothing=0.0)
        total_loss += loss.item() * n
        total_tokens += n
    model.train()
    return total_loss / max(total_tokens, 1)


# --- training ----------------------------------------------------------------------------


def train_one(config_name, train_cfg, device, use_wandb=True, smoke_steps=0, push=False) -> dict:
    model_cfg = get_config(config_name)
    utils.set_seed(ds.SEED)

    data = ds.make_dataloaders(model_cfg, train_cfg)
    loaders, meta = data["loaders"], dict(data["meta"])
    model = build_model(model_cfg, meta).to(device)
    total_params, trainable = utils.count_parameters(model)

    steps_per_epoch = len(loaders["train"])
    total_steps = steps_per_epoch * train_cfg.epochs
    optimizer = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda_factory(train_cfg.warmup_steps, total_steps))

    print(f"\n{'=' * 78}")
    print(f"Training {config_name}: pos={model_cfg.pos_encoding}, attn={model_cfg.attention}, "
          f"norm={model_cfg.norm}, tokenization={model_cfg.tokenization}")
    print(f"  parameters      {total_params / 1e6:.2f}M ({trainable / 1e6:.2f}M trainable)")
    print(f"  train examples  {len(data['datasets']['train']):,}  "
          f"({steps_per_epoch} steps/epoch x {train_cfg.epochs} epochs)")
    print(f"  device          {device}")
    print(f"{'=' * 78}")

    run = _init_wandb(config_name, model_cfg, train_cfg, meta, total_params) if use_wandb else None

    # examples_per_sec is the honest cross-config throughput: a "token" is a BPE subword for
    # C1-C4 but a raw byte for C5, which would make C5 look artificially fast.
    history = {"train_loss": [], "val_loss": [], "epoch_seconds": [],
               "examples_per_sec": [], "peak_memory_mb": [], "lr": [], "teacher_forcing": []}
    best_val, best_epoch, stale, global_step = float("inf"), -1, 0, 0
    ckpt_dir = CKPT_DIR / config_name
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    wall_start = time.perf_counter()

    for epoch in range(1, train_cfg.epochs + 1):
        model.train()
        epoch_loss, epoch_tokens, epoch_examples = 0.0, 0, 0
        utils.reset_peak_memory()

        with utils.Timer() as epoch_timer:
            for batch in loaders["train"]:
                batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
                tf_prob = teacher_forcing_prob(global_step, train_cfg.warmup_steps, total_steps,
                                               train_cfg.scheduled_sampling_floor)
                loss, n_tokens = compute_loss(model_cfg, model, batch, train_cfg.label_smoothing,
                                              tf_prob)
                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                scheduler.step()

                # Read the loss out once, after the step: keeping the graph-attached tensor
                # alive across the epoch would pin every activation it references.
                step_loss = loss.detach().item()
                epoch_loss += step_loss * n_tokens
                epoch_tokens += n_tokens
                epoch_examples += batch["src"].size(0)
                global_step += 1

                if run is not None and global_step % train_cfg.log_every == 0:
                    run.log({"train/loss_step": step_loss, "train/grad_norm": float(grad_norm),
                             "train/lr": scheduler.get_last_lr()[0],
                             "train/teacher_forcing": tf_prob}, step=global_step)
                if smoke_steps and global_step >= smoke_steps:
                    break

        train_loss = epoch_loss / max(epoch_tokens, 1)
        val_loss = evaluate_loss(model_cfg, model, loaders["val"], device)
        peak_mb = utils.peak_memory_mb()
        ex_per_sec = epoch_examples / max(epoch_timer.elapsed, 1e-6)

        for key, value in [("train_loss", train_loss), ("val_loss", val_loss),
                           ("epoch_seconds", epoch_timer.elapsed),
                           ("examples_per_sec", ex_per_sec), ("peak_memory_mb", peak_mb),
                           ("lr", scheduler.get_last_lr()[0]), ("teacher_forcing", tf_prob)]:
            history[key].append(value)

        print(f"  epoch {epoch:>2}/{train_cfg.epochs}  train {train_loss:.4f}  "
              f"val {val_loss:.4f}  ppl {math.exp(min(val_loss, 20)):.2f}  tf {tf_prob:.2f}  "
              f"{utils.human_time(epoch_timer.elapsed)}  {ex_per_sec:.0f} ex/s  {peak_mb:.0f} MiB")

        if run is not None:
            run.log({"epoch": epoch, "train/loss": train_loss, "val/loss": val_loss,
                     "val/perplexity": math.exp(min(val_loss, 20)),
                     "perf/epoch_seconds": epoch_timer.elapsed,
                     "perf/examples_per_sec": ex_per_sec,
                     "perf/peak_memory_mb": peak_mb}, step=global_step)

        if val_loss < best_val - 1e-5:
            best_val, best_epoch, stale = val_loss, epoch, 0
            torch.save({"config_name": config_name, "model_config": model_cfg.to_dict(),
                        "train_config": train_cfg.to_dict(), "meta": meta,
                        "state_dict": model.state_dict(), "epoch": epoch,
                        "val_loss": val_loss}, ckpt_dir / "best.pt")
        else:
            stale += 1
            if stale >= train_cfg.early_stopping_patience:
                print(f"  early stopping: no improvement for {stale} epochs")
                break
        if smoke_steps:
            break

    wall = time.perf_counter() - wall_start
    n_epochs = max(len(history["train_loss"]), 1)
    summary = {
        "config": config_name,
        "params": total_params,
        "params_millions": total_params / 1e6,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": len(history["train_loss"]),
        "wall_seconds": wall,
        "sec_per_epoch": sum(history["epoch_seconds"]) / n_epochs,
        "examples_per_sec": sum(history["examples_per_sec"]) / n_epochs,
        "peak_memory_mb": max(history["peak_memory_mb"], default=0.0),
        "history": history,
        "model_config": model_cfg.to_dict(),
    }
    ds.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    utils.save_json(summary, ds.OUTPUT_DIR / f"history_{config_name}.json")
    print(f"  done in {utils.human_time(wall)}; best val {best_val:.4f} @ epoch {best_epoch}")

    if run is not None:
        run.summary.update({k: v for k, v in summary.items() if k != "history"})
        run.finish()
    if push and not smoke_steps:
        push_to_hub(config_name, ckpt_dir / "best.pt", summary)
    return summary


# --- WandB / HuggingFace -------------------------------------------------------------------


def _init_wandb(config_name, model_cfg, train_cfg, meta, total_params):
    """Start a WandB run; falls back to offline mode when no API key is set."""
    import os

    import wandb

    if not os.environ.get("WANDB_API_KEY"):
        os.environ.setdefault("WANDB_MODE", "offline")
        print("  [wandb] no WANDB_API_KEY in .env -- logging offline to ./wandb/")
    return wandb.init(
        project=WANDB_PROJECT, name=config_name, group="ablation", job_type="train",
        reinit=True,
        config={**model_cfg.to_dict(),
                **{f"train_{k}": v for k, v in train_cfg.to_dict().items()},
                "total_params": total_params,
                **{f"data_{k}": v for k, v in meta.items()}},
    )


def push_to_hub(config_name: str, ckpt_path: Path, summary: dict) -> str | None:
    """Upload the best checkpoint, the tokenizers, and a generated model card."""
    import os

    token, user = os.environ.get("HF_TOKEN"), os.environ.get("HF_USERNAME")
    if not (token and user):
        print("  [hf] HF_TOKEN / HF_USERNAME missing from .env -- skipping upload")
        return None

    from huggingface_hub import HfApi

    repo_id = HF_REPO_TEMPLATE.format(user=user, config=config_name)
    api = HfApi(token=token)
    api.create_repo(repo_id, repo_type="model", exist_ok=True)

    card_path = ckpt_path.parent / "README.md"
    card_path.write_text(_model_card(config_name, summary, repo_id), encoding="utf-8")
    api.upload_file(path_or_fileobj=str(ckpt_path), path_in_repo="best.pt", repo_id=repo_id)
    api.upload_file(path_or_fileobj=str(card_path), path_in_repo="README.md", repo_id=repo_id)
    for tok_file in sorted(ds.TOKENIZER_DIR.glob("*.json")):
        api.upload_file(path_or_fileobj=str(tok_file),
                        path_in_repo=f"tokenizers/{tok_file.name}", repo_id=repo_id)

    url = f"https://huggingface.co/{repo_id}"
    print(f"  [hf] uploaded -> {url}")
    return url


def _model_card(config_name: str, summary: dict, repo_id: str) -> str:
    cfg = summary["model_config"]
    gqa = " (8 query / 2 KV heads)" if cfg["attention"] == "gqa" else ""
    return f"""---
license: mit
tags: [transformer, seq2seq, cryptanalysis, byte-latent-transformer]
---

# ANLP Assignment 1 -- Configuration {config_name}

Encoder-decoder transformer built from scratch, trained to decrypt binary cipher sequences
into English plaintext. One of a five-way controlled ablation; this repo holds **{config_name}**.

| Component | Setting |
|---|---|
| Changed from base | {cfg['changed_from_base']} |
| Positional encoding | {cfg['pos_encoding']} |
| Attention | {cfg['attention']}{gqa} |
| Normalization | {cfg['norm']} |
| Tokenization | {cfg['tokenization']} |
| d_model / heads / layers | {cfg['d_model']} / {cfg['n_heads']} / {cfg['n_encoder_layers']}+{cfg['n_decoder_layers']} |
| Parameters | {summary['params_millions']:.2f}M |
| Best validation loss | {summary['best_val_loss']:.4f} (epoch {summary['best_epoch']}) |

```python
import torch
from huggingface_hub import hf_hub_download

ckpt = torch.load(hf_hub_download("{repo_id}", "best.pt"), map_location="cpu")
# rebuild via train.build_model(ModelConfig(**ckpt["model_config"]), ckpt["meta"])
model.load_state_dict(ckpt["state_dict"])
```

Training and evaluation code is in the submission `{ROLL_NUMBER}_assignment1`.
"""


# --- evaluation ----------------------------------------------------------------------------


def load_checkpoint(config_name: str, device: torch.device):
    """Rebuild a trained model from outputs/checkpoints/<config>/best.pt."""
    path = CKPT_DIR / config_name / "best.pt"
    if not path.exists():
        raise FileNotFoundError(f"No checkpoint for {config_name} at {path}. Train it first.")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model_cfg = ModelConfig(**ckpt["model_config"])
    model = build_model(model_cfg, ckpt["meta"])
    model.load_state_dict(ckpt["state_dict"])
    return model.to(device).eval(), model_cfg, ckpt["meta"], ckpt


@torch.no_grad()
def decode_split(model, model_cfg, data, split, device, limit_lines=None):
    """Greedily decode one split. Returns (predictions, pairs, stats).

    The model decodes chunks (`data["loaders"][split]` is built from `chunk_pairs`), but
    scoring happens at whole-line granularity: chunk predictions are regrouped by
    `chunk_line_ids` and concatenated in order before being returned, one string per entry
    of `pairs`.
    """
    pairs = data["pairs"][split]
    chunk_line_ids = data["chunk_line_ids"][split]
    loader = data["loaders"][split]

    if limit_lines is not None:
        keep_ids = {p.line_id for p in pairs[:limit_lines]}
        # Chunks of the same line are consecutive and in split order, same as `pairs`, so the
        # kept lines' chunks are a contiguous prefix of chunk_line_ids.
        cutoff = next((i for i, lid in enumerate(chunk_line_ids) if lid not in keep_ids),
                     len(chunk_line_ids))
        pairs = pairs[:limit_lines]
        chunk_line_ids = chunk_line_ids[:cutoff]
        loader = torch.utils.data.DataLoader(
            torch.utils.data.Subset(data["datasets"][split], range(cutoff)),
            # .batch_size is None when a loader was built with a batch_sampler.
            batch_size=loader.batch_size or TrainConfig().batch_size,
            shuffle=False, collate_fn=loader.collate_fn)

    meta = data["meta"]
    chunk_predictions: list[str] = []
    utils.reset_peak_memory()
    with utils.Timer() as timer:
        for batch in loader:
            src = batch["src"].to(device, non_blocking=True)
            if model_cfg.is_blt:
                out = model.greedy_decode(src, max_bytes=meta["max_tgt_len"])
                chunk_predictions.extend(bytes_to_text(row) for row in out.cpu())
            else:
                out = model.greedy_decode(src, max_len=meta["max_tgt_len"],
                                          bos_id=ds.BOS_ID, eos_id=ds.EOS_ID)
                chunk_predictions.extend(ds.decode_plain(data["plain_tok"], row)
                                         for row in out.cpu().tolist())

    grouped: dict[int, list[str]] = {}
    for line_id, pred in zip(chunk_line_ids, chunk_predictions):
        grouped.setdefault(line_id, []).append(pred)
    predictions = ["".join(grouped[p.line_id]) for p in pairs]

    return predictions, pairs, {"decode_seconds": timer.elapsed,
                                "decode_peak_memory_mb": utils.peak_memory_mb()}


def evaluate_config(config_name, device, split="test", limit_lines=None, n_samples=8) -> dict:
    """Decode, score, and write qualitative samples for one configuration."""
    print(f"\n[{config_name}] loading checkpoint ...")
    model, model_cfg, meta, ckpt = load_checkpoint(config_name, device)

    data = ds.make_dataloaders(model_cfg, TrainConfig())
    total_params, _ = utils.count_parameters(model)

    print(f"[{config_name}] decoding {split} split greedily ...")
    preds, pairs, stats = decode_split(model, model_cfg, data, split, device, limit_lines)
    golds = [p.plain for p in pairs]

    print(f"[{config_name}] scoring {len(preds)} lines ...")
    metrics = utils.compute_all_metrics(preds, golds)
    metrics.update({
        "config": config_name,
        "changed_from_base": model_cfg.changed_from_base,
        "params_millions": total_params / 1e6,
        "best_val_loss": ckpt.get("val_loss", float("nan")),
        "best_epoch": ckpt.get("epoch", -1),
        "n_test_lines": len(preds),
        "decode_seconds": stats["decode_seconds"],
        "decode_ms_per_line": 1000.0 * stats["decode_seconds"] / max(len(preds), 1),
        "inference_peak_memory_mb": stats["decode_peak_memory_mb"],
    })

    ds.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (ds.OUTPUT_DIR / f"samples_{config_name}.txt").open("w", encoding="utf-8") as fh:
        fh.write(f"Greedy decoding samples -- {config_name} ({model_cfg.changed_from_base})\n")
        fh.write(f"{'=' * 100}\n\n")
        for pair, pred in list(zip(pairs, preds))[:n_samples]:
            bm, bt = utils.bit_accuracy(pred, pair.plain)
            fh.write(f"--- test line {pair.line_id} (bit accuracy {100 * bm / max(bt, 1):.2f}%, "
                     f"edit distance {utils.levenshtein(pred, pair.plain)}) ---\n")
            fh.write(f"GOLD: {pair.plain[:400]}\nPRED: {pred[:400]}\n\n")

    print(f"[{config_name}] bit={metrics['bit_accuracy']:.2f}%  "
          f"char={metrics['char_accuracy']:.2f}%  seq={metrics['sequence_accuracy']:.2f}%  "
          f"lev={metrics['levenshtein_mean']:.1f}  BLEU={metrics['bleu']:.2f}  "
          f"ROUGE-L={metrics['rougeL']:.2f}")
    return metrics


METRIC_COLUMNS = [
    "config", "changed_from_base", "params_millions", "best_val_loss",
    "bit_accuracy", "char_accuracy", "sequence_accuracy",
    "levenshtein_mean", "levenshtein_normalized", "bleu", "rouge1", "rouge2", "rougeL",
    # reference implementations, for comparison against the hand-rolled columns above
    "levenshtein_mean_lib", "levenshtein_normalized_lib",
    "rouge1_lib", "rouge2_lib", "rougeL_lib",
    "decode_ms_per_line", "inference_peak_memory_mb",
]


def write_results(results: dict[str, dict]) -> None:
    """Write results.csv / results.json / runtime_stats.json and regenerate the figures."""
    ds.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with (ds.OUTPUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=METRIC_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for name in sorted(results):
            writer.writerow(results[name])
    utils.save_json(results, ds.OUTPUT_DIR / "results.json")

    histories, runtime = {}, {}
    for name in sorted(results):
        path = ds.OUTPUT_DIR / f"history_{name}.json"
        if not path.exists():
            continue
        summary = json.loads(path.read_text(encoding="utf-8"))
        histories[name] = summary["history"]
        runtime[name] = {
            "params_millions": summary["params_millions"],
            "peak_memory_mb": summary["peak_memory_mb"],
            "sec_per_epoch": summary["sec_per_epoch"],
            "examples_per_sec": summary["examples_per_sec"],
            "epochs_run": summary["epochs_run"],
            "wall_seconds": summary["wall_seconds"],
            "best_val_loss": summary["best_val_loss"],
            "decode_ms_per_line": results[name]["decode_ms_per_line"],
            "inference_peak_memory_mb": results[name]["inference_peak_memory_mb"],
        }
    utils.save_json(runtime, ds.OUTPUT_DIR / "runtime_stats.json")

    if histories:
        utils.plot_loss_curves(histories, ds.OUTPUT_DIR / "loss_curves.png")
    utils.plot_metric_comparison(results, ds.OUTPUT_DIR / "metric_comparison.png")
    if runtime:
        utils.plot_memory_speed(runtime, ds.OUTPUT_DIR / "memory_speed.png")

    print("\n" + "=" * 104)
    print(f"{'Cfg':<5}{'Changed':<22}{'Params':>8}{'Bit%':>8}{'Char%':>8}{'Seq%':>7}"
          f"{'Lev':>9}{'BLEU':>8}{'R-L':>8}{'ms/line':>10}")
    print("-" * 104)
    for name in sorted(results):
        r = results[name]
        print(f"{r['config']:<5}{r['changed_from_base']:<22}{r['params_millions']:>8.2f}"
              f"{r['bit_accuracy']:>8.2f}{r['char_accuracy']:>8.2f}"
              f"{r['sequence_accuracy']:>7.2f}{r['levenshtein_mean']:>9.1f}"
              f"{r['bleu']:>8.2f}{r['rougeL']:>8.2f}{r['decode_ms_per_line']:>10.1f}")
    print("=" * 104)
    print(f"\nWrote results.csv, results.json, runtime_stats.json and 3 figures to "
          f"{ds.OUTPUT_DIR}")


# --- CLI -------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Train or evaluate the ablation configurations")
    parser.add_argument("--config", default="C1", help="C1 .. C5")
    parser.add_argument("--all", action="store_true", help="every configuration in sequence")
    parser.add_argument("--evaluate", action="store_true",
                        help="greedy-decode the test split instead of training")
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--lr", type=float)
    parser.add_argument("--smoke", action="store_true", help="run a few steps as a wiring check")
    parser.add_argument("--smoke-steps", type=int, default=50)
    parser.add_argument("--limit-lines", type=int, help="evaluate only the first N test lines")
    parser.add_argument("--no-wandb", action="store_true")
    parser.add_argument("--push", action="store_true", help="upload the checkpoint to HuggingFace")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    utils.load_env(ENV_FILE)
    utils.set_seed(ds.SEED)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB)")

    if args.evaluate:
        if args.all:
            names = [n for n in CONFIGS if (CKPT_DIR / n / "best.pt").exists()]
            if not names:
                raise SystemExit("No checkpoints found. Train first.")
            missing = [n for n in CONFIGS if n not in names]
            if missing:
                print(f"warning: no checkpoint for {', '.join(missing)} -- skipping")
        else:
            names = [args.config.upper()]
        write_results({n: evaluate_config(n, device, limit_lines=args.limit_lines)
                       for n in names})
        return

    train_cfg = TrainConfig()
    for attr, value in [("epochs", args.epochs), ("batch_size", args.batch_size),
                        ("lr", args.lr)]:
        if value is not None:
            setattr(train_cfg, attr, value)

    names = list(CONFIGS) if args.all else [args.config.upper()]
    summaries, results = {}, {}
    for name in names:
        summaries[name] = train_one(name, train_cfg, device, use_wandb=not args.no_wandb,
                                    smoke_steps=args.smoke_steps if args.smoke else 0,
                                    push=args.push)
        if not args.smoke:
            # Evaluate immediately so results.csv/json and the figures are current after
            # every config, not just at the end of the whole run -- lets a run be judged (and
            # stopped, if a config is clearly off) without waiting for --all to finish.
            results[name] = evaluate_config(name, device, limit_lines=args.limit_lines)
            write_results(results)

    if len(summaries) > 1:
        print("\n" + "=" * 78)
        print(f"{'Config':<8}{'Params(M)':>11}{'Best val':>11}{'Epochs':>9}"
              f"{'s/epoch':>10}{'Peak MiB':>11}")
        print("-" * 78)
        for name, s in summaries.items():
            print(f"{name:<8}{s['params_millions']:>11.2f}{s['best_val_loss']:>11.4f}"
                  f"{s['epochs_run']:>9}{s['sec_per_epoch']:>10.1f}{s['peak_memory_mb']:>11.0f}")


if __name__ == "__main__":
    main()
