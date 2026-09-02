"""Isolated cost measurement for the C1-C5 comparison: seconds/step, peak training memory,
greedy-decode latency and peak inference memory.

Why this exists as a separate pass. The five configurations are *trained* concurrently, which
is the only way to fit the ablation into the available window -- but that makes every cost
number recorded during training worthless for comparison. `torch.cuda.max_memory_allocated` is
a per-process high-water mark and wall-clock throughput on a shared GPU measures the other
tenants as much as the model. So the accuracy numbers come from the real training runs, and
every number in the report's *cost* column comes from here instead: one configuration at a
time, one process each (`--config`), nothing else of ours on the device.

    python src/profile_cost.py --all          # spawns one subprocess per configuration
    python src/profile_cost.py --config C3    # the measurement itself (what the above runs)

Writes outputs/profile.json.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))

import dataset as ds
import train as T
import utils

PROFILE_PATH = ds.OUTPUT_DIR / "profile.json"

WARMUP_STEPS = 5          # discarded: the first steps pay cuDNN autotuning and allocator growth
TRAIN_STEPS = 40          # timed training steps
DECODE_BATCHES = 8        # timed greedy-decoding batches


def profile_one(config_name: str, device) -> dict:
    model_cfg = T.get_config(config_name)
    utils.set_seed(ds.SEED)
    train_cfg = T.TrainConfig()

    data = ds.make_dataloaders(model_cfg, train_cfg)
    meta = dict(data["meta"])
    model = T.build_model(model_cfg, meta).to(device)
    total_params, _ = utils.count_parameters(model)
    opt = torch.optim.Adam(model.parameters(), lr=train_cfg.lr)

    # --- training cost ------------------------------------------------------------------
    model.train()
    batches = []
    for batch in data["loaders"]["train"]:
        batches.append({k: v.to(device) for k, v in batch.items()})
        if len(batches) >= WARMUP_STEPS + TRAIN_STEPS:
            break

    def step(batch):
        loss, n = T.compute_loss(model_cfg, model, batch, train_cfg.label_smoothing, 1.0)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
        opt.step()
        opt.zero_grad(set_to_none=True)
        return n

    for batch in batches[:WARMUP_STEPS]:
        step(batch)
    torch.cuda.synchronize(device)
    utils.reset_peak_memory()

    start = time.perf_counter()
    examples = 0
    for batch in batches[WARMUP_STEPS:]:
        step(batch)
        examples += batch["src"].size(0)
    torch.cuda.synchronize(device)
    train_seconds = time.perf_counter() - start
    train_peak = utils.peak_memory_mb()

    steps_per_epoch = len(data["loaders"]["train"])
    timed = len(batches) - WARMUP_STEPS

    # --- inference cost -----------------------------------------------------------------
    model.eval()
    decode_batches = []
    for batch in data["loaders"]["test"]:
        decode_batches.append({k: v.to(device) for k, v in batch.items()})
        if len(decode_batches) >= DECODE_BATCHES + 1:
            break

    with torch.no_grad():
        first = decode_batches[0]                       # warmup, not timed
        if model_cfg.is_blt:
            model.greedy_decode(first["src"], first)
        else:
            model.greedy_decode(first["src"], max_len=meta["max_tgt_len"],
                                bos_id=ds.BOS_ID, eos_id=ds.EOS_ID)
        torch.cuda.synchronize(device)
        utils.reset_peak_memory()

        start = time.perf_counter()
        chunks = 0
        for batch in decode_batches[1:]:
            if model_cfg.is_blt:
                model.greedy_decode(batch["src"], batch)
            else:
                model.greedy_decode(batch["src"], max_len=meta["max_tgt_len"],
                                    bos_id=ds.BOS_ID, eos_id=ds.EOS_ID)
            chunks += batch["src"].size(0)
        torch.cuda.synchronize(device)
        decode_seconds = time.perf_counter() - start
    decode_peak = utils.peak_memory_mb()

    result = {
        "config": config_name,
        "params_millions": total_params / 1e6,
        "sec_per_step": train_seconds / timed,
        "sec_per_epoch_projected": train_seconds / timed * steps_per_epoch,
        "train_examples_per_sec": examples / train_seconds,
        "train_peak_memory_mb": train_peak,
        "decode_ms_per_chunk": 1000.0 * decode_seconds / max(chunks, 1),
        "decode_chunks_per_sec": chunks / decode_seconds,
        "decode_peak_memory_mb": decode_peak,
        "steps_per_epoch": steps_per_epoch,
        "batch_size": train_cfg.batch_size,
    }
    if model_cfg.is_blt:
        result["mean_patch_size"] = meta.get("mean_patch_size")
        result["max_patch_size"] = meta.get("max_patch_size")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="C1")
    parser.add_argument("--all", action="store_true",
                        help="run every configuration, each in its own subprocess")
    args = parser.parse_args()

    if args.all:
        merged = (json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
                  if PROFILE_PATH.exists() else {})
        for name in T.CONFIGS:
            print(f"\n=== profiling {name} (isolated process) ===")
            subprocess.run([sys.executable, "-u", str(Path(__file__)), "--config", name],
                           check=True)
            merged = json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
        print(f"\n{'Cfg':<5}{'Params(M)':>11}{'s/step':>9}{'s/epoch':>10}{'Train MiB':>12}"
              f"{'ms/chunk':>11}{'Infer MiB':>12}")
        print("-" * 70)
        for name in sorted(merged):
            r = merged[name]
            print(f"{name:<5}{r['params_millions']:>11.2f}{r['sec_per_step']:>9.3f}"
                  f"{r['sec_per_epoch_projected']:>10.1f}{r['train_peak_memory_mb']:>12.0f}"
                  f"{r['decode_ms_per_chunk']:>11.2f}{r['decode_peak_memory_mb']:>12.0f}")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    result = profile_one(args.config.upper(), device)
    # Merge rather than overwrite: each configuration is measured in its own process.
    existing = (json.loads(PROFILE_PATH.read_text(encoding="utf-8"))
                if PROFILE_PATH.exists() else {})
    existing[result["config"]] = result
    utils.save_json(existing, PROFILE_PATH)
    print(f"[{result['config']}] {result['sec_per_step']:.3f} s/step  "
          f"{result['sec_per_epoch_projected']:.0f} s/epoch  "
          f"{result['train_peak_memory_mb']:.0f} MiB train  "
          f"{result['decode_ms_per_chunk']:.2f} ms/chunk  "
          f"{result['decode_peak_memory_mb']:.0f} MiB decode")


if __name__ == "__main__":
    main()
