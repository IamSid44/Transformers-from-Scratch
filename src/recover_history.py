"""Rebuild `outputs/history_<C>.json` from a training log.

`train_one` now writes that file after every epoch, so this is only needed for a run that
predates that change and was stopped on a wall-clock budget before its single end-of-run write
-- which is exactly what happened to C5. The per-epoch line it parses carries everything the
JSON holds except the model config, which comes from the checkpoint:

      epoch  1/40  train 2.6798  val 0.7038  ppl 2.02  tf 1.00  src +54.9pt  16m30s  78 ex/s  10033 MiB

    python src/recover_history.py --config C5
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))

import dataset as ds
import utils

EPOCH_RE = re.compile(
    r"^\s*epoch\s+(?P<epoch>\d+)/(?P<total>\d+)\s+"
    r"train\s+(?P<train>[\d.]+)\s+val\s+(?P<val>[\d.]+)\s+ppl\s+[\d.]+\s+"
    r"tf\s+(?P<tf>[\d.]+)\s+src\s+(?P<src>[+-][\d.]+)pt\s+"
    r"(?P<time>[\dhms]+)\s+(?P<eps>[\d.]+)\s+ex/s\s+(?P<mem>[\d.]+)\s+MiB")
PARAM_RE = re.compile(r"parameters\s+([\d.]+)M")
LR_RE = re.compile(r"train/lr")


def parse_duration(text: str) -> float:
    """'16m30s' / '1h02m03s' / '45s' -> seconds."""
    total, number = 0.0, ""
    for ch in text:
        if ch.isdigit() or ch == ".":
            number += ch
        else:
            total += float(number) * {"h": 3600, "m": 60, "s": 1}[ch]
            number = ""
    return total


def recover(config_name: str, log_path: Path) -> dict:
    history = {"train_loss": [], "val_loss": [], "epoch_seconds": [], "examples_per_sec": [],
               "peak_memory_mb": [], "lr": [], "teacher_forcing": [], "source_gap": []}
    params_millions = None
    for line in log_path.read_text(encoding="utf-8", errors="replace").splitlines():
        found = PARAM_RE.search(line)
        if found:
            params_millions = float(found.group(1))
        m = EPOCH_RE.match(line)
        if not m:
            continue
        history["train_loss"].append(float(m["train"]))
        history["val_loss"].append(float(m["val"]))
        history["epoch_seconds"].append(parse_duration(m["time"]))
        history["examples_per_sec"].append(float(m["eps"]))
        history["peak_memory_mb"].append(float(m["mem"]))
        history["teacher_forcing"].append(float(m["tf"]))
        history["source_gap"].append(float(m["src"]))
        history["lr"].append(float("nan"))     # not printed per epoch; read it off WandB

    if not history["val_loss"]:
        raise SystemExit(f"No epoch lines found in {log_path}")

    ckpt_path = ds.OUTPUT_DIR / "checkpoints" / config_name / "best.pt"
    model_config, best_epoch, best_val = {}, None, min(history["val_loss"])
    if ckpt_path.exists():
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_config = ckpt.get("model_config", {})
        best_epoch, best_val = ckpt.get("epoch"), ckpt.get("val_loss", best_val)
    if best_epoch is None:
        best_epoch = 1 + history["val_loss"].index(best_val)

    n = len(history["train_loss"])
    summary = {
        "config": config_name,
        "params": int((params_millions or 0) * 1e6),
        "params_millions": params_millions,
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "epochs_run": n,
        "wall_seconds": sum(history["epoch_seconds"]),
        "sec_per_epoch": sum(history["epoch_seconds"]) / n,
        "examples_per_sec": sum(history["examples_per_sec"]) / n,
        "peak_memory_mb": max(history["peak_memory_mb"]),
        "history": history,
        "model_config": model_config,
        "recovered_from_log": str(log_path),
        # The run did not reach its epoch budget; say so rather than letting a reader assume
        # the curve ended because it converged.
        "stopped_early_on_time_budget": True,
    }
    out = ds.OUTPUT_DIR / f"history_{config_name}.json"
    utils.save_json(summary, out)
    print(f"recovered {n} epochs from {log_path} -> {out}")
    print(f"  best val {best_val:.4f} @ epoch {best_epoch}; "
          f"{summary['sec_per_epoch']:.0f}s/epoch; final source gap "
          f"{history['source_gap'][-1]:+.1f}pt")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="C5")
    parser.add_argument("--log", type=Path)
    args = parser.parse_args()
    name = args.config.upper()
    recover(name, args.log or ds.PROJECT_ROOT / "logs" / f"train_{name}.log")


if __name__ == "__main__":
    main()
