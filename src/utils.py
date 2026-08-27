"""Metrics, plots, seeding and profiling.

bit accuracy      pred and gold expanded back to 8-bit ASCII and compared position by
                  position; length mismatch is charged against max(len_pred, len_gold).
character accuracy  the same at character granularity.
sequence accuracy   fraction of lines reconstructed exactly.
Levenshtein       edit distance, raw and normalised by gold length.
BLEU              sacreBLEU corpus BLEU.
ROUGE-1/2/L       implemented here (n-gram overlap F1, and LCS-based F1 for ROUGE-L).

Note that bit and character accuracy compare at fixed index, so a single inserted or deleted
character shifts everything after it and reads as wrong. Levenshtein, BLEU and ROUGE-L are
alignment-tolerant and carry the real signal; see the report.
"""

from __future__ import annotations

import json
import os
import random
import time
from collections import Counter
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


def set_seed(seed: int = 42, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def load_env(env_file: Path) -> dict[str, str]:
    """Load KEY=VALUE pairs from .env into os.environ (existing vars win)."""
    if not env_file.exists():
        return {}
    from dotenv import dotenv_values

    values = {k: v for k, v in dotenv_values(env_file).items() if v}
    for key, val in values.items():
        os.environ.setdefault(key, val)
    return values


# --- metrics -----------------------------------------------------------------------------


def text_to_bits(text: str) -> np.ndarray:
    """Expand a string into its flat 8-bit-per-character ASCII bit array, exactly as
    brown_cipher.txt was produced."""
    if not text:
        return np.zeros(0, dtype=np.uint8)
    raw = np.frombuffer(text.encode("latin-1", errors="replace"), dtype=np.uint8)
    return np.unpackbits(raw)


def bit_accuracy(pred: str, gold: str) -> tuple[int, int]:
    """(matching_bits, total_bits); totals are summed by the caller for a true micro-average."""
    p, g = text_to_bits(pred), text_to_bits(gold)
    total = max(p.size, g.size)
    if total == 0:
        return 0, 0
    n = min(p.size, g.size)
    return (int(np.count_nonzero(p[:n] == g[:n])) if n else 0), total


def char_accuracy(pred: str, gold: str) -> tuple[int, int]:
    total = max(len(pred), len(gold))
    if total == 0:
        return 0, 0
    return sum(a == b for a, b in zip(pred, gold)), total


def levenshtein(a: str, b: str) -> int:
    """Edit distance, vectorised row by row.

    Substitution and deletion vectorise directly. The insertion term
    cur[j] = min(cur[j], cur[j-1] + 1) is a running minimum:
    min_{k<=j}(cur[k] - k) + j, which np.minimum.accumulate does in one pass.
    """
    if a == b:
        return 0
    if not a or not b:
        return len(a) or len(b)

    b_arr = np.frombuffer(b.encode("latin-1", errors="replace"), dtype=np.uint8).astype(np.int32)
    idx = np.arange(b_arr.size + 1, dtype=np.int32)
    prev, cur = idx.copy(), np.empty(b_arr.size + 1, dtype=np.int32)

    for i, ch in enumerate(a, start=1):
        cur[0] = i
        cur[1:] = np.minimum(prev[:-1] + (b_arr != ord(ch)), prev[1:] + 1)
        cur = np.minimum.accumulate(cur - idx) + idx
        prev, cur = cur, prev
    return int(prev[b_arr.size])


def _ngrams(tokens: Sequence[str], n: int) -> Counter:
    return Counter(tuple(tokens[i:i + n]) for i in range(len(tokens) - n + 1))


def _f1(overlap: int, n_pred: int, n_gold: int) -> float:
    if not (overlap and n_pred and n_gold):
        return 0.0
    p, r = overlap / n_pred, overlap / n_gold
    return 2 * p * r / (p + r)


def rouge_n(pred: str, gold: str, n: int = 1) -> float:
    p, g = _ngrams(pred.split(), n), _ngrams(gold.split(), n)
    return _f1(sum((p & g).values()), sum(p.values()), sum(g.values()))   # & clips counts


def _lcs_length(a: Sequence[str], b: Sequence[str]) -> int:
    if not a or not b:
        return 0
    prev = [0] * (len(b) + 1)
    for token in a:
        cur = [0] * (len(b) + 1)
        for j, other in enumerate(b, start=1):
            cur[j] = prev[j - 1] + 1 if token == other else max(prev[j], cur[j - 1])
        prev = cur
    return prev[len(b)]


def rouge_l(pred: str, gold: str) -> float:
    p, g = pred.split(), gold.split()
    return _f1(_lcs_length(p, g), len(p), len(g))


def corpus_bleu(preds: Sequence[str], golds: Sequence[str]) -> float:
    import sacrebleu

    return float(sacrebleu.corpus_bleu(list(preds), [list(golds)]).score)


# --- library metrics (cross-check) ---------------------------------------------------------
# The hand-rolled Levenshtein and ROUGE above are the reported numbers; these reference
# implementations are computed alongside them under a `_lib` suffix so the two can be compared.
#
# Levenshtein: rapidfuzz computes the same quantity, so `levenshtein_mean_lib` should match
# `levenshtein_mean` exactly.
#
# ROUGE: rouge_score lowercases before tokenizing, and the hand-rolled version does not. On
# this corpus the alphabet is a-z, A-Z and space only (no punctuation or digits), so its
# default tokenizer is otherwise equivalent to str.split(); stemming is disabled for the same
# reason the task is scored on exact reconstruction. The gap between `rougeL` and
# `rougeL_lib` is therefore the credit the library gives back for capitalization errors.


def levenshtein_lib(a: str, b: str) -> int:
    from rapidfuzz.distance import Levenshtein as _Levenshtein

    return int(_Levenshtein.distance(a, b))


def _rouge_scorer():
    """Cached scorer; constructing one per line dominates the scoring time."""
    global _ROUGE_SCORER
    if _ROUGE_SCORER is None:
        from rouge_score import rouge_scorer

        _ROUGE_SCORER = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"],
                                                 use_stemmer=False)
    return _ROUGE_SCORER


_ROUGE_SCORER = None


def rouge_lib(pred: str, gold: str) -> dict[str, float]:
    """{'rouge1': f1, 'rouge2': f1, 'rougeL': f1} from the reference implementation."""
    scores = _rouge_scorer().score(gold, pred)          # (target, prediction) order
    return {key: value.fmeasure for key, value in scores.items()}


def compute_all_metrics(preds: Sequence[str], golds: Sequence[str],
                        with_library: bool = True) -> dict[str, float]:
    """Hand-rolled metrics, plus the `_lib` reference values when `with_library`."""
    assert len(preds) == len(golds), "prediction/gold count mismatch"
    bit_m = bit_t = char_m = char_t = exact = lev = 0
    lev_norm = r1 = r2 = rl = 0.0
    lev_lib = 0
    lev_norm_lib = r1_lib = r2_lib = rl_lib = 0.0

    for pred, gold in zip(preds, golds):
        bm, bt = bit_accuracy(pred, gold)
        cm, ct = char_accuracy(pred, gold)
        bit_m, bit_t, char_m, char_t = bit_m + bm, bit_t + bt, char_m + cm, char_t + ct
        exact += pred == gold
        dist = levenshtein(pred, gold)
        lev += dist
        lev_norm += dist / max(len(gold), 1)
        r1 += rouge_n(pred, gold, 1)
        r2 += rouge_n(pred, gold, 2)
        rl += rouge_l(pred, gold)

        if with_library:
            dist_lib = levenshtein_lib(pred, gold)
            lev_lib += dist_lib
            lev_norm_lib += dist_lib / max(len(gold), 1)
            lib = rouge_lib(pred, gold)
            r1_lib += lib["rouge1"]
            r2_lib += lib["rouge2"]
            rl_lib += lib["rougeL"]

    n = max(len(preds), 1)
    metrics = {
        "bit_accuracy": 100.0 * bit_m / max(bit_t, 1),
        "char_accuracy": 100.0 * char_m / max(char_t, 1),
        "sequence_accuracy": 100.0 * exact / n,
        "levenshtein_mean": lev / n,
        "levenshtein_normalized": lev_norm / n,
        "bleu": corpus_bleu(preds, golds),
        "rouge1": 100.0 * r1 / n,
        "rouge2": 100.0 * r2 / n,
        "rougeL": 100.0 * rl / n,
    }
    if with_library:
        metrics.update({
            "levenshtein_mean_lib": lev_lib / n,
            "levenshtein_normalized_lib": lev_norm_lib / n,
            "rouge1_lib": 100.0 * r1_lib / n,
            "rouge2_lib": 100.0 * r2_lib / n,
            "rougeL_lib": 100.0 * rl_lib / n,
        })
    return metrics


# --- profiling ---------------------------------------------------------------------------


def count_parameters(model: torch.nn.Module) -> tuple[int, int]:
    return (sum(p.numel() for p in model.parameters()),
            sum(p.numel() for p in model.parameters() if p.requires_grad))


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    """Peak allocated CUDA memory since the last reset, in MiB (0 on CPU)."""
    return torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0


class Timer:
    """`with Timer() as t: ...` -> t.elapsed. Synchronises CUDA so GPU work is counted."""

    def __enter__(self):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.elapsed = time.perf_counter() - self.start
        return False


def human_time(seconds: float) -> str:
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h:d}h{m:02d}m{s:02d}s" if h else f"{m:d}m{s:02d}s"


# --- plots -------------------------------------------------------------------------------

CONFIG_COLORS = {"C1": "#4C72B0", "C2": "#DD8452", "C3": "#55A868",
                 "C4": "#C44E52", "C5": "#8172B3"}


def _style():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"figure.dpi": 150, "font.size": 10, "axes.grid": True,
                         "grid.alpha": 0.3, "axes.spines.top": False,
                         "axes.spines.right": False})
    return plt


def _bar_panels(data: dict[str, dict], panels, path: Path, figsize, nrows=1):
    plt = _style()
    names = sorted(data)
    fig, axes = plt.subplots(nrows, len(panels) // nrows, figsize=figsize)
    for ax, (key, title) in zip(np.atleast_1d(axes).flat, panels):
        values = [data[n].get(key, 0.0) for n in names]
        ax.bar(names, values, color=[CONFIG_COLORS.get(n, "#888") for n in names])
        ax.set_title(title, fontsize=10)
        for i, v in enumerate(values):
            ax.text(i, v, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.margins(y=0.18)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_loss_curves(histories: dict[str, dict], path: Path) -> None:
    plt = _style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for name, hist in sorted(histories.items()):
        color = CONFIG_COLORS.get(name)
        axes[0].plot(range(1, len(hist["train_loss"]) + 1), hist["train_loss"], label=name, color=color)
        axes[1].plot(range(1, len(hist["val_loss"]) + 1), hist["val_loss"], label=name, color=color)
    for ax, title in zip(axes, ("Training loss", "Validation loss")):
        ax.set_title(title)
        ax.set_xlabel("epoch")
        ax.set_ylabel("cross-entropy")
        ax.legend()
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_metric_comparison(results: dict[str, dict], path: Path) -> None:
    _bar_panels(results, [
        ("bit_accuracy", "Bit accuracy (%)"),
        ("char_accuracy", "Character accuracy (%)"),
        ("sequence_accuracy", "Sequence accuracy (%)"),
        ("levenshtein_normalized", "Normalised Levenshtein (lower better)"),
        ("bleu", "BLEU"),
        ("rougeL", "ROUGE-L"),
    ], path, figsize=(13, 7), nrows=2)


def plot_memory_speed(stats: dict[str, dict], path: Path) -> None:
    """Throughput is plotted as examples/second, never tokens/second: a "token" is a BPE
    subword for C1-C4 but a raw byte for C5, so a tok/s chart makes C5 look fastest when it
    is in fact the slowest."""
    _bar_panels(stats, [
        ("peak_memory_mb", "Peak GPU memory, training (MiB)"),
        ("inference_peak_memory_mb", "Peak GPU memory, inference (MiB)"),
        ("examples_per_sec", "Training throughput (examples/s)"),
        ("params_millions", "Parameters (M)"),
    ], path, figsize=(15, 3.6))


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


if __name__ == "__main__":
    assert "".join(map(str, text_to_bits("A"))) == format(ord("A"), "08b")
    assert bit_accuracy("A", "C") == (7, 8)          # differ in exactly one bit
    assert bit_accuracy("abc", "abc") == (24, 24)
    assert bit_accuracy("ab", "abcd") == (16, 32)    # charged against the longer
    assert char_accuracy("abc", "abd") == (2, 3)

    for a, b, want in [("kitten", "sitting", 3), ("flaw", "lawn", 2), ("", "abc", 3),
                       ("abc", "", 3), ("same", "same", 0), ("sunday", "saturday", 3)]:
        assert levenshtein(a, b) == want, (a, b)

    def _reference(a, b):
        prev = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            cur = [i]
            for j, cb in enumerate(b, 1):
                cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
            prev = cur
        return prev[-1]

    rng = random.Random(0)
    for _ in range(200):
        a = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 12)))
        b = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 12)))
        assert levenshtein(a, b) == _reference(a, b), (a, b)
    print("Levenshtein matches an O(nm) reference on 200 random pairs")

    assert abs(rouge_n("the cat sat", "the cat sat", 1) - 1.0) < 1e-9
    assert rouge_n("a b c", "x y z", 1) == 0.0
    assert _lcs_length("the cat sat on the mat".split(), "the cat on the mat".split()) == 5

    out = compute_all_metrics(["hello world", "abc"], ["hello world", "abd"],
                              with_library=False)
    assert abs(out["sequence_accuracy"] - 50.0) < 1e-9 and out["bit_accuracy"] > 90.0

    # The library implementations are optional; skip the cross-check if they are absent.
    try:
        levenshtein_lib("a", "b")
        rouge_lib("a", "b")
    except ImportError:
        print("rapidfuzz / rouge-score not installed -- skipping the library cross-check")
    else:
        for _ in range(200):
            a = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 12)))
            b = "".join(rng.choice("abcde") for _ in range(rng.randint(0, 12)))
            assert levenshtein(a, b) == levenshtein_lib(a, b), (a, b)
        print("hand-rolled Levenshtein matches rapidfuzz on 200 random pairs")

        # Same-case input is the one setting where the two ROUGE paths must agree: the corpus
        # alphabet has no punctuation, so the only remaining difference is lowercasing.
        for pred, gold in [("the cat sat on the mat", "the cat sat on the mat"),
                           ("the cat sat on the mat", "the cat on the mat"),
                           ("a b c d", "a x c y")]:
            lib = rouge_lib(pred, gold)
            assert abs(rouge_n(pred, gold, 1) - lib["rouge1"]) < 1e-6, (pred, gold)
            assert abs(rouge_n(pred, gold, 2) - lib["rouge2"]) < 1e-6, (pred, gold)
            assert abs(rouge_l(pred, gold) - lib["rougeL"]) < 1e-6, (pred, gold)
        print("hand-rolled ROUGE-1/2/L matches rouge_score on lowercase input")

        # ... and the one setting where they must differ, which is what the _lib gap measures.
        assert rouge_n("The Cat", "the cat", 1) == 0.0
        assert abs(rouge_lib("The Cat", "the cat")["rouge1"] - 1.0) < 1e-6
        print("case sensitivity is the only ROUGE divergence (hand-rolled 0.0 vs library 1.0)")

        out = compute_all_metrics(["hello world", "abc"], ["hello world", "abd"])
        assert abs(out["levenshtein_mean"] - out["levenshtein_mean_lib"]) < 1e-9

    print("utils.py self-test passed")
    for k, v in out.items():
        print(f"  {k:<28} {v:.4f}")
