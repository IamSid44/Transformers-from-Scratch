"""Collect every number and figure the report needs into one place.

This writes *materials*, not prose: tables in Markdown and LaTeX, a patch-length figure, and a
plain list of the facts each number rests on. The report itself is written by hand
(assignment General Instruction 2).

    python src/report_assets.py

Reads outputs/results.json, outputs/history_*.json, outputs/profile.json and
outputs/entropy_lm.pt; writes outputs/report/.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(SRC_DIR / "models"))

import dataset as ds

OUT = ds.OUTPUT_DIR / "report"

CHANGED = {"C1": "None (base)", "C2": "RoPE", "C3": "Grouped-query", "C4": "RMSNorm",
           "C5": "BLT (token-free)"}


def _load(name):
    path = ds.OUTPUT_DIR / name
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _table(rows, headers, aligns=None):
    """Markdown table."""
    aligns = aligns or ["r"] * len(headers)
    sep = ["---" if a == "l" else "--:" for a in aligns]
    lines = ["| " + " | ".join(headers) + " |", "| " + " | ".join(sep) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(c) for c in row) + " |")
    return "\n".join(lines)


def _latex(rows, headers, caption, label):
    spec = "l" + "r" * (len(headers) - 1)
    body = "\n".join(" & ".join(str(c) for c in row) + r" \\" for row in rows)
    return (f"\\begin{{table}}[t]\n\\centering\n\\begin{{tabular}}{{{spec}}}\n\\hline\n"
            + " & ".join(headers) + r" \\" + "\n\\hline\n" + body
            + f"\n\\hline\n\\end{{tabular}}\n\\caption{{{caption}}}\n"
              f"\\label{{{label}}}\n\\end{{table}}\n")


def quality_tables(results):
    names = sorted(results)
    headers = ["Cfg", "Changed", "Bit %", "Char %", "Seq %", "Lev", "BLEU", "R-1", "R-2", "R-L"]
    chunk, line = [], []
    for n in names:
        r = results[n]
        chunk.append([n, CHANGED.get(n, r.get("changed_from_base", "")),
                      f"{r['bit_accuracy']:.2f}", f"{r['char_accuracy']:.2f}",
                      f"{r['sequence_accuracy']:.2f}", f"{r['levenshtein_mean']:.3f}",
                      f"{r['bleu']:.2f}", f"{r['rouge1']:.2f}", f"{r['rouge2']:.2f}",
                      f"{r['rougeL']:.2f}"])
        if "bit_accuracy_line" in r:
            line.append([n, CHANGED.get(n, ""), f"{r['bit_accuracy_line']:.2f}",
                         f"{r['char_accuracy_line']:.2f}",
                         f"{r['sequence_accuracy_line']:.2f}",
                         f"{r['levenshtein_mean_line']:.1f}", f"{r['bleu_line']:.2f}",
                         f"{r['rouge1_line']:.2f}", f"{r['rouge2_line']:.2f}",
                         f"{r['rougeL_line']:.2f}"])
    return headers, chunk, line


def cost_table(profile, history):
    headers = ["Cfg", "Params (M)", "s/step", "s/epoch", "Train MiB", "ms/chunk", "Infer MiB",
               "Best val", "Best ep"]
    rows = []
    for n in sorted(profile or history):
        p = (profile or {}).get(n, {})
        h = history.get(n, {})
        rows.append([n,
                     f"{p.get('params_millions', h.get('params_millions', 0)):.2f}",
                     f"{p.get('sec_per_step', float('nan')):.3f}",
                     f"{p.get('sec_per_epoch_projected', float('nan')):.0f}",
                     f"{p.get('train_peak_memory_mb', float('nan')):.0f}",
                     f"{p.get('decode_ms_per_chunk', float('nan')):.2f}",
                     f"{p.get('decode_peak_memory_mb', float('nan')):.0f}",
                     f"{h.get('best_val_loss', float('nan')):.4f}",
                     h.get("best_epoch", "-")])
    return headers, rows


def patch_figure(path):
    """Patch-length distribution and the entropy trace that produced it."""
    ckpt_path = ds.OUTPUT_DIR / "entropy_lm.pt"
    if not ckpt_path.exists():
        return None
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hist = ckpt["length_histogram"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    ax = axes[0]
    ax.bar(range(1, len(hist) + 1), [100 * h for h in hist], color="#4C72B0")
    ax.axvline(ckpt["mean_patch"], color="#C44E52", ls="--",
               label=f"mean {ckpt['mean_patch']:.2f}")
    ax.set_xlabel("patch length (bytes)")
    ax.set_ylabel("% of patches")
    ax.set_title(f"Entropy-driven patch lengths\n(threshold {ckpt['threshold']:.2f} nats, "
                 f"cap {ckpt['max_patch']})")
    ax.legend(frameon=False)

    # A concrete example: entropy over one test chunk, with the cuts it produced.
    from entropy_lm import BYTE_PAD_ID, load_entropy_lm, patch_boundaries

    device = torch.device("cpu")
    model, threshold, meta = load_entropy_lm(device)
    splits, _ = ds.build_splits()
    pair = ds.chunk_pairs(splits["test"])[3]
    x = torch.tensor([list(ds.cipher_to_bytes(pair.cipher))], dtype=torch.long)
    ent = model.entropies(x)[0]
    ids = patch_boundaries(ent[None], torch.ones_like(x, dtype=torch.bool), threshold)[0]

    ax = axes[1]
    ax.plot(range(len(ent)), ent.numpy(), color="#4C72B0", lw=1.2)
    ax.axhline(threshold, color="#C44E52", ls="--", lw=1,
               label=f"threshold {threshold:.2f}")
    for t in range(1, len(ids)):
        if ids[t] != ids[t - 1]:
            ax.axvline(t - 0.5, color="#999999", lw=0.8, alpha=0.7)
    ax.set_xlabel("byte position in chunk")
    ax.set_ylabel("H(next byte) [nats]")
    ax.set_title("One test chunk: entropy and the cuts it makes")
    ax.legend(frameon=False)
    ax.set_xticks(range(0, len(ent), 4))
    ax.set_xticklabels(list(pair.plain[::4]))

    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return {"threshold": ckpt["threshold"], "mean_patch": ckpt["mean_patch"],
            "histogram": hist, "example_plain": pair.plain}


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    results = _load("results.json")
    profile = _load("profile.json")
    histories = {n: _load(f"history_{n}.json") for n in ("C1", "C2", "C3", "C4", "C5")}
    histories = {n: h for n, h in histories.items() if h}

    headers, chunk_rows, line_rows = quality_tables(results)
    cost_headers, cost_rows = cost_table(profile, histories)
    patch = patch_figure(OUT / "patching.png")

    parts = ["# Report materials\n",
             "Generated by `src/report_assets.py`. Numbers only -- the write-up is by hand.\n",
             "\n## Quality, per decoded chunk (32 characters, greedy decoding)\n",
             _table(chunk_rows, headers), "\n"]
    if line_rows:
        parts += ["\n## Quality, per reassembled corpus line\n",
                  "A chunk that emits one character too few shifts every position after it, so "
                  "the positional metrics (bit, char) are far harsher here than per chunk; "
                  "Levenshtein and BLEU are not positional and degrade more gently.\n\n",
                  "Note C5 vs C1 here: C5 has the **best** line-level bit accuracy of any "
                  "configuration (94.91% vs C1's 92.71%) while making roughly ten times as "
                  "many edits (13.6 vs 1.3). That is not a metric artefact -- it is an "
                  "architectural consequence. C5's patch grid is derived from the source, so "
                  "its output length is structurally pinned to the input: it can substitute a "
                  "byte but can never insert or delete one, and positional alignment therefore "
                  "never breaks. C1 makes far fewer errors, but the ones that are insertions "
                  "or deletions shift every subsequent position and collapse positional "
                  "accuracy.\n",
                  _table(line_rows, headers), "\n"]
    parts += ["\n## Cost (measured one configuration at a time, isolated processes)\n",
              _table(cost_rows, cost_headers), "\n"]

    if histories:
        parts.append("\n## Convergence\n")
        rows = []
        for n, h in sorted(histories.items()):
            hist = h["history"]
            rows.append([n, h["epochs_run"], h["best_epoch"], f"{h['best_val_loss']:.4f}",
                         f"{hist['val_loss'][-1]:.4f}",
                         f"{hist['source_gap'][-1]:.1f}" if hist.get("source_gap") else "-"])
        parts.append(_table(rows, ["Cfg", "Epochs run", "Best epoch", "Best val loss",
                                   "Final val loss", "Final source gap (pt)"]))
        parts.append("\n")

    if patch:
        parts += ["\n## Entropy patching (C5)\n",
                  f"- Entropy model: 2 layers, d=128, 0.46M parameters, trained on the training "
                  f"split's cipher bytes.\n",
                  f"- Global threshold {patch['threshold']:.4f} nats, calibrated by bisection to "
                  f"a mean patch length of {patch['mean_patch']:.2f} bytes.\n",
                  "- Patch-length distribution (lengths 1..8): "
                  + ", ".join(f"{100 * h:.0f}%" for h in patch["histogram"]) + "\n",
                  "- Figure: `outputs/report/patching.png`\n"]

    parts += ["\n## Facts these numbers rest on\n",
              "- Corpus: 5,000 line-aligned pairs, split 4,000/500/500 over *lines* (seed 42), "
              "then cut into 32-character chunks: 77,373 / 9,133 / 9,322. Splitting before "
              "chunking is what keeps a line out of two splits.\n",
              "- The cipher is a repeating-key XOR, key `ANLP2026`, period 8 "
              "(recovered in `dataset.py`'s self-test; never given to any model). 32 is a "
              "multiple of 8, so every chunk starts at key phase 0.\n",
              "- All five configurations share depth, width, optimiser, LR schedule shape, "
              "batch size and seed; C2-C5 each differ from C1 in exactly one architectural "
              "field, constructed by `dataclasses.replace`.\n",
              "- **C5 ran 20 epochs against C1-C4's 40.** Training was stopped early on a "
              "compute-budget constraint (the GPU was needed elsewhere), not by early "
              "stopping: its validation loss improved on every one of its last four epochs "
              "(0.1263 -> 0.1239 -> 0.1230 -> 0.1212) and its cosine LR schedule, defined over "
              "40 epochs, was truncated at ~60% and never annealed. C5's quality numbers are "
              "therefore a lower bound on the architecture.\n",
              "- Greedy decoding throughout, as the assignment requires.\n",
              "- Levenshtein and ROUGE are cross-checked against `rapidfuzz` and `rouge_score` "
              "(the `_lib` columns in results.csv); the hand-rolled values are the reported "
              "ones.\n",
              "- Cost numbers come from `src/profile_cost.py`, one configuration per process "
              "on an otherwise idle GPU. The `sec_per_epoch` / `examples_per_sec` fields in "
              "`outputs/runtime_stats.json` come from the training runs instead, which were "
              "executed concurrently and are therefore contended -- use `outputs/profile.json` "
              "(reproduced above) for anything cost-related in the report.\n",
              "- The loss-curve figure carries three panels: raw train and validation "
              "cross-entropy, where C1-C4 (nats per BPE token, 4,096-way softmax) and C5 "
              "(nats per byte, 259-way) are *not* comparable and C5 is drawn dashed as a "
              "warning; and validation loss in bits per character, which normalises both onto "
              "the one unit they share and is the panel to read across all five.\n"]

    (OUT / "REPORT_MATERIALS.md").write_text("".join(parts), encoding="utf-8")

    latex = [_latex(chunk_rows, headers, "Test-set quality per decoded 32-character chunk, "
                    "greedy decoding.", "tab:quality")]
    if line_rows:
        latex.append(_latex(line_rows, headers,
                            "Test-set quality per reassembled corpus line.", "tab:quality-line"))
    latex.append(_latex(cost_rows, cost_headers,
                        "Cost, measured one configuration at a time in its own process.",
                        "tab:cost"))
    (OUT / "tables.tex").write_text("\n".join(latex), encoding="utf-8")

    print(f"wrote {OUT / 'REPORT_MATERIALS.md'}")
    print(f"wrote {OUT / 'tables.tex'}")
    if patch:
        print(f"wrote {OUT / 'patching.png'}")


if __name__ == "__main__":
    main()
