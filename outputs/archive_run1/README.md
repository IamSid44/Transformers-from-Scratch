# Run 1 archive (submitted-as-fallback results)

Everything the first round of training produced, kept so the assignment can be submitted from
it if the second round does not finish or does not improve on it.

| File | What it is |
|---|---|
| `results.csv`, `results.json` | test-set metrics for C1-C5, whole-line granularity |
| `history_C1..C4.json` | per-epoch train/val loss, epoch time, peak memory |
| `samples_C1..C5.txt` | greedy-decoding samples, 8 test lines each |
| `loss_curves.png`, `metric_comparison.png`, `memory_speed.png` | the three figures |
| `runtime_stats.json` | params, s/epoch, peak memory, decode latency |
| `tokenizers/` | the BPE tokenizers trained at CHUNK_CHARS = 64 |
| `logs/` | `train_all.log` (C1-C4) and `train_C5_bitsource.log` |
| `hf_revisions.json` | HuggingFace commit SHAs holding run 1's checkpoints |

## Configuration that produced these

CHUNK_CHARS = 64, batch size 1024, 60 epochs, lr 1e-3, scheduled sampling to a 0.7 floor,
C5 with **fixed** stride-4 patches (no entropy model). 38 optimiser steps per epoch, 2,280 in
total. Every one of the five runs was still descending at its final epoch (`best_epoch == 60`
for all five), so these numbers measure convergence speed, not converged quality.

## Two things are NOT here

* `history_C5.json` and `logs/train_C5.log` from run 1 were untracked and were overwritten
  before this archive was made. C5's run-1 *metrics* survive in `results.csv` and its decoded
  output in `samples_C5.txt`; only its per-epoch loss curve is gone.
* Run-1 checkpoints were overwritten on local disk, but they are still on HuggingFace: the
  new `--push` adds a commit rather than replacing history, so each repo's run-1 state is the
  revision pinned in `hf_revisions.json`, reachable at
  `https://huggingface.co/<repo>/tree/<sha>`.
