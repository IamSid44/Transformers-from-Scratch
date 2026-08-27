# Metrics

Every number reported in `outputs/results.csv` is produced by
[`utils.compute_all_metrics`](src/utils.py#L185), called once per configuration from
[`train.evaluate_config`](src/train.py#L468). It receives two equal-length lists of strings —
the predicted lines and the gold plaintext lines — and returns a flat `dict`.

One example is one whole corpus line, so `decode_split` emits exactly one prediction string
per line and scoring compares it against that line's gold text directly. Val and test loaders
never shuffle, which is what keeps predictions positionally aligned with their `Pair` list.

---

## 1. The two families

The metrics split into two groups that behave very differently on this task, and the split is
the single most important thing to understand when reading the results.

**Position-indexed metrics** — bit accuracy, character accuracy, sequence accuracy — compare
`pred[i]` against `gold[i]` at a fixed index. A single inserted or deleted character shifts
every subsequent position, so the remainder of the line reads as wrong even if it is a perfect
transcription offset by one. These metrics are brutal about alignment.

**Alignment-tolerant metrics** — Levenshtein, BLEU, ROUGE — allow insertions and deletions to
be absorbed. A one-character insertion costs 1 edit, not the rest of the line.

A model that reconstructs the text well but drifts in length will look catastrophic under the
first family and reasonable under the second. When the two families disagree, the alignment-
tolerant numbers carry the real signal.

---

## 2. Metric reference

### Bit accuracy — [`bit_accuracy`](src/utils.py#L65)

```
bit_accuracy(pred, gold) -> (matching_bits, total_bits)
```

Both strings are expanded back into their flat 8-bit-per-character ASCII form by
[`text_to_bits`](src/utils.py#L56) — `np.frombuffer` on the latin-1 bytes, then
`np.unpackbits` — which is exactly the transformation that produced `brown_cipher.txt`. The
two bit arrays are compared elementwise over their common prefix.

`total` is `max(len(pred_bits), len(gold_bits))`, so a length mismatch is charged as error
rather than ignored: predicting half the line correctly and stopping scores ~50%, not 100%.

Returning a `(match, total)` pair rather than a ratio is deliberate — `compute_all_metrics`
sums the numerators and denominators across all lines and divides once, giving a true
micro-average weighted by line length. Averaging per-line percentages would over-weight short
lines.

### Character accuracy — [`char_accuracy`](src/utils.py#L75)

The same comparison at character granularity: `sum(a == b for a, b in zip(pred, gold))` over
`max(len(pred), len(gold))`. Also micro-averaged.

Bit accuracy is always the higher of the two, and never below ~50% even for garbage output,
because ASCII letters share most of their high-order bits. Two unrelated lowercase letters
still agree on the top 3 bits. Read bit accuracy relative to that floor, not against 0.

### Sequence accuracy

Fraction of lines where `pred == gold` exactly. Computed inline in `compute_all_metrics`, not
as a separate function. On a task this hard it is usually 0 and serves as a ceiling check.

### Levenshtein — [`levenshtein`](src/utils.py#L82)

Standard edit distance (insert / delete / substitute, unit cost), reported two ways:

- `levenshtein_mean` — raw distance, averaged over lines.
- `levenshtein_normalized` — each distance divided by `len(gold)` before averaging, so lines
  of different lengths contribute comparably. **Lower is better** for both.

The implementation is a vectorised row-by-row DP. Substitution and deletion vectorise
directly, but the insertion term `cur[j] = min(cur[j], cur[j-1] + 1)` is a left-to-right
dependency. It is rewritten as a running minimum — `min_{k<=j}(cur[k] - k) + j` — which
`np.minimum.accumulate` computes in one pass. That turns an O(nm) Python double loop into
O(n) NumPy row operations, which matters because lines run to 2,670 characters.

`python src/utils.py` checks it against a naive O(nm) reference on 200 random string pairs and
against six hand-computed values.

### BLEU — [`corpus_bleu`](src/utils.py#L139)

`sacrebleu.corpus_bleu(preds, [golds]).score`. Corpus-level, not an average of per-line
scores: n-gram match counts are pooled across the whole test set before the precision ratio is
taken. This is the standard definition and is not comparable to a mean of sentence BLEUs.

The nested list in `[list(golds)]` is sacreBLEU's multi-reference API — one reference stream,
so a list containing a single list.

### ROUGE-1/2/L — [`rouge_n`](src/utils.py#L117), [`rouge_l`](src/utils.py#L134)

All three are **F1** scores, not recall. Tokenization is plain `str.split()`, so tokens are
whitespace-separated words.

- **ROUGE-1 / ROUGE-2** — unigram / bigram overlap. `_ngrams` builds a `Counter` of n-gram
  tuples; `p & g` is `Counter` intersection, which takes the elementwise minimum of the
  counts. That clipping is what stops a prediction from farming credit by repeating one
  correct word many times.
- **ROUGE-L** — based on the longest common subsequence of the two token lists, via
  [`_lcs_length`](src/utils.py#L122). Because a subsequence need not be contiguous, ROUGE-L
  tolerates insertions and reordering that ROUGE-2 punishes.

All three share [`_f1`](src/utils.py#L110): precision is `overlap / n_pred`, recall is
`overlap / n_gold`, and F1 is their harmonic mean, with a 0.0 guard for empty input.

---

## 3. The library cross-check

Levenshtein and ROUGE are additionally computed by reference libraries and reported under a
`_lib` suffix — `levenshtein_mean_lib`, `rouge1_lib`, `rougeL_lib`, and so on. The hand-rolled
values remain the primary reported numbers; the `_lib` columns exist so the two can be
compared directly in `results.csv`.

| Metric | Library | Expected relationship |
|---|---|---|
| Levenshtein | `rapidfuzz.distance.Levenshtein` | **Identical.** Same quantity, same unit costs. Any difference is a bug. |
| ROUGE-1/2/L | `rouge_score.rouge_scorer` | **Library ≥ hand-rolled**, differing only by capitalization. |
| BLEU | `sacrebleu` | Only one implementation; there is no `bleu_lib`. |

### Why the ROUGE numbers differ, and why that is informative

`rouge_score`'s default tokenizer lowercases the input, strips non-alphanumeric characters,
and splits on whitespace. The hand-rolled version only splits on whitespace.

On this corpus the second and third of those are no-ops. The plaintext alphabet is exactly
`a–z`, `A–Z` and space — no punctuation, no digits — so there is nothing for the
non-alphanumeric strip to remove, and both implementations end up splitting on whitespace.
**Lowercasing is the only substantive difference.** Stemming is explicitly disabled
(`use_stemmer=False`), because the task is exact text reconstruction and giving credit for a
matching word stem would measure something other than what the task asks for.

This makes the gap between the two columns meaningful rather than noise:

```
rougeL_lib - rougeL  ==  the credit recovered by ignoring case
```

A model that reconstructs words correctly but gets capitalization wrong shows a large gap. A
model that produces the wrong words entirely shows a small one, because lowercasing does not
rescue a wrong word. The hand-rolled column is the stricter, task-appropriate number; the
library column is the reference-standard definition.

`python src/utils.py` asserts both halves of this: the two ROUGE paths agree exactly on
already-lowercase input, and they diverge maximally on `"The Cat"` vs `"the cat"` (hand-rolled
0.0, library 1.0). The Levenshtein cross-check asserts exact equality on 200 random pairs.

---

## 4. Output keys

`compute_all_metrics` returns these keys. `evaluate_config` then adds identification and
runtime fields (`config`, `params_millions`, `decode_ms_per_line`, …) before the row is written.

| Key | Range | Direction |
|---|---|---|
| `bit_accuracy` | 0–100 (floor ≈ 50) | higher better |
| `char_accuracy` | 0–100 | higher better |
| `sequence_accuracy` | 0–100 | higher better |
| `levenshtein_mean` | ≥ 0, unbounded | **lower better** |
| `levenshtein_normalized` | ≥ 0, ~1.0 = as bad as empty | **lower better** |
| `bleu` | 0–100 | higher better |
| `rouge1`, `rouge2`, `rougeL` | 0–100 | higher better |
| `levenshtein_mean_lib`, `levenshtein_normalized_lib` | — | must match the non-`_lib` value |
| `rouge1_lib`, `rouge2_lib`, `rougeL_lib` | 0–100 | ≥ the non-`_lib` value |

Passing `with_library=False` skips the `_lib` computation and returns only the first nine keys,
which is what the self-test uses so it can run without the optional dependencies installed.

---

## 5. Where the metrics are consumed

- [`evaluate_config`](src/train.py#L468) calls `compute_all_metrics` and writes
  `outputs/samples_<config>.txt`, which shows per-line bit accuracy and edit distance for the
  first few test lines — the qualitative counterpart to the aggregate numbers.
- [`write_results`](src/train.py#L522) writes `results.csv` (columns fixed by
  `METRIC_COLUMNS`), `results.json`, and `runtime_stats.json`, then regenerates the figures.
- [`plot_metric_comparison`](src/utils.py#L329) plots six panels: bit, character and sequence
  accuracy, normalised Levenshtein, BLEU and ROUGE-L. It reads the hand-rolled columns.
- Throughput is always plotted as **examples/second**, never tokens/second — a "token" is a
  BPE subword for C1–C4 but a raw byte for C5, so a tokens/s chart would make C5 look fastest
  when it is in fact the slowest.
