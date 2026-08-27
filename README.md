# ANLP Assignment 1 — Transformers from Scratch, Architectural Variants, and BLT

Roll number: **2023102040**

An encoder–decoder Transformer built entirely from basic PyTorch operations, trained to
decrypt binary cipher sequences into English plaintext, plus a five-way controlled ablation
(C1–C5) covering positional encoding, attention, normalization and tokenization.

---

## 1. Setup

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt

# all commands below are run from the project root
.venv/bin/python src/<file>.py
```

### Credentials

Create a `.env` in the project root. It is gitignored and excluded from the submission
archive.

```
WANDB_API_KEY=...      # https://wandb.ai/authorize
HF_TOKEN=...           # https://huggingface.co/settings/tokens  (write scope)
HF_USERNAME=...
```

If `WANDB_API_KEY` is absent everything still runs; WandB silently drops to offline mode and
writes to `./wandb/`, which can be uploaded later with `wandb sync wandb/offline-run-*`.

---

## 2. Dataset

Two parallel, strictly line-aligned files in this folder:

| File | Description | Format | Lines |
| :--- | :--- | :--- | :--- |
| `brown_cipher.txt` | Encrypted binary string | String of `0`s and `1`s | 5,000 |
| `brown_plain.txt` | Plaintext English sentence | UTF-8 text string | 5,000 |

- **Strict 1-to-1 line alignment**: line *k* of `brown_cipher.txt` corresponds to line *k* of
  `brown_plain.txt`. No empty lines or header rows.
- `len(cipher_line) == 8 * len(plain_line)` holds on all 5,000 lines: every plaintext
  character is expanded to exactly 8 bits.
- Plaintext alphabet is 53 symbols (`a–z`, `A–Z`, space). No punctuation, digits or
  non-ASCII.
- Line lengths are long: plaintext median 553 / max 2,670 characters, i.e. cipher median
  4,424 / max 21,360 bits.

### What the cipher actually is

Analysis of all 5,000 pairs shows the encryption is a **repeating-key XOR**:

```
cipher_bits[i] = ascii(plain[i]) XOR key[i mod 8],  key = "ANLP2026"
```

verified exact on 5,000/5,000 lines, with the key phase resetting at the start of every line.

This is reported as analysis only. **The models never use it** — no part of the data
pipeline, loss or decoding exploits the key. It is included because it explains *why* the
task is learnable at all and why positional information matters so much: recovering character
*i* requires knowing `i mod 8`, so the model has to track its absolute offset in the bit
stream.

### Examples and batching

**One training example is one complete line.** Splitting into train/val/test is done over
lines (4,000 / 500 / 500, seed 42). There is no windowing or chunking anywhere in the
pipeline: a line goes in whole and is scored whole.

Because lines range from 21 to 2,670 characters, the training loader batches examples of
similar length (`dataset.LengthGroupedBatchSampler`), as in Vaswani et al. §5.1. Without it,
uniform random batching pads to a mean of 1,113 source tokens against a mean real length of
432 — **61% of attention compute spent on padding**; with it, 5%. Validation and test keep
their natural order so predictions stay aligned with their gold lines.

Sequence-length caps sit at the observed maximum (`max_src_len=1944`, `max_tgt_len=728`,
**0.00% truncation on both sides**): an example is a whole line, so truncating an outlier
would delete the end of a document.

---

## 3. Running

```bash
# Build the BPE tokenizers and verify the data pipeline (round-trip assertions)
python src/dataset.py

# Train a single configuration / all five
python src/train.py --config C1
python src/train.py --all --epochs 60

# Quick wiring check: a few steps, no WandB
python src/train.py --config C1 --smoke --no-wandb

# Train and upload the checkpoint to HuggingFace
python src/train.py --config C1 --push

# Greedy-decode the test split and write outputs/
python src/train.py --evaluate --all
python src/train.py --evaluate --config C5 --limit-lines 50    # quick check
```

Every module carries a self-test:

```bash
python src/utils.py                # metric implementations vs hand-computed values
python src/dataset.py              # split disjointness, round-trip, tokenizer losslessness
python src/models/norm.py          # LayerNorm vs F.layer_norm; RMSNorm invariants
python src/models/positional.py    # RoPE norm preservation and translation invariance
python src/models/attention.py     # masking, GQA reduces to MHA, causality, weight tying
python src/models/blt.py           # shapes, causality audit, overfit-one-batch, source use
```

---

## 4. The five configurations

| Config | Changed from base | Positional Encoding | Attention | Normalization | Tokenization |
|---|---|---|---|---|---|
| C1 | None (base) | Sinusoidal Absolute | Multi-Head | LayerNorm | Standard Subword |
| C2 | Positional encoding | **RoPE** | Multi-Head | LayerNorm | Standard Subword |
| C3 | Attention mechanism | Sinusoidal Absolute | **Grouped-Query** | LayerNorm | Standard Subword |
| C4 | Normalization | Sinusoidal Absolute | Multi-Head | **RMSNorm** | Standard Subword |
| C5 | Tokenization | Sinusoidal Absolute | Multi-Head | LayerNorm | **BLT (token-free)** |

C2–C5 are constructed with `dataclasses.replace` from the C1 object
([src/train.py](src/train.py)), so it is structurally impossible for more than the named
field to differ.

### Shared hyperparameters

Identical across all five configurations, so any difference in the results is architectural:

| | Value |
|---|---|
| Encoder / decoder layers | 4 + 4 |
| `d_model` | 256 |
| `d_ff` | 2048 |
| Heads `h` (`d_head`) | 8 (32) |
| Dropout / label smoothing | 0.1 / 0.1 |
| Batching | grouped by approximate length, batch size 24 |

Parameter counts: **C1/C2 12.89M, C3 11.70M, C4 12.88M, C5 22.75M.**

Optimisation: Adam at 6e-4 with 250 steps of linear warmup then cosine decay to **75% of the
peak**, gradient clipping at 1.0, batch size 24, up to 60 epochs with early stopping (patience
5 on validation loss). Training runs in plain fp32.

The decay floor is deliberately shallow. An earlier run decayed to 10% of peak and ended with
both training and validation loss still falling at the final epoch — the schedule was winding
down while the model was still learning. A 75% floor leaves the last epochs at 4.5e-4 rather
than 6e-5, so the extra epochs do useful work.

Two deliberate design choices:

1. **Pre-LN, not Post-LN.** The assignment (§2) explicitly requires "Pre-Layer
   Normalization" modules, and Pre-LN is the more trainable choice regardless.
2. **Separate 1,024 / 4,096 vocabularies, not one shared BPE.** Sharing assumes source and
   target draw on common subwords; here the source alphabet is `{0,1}` and the target is
   English, so the overlap is empty.

**No weight decay and no mixed precision.** Both were removed as unnecessary: the models are
12–23M parameters trained for at most 60 epochs on 4,000 examples with dropout 0.1 and label
smoothing 0.1 already regularising, and the 96 GB card has no memory pressure that fp16 would
relieve. Dropping AMP also removes the `GradScaler`, the autocast contexts, and the float32
softmax/norm upcasts that existed solely to stop fp16 underflow — see §5.

---

## 5. Implementation notes

### Constraint compliance

`nn.Transformer`, `nn.TransformerEncoder`, `nn.MultiheadAttention`, `nn.LayerNorm` and
`F.scaled_dot_product_attention` appear **nowhere** in the model code. Only `nn.Linear`,
`nn.Embedding`, `nn.Dropout`, `nn.GELU` and elementary tensor operations are used. Verify
with:

```bash
grep -rnE "nn\.Transformer|nn\.MultiheadAttention|nn\.LayerNorm|scaled_dot_product_attention" src/models/*.py src/train.py
```

(The only hits are inside `if __name__ == "__main__"` self-test blocks, where the PyTorch
reference implementations are used deliberately to *check* the hand-written ones agree.)

### Tokenization (C1–C4)

Two independent BPE tokenizers, both trained on the training split only:

- **cipher**: initial alphabet `{"0", "1"}`, no pre-tokenizer, vocab 1,024. Achieves ~11.2×
  compression (a mean line of 4,827 bits → ~432 tokens).
- **plaintext**: GPT-2-style ByteLevel pre-tokenizer, vocab 4,096, ~3.9× compression (a mean
  line of 603 characters → ~156 tokens). Decoding is verified lossless in
  `python src/dataset.py`.

Sequence caps sit at the observed maximum — 1,944 source and 728 target — so nothing is
truncated.

### Mask convention

A mask is a boolean tensor broadcastable to `(B, H, T_q, T_k)` where **`True` means the
position may be attended to**. Blocked logits are filled with the smallest *finite* value of
the score dtype, so a fully-masked query row softmaxes to a uniform distribution instead of
producing NaNs.

### RoPE and cross-attention

RoPE encodes a *relative* offset between a query and a key position. In cross-attention the
query indexes the target and the key indexes the source — two unrelated coordinate systems,
where "distance" is not meaningful. RoPE is therefore applied in **encoder self-attention and
decoder self-attention only**. The decoder still receives source positional information,
because the encoder states it attends to were themselves built with RoPE.

### BLT (C5)

```
source bytes --LocalByteEncoder--> byte states --PatchPooler--> patches (stride 16)
target bytes --LocalByteEncoder--> byte states --PatchPooler--> patches (stride 8)
                              GlobalTransformer  (the same class as C1, in latent mode)
                              patch latents --LocalByteDecoder--> byte logits
```

A mean line is 4,827 source bytes → 302 source patches and 604 target bytes → 76 target
patches.

- **Byte embeddings** are a 259-entry table plus hashed byte n-gram embeddings (n = 3, 4),
  with backward-looking windows. Bucket counts are sized per side: 8,192 on the target
  (English, ~148k possible 3-grams) but only 512 on the source, whose alphabet is `{0,1}`
  and therefore has just 8 distinct 3-grams.
- **Local attention is block-local** (128-byte blocks), which is what makes a multi-thousand
  byte source affordable: independent 128×128 blocks instead of one quadratic matrix.
- **The target-side local encoder is causal.** Patch *t*'s pooled vector is what the global
  decoder consumes at step *t*, and step *t* predicts patch *t+1* — so a byte in patch *t*
  attending to patch *t+1* would be reading its own answer. `python src/models/blt.py` audits
  this by perturbing individual target bytes and asserting no logit at or before that
  position moves.
- **Patching is fixed-size**, not entropy-driven. The BLT paper trains a separate byte-level
  LM to place patch boundaries at entropy spikes; the assignment asks for a *simplified* BLT,
  and fixed strides keep the patch grid rectangular and batchable.
- **The global transformer is literally `Seq2SeqTransformer` in latent mode** — same depth,
  width, sinusoidal encoding, MHA and LayerNorm as C1. Only the representation layer differs.

### GPU memory

Three changes keep training within a sane footprint, none of which alters what the models
compute:

1. **The local byte decoder never copies the source memory per patch.** Every target patch
   cross-attends to the same encoder memory. Materialising one copy per patch is a
   `(B·N, Ns, d)` tensor — for the longest batch in this corpus, ~11 GB *per decoder layer*.
   Instead patches fold into the batch axis for self-attention and into the *query* axis for
   cross-attention, so the memory stays `(B, Ns, d)`. `DecoderLayer` takes an optional
   `reshape` pair to support this.
2. **Attention scores are masked and softmaxed in place.** The `(B, H, T_q, T_k)` score
   tensor is the largest allocation in the model; `masked_fill_` and a same-dtype softmax
   avoid rewriting it three times over.
3. **Attention weights are not returned.** They were threaded through every block but no
   caller used them, which kept a full score tensor alive per layer.

Together these keep C5 training within a workable footprint. Without the first change alone,
an earlier build of this model peaked at ~61 GB; the exact figure for the current geometry
(4 + 4 layers, 8 heads, `d_ff=2048`) is reported in `outputs/runtime_stats.json` after a run.

### Comparability caveats

Two numbers must not be compared directly across the C1–C4 / C5 boundary, and the report
says so explicitly:

- **Validation loss.** C1–C4 predict over a 4,096-way subword vocabulary, C5 over a 259-way
  byte vocabulary. A lower cross-entropy for C5 is partly just a smaller output space.
- **Tokens/second.** A "token" is a BPE subword (~156 per line) for C1–C4 but a raw byte
  (~604 per line) for C5. `examples_per_sec` and `sec_per_epoch` are the honest throughput
  comparisons and are logged alongside.

---

## 6. Metrics

Computed on the 500 reconstructed test lines, using **greedy decoding** throughout.

| Metric | Definition |
|---|---|
| Bit accuracy | Prediction and gold are expanded back to their 8-bit ASCII form and compared bitwise; length mismatch is charged against `max(len_pred, len_gold)`. |
| Character accuracy | The same comparison at character granularity. |
| Sequence accuracy | Fraction of test lines reconstructed exactly. |
| Levenshtein | Edit distance, reported raw and normalised by gold length. |
| BLEU | sacreBLEU corpus BLEU. |
| ROUGE-1/2/L | Unigram / bigram / LCS F1, implemented in `src/utils.py`. |

Levenshtein and ROUGE are hand-implemented (see [src/utils.py](src/utils.py)) rather than
imported; `python src/utils.py` cross-checks the edit distance against a naive O(nm)
reference on 200 random string pairs and against hand-computed values.

---

## 7. Repository layout

```
src/
  models/
    attention.py     SDPA, MHA, GQA, FeedForward, encoder/decoder blocks, Seq2SeqTransformer
    positional.py    Sinusoidal absolute encoding, RoPE
    norm.py          LayerNorm (hand-written), RMSNorm
    blt.py           Byte vocabulary, local byte encoder/decoder, patch pooling, BLTSeq2Seq
  dataset.py         Paths, data constants, BPE tokenizers, tokenized and token-free loaders
  train.py           The five configurations, model building, training loop, evaluation, CLI
  utils.py           Metrics, plots, seeding, profiling
outputs/             results.csv/json, runtime_stats.json, figures, sample decodes,
                     tokenizers/, checkpoints/ (excluded from the zip; mirrored on HuggingFace)
README.md            This file
Report.pdf           Final report
```

The file list is fixed by the assignment, so two things live where they otherwise would not:
the encoder–decoder assembly sits in `models/attention.py` (putting it in `train.py` would
create the import cycle train → blt → train), and evaluation sits in `train.py` rather than a
separate `evaluate.py`. There are no `__init__.py` files — `src/` is not a package; each file
is a plain script, and `train.py` / `dataset.py` add `src/models/` to `sys.path` so the model
files can import each other as siblings.

---

## 8. Links

With `WANDB_API_KEY` and `HF_TOKEN` / `HF_USERNAME` set in `.env`, `python src/train.py
--config C1 --push` logs live to WandB and uploads the checkpoint when the run finishes.
Without a WandB key the run falls back to offline mode in `wandb/`, which `wandb sync` can
upload later.

**Weights & Biases:** `https://wandb.ai/<entity>/anlp-a1-2023102040`

**HuggingFace checkpoints:**

| Config | Repository |
|---|---|
| C1 | `https://huggingface.co/<HF_USERNAME>/anlp-a1-2023102040-C1` |
| C2 | `https://huggingface.co/<HF_USERNAME>/anlp-a1-2023102040-C2` |
| C3 | `https://huggingface.co/<HF_USERNAME>/anlp-a1-2023102040-C3` |
| C4 | `https://huggingface.co/<HF_USERNAME>/anlp-a1-2023102040-C4` |
| C5 | `https://huggingface.co/<HF_USERNAME>/anlp-a1-2023102040-C5` |

---

## 9. Results

Not yet trained under the current configuration. Run:

```bash
python src/train.py --all
python src/train.py --evaluate --all
```

which writes `outputs/results.csv`, `outputs/results.json`,
`outputs/runtime_stats.json`, per-config `samples_<C>.txt` decodes, and three figures
(`loss_curves.png`, `metric_comparison.png`, `memory_speed.png`). The analysis of those
numbers lives in `Report.pdf`.

### One caveat to carry into the report

**Bit accuracy cannot rank these systems.** Lowercase ASCII letters share their top four
bits, so a constant prediction of the letter `e`, padded to the gold length, scores around
73% — higher than a genuinely trained model can be expected to reach early on. Levenshtein,
BLEU and ROUGE-L are the metrics that carry real signal here; bit accuracy is reported
because the assignment asks for it, not because it discriminates.
