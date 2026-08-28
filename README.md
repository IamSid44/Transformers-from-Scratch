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

Splitting into train/val/test is done over **lines** (4,000 / 500 / 500, seed 42), but a line
is not what a model trains or decodes on. `dataset.chunk_pairs` cuts each line into consecutive
64-character pieces (`CHUNK_CHARS = 64`; cipher and plaintext are cut at the same character
offset, so `cipher[8·start : 8·end]` is always exactly the bits of `plain[start:end]`, no more
and no less — the only fact this uses is the disclosed 8-bits-per-character encoding, nothing
about the key). A line's last chunk is whatever is left over, from 1 up to 64 characters.
**One training/decoding example is one chunk.** This turns 4,000 / 500 / 500 lines into
39,683 / 4,685 / 4,782 chunks — about 9.9 chunks per line.

Why chunk at all, given the pipeline already worked one whole line at a time:

- **Compute.** `max_src_len`/`max_tgt_len` collapse from 1,488/720 *tokens* (whole-line) to
  64/64 tokens (chunked) — attention cost per example drops by roughly two orders of
  magnitude, comfortably outweighing the ~10x increase in example count.
- **Possibly, learning.** The cipher's key resets every line, so position 0 of every example
  already carries a fixed phase in both the old whole-line scheme and the new chunked one —
  chunking doesn't create this signal, it was already implicit in training on lines at all.
  What chunking changes is *density*: position indices 0–63 now recur on every single example
  instead of only at the start of a ~550-character line, giving a learned positional encoding
  far more repetition to pick the phase pattern up from, over a much shorter span to track it
  across. This is a hypothesis, not a measured result — it's reported as one, and the
  compute case above holds regardless of whether it pans out.

Evaluation decodes each test chunk independently (still fully greedy, one chunk at a time —
this changes nothing about the mandated decoding strategy) and reassembles a line's predicted
chunks back into one string, in order, before scoring against the whole gold line.

Because chunks vary a little in length (a full 64-character chunk vs. a line's short leftover
tail), the training loader still batches examples of similar length
(`dataset.LengthGroupedBatchSampler`), as in Vaswani et al. §5.1 — mean padded batch 34 tokens
vs. mean real length 33, **1% of attention compute spent on padding** (down from 5% in the
whole-line scheme, since chunk lengths already vary far less than line lengths did).

Sequence-length caps sit at the observed maximum over chunks (`max_src_len=64`,
`max_tgt_len=64`, **0.00% truncation on both sides**).

---

## 3. Running

```bash
# Build the BPE tokenizers and verify the data pipeline (round-trip assertions)
python src/dataset.py

# Check the from-scratch BPE against HuggingFace `tokenizers`, once, on this corpus
python src/verify_tokenizer.py           # --quick to skip the full-corpus retraining

# Train, then evaluate, a single configuration / all five (each config evaluates immediately
# after it finishes training, before the next one starts -- results.csv/json and the figures
# are current after every config)
python src/train.py --config C1
python src/train.py --all --epochs 50

# Quick wiring check: a few steps, no WandB, no evaluation
python src/train.py --config C1 --smoke --no-wandb

# Train, evaluate, and upload the checkpoint to HuggingFace
python src/train.py --config C1 --push

# Re-evaluate existing checkpoints without retraining
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
| Batching | grouped by approximate length, batch size 1024 |

Parameter counts: **C1/C2 12.89M, C3 11.70M, C4 12.88M, C5 22.75M.**

Optimisation: Adam at 1e-3 with 1 epoch of linear warmup then cosine decay to **10% of the
peak** (1e-4), gradient clipping at 1.0, batch size 1024, up to 60 epochs with early stopping
(patience 5 on validation loss). Training runs in plain fp32.

Batch size moved from 16 to 1024 once chunking (§2) cut sequence lengths to 64 tokens: at 1024,
C5 (the heaviest configuration) peaks at ~49 GB and processes ~2x the examples/second of
batch 256, on a card with 75-80 GB free even alongside other jobs -- comfortably using the GPU
`batch_size=16` (tuned for the old ~1,488-token sequences) left mostly idle. Warmup is specified
in epochs (`warmup_epochs`), not a fixed step count, because `steps_per_epoch` now depends on
batch size: at 1024 that's 39 steps/epoch, so a fixed `warmup_steps=250` (right for the old
~2,480 steps/epoch) would have meant over 6 epochs of warmup instead of a small fraction of one.
`lr=6e-4`/`warmup_epochs=2` was the first setting tried at this batch size; `lr=1e-3` (higher,
since a much bigger batch gives a lower-variance gradient estimate that tolerates a larger step)
and `warmup_epochs=1`/`epochs=60` followed after the first several epochs of that run showed
validation loss still dropping steadily with no sign of a plateau, prompting a higher LR and a
larger epoch budget rather than waiting out the original one.

An earlier run used a 75% floor, on the reading that both losses still falling at the final
epoch meant the schedule was winding down too early. That was the wrong diagnosis: the losses
plateaued high because the old tokenization gave the model nothing aligned to learn from (see
§5), and a rate that never really came down just kept the late epochs noisy. With the cipher
side now BPE over byte-boundary-respecting symbols rather than arbitrary bit runs, there are
real character-aligned units to fit, which is what the 10% floor is for.

Two deliberate design choices:

1. **Pre-LN, not Post-LN.** The assignment (§2) explicitly requires "Pre-Layer
   Normalization" modules, and Pre-LN is the more trainable choice regardless.
2. **Separate cipher / plaintext vocabularies, not one shared BPE.** Sharing assumes source
   and target draw on common subwords; here the source alphabet is `{0,1}` and the target is
   English, so the overlap is empty.

**No weight decay and no mixed precision.** Both were removed as unnecessary: the models are
12–23M parameters trained for at most 60 epochs on 39,683 chunks with dropout 0.1 and label
smoothing 0.1 already regularising, and the 96 GB card has no memory pressure that fp16 would
relieve. Dropping AMP also removes the `GradScaler`, the autocast contexts, and the float32
softmax/norm upcasts that existed solely to stop fp16 underflow — see §5.

**Scheduled sampling, training-time only.** The assignment mandates greedy decoding for every
reported metric (§4), so evaluation can't be changed to fix exposure bias — but training is
otherwise 100% teacher-forced (the decoder is always handed the true previous token), which
means the model never practices recovering from a mistake, while greedy decoding at evaluation
feeds back exactly that, letting one early error compound through everything after it.
`teacher_forcing_prob(step, warmup_steps, total_steps, floor)` stays at 1.0 through the same
warmup the LR schedule uses, then linearly decays to `scheduled_sampling_floor` (0.7) by the
end of training: some decoder-input positions get swapped for the model's own prediction
instead of the gold token, via one extra `torch.no_grad()` forward pass per training step
(the usual parallelizable approximation for a Transformer decoder). This never touches
`greedy_decode` or any evaluation path — see §5 for the mechanics on each of C1–C4 and C5.

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

Two independent BPE tokenizers, trained on the training split's **chunks** (see
[§2, Examples and batching](#2-dataset)), not whole lines — each chunk is encoded
independently, so no token can straddle a chunk boundary:

- **cipher**: each 8-bit unit (one plaintext character's cipher byte) is mapped to a single
  atomic symbol first (`dataset.cipher_to_symbols`, the same byte<->character bijection
  `ByteLevel` uses over the full 256-value byte alphabet), then BPE runs with **no**
  pre-tokenizer, so a whole chunk is one "word" and merges are free to combine adjacent
  symbols into tokens spanning multiple characters — the same way ordinary BPE combines
  characters into words. Vocab cap 1,024, and this side actually reaches it: 4 special tokens
  + 256 single-byte tokens (`ByteLevel.alphabet()`, forced into the vocabulary regardless of
  frequency so no byte value the cipher could in principle produce is ever `<unk>`) + 764
  learned multi-character merge tokens, all 764 of which are actually exercised. Mean chunk is
  486.6 bits (60.8 characters) → 33.3 tokens, **14.60× compression**.

  Of the 256 single-byte tokens, only **126 ever occur in this corpus** (all with the high bit
  clear — XORing two mostly-7-bit alphabets never sets it); the other **130 sit in the
  vocabulary permanently unused** — reserved capacity, never chosen by a merge, never emitted,
  by design rather than by accident.
- **plaintext**: `Split(" ?[A-Za-z]+", "isolated")` pre-tokenization over the **53 characters
  the corpus actually contains** (`a-z`, `A-Z`, space), vocab 4,096, **3.31× compression** (a
  mean chunk of 60.8 characters → 18.4 tokens). Each pre-token is a word carrying its own
  leading space, so the space is an ordinary vocabulary character rather than a marker and
  `decode` is plain concatenation. A byte-level alphabet would have spent 203 of its 256
  symbols on bytes that never occur. Decoding is verified lossless in `python src/dataset.py`.

This replaces an earlier scheme — free-form BPE over the raw bit string — under which token
boundaries had nothing to do with the character grid at all: a source token could cover some
arbitrary run of 11 bits, cutting across the middle of a character's 8-bit encoding, so no
source position corresponded to any character the decoder had to emit. C1–C4 plateaued around
4.5–4.7 validation loss as a result, while C5, which reads raw bytes and never had that
problem, reached 1.96. An intermediate fix (hard 8-bit word boundaries, guaranteeing exactly
one source token per character) was tried and discarded in favor of the scheme above: it
forced perfect alignment but gave up real compression, doing hardly more than mapping each
byte value to a token id (179 tokens used out of a 1,024 cap, 8.0× compression by
construction). The current scheme keeps merges confined to whole characters — a token can
span several characters but never a fraction of one — while still letting BPE find and
exploit repeated cipher-byte patterns, same as it does on the plaintext side. The cipher's
periodic substitution (each character maps to one of ~7 byte patterns depending on position)
means any given plaintext bigram is diluted across up to 7 distinct byte-pair realizations,
so this is short of the compression a comparable English BPE tokenizer would reach on
unshifted text — but the 1,024-token vocabulary is fully used regardless.

### Scheduled sampling mechanics

`compute_loss` (`train.py`) runs the extra no-grad forward pass and builds the mix only while
`model.training` and `tf_prob < 1.0`; `evaluate_loss` never passes `tf_prob`, so validation
loss stays a clean teacher-forced likelihood throughout.

- **Tokenized (C1–C4)** — positions 1..end of the decoder input `tgt_in` get swapped for the
  model's own prediction from the position before them, at rate `1 - tf_prob`. Position 0
  (`<bos>`) is never touched.
- **BLT (C5)** — `BLTSeq2Seq.forward` gained an optional `ctx_bytes` argument (default
  `tgt_bytes`) that drives the patch pooling feeding the global decoder *and* the within-patch
  shift feeding the local decoder, while `tgt_bytes` still supplies the loss labels
  unconditionally. `compute_loss` mixes the model's own byte predictions into `ctx_bytes` at
  non-`<pad>` positions, hitting both of that architecture's exposure-bias points (patch-to-
  patch and byte-within-patch) with one mix instead of two separate ones.

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
an earlier build of this model — trained one whole line (up to 2,670 characters) at a time —
peaked at ~61 GB; chunking (§2) has since cut the sequence lengths involved by more than an
order of magnitude, so this optimization matters far less now than when it was written, but
it's still in place. The exact current figure (4 + 4 layers, 8 heads, `d_ff=2048`, 64-character
chunks) is reported in `outputs/runtime_stats.json` after a run.

### Comparability caveats

Two numbers must not be compared directly across the C1–C4 / C5 boundary, and the report
says so explicitly:

- **Validation loss.** C1–C4 predict over a 4,096-way subword vocabulary, C5 over a 259-way
  byte vocabulary. A lower cross-entropy for C5 is partly just a smaller output space.
- **Tokens/second.** A "token" is a BPE subword (~18 per chunk) for C1–C4 but a raw byte
  (~62 per chunk: a chunk's characters plus one EOS byte) for C5. `examples_per_sec` and
  `sec_per_epoch` are the honest throughput comparisons and are logged alongside.

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
  bpe.py             Byte-Pair Encoding from scratch: pre-tokenizers, model, trainer, decoders
  verify_tokenizer.py  Differential test of bpe.py against HuggingFace `tokenizers`
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
