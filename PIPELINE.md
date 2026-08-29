# Pipeline Walkthrough

What actually happens, in call order, from `python src/train.py` to the files in `outputs/`.
Every shape is written out. For per-function detail see [CODE_REFERENCE.md](CODE_REFERENCE.md);
for metric definitions see [METRICS.md](METRICS.md).

The task: given a bit string, recover the English text it encodes. Each plaintext character was
expanded to exactly 8 bits, so the source is always 8× longer than the target. The model is
never told this relationship — it has to learn it.

---

## Stage 0 — Entry

```
python src/train.py --config C1
```

[`main()`](src/train.py#L588) parses arguments, then:

1. **`utils.load_env(ENV_FILE)`** — pulls `WANDB_API_KEY`, `HF_TOKEN`, `HF_USERNAME` from
   `.env` into `os.environ`. Only the optional integrations read these; training runs fine
   without the file.
2. **`utils.set_seed(ds.SEED)`** — seeds `random`, `numpy`, `torch`, CUDA, `PYTHONHASHSEED`.
3. **Device selection** — `cuda` unless `--cpu` or no GPU is visible.
4. **Dispatch** — `--evaluate` goes straight to Stage 6, against existing checkpoints, without
   retraining. Otherwise Stage 1, and (unless `--smoke`) Stage 6 runs again automatically right
   after Stage 5 finishes for that configuration — so with `--all`, C1 trains *and* evaluates
   before C2 starts, and `results.csv`/`results.json`/the figures are rewritten after every
   configuration, not just at the end of the run.

`TrainConfig()` is instantiated once and shared by every configuration in the run, so optimiser,
schedule, batch size and epoch budget are identical by construction. Any difference in results
comes from the one architectural axis that changed.

---

## Stage 1 — Corpus to examples

Triggered by `train_one` → `ds.make_dataloaders` → `ds.build_splits`.

### 1a. Read and verify

**`read_corpus()`** reads both files, drops blank lines, and asserts two invariants: the two
files have equal line counts, and `len(cipher_line) == 8 * len(plain_line)` on every line. A
violation raises immediately rather than surfacing as a silent misalignment during training.

```
brown_cipher.txt  ──►  cipher_lines: list[str]   5000 bit strings
brown_plain.txt   ──►  plain_lines:  list[str]   5000 text lines
```

### 1b. Split

**`split_line_ids(5000, seed=42)`** shuffles `range(5000)` with a seeded RNG and slices
4000 / 500 / 500.

The split is over lines. A line is **not** the training/decoding unit, though — see 1d.

### 1c. Pair up

**`build_splits()`** wraps each selected line as a `Pair`, so each split is a list of
whole-line examples in `line_id` order.

```
build_splits() ──► ({"train": [Pair]×4000,
                     "val":   [Pair]×500,
                     "test":  [Pair]×500},
                    plain_lines)          # gold references, indexed by line_id
```

Each `Pair` holds `line_id`, `cipher` (bit string), `plain` (text).

### 1d. Chunk

**`chunk_pairs(pairs, chunk_chars=CHUNK_CHARS=64)`** expands every whole-line `Pair` into
consecutive fixed-size pieces, cut on the raw character/bit strings before either tokenizer
runs: `Pair(line_id, cipher[8*start:8*end], plain[start:end])` for `start` stepping by 64
characters. `cipher[8*start:8*end]` is always exactly `8*(end-start)` bits — the bits of
exactly `plain[start:end]`'s characters, no more, no less — so alignment is exact by
construction, using only the disclosed 8-bits-per-character fact. A line's last chunk is
whatever remains (1 to 64 characters). All of one line's chunks keep its `line_id` and stay
consecutive and in order in the output list, which is what lets evaluation regroup them later.

**A chunk, not a line, is the actual training/decoding example** from here on:
4,000/500/500 lines become 39,683/4,685/4,782 chunks (train/val/test). `info["pairs"]` in
`make_dataloaders` (Stage 3) stays at whole-line granularity — it's the ground truth
evaluation reconstructs against — while every `Dataset`/`DataLoader` is built from the chunked
lists instead.

---

## Stage 2 — Tokenization

Only on the C1–C4 path. C5 skips this stage entirely — that *is* the C5 ablation.

**`build_tokenizers(train_ciphers, train_plains)`** reloads from `outputs/tokenizers/` if
present, otherwise trains and saves. **Inputs are the training split's chunks** (Stage 1d),
not whole lines — merge frequencies should reflect what the model actually encodes at both
train and eval time. Training a vocabulary on val or test text would also leak.

Two separate tokenizers, because the alphabets do not overlap at all:

| | Cipher | Plaintext |
|---|---|---|
| Alphabet | 256 byte values (`ByteLevel.alphabet()`) | `a-z`, `A-Z`, space (53 characters) |
| Pre-processing | `dataset.cipher_to_symbols`: each 8-bit unit -> one atomic symbol | none |
| Pre-tokenizer | none (whole chunk is one BPE "word") | `Split(" ?[A-Za-z]+", "isolated")` |
| Decoder | `Fuse`, then `dataset.symbols_to_cipher` | `Fuse` |
| Vocabulary cap | 1024 | 4096 |
| Why | Each 8-bit unit is one plaintext character's cipher byte. Mapping it to a single symbol first (the same byte<->character bijection `ByteLevel` uses) and then running BPE with **no** boundary restriction inside a chunk lets merges combine adjacent symbols into tokens spanning multiple characters — the same way ordinary BPE combines characters into words. A token can never span two *chunks*, since each chunk is encoded independently (Stage 1d). | Those 53 characters are all the corpus contains, so a byte-level alphabet would spend 203 of its 256 symbols on bytes that never occur. Each pre-token is a word carrying its own leading space, which makes the space an ordinary vocabulary character and `decode` plain concatenation — still exact. |

Measured over the trained cipher vocabulary: 4 special ids + 256 single-byte tokens (forced in
by `ByteLevel.alphabet()` regardless of frequency) + 764 learned multi-character merge tokens
= 1024, and all 764 merge tokens are exercised. Of the 256 single-byte tokens, only 126 ever
occur in this corpus (all with the high bit clear); the other 130 sit in the vocabulary
permanently unused — reserved capacity, not a bug.

An earlier version of the cipher tokenizer pre-tokenized into hard 8-bit words so no merge
could cross an 8-bit-unit boundary, guaranteeing exactly one source token per plaintext
character. That traded away real compression for positional alignment; this version restores
ordinary BPE behavior on the cipher side, on the view that unit-boundary-respecting
multi-character tokens are not the same failure mode as the original (pre-alignment) scheme,
where a token could span a *fraction* of a character.

Both vocabulary numbers are **ceilings, not targets**: each side runs out of pairs worth
merging (`min_frequency=2`) well before the cap, and `meta` records what was actually learned.

Both reserve the same four special ids: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`.

The function also derives **`meta`**, computed over the training split and then carried
everywhere — into the model constructor, into WandB config, and into every checkpoint:

```
meta = {cipher_vocab_size, plain_vocab_size,   # embedding table sizes
        max_src_len, max_tgt_len,              # 100th percentile, rounded up to a multiple of 8
        src_len_mean, tgt_len_mean, src_len_max, tgt_len_max,
        src_compression, tgt_compression}      # mean characters per token
```

The caps sit at the **observed maximum**, so nothing is ever truncated. An example is one
chunk (`max_src_len=64`, `max_tgt_len=64` measured); truncating would delete the end of one.

---

## Stage 3 — Datasets, collation, batching

**`make_dataloaders(model_cfg, train_cfg)`** branches on `model_cfg.is_blt`.

### 3a. Encoding

**Tokenized path (C1–C4)** — `TokenizedSeq2SeqDataset` encodes **both sides eagerly in
`__init__`** with `encode_batch`; per-item encoding would dominate step time.

```
cipher str ──encode──► src_ids           [:max_src_len]
plain  str ──encode──► [BOS] + ids + [EOS]   body [:max_tgt_len - 2]
```

**Byte path (C5)** — `ByteSeq2SeqDataset` encodes lazily per item with `latin-1`, so cipher
character `'0'` becomes byte 48 and `'1'` becomes 49. C5 consumes the same input file with only
the vocabulary removed. Target gets a trailing `BYTE_EOS_ID` and **no BOS** — the BLT model
supplies a learned `bos_patch` internally. Nothing is truncated.

### 3b. Collation — padding is per batch, not global

**`collate_tokenized`** receives `[(src, tgt), …]`, transposes it with `zip(*batch)` into a
tuple of sources and a tuple of targets, then pads each side to **that batch's own longest
sequence** with `PAD_ID`. Padding to the global cap instead would waste most of the compute.

**`CollateBytes`** does the same for C5 but rounds up to a whole multiple of the patch size
(`SRC_PATCH_SIZE=32`, `TGT_PATCH_SIZE=4`), so the patch grid is rectangular. The two are
locked together as `SRC_PATCH_SIZE = BITS_PER_CHAR * TGT_PATCH_SIZE` so a source patch and a
target patch cover the same four characters -- see [BLT.md](BLT.md).

Both return `{"src": (B, S), "tgt": (B, T)}`.

### 3c. Length-grouped batching — training only

Chunks are mostly 64 characters, with a shorter leftover at some lines' ends, so padding waste
is already small — mean 34 padded tokens vs. mean 33 real per batch under uniform random
batching (measured).

**`LengthGroupedBatchSampler`** shuffles indices, sorts by length only *within* megabatches of
`batch_size × 50`, cuts batches, then shuffles the batch order. Padding drops to ~1% while
batch membership still changes every epoch, because the internal `epoch` counter advances the
RNG on each `__iter__`.

**Val and test never use it.** They keep natural order, so predictions stay positionally
aligned with their `Pair` list and each one is scored against the right gold line.

```
make_dataloaders ──► {"pairs":    {split: [Pair]},
                      "datasets": {split: Dataset},
                      "loaders":  {split: DataLoader},
                      "meta":     {...},
                      "cipher_tok", "plain_tok"}   # tokenized path only
```

---

## Stage 4 — Model construction

**`build_model(model_cfg, meta)`**, with `max_len = max(max_src_len, max_tgt_len) + 64`.

### C1–C4: `Seq2SeqTransformer`

Pre-LN encoder–decoder, 4 + 4 layers, `d_model=256`, 8 heads (`d_head=32`), `d_ff=2048`. Source
and target embeddings; output projection tied to the target embedding. Embeddings are scaled by
`√d_model` so their magnitude stays comparable to the positional encoding.

The four switches resolve at construction:

| Axis | C1 base | Variant | Where it takes effect |
|---|---|---|---|
| Position | `SinusoidalPositionalEncoding` | C2: `RotaryPositionalEmbedding` | Sinusoidal adds a vector once before layer 1; RoPE rotates q/k inside every self-attention. |
| Attention | `MultiHeadAttention` | C3: `GroupedQueryAttention` | 8 query heads, 2 KV heads shared across groups of 4. |
| Norm | `LayerNorm` | C4: `RMSNorm` | Every normalization site. |
| Tokenization | BPE | C5: BLT | Whole representation layer. |

**RoPE is applied in self-attention only, never cross-attention** — there the query indexes the
target and the key indexes the source, two unrelated coordinate systems, so a relative offset
between them is meaningless.

### C5: `BLTSeq2Seq`

```
src bytes (B,Ls) ─LocalByteEncoder─► (B,Ls,256) ─PatchPooler/16─► (B,Ns,256)
                                                                       │
tgt bytes (B,Lt) ─LocalByteEncoder─► (B,Lt,256) ─PatchPooler/8──► (B,Nt,256)
                                        (causal)                       │
                          Seq2SeqTransformer in latent mode ◄──────────┘
                          latents (B,Nt,256) ─LocalByteDecoder─► (B,Nt,8,259)
```

The global transformer is **literally the same class** as C1, with the same depth, width,
sinusoidal encoding, MHA and LayerNorm — only the representation layer differs. A learned
vocabulary is replaced by learned pooling over raw bytes.

Byte ids are their own numeric value; the three control ids sit above 255 (`PAD=256`,
`EOS=257`, `PATCH_START=258`) so no offset arithmetic is needed anywhere. `ByteEmbedding` adds
hashed 3- and 4-gram embeddings, because a single `'0'`/`'1'` byte carries almost no
information. The source side uses only 512 hash buckets — a binary alphabet has just 2³
distinct 3-grams — against 8192 on the English side.

---

## Stage 5 — Training loop

[`train_one`](src/train.py#L203) after building data and model:

- **Optimizer** — Adam, `lr=1e-3`. No weight decay: dropout 0.1 and label smoothing 0.1
  already regularise a 12–23M model trained for at most 60 epochs on 39,683 chunks.
- **Schedule** — `LambdaLR` with `warmup_epochs` (1) worth of linear warmup, then cosine decay
  to 10% of peak. Warmup is specified in epochs, not a fixed step count, and converted to
  `warmup_steps = round(warmup_epochs * steps_per_epoch)` inside `train_one`: `steps_per_epoch`
  depends on `batch_size`, so a fixed step count picked for one batch size silently means a
  different fraction of training at another. At `batch_size=1024` that's 39 steps/epoch, so 1
  epoch is 39 steps.
  An earlier run used a 75% floor, on the reading that the losses were still falling at the
  last epoch because the rate had wound down too far. That was the wrong diagnosis: they
  plateaued high because the old tokenization gave the model no aligned units to learn from
  (Stage 2), and a rate that never came down just kept the late epochs noisy.
- **Precision** — plain fp32. There is no autocast and no `GradScaler`.
- **`utils.set_seed(ds.SEED)`** is called again at the top, so each configuration starts from
  identical initialization regardless of what ran before it.
- **Scheduled sampling** — `teacher_forcing_prob(step, warmup_steps, total_steps, floor)`
  decays from 1.0 through the same warmup the LR schedule uses, down to
  `scheduled_sampling_floor` (default 0.7) by the end of training. Training is otherwise
  100% teacher-forced: the decoder is always handed the true previous token, so it never
  practices recovering from one of its own mistakes -- but at evaluation, greedy decoding
  (mandated by the assignment for every reported metric) feeds back its *own* output, and any
  early error compounds through everything after it. Scheduled sampling closes that train/
  inference gap by occasionally swapping in the model's own prediction during training too.
  Training-time only: it never touches `greedy_decode` or the evaluation path.

### One step

```python
batch → device
tf_prob = teacher_forcing_prob(global_step, warmup_steps, total_steps, scheduled_sampling_floor)
loss, n_tokens = compute_loss(model_cfg, model, batch, label_smoothing=0.1, tf_prob=tf_prob)
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
scheduler.step()
step_loss = loss.detach().item()          # after the step, so no graph is pinned
```

### Scheduled sampling mechanics

When `tf_prob < 1.0` and the model is training, `compute_loss` runs one extra `torch.no_grad()`
forward pass to get the model's own predictions, then swaps some decoder-input positions for
them at rate `1 - tf_prob` before the real (gradient-tracked) forward pass. This is the usual
parallelizable approximation of scheduled sampling for a Transformer decoder — a fully
sequential mix would cost one forward pass per position instead of one per batch.

- **Tokenized (C1–C4)**: positions of `tgt_in` (the decoder input) are swapped for the model's
  own prediction from the previous position. `<bos>` (position 0) is never touched — it has
  no "own prediction" to be replaced with.
- **BLT (C5)**: `BLTSeq2Seq.forward` takes an optional `ctx_bytes` argument (default `tgt_bytes`)
  that drives *both* exposure-bias points in that architecture at once — the patch pooling
  feeding the global decoder, and the within-patch shift feeding the local decoder — while
  the loss still supervises against the true `tgt_bytes`. `compute_loss` builds `ctx_bytes` by
  mixing in the model's own byte predictions at non-`<pad>` positions.

### The two label conventions

This is where the paths differ most, and it is easy to misread:

**Tokenized (C1–C4)** — target is `<bos> w1 … wn <eos>`. Feed all but the last, supervise all
but the first, so position `t` predicts `t+1`:

```
input:   <bos>  w1   w2  …  wn        = tgt[:, :-1]
labels:   w1    w2   w3  …  <eos>     = tgt[:, 1:]
ignore_index = PAD_ID
```

**BLT (C5)** — the model shifts internally on **two** levels: the patch stream is shifted right
by one (slot `t` holds patch `t-1`, with the learned `bos_patch` at slot 0), and the bytes
within each patch are rolled right by one with `PATCH_START` in slot 0. So its logits already
align with the raw target:

```
input:  tgt (unshifted)
labels: tgt (unshifted)
ignore_index = BYTE_PAD_ID
```

`compute_loss` returns `(loss, n_supervised_tokens)` so the epoch mean can be **token-weighted**
rather than a mean of per-batch means, which would over-weight short batches.

### Per epoch

1. Train over `loaders["train"]`, accumulating token-weighted loss, example count, peak memory.
2. **`evaluate_loss`** on `loaders["val"]` — with **label smoothing disabled**, so validation
   loss is a true likelihood and comparable across configurations.
3. Append to `history`; log to WandB if enabled.
4. **Checkpoint** to `outputs/checkpoints/<config>/best.pt` if `val_loss` improved. The
   checkpoint stores `state_dict`, `model_config`, `train_config`, `meta`, `epoch`, `val_loss`
   — everything needed to rebuild the model without consulting the current source.
5. **Early stop** after `patience=5` epochs without improvement.

Throughput is recorded as **examples/second, never tokens/second** — a "token" is a BPE subword
for C1–C4 but a raw byte for C5, so a tokens/s figure would make C5 look fastest when it is in
fact the slowest.

On exit: `outputs/history_<config>.json`, and optionally a HuggingFace push.

---

## Stage 6 — Evaluation

Runs automatically right after Stage 5 for the configuration that just trained (see Stage 0),
and can also be re-run standalone against saved checkpoints without retraining:

```
python src/train.py --evaluate --all
```

[`evaluate_config`](src/train.py#L468) per configuration:

### 6a. Rebuild

**`load_checkpoint`** reads `best.pt` and reconstructs `ModelConfig(**ckpt["model_config"])`,
then `build_model(model_cfg, ckpt["meta"])`. Architecture and metadata come **from the file**,
so editing `CONFIGS` afterwards cannot silently mismatch an old checkpoint.

Data is rebuilt with `make_dataloaders(...)` using the same `SEED`, which is what makes the
test split reproducible across processes.

### 6b. Greedy decode

**`decode_split`** runs `model.greedy_decode` over the split's loader — one **chunk** at a
time, same as training. Greedy is mandated; no sampling, no beam search; chunking doesn't
change that, it only changes what one decoded sequence covers.

**Tokenized** — start from `<bos>`, repeatedly run the decoder over the whole prefix, take
`argmax` of the last position, append. Rows that have emitted `<eos>` are frozen to `PAD`. There
is **no KV cache** — the prefix is re-run each step, so decoding is O(L²) forward passes. This
dominates evaluation wall time. Ids → text via `ds.decode_plain` (stop at `<eos>`, drop
specials).

**BLT** — patch by patch: one global decoder step produces latent `h_t`, then the local decoder
emits `TGT_PATCH_SIZE = 4` bytes autoregressively within that patch. Bytes generated so far are
re-encoded and re-pooled exactly as in training, so inference matches the training computation.
Control symbols other than EOS are replaced with a space. Bytes → text via `bytes_to_text`.

Decode time and inference peak memory are recorded here.

### 6c. Reassemble and pair with gold

Chunk predictions come back in loader order, one string per chunk. `decode_split` regroups
them by `chunk_line_ids[split]` (recorded by `make_dataloaders` when it chunked the split) and
concatenates each line's chunks in order, giving one predicted string per *line* — `preds[i]`
then corresponds to whole-line `pairs[i]`. This is why chunk order (and val/test order) is
never shuffled: it's what makes the regrouping correct.

Gold comes from `pair.plain` — the original whole-line text, not a detokenized round trip.

### 6d. Score

**`utils.compute_all_metrics(preds, golds)`** over the predicted lines. Position-indexed
metrics (bit, character, sequence accuracy) plus alignment-tolerant ones (Levenshtein, BLEU,
ROUGE), each also computed by a reference library under a `_lib` suffix. See
[METRICS.md](METRICS.md).

`evaluate_config` then adds `config`, `changed_from_base`, `params_millions`, `best_val_loss`,
`best_epoch`, `n_test_lines`, `decode_seconds`, `decode_ms_per_line`,
`inference_peak_memory_mb`, and writes `outputs/samples_<config>.txt` with per-line bit accuracy
and edit distance for the first few test lines.

### 6e. Aggregate

**`write_results`** writes `results.csv` (column order fixed by `METRIC_COLUMNS`),
`results.json`, `runtime_stats.json`, regenerates `loss_curves.png`, `metric_comparison.png`
and `memory_speed.png`, and prints the comparison table.

---

## Data shapes end to end

Tracing one C1 batch of 32 examples:

| Point | Object | Shape / type |
|---|---|---|
| Corpus | `cipher_lines[k]` | `str`, ~4,827 bits mean |
| Pair (whole line) | `Pair.cipher` / `.plain` | `str` / `str`, 8:1 length |
| Chunk (`chunk_pairs`) | `Pair.cipher` / `.plain` | `str` / `str`, 8:1 length, ≤512 bits / ≤64 chars |
| Dataset item | `(src, tgt)` | `(S,)`, `(T,)` `long`, variable |
| Collated | `batch["src"]`, `batch["tgt"]` | `(16, S_max)`, `(16, T_max)` |
| Embedded | after `_prepare` | `(16, S_max, 256)` |
| Encoder out | `memory` | `(16, S_max, 256)` |
| Decoder in | `tgt[:, :-1]` | `(16, T_max - 1)` |
| Decoder out | hidden | `(16, T_max - 1, 256)` |
| Logits | `output_proj(hidden)` | `(16, T_max - 1, 4096)` |
| Labels | `tgt[:, 1:]` | `(16, T_max - 1)` |
| Loss | scalar | `()` |

For C5 the same batch:

| Point | Shape |
|---|---|
| `batch["src"]` (bytes, padded to ×16) | `(16, Ls)` |
| Local encoder out | `(16, Ls, 256)` |
| Source patches → `memory` | `(16, Ls/16, 256)` |
| `batch["tgt"]` (bytes, padded to ×8) | `(16, Lt)` |
| Target patches | `(16, Lt/8, 256)` |
| Global latents | `(16, Lt/8, 256)` |
| Byte logits (reshaped) | `(16, Lt, 259)` |
| Labels | `(16, Lt)` unshifted |

---

## Outputs

```
outputs/
├── tokenizers/
│   ├── cipher_bpe.json      trained on the training split only
│   ├── plain_bpe.json
│   └── meta.json            vocab sizes, length caps, compression ratios
├── checkpoints/<config>/
│   ├── best.pt              state_dict + model_config + meta + epoch + val_loss
│   └── README.md            generated model card (only when --push)
├── history_<config>.json    per-epoch losses, timings, memory; plus the run summary
├── samples_<config>.txt     gold vs predicted for the first 8 test lines
├── results.csv              one row per configuration, METRIC_COLUMNS order
├── results.json             the same data, nested
├── runtime_stats.json       params, memory, throughput, epochs, wall time
├── loss_curves.png          train / val cross-entropy per epoch
├── metric_comparison.png    six accuracy and quality panels
└── memory_speed.png         memory, throughput, parameter count
```

---

## Verifying without training

Every module has a `__main__` self-test that runs standalone:

```bash
python src/models/norm.py         # LayerNorm vs F.layer_norm; RMSNorm scale invariance
python src/models/positional.py   # sinusoidal values; RoPE translation invariance
python src/models/attention.py    # mask correctness, GQA→MHA reduction, causality, greedy
python src/models/blt.py          # causality per byte, overfits one batch, uses its source
python src/utils.py               # Levenshtein vs O(nm) reference; library cross-check
python src/dataset.py             # round-trip, reassembly, BPE losslessness, padding stats
```

Two are worth calling out. `attention.py` verifies causality by perturbing one target position
and asserting that no earlier logit changes — the standard way to catch information leakage.
`blt.py` goes further: it overfits a single batch to confirm the stack can learn at all, then
feeds **mismatched sources** and asserts teacher-forced accuracy drops sharply. A seq2seq model
that quietly ignores its encoder still scores well under teacher forcing by modelling the target
language alone, then collapses into repetition at decode time; that check catches it.

For an end-to-end wiring check without a full run:

```bash
python src/train.py --config C1 --smoke --no-wandb
```
