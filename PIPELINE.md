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
4. **Dispatch** — `--evaluate` goes to Stage 6; otherwise Stage 1.

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

The split is over lines, and **one example is one whole line** — there is no windowing or
chunking anywhere in the pipeline. A line goes in whole and is scored whole.

### 1c. Pair up

**`build_splits()`** wraps each selected line as a `Pair`, so each split is a list of examples
in `line_id` order.

```
build_splits() ──► ({"train": [Pair]×4000,
                     "val":   [Pair]×500,
                     "test":  [Pair]×500},
                    plain_lines)          # gold references, indexed by line_id
```

Each `Pair` holds `line_id`, `cipher` (bit string), `plain` (text).

---

## Stage 2 — Tokenization

Only on the C1–C4 path. C5 skips this stage entirely — that *is* the C5 ablation.

**`build_tokenizers(train_ciphers, train_plains)`** reloads from `outputs/tokenizers/` if
present, otherwise trains and saves. **Inputs are the training split only**; training a
vocabulary on val or test text would leak.

Two separate tokenizers, because the alphabets do not overlap at all:

| | Cipher | Plaintext |
|---|---|---|
| Alphabet | `{"0", "1"}` | ByteLevel (all 256 bytes) |
| Pre-tokenizer | **none** | `ByteLevel(add_prefix_space=False)` |
| Vocabulary | 1024 | 4096 |
| Why | A bit string has no word boundaries, so BPE may merge anywhere. Merges become learned groupings of bits. | Whitespace round-trips byte-for-byte through `decode`. |

Both reserve the same four special ids: `<pad>=0`, `<unk>=1`, `<bos>=2`, `<eos>=3`.

The function also derives **`meta`**, computed over the training split and then carried
everywhere — into the model constructor, into WandB config, and into every checkpoint:

```
meta = {cipher_vocab_size, plain_vocab_size,   # embedding table sizes
        max_src_len, max_tgt_len,              # 100th percentile, rounded up to a multiple of 8
        src_len_mean, tgt_len_mean, src_len_max, tgt_len_max,
        src_compression, tgt_compression}      # mean characters per token
```

The caps sit at the **observed maximum**, so nothing is ever truncated. An example is a whole
line; truncating would delete the end of a document rather than trim a window.

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
(`SRC_PATCH_SIZE=16`, `TGT_PATCH_SIZE=8`), so the patch grid is rectangular.

Both return `{"src": (B, S), "tgt": (B, T)}`.

### 3c. Length-grouped batching — training only

Line lengths run from 21 to 2,670 characters. Under uniform random batching, batches pad to a
mean of ~1,113 source tokens against a mean real length of 432 — **61% of attention compute
spent on padding**.

**`LengthGroupedBatchSampler`** shuffles indices, sorts by length only *within* megabatches of
`batch_size × 50`, cuts batches, then shuffles the batch order. Padding drops to ~5% while
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

- **Optimizer** — Adam, `lr=6e-4`. No weight decay: dropout 0.1 and label smoothing 0.1
  already regularise a 12–23M model trained for at most 60 epochs on 4,000 examples.
- **Schedule** — `LambdaLR` with 250 steps of linear warmup, then cosine decay to 75% of peak.
  The shallow floor is deliberate: at a 10% floor both losses were still falling at the final
  epoch, so the schedule was winding down while the model was still learning.
- **Precision** — plain fp32. There is no autocast and no `GradScaler`.
- **`utils.set_seed(ds.SEED)`** is called again at the top, so each configuration starts from
  identical initialization regardless of what ran before it.

### One step

```python
batch → device
loss, n_tokens = compute_loss(model_cfg, model, batch, label_smoothing=0.1)
loss.backward()
grad_norm = clip_grad_norm_(model.parameters(), 1.0)
optimizer.step()
optimizer.zero_grad(set_to_none=True)
scheduler.step()
step_loss = loss.detach().item()          # after the step, so no graph is pinned
```

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

**`decode_split`** runs `model.greedy_decode` over the split's loader. Greedy is mandated; no
sampling, no beam search.

**Tokenized** — start from `<bos>`, repeatedly run the decoder over the whole prefix, take
`argmax` of the last position, append. Rows that have emitted `<eos>` are frozen to `PAD`. There
is **no KV cache** — the prefix is re-run each step, so decoding is O(L²) forward passes. This
dominates evaluation wall time. Ids → text via `ds.decode_plain` (stop at `<eos>`, drop
specials).

**BLT** — patch by patch: one global decoder step produces latent `h_t`, then the local decoder
emits `TGT_PATCH_SIZE = 8` bytes autoregressively within that patch. Bytes generated so far are
re-encoded and re-pooled exactly as in training, so inference matches the training computation.
Control symbols other than EOS are replaced with a space. Bytes → text via `bytes_to_text`.

Decode time and inference peak memory are recorded here.

### 6c. Pair with gold

`decode_split` returns predictions in loader order alongside the split's `Pair` list, so
`preds[i]` corresponds to `pairs[i]`. This is why val/test order is never shuffled.

Gold comes from `pair.plain` — the original text, not a detokenized round trip.

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
| Corpus | `cipher_lines[k]` | `str`, ~4,827 chars mean |
| Pair | `Pair.cipher` / `.plain` | `str` / `str`, 8:1 length |
| Dataset item | `(src, tgt)` | `(S,)`, `(T,)` `long`, variable |
| Collated | `batch["src"]`, `batch["tgt"]` | `(32, S_max)`, `(32, T_max)` |
| Embedded | after `_prepare` | `(32, S_max, 256)` |
| Encoder out | `memory` | `(32, S_max, 256)` |
| Decoder in | `tgt[:, :-1]` | `(32, T_max - 1)` |
| Decoder out | hidden | `(32, T_max - 1, 256)` |
| Logits | `output_proj(hidden)` | `(32, T_max - 1, 4096)` |
| Labels | `tgt[:, 1:]` | `(32, T_max - 1)` |
| Loss | scalar | `()` |

For C5 the same batch:

| Point | Shape |
|---|---|
| `batch["src"]` (bytes, padded to ×16) | `(32, Ls)` |
| Local encoder out | `(32, Ls, 256)` |
| Source patches → `memory` | `(32, Ls/16, 256)` |
| `batch["tgt"]` (bytes, padded to ×8) | `(32, Lt)` |
| Target patches | `(32, Lt/8, 256)` |
| Global latents | `(32, Lt/8, 256)` |
| Byte logits (reshaped) | `(32, Lt, 259)` |
| Labels | `(32, Lt)` unshifted |

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
