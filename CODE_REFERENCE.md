# Code Reference

Every module, class and function in `src/`, with inputs, outputs and what each one does.
For how these fit together at runtime, see [PIPELINE.md](PIPELINE.md). For metric definitions,
see [METRICS.md](METRICS.md).

```
src/
├── bpe.py              Byte-Pair Encoding from scratch: pre-tokenizers, model, trainer, decoders
├── verify_tokenizer.py differential test of bpe.py against HuggingFace `tokenizers`
├── dataset.py          corpus loading, BPE tokenizers, datasets, collation, batching
├── train.py            configs, model construction, training loop, evaluation, CLI
├── utils.py            metrics, seeding, profiling, plots
└── models/
    ├── attention.py    attention primitives + the encoder-decoder transformer
    ├── blt.py          the byte-latent (token-free) pathway for C5
    ├── norm.py         LayerNorm / RMSNorm
    └── positional.py   sinusoidal / rotary position encoding
```

Every module has a `__main__` self-test. Run any file directly to execute it.

---

## `src/dataset.py`

Turns two parallel text files into padded batches of tensors.

### Constants

| Name | Value | Meaning |
|---|---|---|
| `SEED` | 42 | Seeds the split shuffle and the batch sampler. |
| `BITS_PER_CHAR` | 8 | Each plaintext character is 8 cipher bits. |
| `N_TRAIN/VAL/TEST_LINES` | 4000/500/500 | Split sizes, over lines. |
| `MAX_LINE_CHARS` | 2670 | Longest line in the corpus, measured. No longer a model-facing length cap -- see `CHUNK_CHARS`. |
| `MAX_LINE_BITS` | 21360 | `MAX_LINE_CHARS * 8`. |
| `CHUNK_CHARS` | 64 | The actual training/decoding unit is a chunk this many plaintext characters long (512 cipher bits), not a whole line -- see `chunk_pairs`. This is the BLT source/target-length basis now. |
| `CIPHER_VOCAB_SIZE` | 1024 | Ceiling on the cipher BPE vocabulary. |
| `PLAIN_VOCAB_SIZE` | 4096 | Ceiling on the plaintext BPE vocabulary. |
| `CIPHER_ALPHABET` | `["0", "1"]` | The raw cipher text's whole character set (checked by `read_corpus`; not the BPE alphabet -- see `cipher_to_symbols`). |
| `PLAIN_ALPHABET` | space + `A-Z` + `a-z` | The 53 characters the plaintext uses, and nothing else. |
| `PLAIN_SPLIT_PATTERN` | `" ?[A-Za-z]+"` | Word pre-tokenization; each word keeps its leading space. |
| `PAD_ID, UNK_ID, BOS_ID, EOS_ID` | 0, 1, 2, 3 | Special ids, identical in both tokenizers. |

### `Pair` (frozen dataclass)

One whole corpus line, or (after `chunk_pairs`) one chunk of one -- the same class serves
both; only `chunk_pairs` changes what's actually inside. Fields: `line_id: int`, `cipher: str`
(bit string), `plain: str`. The invariant `len(cipher) == 8 * len(plain)` holds by
construction in both cases. Multiple chunks of one line share its `line_id`.

### Corpus functions

**`read_corpus() -> (list[str], list[str])`**
Reads both files, drops blank lines, and raises if the line counts differ, if any line
violates the 8× length relation, or if either side uses a character outside its declared
alphabet. That last check is what lets the two vocabularies below be closed sets: a stray
character would otherwise become `<unk>` and its line could never be reproduced. Returns
`(cipher_lines, plain_lines)`.

**`split_line_ids(n_lines, seed=SEED) -> dict[str, list[int]]`**
Shuffles `range(n_lines)` with a seeded RNG, slices 4000/500/500, returns sorted id lists under
`"train"`, `"val"`, `"test"`. Splitting over lines guarantees no line appears in two splits.

**`build_splits() -> (dict[str, list[Pair]], list[str])`**
Wraps each selected line as a `Pair`. Returns the per-split whole-line lists (the ground truth
evaluation scores against) and the raw plaintext lines (indexed by `line_id`).

**`chunk_pairs(pairs, chunk_chars=CHUNK_CHARS) -> list[Pair]`**
Expands each whole-line `Pair` into consecutive `chunk_chars`-character pieces, cutting the
raw strings (`cipher[8*start:8*end]`, `plain[start:end]`) before either tokenizer runs, so
alignment between the two sides is exact by construction — no BPE token can ever need to
know where a chunk boundary falls, since chunking happens first. A line's last chunk is
whatever remains (1 to `chunk_chars` characters). All of one line's chunks keep its `line_id`
and stay consecutive, in order, in the returned list — this is what lets callers regroup a
line's chunk predictions back together (`train.py`'s `decode_split`). This is the function
that turns "one example is one whole line" into "one example is one chunk": every dataset in
`make_dataloaders` is built from its output, not from `build_splits`'s output directly.

### Tokenizer functions

**`_train_bpe(corpus, vocab_size, initial_alphabet, pre_tokenizer) -> Tokenizer`**
Trains one BPE model — from `src/bpe.py`, not the `tokenizers` library — with
`min_frequency=2`, the four special tokens, and a `Fuse` decoder. `vocab_size` is a ceiling,
not a guarantee: training stops early once no pair occurs more than once.

**`cipher_to_symbols(cipher: str) -> str`** / **`symbols_to_cipher(symbols: str) -> str`**
The preprocessing that makes cross-byte BPE possible on the cipher side. Each 8-bit unit
(one plaintext character's cipher byte) is interpreted as a byte value 0-255 and mapped to a
single character via the same byte<->character bijection `ByteLevel` uses (`BYTE_TO_CHAR` /
`CHAR_TO_BYTE` in `bpe.py`). `symbols_to_cipher` is the exact inverse.

**`train_cipher_tokenizer(cipher_texts) -> Tokenizer`**
Runs `cipher_to_symbols` over the corpus, then trains BPE with **no pre-tokenizer** and the
full 256-value byte alphabet (`ByteLevel.alphabet()`, not just the 126 byte values this
corpus's cipher actually realizes — all <128, since XORing two mostly-7-bit alphabets never
sets the high bit; the other 130 sit in the trained vocabulary as permanently zero-frequency
tokens). With no pre-tokenizer, `Tokenizer.train_from_iterator` treats each whole *chunk* as
one "word" (see its `if self.pre_tokenizer is None` branch in `bpe.py`), so merges are free to
combine adjacent symbols into tokens spanning multiple plaintext characters — never a fraction
of one, since the symbol *is* the character's whole byte, but also never confined to just one,
and never across a chunk boundary, since `cipher_texts` here is already chunk-level (see
`chunk_pairs`).

**`decode_cipher(cipher_tok, ids) -> str`**
`cipher_tok.decode` followed by `symbols_to_cipher`. Unlike `decode_plain`, nothing wraps the
source in `<bos>`/`<eos>`, since the cipher is only ever encoder input.

**`train_plain_tokenizer(plain_texts) -> Tokenizer`**
`Split(" ?[A-Za-z]+", "isolated")` over `PLAIN_ALPHABET`. Each pre-token is a word carrying
its own leading space, so the space is an ordinary vocabulary character rather than a marker
and decoding is plain concatenation — still exact, and without spending 203 of a byte-level
alphabet's 256 symbols on bytes the corpus never contains.

**`build_tokenizers(cipher_texts, plain_texts, force=False) -> (Tokenizer, Tokenizer, dict)`**
Trains both tokenizers or reloads them from `outputs/tokenizers/` if already present. Inputs
must be *lists of strings from the training split's chunks* (`chunk_pairs`, not `build_splits`
directly) — merge frequencies should reflect what the model actually encodes, and passing
val/test text would also leak.
A shared vocabulary would be meaningless — the two alphabets do not overlap.

Also derives `meta`, which is persisted to `meta.json` and later travels inside every
checkpoint:

| Key | Meaning |
|---|---|
| `cipher_vocab_size`, `plain_vocab_size` | Actual vocabulary sizes; feed the embedding tables. |
| `max_src_len`, `max_tgt_len` | Length caps, at the observed maximum rounded up to a multiple of 8. |
| `src_len_mean`, `tgt_len_mean`, `src_len_max`, `tgt_len_max` | Length statistics. |
| `src_compression`, `tgt_compression` | Mean characters per token — how much the BPE actually compresses. |

**`load_tokenizers() -> (Tokenizer, Tokenizer, dict)`**
Reloads all three artefacts from disk. Raises `FileNotFoundError` with a hint if absent.

**`decode_plain(plain_tok, ids) -> str`**
Ids back to text: stops at the first `EOS_ID`, drops `PAD_ID` and `BOS_ID`, then calls the
tokenizer's `decode`. This is the inverse used on every prediction.

### Dataset classes

**`TokenizedSeq2SeqDataset(pairs, cipher_tok, plain_tok, max_src_len, max_tgt_len)`** — C1–C4

Holds two parallel lists: `src_ids` (cipher BPE ids, truncated to `max_src_len`) and
`tgt_ids` (plaintext ids wrapped as `[BOS] … [EOS]`, body truncated to `max_tgt_len - 2`).
Both sides are encoded eagerly in `__init__` via `encode_batch`, because per-item encoding
would dominate step time.

- `__getitem__(i)` → `(src_tensor, tgt_tensor)`, both 1-D `torch.long`, unpadded.
- `source_lengths()` → `list[int]` of source id lengths, for the length-grouped sampler.

**`ByteSeq2SeqDataset(pairs)`** — C5

No tokenizer at all. `__getitem__` encodes on the fly with `latin-1`, so a cipher character
`'0'` becomes byte 48 and `'1'` becomes 49 — C5 consumes the same input file as C1–C4 with only
the vocabulary removed. The target gets a trailing `BYTE_EOS_ID` and **no BOS** (the BLT model
supplies a learned `bos_patch` internally). Nothing is truncated.

- `source_lengths()` → character lengths of `pair.cipher` (1 byte = 1 token here).

### Collation and batching

**`collate_tokenized(batch) -> {"src": (B,S), "tgt": (B,T)}`**
`zip(*batch)` transposes the list of `(src, tgt)` pairs into a tuple of sources and a tuple of
targets, then each side is padded with `PAD_ID` to that batch's own longest sequence — not to
the global cap, which would waste most of the compute.

**`CollateBytes(src_patch=SRC_PATCH_SIZE, tgt_patch=TGT_PATCH_SIZE)`**
Same idea for C5, but pads up to a whole multiple of the patch size so the patch grid is
rectangular, using `BYTE_PAD_ID`. A class rather than a closure so it stays picklable for
DataLoader workers.

**`LengthGroupedBatchSampler(lengths, batch_size, megabatch_factor=50, seed=SEED)`**
Yields lists of dataset indices. Chunks are mostly `CHUNK_CHARS` characters, with an
occasional shorter one at a line's end, so uniform random batching already pads little (mean
34 padded vs. mean 33 real tokens per batch, measured). This shuffles first, sorts by length
only *within* megabatches of `batch_size * 50`, cuts batches, then shuffles the batch order.
Padding drops to ~1% while batch membership still changes every epoch (the `epoch` counter
advances the RNG on each `__iter__`). Training loader only — val and test keep their natural
order so predictions stay aligned with their `Pair` list.

**`make_dataloaders(model_cfg, train_cfg, splits=None, tokenizers=None) -> dict`**
The single entry point that assembles everything. Branches on `model_cfg.is_blt` to pick the
byte or tokenized path. Returns:

| Key | Contents |
|---|---|
| `pairs` | `{split: list[Pair]}`, **whole-line** — carries `line_id` and gold text for evaluation. |
| `chunk_line_ids` | `{split: list[int]}` — the `line_id` of each chunk in `datasets`/`loaders` order, same length as the chunked dataset; lets `decode_split` regroup chunk predictions back into lines. |
| `datasets` | `{split: Dataset}`, built from `chunk_pairs(pairs[split])`, not `pairs[split]` directly. |
| `loaders` | `{split: DataLoader}`; train uses the length sampler when enabled. |
| `meta` | Tokenizer metadata, or `{max_src_len: CHUNK_CHARS * BITS_PER_CHAR, max_tgt_len: CHUNK_CHARS + 1}` for BLT. |
| `cipher_tok`, `plain_tok` | Present only on the tokenized path. |

---

## `src/models/norm.py`

**`LayerNorm(d_model, eps=1e-5)`** — `(x - mean) / sqrt(var + eps) * gamma + beta`. Two
parameter vectors.

**`RMSNorm(d_model, eps=1e-6)`** — `x / sqrt(mean(x²) + eps) * gamma`. Rescaling only: no
centring, no bias, so one parameter vector and one fewer reduction.

Both compute their statistics in the input dtype. Training is fp32 throughout, so the float32
upcast these once carried (a guard against fp16 variance underflow under AMP) is gone.

**`build_norm(kind, d_model) -> nn.Module`** — dispatches on `"layernorm"` / `"rmsnorm"`.
This is the C4 switch, and it is called at every normalization site in the codebase.

---

## `src/models/positional.py`

**`SinusoidalPositionalEncoding(d_model, max_len=4096, dropout=0.0)`**
Precomputes the fixed `PE[pos, 2i] = sin(pos / 10000^(2i/d))` table as a non-persistent
buffer. `forward(x, offset=0)` adds the slice `pe[offset : offset+T]` to `(B, T, d_model)` and
applies dropout. Absolute position, added **once** before the first layer. The `offset`
argument supports incremental decoding.

**`rotate_half(x)`** — maps `(x1, x2) -> (-x2, x1)` over the two halves of the last dimension.
The LLaMA / GPT-NeoX layout, where element `i` pairs with element `i + d/2`.

**`RotaryPositionalEmbedding(d_head, max_len=4096, base=10000.0)`**
Caches `cos`/`sin` tables. `forward(q, k, offset=0)` returns rotated `q` and `k` — never `v`.
Query and key head counts may differ, which is what lets GQA rotate before KV expansion.
`get_tables(seq_len, offset, device, dtype)` returns tables shaped `(1, 1, T, d_head)` ready to
broadcast over `(B, H, T, D)`.

The distinction that matters: sinusoidal adds a vector once and encodes **absolute** position;
RoPE adds nothing and instead rotates inside every attention block, so the logit `q_m · k_n`
depends only on `m - n` — **relative** position, refreshed at every layer. This is the C2 switch.

---

## `src/models/attention.py`

`nn.MultiheadAttention`, `nn.Transformer` and `F.scaled_dot_product_attention` are not used
anywhere.

**Mask convention (used everywhere):** a boolean tensor broadcastable to `(B, H, T_q, T_k)`
where **`True` means attention is allowed**. Blocked logits are filled with the dtype minimum
rather than `-inf`, so a fully-blocked row softmaxes to uniform instead of `NaN`.

**`scaled_dot_product_attention(q, k, v, mask=None, dropout=None) -> (context, weights)`**
`softmax(QKᵀ / √d_k)V`. Inputs `(B, H, T, D)`. The `(B, H, T_q, T_k)` score tensor is the
largest allocation in the model, so the mask fill is done in place (`masked_fill_`) and the
softmax keeps the input dtype rather than round-tripping through float32.

**`MultiHeadAttention(d_model, n_heads, dropout=0.1)`**
Four square projections. `forward(query, key_value=None, mask=None, rope=None)` — `key_value`
defaults to `query`, so the same module serves both self- and cross-attention. Returns the
output tensor. Attention weights are not returned: no caller used them, and holding one score
tensor per layer alive was pure memory cost.

**`GroupedQueryAttention(d_model, n_heads, n_kv_heads, dropout=0.1)`**
Keeps all `n_heads` query heads but projects only `n_kv_heads` key/value heads. `w_k` and `w_v`
are `d_model → n_kv_heads * d_head` — a smaller KV cache and fewer parameters, for some loss of
capacity. `_expand_kv` uses `repeat_interleave`, so contiguous query heads share a KV head.
RoPE is applied *before* expansion, so each KV head is rotated once. With
`n_kv_heads == n_heads` it reduces exactly to MHA, which the self-test asserts numerically.

**`build_attention(cfg, d_model=None, n_heads=None)`** — the C3 switch. Optional overrides let
`blt.py` build narrow local blocks from the same config object.

**`causal_mask(seq_len, device) -> (1, 1, T, T)`** — lower-triangular, `True` where allowed.

**`padding_mask(tokens, pad_id) -> (B, 1, 1, T)`** — `True` on real tokens.

**`FeedForward(d_model, d_ff, dropout)`** — `Linear → GELU → dropout → Linear`.

**`EncoderLayer(cfg)`** — Pre-LN: `x = x + Sublayer(Norm(x))`, self-attention then FFN.

**`DecoderLayer(cfg)`** — Pre-LN with three sublayers: masked self-attention, cross-attention
over `memory`, FFN. RoPE is passed to self-attention only. Cross-attention is skipped when
`memory is None`. The optional `reshape=(fold, unfold)` argument wraps the cross-attention
query, which is how `blt.py` runs self-attention with patches on the batch axis and
cross-attention with them on the query axis — see the `LocalByteDecoder` note below.

**`Seq2SeqTransformer(cfg, src_vocab_size=None, tgt_vocab_size=None, pad_id=0, max_len=4096, tie_embeddings=True)`**

The full stack, and the model for C1–C4. Two modes:

- **Token mode** (vocab sizes given) — owns the embeddings, returns logits. Output projection
  weights are tied to the target embedding.
- **Latent mode** (both `None`) — consumes and returns `d_model` vectors. `blt.py` feeds it
  patch representations, which makes C5's global model *literally this same class*.

Exactly one positional mechanism is active, selected by `cfg.pos_encoding`. **RoPE is applied
in encoder and decoder self-attention only, never cross-attention** — there the query indexes
the target and the key indexes the source, two unrelated coordinate systems, so a relative
offset between them is meaningless.

| Method | Signature | Returns |
|---|---|---|
| `_init_weights` | static | Xavier-uniform on Linear, `N(0, 0.02)` on Embedding, zeros at `padding_idx`. |
| `_prepare(x, embed, offset=0)` | — | Embeds and scales by `√d_model` (keeping magnitude comparable to the PE), then adds absolute PE if active. |
| `encode(src, src_mask=None)` | `(B,S)` ids or `(B,S,d)` vectors | `memory (B, S, d_model)`. |
| `decode(tgt_in, memory, tgt_mask, memory_mask)` | — | hidden states `(B, T, d_model)`. |
| `build_tgt_mask(tgt_in)` | — | causal, ANDed with target padding when ids are available. |
| `forward(src, tgt_in, ...)` | teacher-forced | logits `(B, T, V)` in token mode, hidden states in latent mode. |
| `greedy_decode(src, max_len, bos_id, eos_id)` | — | `(B, L)` ids starting at `<bos>`, padded after each row's `<eos>`. |

`greedy_decode` re-runs the whole prefix each step — no KV cache. Correct but O(L²) forward
passes, which is why decoding dominates evaluation wall time. It early-exits once every row has
emitted `<eos>`.

---

## `src/models/blt.py`

The token-free pathway for C5.

```
source bytes --LocalByteEncoder--> byte states --PatchPooler--> source patches
target bytes --LocalByteEncoder--> byte states --PatchPooler--> target patches
                      GlobalTransformer (Seq2SeqTransformer in latent mode)
                      patch latents --LocalByteDecoder--> byte logits
```

The global transformer is the same class as C1 with the same depth, width, sinusoidal encoding,
MHA and LayerNorm, so **only the representation layer differs**: a learned vocabulary is
replaced by learned pooling over raw bytes.

Patching is fixed-stride rather than entropy-driven — a simplification of Meta's BLT that the
assignment permits. Hash n-gram byte embeddings and the local/global/local structure are kept.

### Constants

`BYTE_PAD_ID=256`, `BYTE_EOS_ID=257`, `BYTE_PATCH_START_ID=258`, `BYTE_VOCAB_SIZE=259`.
Control ids sit *above* 255 so a raw byte's id is its own numeric value — no offset arithmetic
anywhere. `SRC_PATCH_SIZE=16` (16 cipher bits = 2 characters per source patch),
`TGT_PATCH_SIZE=8` (8 plaintext bytes per target patch).

### Components

**`LocalConfig` / `_local_cfg(cfg)`** — projects the main `ModelConfig` down to the narrow
geometry the byte-level blocks run on (`d_local`, `local_n_heads`, `4 * d_local` FFN). Local
blocks are always MHA — they are not part of the C3 ablation — but inherit `norm` and `dropout`.

**`ByteEmbedding(d_local, ngram_sizes=(3,4), ngram_buckets=8192)`**
Byte lookup plus hashed byte n-gram embeddings. A single byte carries almost nothing —
`'0'`/`'1'` on the source side — so each position is augmented with embeddings of the n-grams
*ending* there. `_hash_ngrams` is a polynomial rolling hash modulo `ngram_buckets`, so 260⁴
possible 4-grams cost only `ngram_buckets` rows. Windows look strictly backwards, which keeps
this usable inside the causal target encoder.

**`_to_blocks` / `_from_blocks` / `_mask_to_blocks`**
Reshape helpers that fold a `(B, L, d)` sequence into `(B * n_blocks, window, d)` so attention
is block-local: the cross-block quadrants of the attention matrix are never materialised, which
is what keeps the local modules affordable over thousands of bytes.

**`LocalByteEncoder(cfg, causal=False, max_len=4096, ngram_buckets=None)`**
Shallow narrow transformer over raw bytes. Absolute position is added over the *whole* sequence
before blocking, so a byte knows where it sits globally even though it only attends locally.
`causal=True` additionally restricts each byte to the past — required on the target side.
`(B, L)` byte ids → `(B, L, d_local)`.

**`PatchPooler(cfg, patch_size)`**
Compresses a fixed stride of byte states into one patch vector by cross-attention: a learned
query attends over its patch's bytes, so the model decides which bytes matter. A mean-pool
residual (over valid bytes only) keeps the output sensible before attention has learned
anything. The patch axis is folded into the batch axis so every patch pools independently.
Returns `(patches (B, N, d_model), patch_valid (B, N))`.

**`LocalByteDecoder(cfg, patch_size, byte_embed)`**
Generates one patch's bytes autoregressively from that patch's global latent `h_t`. Two design
details, both found empirically:

- **Latent expansion.** `h_t` is expanded into `patch_size` conditioning vectors, one per byte
  slot, rather than used as a single vector. One vector describing 8 characters starves the
  decoder, which then falls back on within-patch English statistics — enough to score well
  under teacher forcing, but it collapses into repetition at inference. Causality is safe:
  every slot is a function of `h_t` alone, and `h_t` came from patches `0..t-1`.
- **Direct source access.** The decoder also cross-attends to the global *encoder* memory.
  Without it, teacher forcing lets it see the 7 preceding true bytes of its own patch, which is
  enough to emit plausible English unaided — so the global path gets little gradient and `h_t`
  collapses into a positional code. The memory depends only on the source, so it cannot leak a
  target byte.
- **The memory is never copied per patch.** Self-attention runs per patch, so patches fold into
  the batch axis: `(B·N, P, d)`. Cross-attention runs against the shared source memory, so
  there the patch axis folds into the *query* axis instead — `(B, N·P, d)` against an
  unexpanded `(B, Ns, d)` — via the `reshape` argument on `DecoderLayer`. Materialising one
  memory copy per patch is a `(B·N, Ns, d)` tensor, roughly 11 GB per decoder layer for the
  longest batch in this corpus. Avoiding it is what keeps C5 training within ~14 GB; an
  earlier build that materialised the copy peaked at ~61 GB.

Returns byte logits `(B, N, P, BYTE_VOCAB_SIZE)`.

**`BLTSeq2Seq(cfg, max_len=4096)`**

| Method | Purpose |
|---|---|
| `encode_source(src_bytes)` | Local-encode → pool → global encode. Returns `(memory, memory_mask)`. |
| `_target_patches(tgt_bytes)` | Causal local-encode → pool. Returns `(patches, valid)`. |
| `_shift_for_local_decoder(tgt_bytes)` | `(B, Lt)` → `(B, Nt, P)` with each patch's bytes rolled right by one and slot 0 set to `PATCH_START`, so byte `j` conditions on `0..j-1` and never on itself. |
| `forward(src_bytes, tgt_bytes, ctx_bytes=None)` | Teacher-forced. Returns `(B, Lt, BYTE_VOCAB_SIZE)`. |
| `greedy_decode(src_bytes, max_bytes)` | Patch by patch: one global step per latent, then the local decoder emits `tgt_patch` bytes. |

`forward` shifts on **two levels**: the patch stream is shifted right by one (slot `t` holds
patch `t-1`, with the learned `bos_patch` at slot 0), and the bytes within each patch are
shifted by one. That is why `compute_loss` supervises BLT against the *unshifted* `tgt` — the
logits already align with the raw target bytes, unlike the tokenized path.

`ctx_bytes` (default `tgt_bytes`) is what actually drives both shifts above — `tgt_bytes`
still supplies the loss labels regardless of what `ctx_bytes` is. `compute_loss` uses this to
inject scheduled sampling: a version of `ctx_bytes` with some positions replaced by the
model's own prediction, so the conditioning (not the labels) gets a taste of the model's own
mistakes at both exposure-bias points in this architecture (patch-to-patch and
byte-within-patch) at once.

The source-side encoder uses `src_ngram_buckets=512` rather than 8192, because only 2³ distinct
3-grams exist in a binary alphabet.

**`bytes_to_text(byte_row) -> str`** — one decoded row to a string, stopping at EOS and
dropping control ids.

---

## `src/utils.py`

Metrics are documented in full in [METRICS.md](METRICS.md); this section covers the rest.

### Seeding and environment

**`set_seed(seed=42, deterministic=True)`** — seeds `random`, `numpy`, `torch`, CUDA, and
`PYTHONHASHSEED`; sets `cudnn.deterministic` and disables `cudnn.benchmark`. Called at the top
of `main()` and again at the start of every `train_one`, so each configuration starts from
identical initialization regardless of what ran before it.

**`load_env(env_file) -> dict[str, str]`**
Reads `KEY=VALUE` pairs from `.env` into `os.environ`, skipping empty values. Uses
`os.environ.setdefault`, so a variable already exported in the shell **wins over the file**.
Returns the values it read.

Called once, from [`main()`](src/train.py#L605). It exists solely to supply the three
credentials used by the optional integrations: `WANDB_API_KEY` (read by `_init_wandb`, which
falls back to offline mode without it) and `HF_TOKEN` / `HF_USERNAME` (read by `push_to_hub`,
which skips the upload without them). **Nothing in training, decoding or metric computation
depends on it** — the pipeline runs identically if `.env` is absent.

### Profiling

| Function | Returns |
|---|---|
| `count_parameters(model)` | `(total, trainable)` parameter counts. |
| `reset_peak_memory()` | Resets CUDA peak-memory stats; no-op on CPU. |
| `peak_memory_mb()` | Peak allocated CUDA memory since reset, in MiB; `0.0` on CPU. |
| `Timer()` | Context manager exposing `.elapsed`. **Synchronises CUDA** on both enter and exit, so GPU work is actually counted rather than timing the async launch. |
| `human_time(seconds)` | `"1h02m03s"` / `"5m07s"`. |

### Plots

`CONFIG_COLORS` fixes one colour per configuration so C1–C5 are visually consistent across
every figure. `_style()` selects the Agg backend and sets shared rcParams; `_bar_panels` is the
shared grid-of-bars helper.

- **`plot_loss_curves(histories, path)`** — two panels, train and validation cross-entropy
  against epoch, one line per configuration.
- **`plot_metric_comparison(results, path)`** — six panels: bit, character and sequence
  accuracy, normalised Levenshtein, BLEU, ROUGE-L.
- **`plot_memory_speed(stats, path)`** — four panels: training and inference peak memory,
  training throughput, parameter count.

**`save_json(obj, path)`** — creates parent directories and writes indented JSON with
`default=str`.

---

## `src/train.py`

### Configuration

**`ModelConfig`** — architecture of one configuration. Four ablated axes (`pos_encoding`,
`attention`, `norm`, `tokenization`) plus shared geometry: `d_model=256`, `n_heads=8`
(`d_head=32`), `n_kv_heads=2`, 4 encoder and 4 decoder layers, `d_ff=2048`, `dropout=0.1`.
BLT-only fields (`d_local`, `local_attn_window`, `ngram_buckets`, …) are ignored unless
`tokenization == "blt"`. Properties: `is_blt`, `d_head`; `to_dict()` for serialization.

**`TrainConfig`** — shared by *all five* configurations: 50 epochs, batch size 16, Adam at
`lr=6e-4` with 250 warmup steps, gradient clipping at 1.0, label smoothing 0.1,
`scheduled_sampling_floor=0.7`, early-stopping patience 5, length grouping on, `log_every=50`.
No weight decay and no mixed precision — training is plain fp32.

**`CONFIGS`** — C2–C5 are built from C1 with `dataclasses.replace`, changing exactly one named
field each. This makes it structurally impossible for more than the named axis to differ:

| Config | Changed | Setting |
|---|---|---|
| C1 | — (base) | sinusoidal, MHA, LayerNorm, BPE |
| C2 | Positional encoding | `pos_encoding="rope"` |
| C3 | Attention | `attention="gqa"` (8 query / 2 KV heads) |
| C4 | Normalization | `norm="rmsnorm"` |
| C5 | Tokenization | `tokenization="blt"` |

**`get_config(name)`** — case-insensitive lookup, raises `KeyError` listing valid names.
**`describe_configs()`** — formatted comparison table as a string.

### Model and loss

**`build_model(model_cfg, meta) -> nn.Module`**
`BLTSeq2Seq` when `is_blt`, otherwise `Seq2SeqTransformer` with the vocabulary sizes from
`meta`. `max_len` is `max(max_src_len, max_tgt_len) + 64`, sized from the data.

**`teacher_forcing_prob(step, warmup_steps, total_steps, floor) -> float`**
1.0 through the same warmup the LR schedule uses, then linear decay to `floor` by the end of
training. The probability that a given decoder-input position uses the true previous token
rather than the model's own prediction for it (scheduled sampling). Training is otherwise
100% teacher-forced, so the model never practices recovering from its own mistakes, while
`greedy_decode` (mandated for every reported metric) feeds back exactly that at evaluation
time. Training-time only — evaluation and `greedy_decode` are untouched.

**`compute_loss(model_cfg, model, batch, label_smoothing, tf_prob=1.0) -> (loss, n_supervised_tokens)`**
Teacher-forced forward pass, branching on the two label conventions:

- **Tokenized** — target is `<bos> w1..wn <eos>`; feed `tgt[:, :-1]`, supervise `tgt[:, 1:]`,
  so position `t` predicts `t+1`. Ignore index `PAD_ID`.
- **BLT** — the model shifts internally on both levels, so its logits already align with the
  raw `tgt`. Ignore index `BYTE_PAD_ID`.

When `tf_prob < 1.0` and `model.training`, one extra `torch.no_grad()` forward pass gets the
model's own predictions, and some decoder-input positions (`ctx_bytes` for BLT, `tgt_in` for
the tokenized path) are swapped for them at rate `1 - tf_prob` before the real forward pass —
the parallelizable approximation of scheduled sampling for a Transformer decoder. `<bos>` is
never replaced on the tokenized path. `evaluate_loss` never passes `tf_prob`, so validation
stays a pure teacher-forced likelihood.

Returns the token count alongside the loss so the caller can compute a token-weighted mean
rather than a mean-of-means.

**`lr_lambda_factory(warmup_steps, total_steps)`** — linear warmup, then cosine decay to 10% of
the peak. Returns the multiplier function for `LambdaLR`.

**`evaluate_loss(model_cfg, model, loader, device) -> float`**
Token-weighted mean cross-entropy with **label smoothing disabled**, so it is a true
likelihood and comparable across configurations. Restores `model.train()` on exit.

### Training

**`train_one(config_name, train_cfg, device, use_wandb=True, smoke_steps=0, push=False) -> dict`**

Builds data, model, optimizer and scheduler; runs the epoch loop; checkpoints on improvement;
early-stops; writes `outputs/history_<config>.json`; optionally pushes to HuggingFace. The
step loss is read out with `.detach().item()` *after* `optimizer.step()`, so no autograd graph
is pinned across the epoch.

Tracks per-epoch `train_loss`, `val_loss`, `epoch_seconds`, `examples_per_sec`,
`peak_memory_mb`, `lr`. Throughput is measured in **examples/second**, never tokens/second — a
"token" is a BPE subword for C1–C4 but a raw byte for C5, which would make C5 look artificially
fast.

Returns a summary dict with parameter counts, best validation loss and epoch, epochs run, wall
time, mean seconds per epoch, peak memory, the full history, and the model config.

**`_init_wandb(...)`** — starts a run logging the merged model/train/data config. Falls back to
offline mode when `WANDB_API_KEY` is unset.

**`push_to_hub(config_name, ckpt_path, summary) -> str | None`**
Uploads `best.pt`, a generated model card, and both tokenizer JSONs. Returns the repo URL, or
`None` (with a message) when `HF_TOKEN` / `HF_USERNAME` are missing.

**`_model_card(config_name, summary, repo_id) -> str`** — generates the HuggingFace README with
the configuration table and a load snippet.

### Evaluation

**`load_checkpoint(config_name, device) -> (model, model_cfg, meta, ckpt)`**
Rebuilds a trained model from `outputs/checkpoints/<config>/best.pt`. The checkpoint carries
its own `model_config` and `meta`, so the architecture is reconstructed from the file rather
than from the current source — a config edit cannot silently mismatch an old checkpoint.

**`decode_split(model, model_cfg, data, split, device, limit_lines=None) -> (predictions, pairs, stats)`**
Greedily decodes one split **one chunk at a time** (`data["loaders"][split]` is chunk-level),
dispatching to the byte or token decoder and converting ids back to text (`bytes_to_text` or
`ds.decode_plain`). With `limit_lines`, restricts to the first N *lines*: since a line's chunks
are consecutive in `chunk_line_ids[split]`, their kept-line's chunks form a contiguous prefix,
found with a linear scan, and a `Subset` loader is rebuilt over just that prefix — note
`loader.batch_size` is `None` when a `batch_sampler` was used, hence the fallback. Chunk
predictions are then regrouped by `chunk_line_ids[split]` and concatenated in order. Returns
one prediction string **per line** (post-reassembly), the matching whole-line `Pair` list, and
decode time and peak memory.

**`evaluate_config(config_name, device, split="test", limit_lines=None, n_samples=8) -> dict`**
Loads the checkpoint, decodes, scores against the gold lines, writes
`outputs/samples_<config>.txt` with per-line bit accuracy and edit distance for the first few
lines, and returns the metric dict augmented with identification and runtime fields.

**`METRIC_COLUMNS`** — fixed column order for `results.csv`, including the `_lib` cross-check
columns.

**`write_results(results)`** — writes `results.csv`, `results.json` and `runtime_stats.json`,
regenerates all three figures, and prints the comparison table.

### CLI

**`main()`** — argument parsing and dispatch. Loads `.env`, seeds, selects the device
(`cuda` unless `--cpu` or unavailable), then either evaluates or trains.

Without `--evaluate`, `main()` loops over the requested configuration(s) and, for each one,
calls `train_one` and then — unless `--smoke` — immediately `evaluate_config` and
`write_results` on the accumulated results dict, before moving to the next configuration. So
`--all` trains and evaluates C1, then C2, and so on; `results.csv`/`results.json`/the three
figures are current after every configuration finishes, not just at the end of the run.

| Flag | Effect |
|---|---|
| `--config C1..C5` | Which configuration (default `C1`). |
| `--all` | Every configuration in sequence, train-then-evaluate each before the next. |
| `--evaluate` | Decode the test split against existing checkpoints instead of training. |
| `--epochs`, `--batch-size`, `--lr` | Override the corresponding `TrainConfig` field. |
| `--smoke`, `--smoke-steps N` | Run a few steps as a wiring check (default 50); skips the automatic post-train evaluation. |
| `--limit-lines N` | Evaluate only the first N test lines. |
| `--no-wandb` | Disable logging. |
| `--push` | Upload the checkpoint to HuggingFace. |
| `--cpu` | Force CPU. |

With `--evaluate --all`, configurations without a checkpoint are skipped with a warning rather
than failing.
