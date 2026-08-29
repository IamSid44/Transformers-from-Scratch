# The Byte Latent Transformer (C5)

What a BLT is, how this repository's simplified version is built, and why the first
implementation of it collapsed. Everything here refers to
[`src/models/blt.py`](src/models/blt.py) unless another file is named.

---

## 1. The idea in one paragraph

A normal transformer needs a vocabulary. Text is cut into subword tokens by BPE, each token
gets a row in an embedding table, and the model never sees a byte. That buys compression — a
sequence of 60 characters becomes ~18 tokens, and attention cost falls with the square of
that — at the price of a fixed, corpus-specific vocabulary, an `<unk>` failure mode, and a
tokenizer that has to be trained and shipped alongside the model.

A **Byte Latent Transformer** deletes the vocabulary. It reads raw bytes, groups them into
fixed-size groups called **patches**, learns one vector per patch, and runs an ordinary
transformer over *those vectors* instead of over token embeddings. The compression that BPE
got from a lookup table, BLT gets from a learned pooling operation. Meta's paper (Pagnoni et
al., 2024) places patch boundaries where a small byte-level language model is surprised;
the assignment asks for "a simplified BLT", so here the boundaries are at fixed strides,
which keeps the patch grid rectangular and therefore batchable.

The consequence that matters for the ablation: **C5 differs from C1 only in its
representation layer.** The transformer in the middle is literally the same
`Seq2SeqTransformer` class, at the same depth, width, head count, normalization and
positional encoding. Only the way text becomes vectors — and the way vectors become text —
has changed.

---

## 2. The pipeline

```
   cipher bytes                                                    plaintext bytes
   "0110...", 512 of them                                          "The quick...", 65 of them
        │                                                                   ▲
        ▼                                                                   │
  ┌───────────────────┐                                        ┌────────────────────────┐
  │ LocalByteEncoder  │  2 layers, d_local=256                  │   LocalByteDecoder     │
  │ block-local attn  │  byte + hashed n-gram embeddings        │ emits P bytes per patch│
  └───────────────────┘                                        └────────────────────────┘
        │ (B, 512, 256)                                                     ▲ (B, N, P, 259)
        ▼                                                                   │ latents (B, N, 256)
  ┌───────────────────┐                                        ┌────────────────────────┐
  │   PatchPooler     │  stride 32 → 16 patches                │  patch latents h_0..h_N│
  │ learned query +   │                                        └────────────────────────┘
  │ masked mean       │                                                     ▲
  └───────────────────┘                                                     │
        │ (B, 16, 256)                                                      │
        ▼                                                                   │
  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  Seq2SeqTransformer in LATENT MODE — the same class C1 uses, vocab sizes = None   │
  │  4 encoder + 4 decoder layers, d_model 256, d_ff 2048, 8 heads,                   │
  │  sinusoidal absolute PE, MHA, LayerNorm                                           │
  └──────────────────────────────────────────────────────────────────────────────────┘
                                        ▲
                                        │ target patches (B, N, 256)
                            ┌───────────────────────┐
                            │ LocalByteEncoder      │  causal
                            │ + PatchPooler (stride 4)│
                            └───────────────────────┘
                                        ▲
                                 plaintext bytes
```

Read it as four stages.

### Stage 1 — bytes become vectors (`ByteEmbedding`, `LocalByteEncoder`)

`ByteEmbedding` is a 259-row table: the 256 real byte values plus `<pad>`, `<eos>` and a
`<patch-start>` marker. Control ids sit *above* 255 so a raw byte's id is just its own
numeric value; there is no offset arithmetic anywhere in the file.

A single byte carries almost nothing. On the source side it is literally the ASCII character
`'0'` or `'1'` — **one bit of information**. So each position is additionally given the
embedding of the byte *n*-grams ending there (n = 3 and 4), hashed into a fixed number of
buckets by a polynomial rolling hash. Hashing is what makes this affordable: there are 259⁴
possible 4-grams and only `ngram_buckets` rows. The windows look strictly backwards, which is
what lets the same module be reused inside the causal target-side encoder.

Bucket counts are sized per side and deliberately asymmetric — 8192 on the target (English,
~148k plausible 3-grams) but 512 on the source, whose alphabet is `{'0','1'}` and therefore
has exactly 8 distinct 3-grams.

`LocalByteEncoder` then runs 2 narrow transformer layers over those embeddings with
**block-local attention**: the sequence is cut into 128-byte blocks and attention runs
independently inside each. This is what makes a multi-hundred-byte source affordable — the
cross-block quadrants of the attention matrix are never materialised. An absolute sinusoidal
position is added *before* blocking, so a byte still knows where it sits globally even though
it only attends locally.

The target-side instance is additionally **causal**, and this is load-bearing: patch *t*'s
pooled vector is what the global decoder consumes at step *t*, and step *t* predicts patch
*t+1*. A byte in patch *t* attending into patch *t+1* would be reading its own answer. The
self-test audits this by perturbing individual target bytes and asserting that no logit at or
before that position moves.

### Stage 2 — vectors become patches (`PatchPooler`)

A patch is `patch_size` consecutive byte states compressed into one `d_model` vector. The
pooling is a small cross-attention: a single learned query attends over the patch's bytes, so
the model decides which bytes matter, plus a **masked mean residual** over the valid bytes so
the output is sensible before attention has learned anything. Padding bytes are excluded from
both.

The patch axis is folded into the batch axis, so every patch pools independently and the
operation is one batched attention call rather than a loop.

### Stage 3 — the global transformer

`Seq2SeqTransformer(cfg, None, None)` — both vocabulary sizes `None`. In that **latent mode**
it owns no embedding table and no output projection; it consumes `d_model` vectors and returns
`d_model` vectors. Everything else is identical to C1: same 4+4 layers, same `d_ff`, same
sinusoidal absolute positional encoding over the *patch* index, same multi-head attention,
same LayerNorm, same Pre-LN residual structure.

The decoder input is the target patch stream shifted right by one, with a learned `bos_patch`
parameter in slot 0 playing the role `<bos>` plays for C1–C4. So the decoder output at slot
*t* has seen patches 0..*t*−1 and is the conditioning needed to produce patch *t*.

### Stage 4 — patches become bytes (`LocalByteDecoder`)

The global decoder produces one latent `h_t` per target patch. The local decoder turns each
into `patch_size` actual bytes, autoregressively within the patch. Four mechanisms:

1. **Latent expansion.** `h_t` is not used as a single vector. One vector describing a whole
   patch starves the decoder, which then falls back on within-patch English statistics —
   enough to score well under teacher forcing, and a repetition loop at inference. `h_t` is
   expanded by a `Linear(d_model, patch_size * d_local)` into one conditioning vector per byte
   slot. Causality is safe because every slot is a function of `h_t` alone, and `h_t` came
   from patches 0..*t*−1.

2. **Direct source access.** The local decoder *also* cross-attends to the global encoder
   memory. Without it, teacher forcing lets it see the preceding true bytes of its own patch,
   which is enough to emit plausible English unaided — so the global path gets little gradient
   and `h_t` degenerates into a positional code. The memory depends only on the source, so it
   cannot leak a target byte.

3. **Absolute byte position.** The cross-attention query has to *name* a source patch. It
   therefore carries the sinusoidal absolute position of the byte being emitted,
   `(patch_offset + i) * P + j` — the same "Sinusoidal Absolute" scheme C1–C4 put on their
   decoder positions. §4 explains why the earlier learned within-patch code was not enough.

4. **No per-patch memory copy.** All patches decode in parallel. They fold into the *batch*
   axis for self-attention (each patch is its own causal sequence) and into the *query* axis
   for cross-attention (every patch attends to the same source memory). Materialising one
   memory copy per patch would be a `(B·N, Ns, d)` tensor — around 11 GB per decoder layer for
   the longest batch this corpus produced before chunking. `DecoderLayer` takes an optional
   `reshape` pair to support exactly this.

### Decoding

`greedy_decode` alternates the two levels: one global step produces the next latent, then the
local decoder emits `TGT_PATCH_SIZE` bytes one at a time from it. The bytes generated so far
are re-encoded and re-pooled exactly as in training, so there is no train/decode mismatch. A
row that has emitted `<eos>` is frozen at `<pad>`; control symbols other than `<eos>` are
never emitted, substituted with a space if argmax picks one.

---

## 3. Why the patch grids have to line up

This is the part that matters most, and the part the first implementation got wrong.

The corpus spends **exactly 8 cipher characters on every plaintext character**. So the two
byte streams are not independent sequences that happen to be paired — they are the same text
at two different scales, locked together at a known ratio.

A patch is the unit the global transformer reasons over. If a source patch covers a different
span of text than a target patch, then the cross-attention between them has no diagonal to
find. Concretely, with the original `SRC_PATCH_SIZE = 16` (2 characters) and
`TGT_PATCH_SIZE = 8` (8 characters):

- a 64-character chunk became **32 source patches** and **9 target patches**
- target character at (patch *n*, slot *j*) is determined by source patch `4n + j//2`
- and by *one half* of that patch, whose pooling had already averaged its two characters
  together

So the model had to learn a 4:1 non-diagonal alignment *and* un-average a pooled pair. It
never did. See §4.

The invariant is now stated in code and asserted in the self-test:

```python
BITS_PER_CHAR = 8
TGT_PATCH_SIZE = 4                                 # plaintext characters per target patch
SRC_PATCH_SIZE = BITS_PER_CHAR * TGT_PATCH_SIZE    # the same 4 characters, as 32 cipher bits
```

With that, a 64-character chunk is **16 source patches and 17 target patches** (the extra one
holds `<eos>`), source patch *k* and target patch *k* describe the same four characters, and —
because the global transformer applies the same sinusoidal table to both sides — they receive
the *same* positional vector. The alignment the model has to learn is the identity.

**Why 4 characters and not 2 or 8?** It is the one free knob left, and it trades two things
off:

| | smaller patch | larger patch |
|---|---|---|
| global sequence length | longer | shorter |
| bits each latent `h_t` must carry | fewer | more |
| sequential steps at decode | more global, fewer local | fewer global, more local |

At 8 characters, one 256-dimensional latent has to convey ~45 bits of source-determined
content, and the local decoder must autoregress 8 bytes from it — the starvation the
`LocalByteDecoder` docstring already warned about. At 2, the global decoder runs 33 steps per
chunk, and the "compression" that justifies patching at all mostly evaporates. 4 gives a
target-side compression of 4×, which is close to the plaintext BPE's 3.31× in C1–C4 — so the
global transformer in C5 operates on a sequence of comparable length to C1's, which is what
makes the training-speed and memory comparison the assignment asks for meaningful rather than
an artefact of sequence length.

---

## 4. What went wrong the first time, and what was measured

C5 trained to a validation loss of **1.9525 nats/byte** and then produced, for all 500 test
lines, the identical string:

```
 the sea the sea the sea the sea the sea the sea the sea ...
```

That is not a degraded translation. It is a model that has learned English and is not reading
its input at all. Four measurements on the trained checkpoint:

**(a) The source is worth nothing.** Teacher-forced byte accuracy over 256 test chunks:

| source fed to the model | byte accuracy | cross-entropy |
|---|---|---|
| the example's own cipher | 46.54% | 1.9356 |
| a *different* example's cipher | 46.29% | 1.9439 |
| a constant string of `'0'` | 46.55% | 1.9376 |

0.25 accuracy points and 0.008 nats. A constant source is as good as the real one.

**(b) It is exactly a source-blind character model.** An interpolated character n-gram of
order ≤ 7 whose context resets at every patch boundary — precisely the local decoder's
receptive field once `h_t` is uninformative — scores **1.9721 nats/byte** on the same
validation split. The 22.75M-parameter model beat a count table by 0.02 nats.

**(c) The source content was destroyed, not merely ignored.** Fraction of each
representation's variance explained by *which example it is* (the remainder being position
alone):

| stage | example-dependent variance |
|---|---|
| `src_encoder` byte states | 0.01% |
| `src_pooler` patches | 0.03% |
| global encoder memory | 4.47% |

And a linear probe trained to recover the plaintext character from those vectors:

| representation | probe accuracy (both characters of the patch) |
|---|---|
| byte states | 16.3% / 15.9% |
| source patches | 13.4% / 11.5% |
| global memory | 10.7% / 11.4% |
| *majority-class baseline (always `' '`)* | *16.9%* |

Every one at or below the trivial baseline.

**(d) The loss curve shows the moment it happened.** Training loss fell 3.83 → 2.60 over the
first 8 epochs and then *rose* monotonically to 2.70 over the remaining 25 while validation
loss sat flat at ~1.96. The model found the English-only solution in the first few epochs and
never left it; the rising training loss is scheduled sampling mixing in its own (useless)
predictions.

### The two causes

**Cause 1 — the byte embeddings were 13× smaller than the positional encoding added to them.**

`LocalByteEncoder.forward` was `self.pos(self.embed(byte_ids))`. A sinusoidal row has norm
√(d/2) ≈ 11.31 by construction. A freshly initialised `ByteEmbedding` produces vectors of norm
≈ 0.55. So the byte's identity was **4.9% of the vector the first layer saw**.

`Seq2SeqTransformer._prepare` ([attention.py:262](src/models/attention.py#L262)) multiplies
embeddings by √`d_model` = 16 before adding the *identical* table — the standard Vaswani §3.4
scaling — which puts C1–C4 at **45.4%**. The local encoders simply skipped it.

On the source side this is fatal rather than merely slow, because a cipher byte carries one
bit: the entire content signal is `emb('0') − emb('1')`, measured norm **0.87**, against a
positional vector of norm **11.31**. Two rounds of block-local attention averaging took it to
the 0.01% in table (c).

It also spared the target *asymmetrically*, which is exactly the observed failure. The local
byte **decoder** read `byte_embed + slot_pos + latent_expand(h)` with no sinusoidal table
added at all — so the English-language path got clean embeddings while the source path was
buried. Fluent English, blind to the input.

The fix is one multiplication in each local block, applying the same rule the tokenized models
already used.

**Cause 2 — the patch grids did not line up**, as described in §3. The alignment was 4:1 and
required un-averaging a pooled character pair.

These two compound into a deadlock. The local decoder's cross-attention query needed to know
which patch it was in to select a source region; its only handle on that was `h_t`; and `h_t`
only becomes informative once the query is already working. Meanwhile a shortcut — predict
English from the preceding bytes in the patch — pays off from the first gradient step. The
shortcut won in four epochs and the source path never received a useful gradient again.

---

## 5. What changed

All in [`src/models/blt.py`](src/models/blt.py) unless noted.

| # | change | why it is ablation-safe |
|---|---|---|
| 1 | `LocalByteEncoder.forward` scales embeddings by √`d_local` before the positional table | makes C5 obey the **same rule** as C1–C4 ([attention.py:262](src/models/attention.py#L262)); it increases consistency between configurations rather than reducing it |
| 2 | `LocalByteDecoder` likewise scales its byte embedding | same rule, same reason |
| 3 | `SRC_PATCH_SIZE = BITS_PER_CHAR * TGT_PATCH_SIZE`, with `TGT_PATCH_SIZE = 4` | patch stride is a BLT-only quantity with no C1–C4 analogue — the assignment's "consistent hyperparameters (depth, width, learning rate, batch size) **where applicable**" |
| 4 | `LocalByteDecoder` replaces the learned `slot_pos` with sinusoidal absolute PE over the byte index, and takes a `patch_offset` so incremental decoding stays consistent | Table 1 of the assignment *requires* Sinusoidal Absolute for C5; a learned within-patch code was not that |
| 5 | the shifted patch stream's key-padding mask is shifted with it | plain correctness fix; no effect on full-length chunks |
| 6 | `train.source_dependence`, logged every epoch for **all five** configurations | instrumentation, identical across configurations |
| 7 | the `blt.py` self-test asserts the grid-alignment invariant, builds a toy whose target is genuinely produced from its source, and checks source usage at **byte 0** | see below |

### On testing "does it read the source?"

The old self-test asserted that teacher-forced accuracy drops by more than 20 points when the
batch's sources are rotated. That assertion passed throughout the collapsed run, because it
ran on three examples of unrelated random noise, where the only way to memorise anything *is*
via the source.

Rebuilding the toy so its target is genuinely produced from its source made the aggregate
worse, not better: 100% with the right source against 92.4% with the wrong one. With six
examples the target's own prefix identifies which example it is, so a source-blind model
recalls most of the rest regardless. Aggregate accuracy simply cannot separate the two
hypotheses at this scale.

**Byte 0 can.** Its local-decoder input is the `<patch-start>` marker and its global-decoder
input is the `bos_patch` parameter — both constants. Its prediction is therefore a pure
function of the source, with no target context of any kind. The self-test now requires it to
be correct with the example's own source and wrong with somebody else's, which is a sharp,
scale-independent claim.

The aggregate comparison is still the right instrument on real data, where the set is far too
large to memorise — that is `train.source_dependence`, run every epoch on the validation
split. Passing `blt.py` is necessary and nowhere near sufficient.

### Deliberately *not* changed

- **`scheduled_sampling_floor` stays at 0.7 for every configuration.** Turning it off for C5
  alone would break the control the assignment asks for — it is a shared `TrainConfig`
  hyperparameter, in the same class as learning rate and batch size. Its visible harm on the
  first C5 run (the rising training loss) was a *consequence* of the collapse, not a cause; it
  is fine for C1–C4 and should be fine for C5 once the encoder is being read.
- **`ngram_buckets` stays at 8192 / 512.** The target-side tables are 18.4% of C5's
  parameters, which is capacity spent on the language-model path — but that is a
  representation-layer design choice, not a defect, and shrinking it would be tuning C5 in a
  way C1–C4 were not tuned.
- **C1–C4 are untouched.** No file they execute has changed except the additive diagnostic in
  `train.py`, so their checkpoints, results and figures remain valid and do not need
  regenerating.

### What has to be re-run

Only C5. Its architecture changed, so `outputs/checkpoints/C5/best.pt` is no longer loadable
and its row in `results.csv` is stale:

```bash
python src/train.py --config C5           # trains, then evaluates and rewrites results
```

Watch the `src` column in the per-epoch line. It is the matched-minus-mismatched source
accuracy in percentage points, and it was **+0.25** for the whole of the failed run. If it is
still near zero by epoch 3–5, the model is collapsing again and there is no point waiting for
the remaining 55 epochs.

---

## 6. What to expect in the comparison

The assignment asks C5 to be judged against C1 on training speed, peak GPU memory, and
reconstruction quality. Some of that is structural and will hold regardless of how the retrain
goes:

- **Memory and speed.** C5 pays for its lack of a vocabulary at the byte level: the local
  encoders run over 512 source bytes where C1's encoder runs over ~33 BPE tokens. Even with
  block-local attention that is the dominant cost, and it is why the first run peaked at
  48.7 GB against C1's 22.8 GB and managed 223 examples/second against 610. Aligning the patch
  grids does not change this; the local blocks are where the time goes.
- **Parameters.** 22.49M against C1's 12.89M, most of the difference being the hashed n-gram
  tables (4.46M) and the two local transformer stacks. (Slightly down from the first run's
  22.75M: `latent_expand` now produces 4 slot vectors per patch instead of 8, and the learned
  `slot_pos` is gone.)
- **BLEU and ROUGE.** The assignment scopes these to "tokenized models only" (§4 of the PDF).
  They are still computed for C5 so the table is complete, but the honest cross-family
  comparisons are Levenshtein, character accuracy and sequence accuracy.
- **Validation loss is not comparable across the C1–C4 / C5 boundary.** C1–C4 predict over a
  4,096-way subword vocabulary; C5 predicts over a 259-way byte vocabulary. A lower
  cross-entropy for C5 is partly just a smaller output space. The same applies to tokens/second
  — a "token" is a BPE subword on one side and a raw byte on the other, so `examples_per_sec`
  is the honest throughput number.
