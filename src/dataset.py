"""Corpus loading, BPE tokenizers, and the tokenized / token-free datasets.

The corpus is 5000 line-aligned pairs: line k of brown_cipher.txt is line k of
brown_plain.txt expanded to exactly 8 bits per character.

Splitting is over lines (4000/500/500, seed 42). A line is not the unit a model trains or
decodes on, though: `chunk_pairs` cuts each line into consecutive CHUNK_CHARS-character
pieces (cipher and plaintext cut at the same character offset, so alignment is exact by
construction -- see its docstring), and a chunk is one training/decoding example. Evaluation
regroups a line's chunk predictions back into one string before scoring against the whole-line
gold text.

Two tokenizers are trained on the training split's chunks, both by `src/bpe.py` rather than the
`tokenizers` library, and both with a vocabulary that says what the corpus actually contains:

  * cipher -- each 8-bit unit (one plaintext character's worth of bits) is mapped to a
    single atomic symbol first (`cipher_to_symbols`, the same byte<->character bijection
    `ByteLevel` uses for arbitrary text), then BPE runs with no pre-tokenizer: merges are free
    to combine adjacent symbols into tokens spanning multiple characters, exactly as ordinary
    BPE combines characters into words. A token can never straddle an 8-bit unit, since the
    unit is now the atomic quantity merges are built from -- but it can, and is meant to,
    straddle several plaintext characters once BPE finds a unit sequence common enough to
    merge (bounded by the CHUNK_CHARS-character chunk it's encoded within -- see below).
  * plaintext -- split into words carrying their own leading space (` ?[A-Za-z]+`), then BPE
    over the 53 characters the corpus uses: a-z, A-Z and the space. Nothing else appears in
    it, so a byte-level alphabet would have spent 203 of its 256 symbols on bytes that never
    occur; here `decode` is plain concatenation and is still exact.

Two dataset classes consume the same pair list:
  * TokenizedSeq2SeqDataset -- C1-C4, BPE ids on both sides.
  * ByteSeq2SeqDataset      -- C5, raw bytes, no vocabulary. Both share `cipher_to_bytes`, so
    the cipher reaches either pathway as one byte per plaintext character.
"""

from __future__ import annotations

import json
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from bpe import BYTE_TO_CHAR, CHAR_TO_BYTE, Regex, Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from blt import BYTE_EOS_ID, BYTE_PAD_ID
from entropy_lm import (MAX_PATCH_SIZE, load_entropy_lm, patch_boundaries)

# --- paths -------------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CIPHER_FILE = PROJECT_ROOT / "brown_cipher.txt"
PLAIN_FILE = PROJECT_ROOT / "brown_plain.txt"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TOKENIZER_DIR = OUTPUT_DIR / "tokenizers"

CIPHER_TOKENIZER_PATH = TOKENIZER_DIR / "cipher_bpe.json"
PLAIN_TOKENIZER_PATH = TOKENIZER_DIR / "plain_bpe.json"
TOKENIZER_META_PATH = TOKENIZER_DIR / "meta.json"

# --- data --------------------------------------------------------------------------------
SEED = 42
BITS_PER_CHAR = 8                     # the dataset expands each character into 8 bits
N_TRAIN_LINES, N_VAL_LINES, N_TEST_LINES = 4000, 500, 500

MAX_LINE_CHARS = 2670                 # measured over all 5000 lines
MAX_LINE_BITS = MAX_LINE_CHARS * BITS_PER_CHAR

# A training/val/test example is a chunk of this many plaintext characters (and the 8x-as-many
# cipher bits at the same character offset), not a whole line. Only the disclosed 8-bits-per-
# character fact is used to place the cut points -- chunk boundaries fall on character
# boundaries, never mid-character. The last chunk of a line is whatever is left over, possibly
# shorter than CHUNK_CHARS. See `chunk_pairs`.
#
# 32 rather than 64: the cipher is a repeating-key XOR of period 8 (see this file's self-test),
# so any multiple of 8 keeps every chunk starting at key phase 0 and solvable in isolation, and
# the smaller size buys ~2x the training examples at half the sequence length -- roughly 4x the
# optimiser steps per unit of compute, which is what the first round of runs was short of.
CHUNK_CHARS = 32

# --- subword vocabularies (C1-C4) --------------------------------------------------------
# Two separate tokenizers; a shared vocabulary would be meaningless here since the source
# alphabet is {"0","1"} and the target is English -- the overlap is empty.
CIPHER_VOCAB_SIZE = 1024
PLAIN_VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3

# Every character the plaintext contains, and nothing else -- checked against the corpus by
# `read_corpus`. Ordered so the ids that follow the specials are contiguous and predictable.
PLAIN_ALPHABET = [" "] + [chr(c) for c in range(ord("A"), ord("Z") + 1)] \
                       + [chr(c) for c in range(ord("a"), ord("z") + 1)]
CIPHER_ALPHABET = ["0", "1"]

# One pre-token per word, the leading space included, so the space is an ordinary vocabulary
# character rather than a marker and decoding is concatenation. The corpus has no runs of
# spaces and no leading or trailing space, so this covers every line exactly.
PLAIN_SPLIT_PATTERN = " ?[A-Za-z]+"


# --- corpus ------------------------------------------------------------------------------


@dataclass(frozen=True)
class Pair:
    """One aligned example: a whole corpus line."""

    line_id: int
    cipher: str        # bit string, exactly 8 * len(plain) characters
    plain: str


def read_corpus() -> tuple[list[str], list[str]]:
    """Read both files, drop blank lines, and verify the 8x alignment."""
    cipher = [l for l in CIPHER_FILE.read_text(encoding="utf-8").split("\n") if l]
    plain = [l for l in PLAIN_FILE.read_text(encoding="utf-8").split("\n") if l]
    if len(cipher) != len(plain):
        raise ValueError(f"Not line-aligned: {len(cipher)} cipher vs {len(plain)} plain")
    for i, (c, p) in enumerate(zip(cipher, plain)):
        if len(c) != BITS_PER_CHAR * len(p):
            raise ValueError(f"Line {i}: {len(c)} bits for {len(p)} chars")
    # The two vocabularies below are the corpus's own alphabets, so say so out loud: a
    # character outside them would silently become <unk> and the target could never be
    # reproduced.
    for name, lines, allowed in (("plain", plain, set(PLAIN_ALPHABET)),
                                 ("cipher", cipher, set(CIPHER_ALPHABET))):
        stray = set().union(*(set(l) for l in lines)) - allowed
        if stray:
            raise ValueError(f"{name} text contains {sorted(stray)}, outside its alphabet")
    return cipher, plain


def split_line_ids(n_lines: int, seed: int = SEED) -> dict[str, list[int]]:
    ids = list(range(n_lines))
    random.Random(seed).shuffle(ids)
    a, b = N_TRAIN_LINES, N_TRAIN_LINES + N_VAL_LINES
    c = b + N_TEST_LINES
    return {"train": sorted(ids[:a]), "val": sorted(ids[a:b]), "test": sorted(ids[b:c])}


def build_splits() -> tuple[dict[str, list[Pair]], list[str]]:
    """Split the corpus into train/val/test lists of Pair, plus the full plaintext lines."""
    cipher_lines, plain_lines = read_corpus()
    ids = split_line_ids(len(plain_lines))
    splits = {name: [Pair(i, cipher_lines[i], plain_lines[i]) for i in lids]
              for name, lids in ids.items()}
    return splits, plain_lines


def chunk_pairs(pairs: list[Pair], chunk_chars: int = CHUNK_CHARS) -> list[Pair]:
    """Expand each whole-line Pair into consecutive fixed-size chunks.

    Cutting happens on the raw character/bit strings, before either tokenizer sees anything,
    so alignment between the two sides is exact by construction: `cipher[8*start:8*end]` is
    always exactly `8 * (end - start)` bits, the bits of exactly `plain[start:end]`'s
    characters, no more and no less. A line's last chunk is whatever remains, from 1 up to
    `chunk_chars` characters.

    All chunks of one line keep that line's `line_id` and appear consecutively in the
    returned list, in left-to-right order -- callers that need to put a line back together
    (evaluation) can regroup by `line_id` and concatenate in list order.
    """
    out = []
    for p in pairs:
        n = len(p.plain)
        for start in range(0, n, chunk_chars):
            end = min(start + chunk_chars, n)
            out.append(Pair(p.line_id, p.cipher[8 * start:8 * end], p.plain[start:end]))
    return out


def cipher_to_bytes(cipher: str) -> bytes:
    """The 8 characters '0'/'1' the corpus spends on each plaintext character, packed back into
    the single byte they denote: one cipher byte per plaintext character.

    This is a lossless re-reading of the file, not a tokenization -- no vocabulary, no merges,
    nothing trained. Both pathways start here: C1-C4 run BPE over the symbols below, and C5
    (`ByteSeq2SeqDataset`) feeds these bytes straight to its local encoder. Feeding C5 the
    literal '0'/'1' characters instead would spend one of the 259 byte rows per *bit*, leaving
    every source position carrying one bit of content and the local encoder to reassemble
    characters from 8 positions whose n-gram features cannot even see their own bit offset."""
    return bytes(int(cipher[i:i + BITS_PER_CHAR], 2) for i in range(0, len(cipher), BITS_PER_CHAR))


def cipher_to_symbols(cipher: str) -> str:
    """One character per 8-bit unit, via the byte<->character bijection `ByteLevel` uses for
    arbitrary text. This is what lets BPE merge *across* the original 8-bit boundaries: once
    every unit is a single symbol, merging two adjacent symbols builds a token spanning two
    plaintext characters, and so on -- the same freedom `train_plain_tokenizer` has to merge
    characters into words."""
    return cipher_to_bytes(cipher).decode("latin-1").translate(BYTE_TO_CHAR)


def symbols_to_cipher(symbols: str) -> str:
    """Inverse of `cipher_to_symbols`."""
    return "".join(f"{CHAR_TO_BYTE[c]:0{BITS_PER_CHAR}b}" for c in symbols)


# --- tokenizers --------------------------------------------------------------------------


def _train_bpe(corpus, vocab_size, initial_alphabet, pre_tokenizer) -> Tokenizer:
    """One BPE tokenizer. `vocab_size` is a ceiling, not a guarantee: training stops early
    once no pair occurs more than once (`min_frequency=2`)."""
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = pre_tokenizer
    tok.decoder = decoders.Fuse()          # the vocabularies are the texts' own characters
    tok.train_from_iterator(corpus, trainer=trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=initial_alphabet,
        show_progress=False,
        min_frequency=2,
    ))
    return tok


def train_cipher_tokenizer(cipher_texts) -> Tokenizer:
    """`cipher_to_symbols` first, then BPE with no pre-tokenizer: each chunk is one "word", so
    merges are free to build tokens spanning multiple plaintext characters (but never across a
    chunk boundary, since chunks are encoded independently -- see `chunk_pairs`). The alphabet
    is the full 256-byte range (`ByteLevel.alphabet()`, reused rather than the 126 byte values
    (all of them <128 -- XORing two mostly-7-bit alphabets never sets the high bit) this
    corpus's cipher actually realizes) so no byte value the cipher could in principle produce
    falls outside the vocabulary. The other 130 values sit in the trained vocabulary as
    permanently zero-frequency single-byte tokens: never chosen by a merge, never emitted,
    genuine unused capacity rather than a bug."""
    symbols = [cipher_to_symbols(c) for c in cipher_texts]
    return _train_bpe(symbols, CIPHER_VOCAB_SIZE, pre_tokenizers.ByteLevel.alphabet(), None)


def train_plain_tokenizer(plain_texts) -> Tokenizer:
    """Words with their leading space, then BPE over the corpus's 53 characters."""
    return _train_bpe(plain_texts, PLAIN_VOCAB_SIZE, PLAIN_ALPHABET,
                      pre_tokenizers.Split(Regex(PLAIN_SPLIT_PATTERN), behavior="isolated"))


def decode_cipher(cipher_tok: Tokenizer, ids: Iterable[int]) -> str:
    """Ids back to the original bit string. No <bos>/<eos> to strip: the cipher is only ever
    encoder input, never generated, so nothing wraps it the way `decode_plain` unwraps."""
    return symbols_to_cipher(cipher_tok.decode(list(ids), skip_special_tokens=True))


def build_tokenizers(cipher_texts, plain_texts, force: bool = False):
    """Train (or reload) both tokenizers and derive the sequence-length caps.

    Inputs must come from the training split's chunks (`chunk_pairs`), not whole lines.
    Returns (cipher_tok, plain_tok, meta). Caps sit at the observed maximum: an example is one
    chunk, so truncating an outlier would delete the end of it.
    """
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all(p.exists() for p in
                         (CIPHER_TOKENIZER_PATH, PLAIN_TOKENIZER_PATH, TOKENIZER_META_PATH)):
        return (Tokenizer.from_file(str(CIPHER_TOKENIZER_PATH)),
                Tokenizer.from_file(str(PLAIN_TOKENIZER_PATH)),
                json.loads(TOKENIZER_META_PATH.read_text(encoding="utf-8")))

    print(f"[tokenizer] training BPE on {len(cipher_texts):,} training examples ...")
    cipher_tok = train_cipher_tokenizer(cipher_texts)
    plain_tok = train_plain_tokenizer(plain_texts)

    src_lengths = [len(e.ids)
                  for e in cipher_tok.encode_batch([cipher_to_symbols(c) for c in cipher_texts])]
    tgt_lengths = [len(e.ids) + 2 for e in plain_tok.encode_batch(list(plain_texts))]

    def cap(lengths, multiple=8):
        return int(-(-max(lengths) // multiple) * multiple)

    meta = {
        "cipher_vocab_size": cipher_tok.get_vocab_size(),
        "plain_vocab_size": plain_tok.get_vocab_size(),
        "max_src_len": cap(src_lengths),
        "max_tgt_len": cap(tgt_lengths),
        "src_len_mean": sum(src_lengths) / len(src_lengths),
        "tgt_len_mean": sum(tgt_lengths) / len(tgt_lengths),
        "src_len_max": max(src_lengths),
        "tgt_len_max": max(tgt_lengths),
        "src_compression": (sum(map(len, cipher_texts)) / len(cipher_texts))
                           / (sum(src_lengths) / len(src_lengths)),
        "tgt_compression": (sum(map(len, plain_texts)) / len(plain_texts))
                           / (sum(tgt_lengths) / len(tgt_lengths)),
    }

    cipher_tok.save(str(CIPHER_TOKENIZER_PATH))
    plain_tok.save(str(PLAIN_TOKENIZER_PATH))
    TOKENIZER_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[tokenizer] cipher vocab={meta['cipher_vocab_size']} "
          f"mean={meta['src_len_mean']:.1f} cap={meta['max_src_len']}")
    print(f"[tokenizer] plain  vocab={meta['plain_vocab_size']} "
          f"mean={meta['tgt_len_mean']:.1f} cap={meta['max_tgt_len']}")
    return cipher_tok, plain_tok, meta


def load_tokenizers():
    if not CIPHER_TOKENIZER_PATH.exists():
        raise FileNotFoundError(f"No tokenizers in {TOKENIZER_DIR}; run `python src/dataset.py`")
    return (Tokenizer.from_file(str(CIPHER_TOKENIZER_PATH)),
            Tokenizer.from_file(str(PLAIN_TOKENIZER_PATH)),
            json.loads(TOKENIZER_META_PATH.read_text(encoding="utf-8")))


def decode_plain(plain_tok: Tokenizer, ids: Iterable[int]) -> str:
    """Ids back to text, stopping at <eos> and dropping specials."""
    out = []
    for i in ids:
        i = int(i)
        if i == EOS_ID:
            break
        if i not in (PAD_ID, BOS_ID):
            out.append(i)
    return plain_tok.decode(out, skip_special_tokens=True)


# --- datasets ----------------------------------------------------------------------------


class TokenizedSeq2SeqDataset(Dataset):
    """BPE ids on both sides. Targets are wrapped as <bos> ... <eos>."""

    def __init__(self, pairs, cipher_tok, plain_tok, max_src_len, max_tgt_len):
        pairs = list(pairs)
        # Encode once up front; per-item encoding would dominate the step time.
        self.src_ids = [e.ids[:max_src_len]
                        for e in cipher_tok.encode_batch(
                            [cipher_to_symbols(p.cipher) for p in pairs])]
        self.tgt_ids = [[BOS_ID] + e.ids[:max_tgt_len - 2] + [EOS_ID]
                        for e in plain_tok.encode_batch([p.plain for p in pairs])]

    def __len__(self):
        return len(self.src_ids)

    def __getitem__(self, i):
        return (torch.tensor(self.src_ids[i], dtype=torch.long),
                torch.tensor(self.tgt_ids[i], dtype=torch.long))

    def source_lengths(self) -> list[int]:
        return [len(s) for s in self.src_ids]


def collate_tokenized(batch):
    """Pad to the batch's own longest sequence, not to the global cap."""
    srcs, tgts = zip(*batch)
    src = torch.full((len(batch), max(len(s) for s in srcs)), PAD_ID, dtype=torch.long)
    tgt = torch.full((len(batch), max(len(t) for t in tgts)), PAD_ID, dtype=torch.long)
    for i, (s, t) in enumerate(zip(srcs, tgts)):
        src[i, :len(s)] = s
        tgt[i, :len(t)] = t
    return {"src": src, "tgt": tgt}


class ByteSeq2SeqDataset(Dataset):
    """Raw bytes on both sides -- no vocabulary, no merges, nothing trained.

    The source is `cipher_to_bytes`: the corpus writes each byte as 8 characters '0'/'1', and
    those are packed back into the byte they denote, so a chunk is CHUNK_CHARS source bytes
    rather than 8x that many. C5 still consumes the same input file as C1-C4 with only the
    tokenizer removed -- the grouping is the same one `cipher_to_symbols` already performs for
    the BPE pathway, and it uses only the 8-bits-per-character fact that `chunk_pairs` and the
    patch grid depend on anyway.

    Each item also carries its **patch lengths**: the variable-width segmentation
    `entropy_patch_lengths` derived from the entropy model's next-byte surprise, not a fixed
    stride. `patch_lengths[i]` sums to the source length; the target grid is the same one with
    the EOS byte folded into the final patch (or given its own, if that patch is already full).
    """

    def __init__(self, pairs, patch_lengths):
        self.pairs = list(pairs)
        self.patch_lengths = list(patch_lengths)
        assert len(self.pairs) == len(self.patch_lengths)

    def __len__(self):
        return len(self.pairs)

    @staticmethod
    def target_patch_lengths(src_lengths: list[int]) -> list[int]:
        """Source grid -> target grid. The target is the plaintext plus one EOS byte, so it is
        one byte longer than the source; that byte joins the last patch unless doing so would
        exceed the cap, in which case it gets a patch of its own."""
        out = list(src_lengths)
        if out and out[-1] < MAX_PATCH_SIZE:
            out[-1] += 1
        else:
            out.append(1)
        return out

    def __getitem__(self, i):
        pair = self.pairs[i]
        src_patches = self.patch_lengths[i]
        return {
            "src": torch.tensor(list(cipher_to_bytes(pair.cipher)), dtype=torch.long),
            "tgt": torch.tensor(list(pair.plain.encode("latin-1")) + [BYTE_EOS_ID],
                                dtype=torch.long),
            "src_patches": src_patches,
            "tgt_patches": self.target_patch_lengths(src_patches),
        }

    def source_lengths(self) -> list[int]:
        return [len(p.cipher) // BITS_PER_CHAR for p in self.pairs]


def _patch_grid(lengths_per_example: list[list[int]], n_patches: int, max_patch: int):
    """Patch lengths -> the (B, N, P) gather grid the BLT modules index bytes with.

    Returns (index, mask, patch_valid). `index[b, n, j]` is the byte position of slot j of
    patch n -- clamped to 0 where the slot is unused, with `mask[b, n, j]` saying so. Ordering
    matters and is exact: reading the valid slots of `index` in row-major order yields
    0, 1, 2, ... for every example, because patches partition the byte sequence contiguously
    and in order. That is what lets `BLTSeq2Seq.forward` scatter its per-slot logits straight
    back onto the byte axis with a boolean mask instead of a second gather.
    """
    b = len(lengths_per_example)
    index = torch.zeros(b, n_patches, max_patch, dtype=torch.long)
    mask = torch.zeros(b, n_patches, max_patch, dtype=torch.bool)
    for i, lengths in enumerate(lengths_per_example):
        pos = 0
        for n, size in enumerate(lengths):
            index[i, n, :size] = torch.arange(pos, pos + size)
            mask[i, n, :size] = True
            pos += size
    return index, mask, mask.any(-1)


class CollateBytes:
    """Pad a batch of byte examples and build both patch grids.

    Nothing is rounded up to a stride any more: patch widths are whatever the entropy model
    chose, so the batch is padded to its own longest byte sequence and its own largest patch
    count. Class-based to stay picklable for DataLoader workers.
    """

    def __init__(self, max_patch: int = MAX_PATCH_SIZE):
        self.max_patch = max_patch

    def __call__(self, batch):
        b = len(batch)
        max_src = max(len(e["src"]) for e in batch)
        max_tgt = max(len(e["tgt"]) for e in batch)
        src = torch.full((b, max_src), BYTE_PAD_ID, dtype=torch.long)
        tgt = torch.full((b, max_tgt), BYTE_PAD_ID, dtype=torch.long)
        for i, e in enumerate(batch):
            src[i, :len(e["src"])] = e["src"]
            tgt[i, :len(e["tgt"])] = e["tgt"]

        out = {"src": src, "tgt": tgt}
        for side in ("src", "tgt"):
            lengths = [e[f"{side}_patches"] for e in batch]
            n = max(len(l) for l in lengths)
            index, mask, valid = _patch_grid(lengths, n, self.max_patch)
            out[f"{side}_patch_index"] = index
            out[f"{side}_patch_mask"] = mask
            out[f"{side}_patch_valid"] = valid
        return out


def entropy_patch_lengths(pairs, device=None, batch_size: int = 1024) -> list[list[int]]:
    """Segment every chunk's cipher bytes into entropy-driven patches.

    This is BLT's dynamic patching, run once up front rather than inside the training step:
    the entropy model is frozen, so a chunk's boundaries never change and recomputing them
    every epoch would only burn time. `entropy_lm.py` explains why the boundaries are read off
    the *cipher* and then reused for the plaintext.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, threshold, ckpt = load_entropy_lm(device)
    max_patch = ckpt.get("max_patch", MAX_PATCH_SIZE)

    out: list[list[int]] = []
    for start in range(0, len(pairs), batch_size):
        window = pairs[start:start + batch_size]
        width = max(len(p.plain) for p in window)
        x = torch.full((len(window), width), BYTE_PAD_ID, dtype=torch.long)
        for i, pair in enumerate(window):
            row = cipher_to_bytes(pair.cipher)
            x[i, :len(row)] = torch.tensor(list(row), dtype=torch.long)
        x = x.to(device)
        valid = x != BYTE_PAD_ID
        ids = patch_boundaries(model.entropies(x), valid, threshold, max_patch)
        ids, valid = ids.cpu(), valid.cpu()
        for i, pair in enumerate(window):
            n = len(pair.plain)
            row = ids[i, :n]
            out.append(torch.bincount(row, minlength=int(row.max()) + 1).tolist())
    return out


class LengthGroupedBatchSampler(torch.utils.data.Sampler):
    """Batch similar-length chunks together.

    Chunks are mostly CHUNK_CHARS characters with an occasional shorter one at a line's end,
    so padding waste is already small; grouping still removes what's left from mixing full-
    size and leftover chunks in the same batch. Randomness is kept by shuffling first and
    sorting only within megabatches, so batch membership still changes every epoch.

    Training loader only; val/test keep their natural order so predictions stay aligned with
    their Pair list.
    """

    def __init__(self, lengths, batch_size, megabatch_factor=50, seed=SEED):
        self.lengths = list(lengths)
        self.batch_size = batch_size
        self.megabatch_size = batch_size * megabatch_factor
        self.seed, self.epoch = seed, 0

    def __len__(self):
        return -(-len(self.lengths) // self.batch_size)

    def __iter__(self):
        rng = random.Random(self.seed + self.epoch)
        self.epoch += 1
        order = list(range(len(self.lengths)))
        rng.shuffle(order)
        batches = []
        for start in range(0, len(order), self.megabatch_size):
            mega = sorted(order[start:start + self.megabatch_size],
                          key=lambda i: self.lengths[i], reverse=True)
            batches += [mega[b:b + self.batch_size] for b in range(0, len(mega), self.batch_size)]
        rng.shuffle(batches)
        return iter(batches)


def make_dataloaders(model_cfg, train_cfg, splits=None, tokenizers=None) -> dict:
    """Build train/val/test loaders plus the Pair lists the evaluation needs.

    `info["pairs"]` stays at whole-line granularity -- it is the ground truth evaluation
    reconstructs against. Every dataset/loader is built from `chunk_pairs(splits[...])`
    instead: a chunk, not a line, is the actual training/decoding example. `chunk_line_ids`
    records which original line each chunk (in loader order) belongs to, so a chunked
    split's per-chunk outputs can be regrouped back into per-line strings.
    """
    if splits is None:
        splits, _ = build_splits()
    chunked = {name: chunk_pairs(p) for name, p in splits.items()}
    info: dict = {"pairs": splits, "chunks": chunked,
                 "chunk_line_ids": {name: [p.line_id for p in cp] for name, cp in chunked.items()}}

    if model_cfg.is_blt:
        collate = CollateBytes()
        datasets = {name: ByteSeq2SeqDataset(p, entropy_patch_lengths(p))
                    for name, p in chunked.items()}
        sizes = [n for d in datasets.values() for l in d.patch_lengths for n in l]
        info["meta"] = {"max_src_len": CHUNK_CHARS, "max_tgt_len": CHUNK_CHARS + 1,
                        "mean_patch_size": sum(sizes) / len(sizes),
                        "max_patch_size": MAX_PATCH_SIZE}
    else:
        if tokenizers is None:
            tokenizers = build_tokenizers([p.cipher for p in chunked["train"]],
                                          [p.plain for p in chunked["train"]])
        cipher_tok, plain_tok, meta = tokenizers
        collate = collate_tokenized
        datasets = {name: TokenizedSeq2SeqDataset(p, cipher_tok, plain_tok,
                                                  meta["max_src_len"], meta["max_tgt_len"])
                    for name, p in chunked.items()}
        info["cipher_tok"], info["plain_tok"], info["meta"] = cipher_tok, plain_tok, meta

    info["datasets"] = datasets
    common = dict(collate_fn=collate, num_workers=train_cfg.num_workers,
                  pin_memory=torch.cuda.is_available())
    info["loaders"] = {
        name: (DataLoader(dset, batch_sampler=LengthGroupedBatchSampler(
                   dset.source_lengths(), train_cfg.batch_size), **common)
               if name == "train" and train_cfg.group_by_length
               else DataLoader(dset, batch_size=train_cfg.batch_size, shuffle=False, **common))
        for name, dset in datasets.items()
    }
    return info


if __name__ == "__main__":
    cipher_lines, plain_lines = read_corpus()
    print(f"corpus: {len(plain_lines):,} lines, {sum(map(len, plain_lines)):,} characters")

    ids = split_line_ids(len(plain_lines))
    assert set(ids["train"]).isdisjoint(ids["val"] + ids["test"])
    assert set(ids["val"]).isdisjoint(ids["test"])

    splits, _ = build_splits()
    for name, pairs in splits.items():
        assert all(p.plain == plain_lines[p.line_id] for p in pairs)
        assert all(p.cipher == cipher_lines[p.line_id] for p in pairs)
    print(f"  {sum(len(p) for p in splits.values()):,} corpus lines split "
          f"{'/'.join(str(len(p)) for p in splits.values())} (train/val/test)")

    chunked = {name: chunk_pairs(p) for name, p in splits.items()}
    for name, pairs in chunked.items():
        print(f"  {name:<6} {len(pairs):>6,} chunks from {len(splits[name]):>5,} lines")
        assert all(len(p.plain) <= CHUNK_CHARS for p in pairs)
        assert all(len(p.cipher) == 8 * len(p.plain) for p in pairs)
    by_line: dict[int, list] = {}
    for p in chunked["test"]:
        by_line.setdefault(p.line_id, []).append(p)
    for line in splits["test"]:
        pieces = by_line[line.line_id]
        assert "".join(c.plain for c in pieces) == line.plain
        assert "".join(c.cipher for c in pieces) == line.cipher
    print("  chunks reassemble (concatenated, in order) into the exact original line")

    rng = random.Random(0)
    test = chunked["test"]
    cipher_tok, plain_tok, meta = build_tokenizers([p.cipher for p in chunked["train"]],
                                                   [p.plain for p in chunked["train"]])
    for k, v in meta.items():
        print(f"  {k:<20} {v}")
    sample = rng.sample(test, k=200)
    for pair in sample:
        assert decode_plain(plain_tok, plain_tok.encode(pair.plain).ids) == pair.plain
    print("  plaintext BPE round-trip is lossless (200 chunks)")

    for pair in sample:
        ids = cipher_tok.encode(cipher_to_symbols(pair.cipher)).ids
        assert decode_cipher(cipher_tok, ids) == pair.cipher, "cipher round-trip lost bits"
    print(f"  cipher BPE round-trip is lossless (200 chunks), "
          f"{meta['src_compression']:.2f} bits/token")

    plain_singles = {t for t in plain_tok.get_vocab() if len(t) == 1}
    assert plain_singles - set(PLAIN_ALPHABET) <= set(SPECIAL_TOKENS), \
        f"characters outside a-z/A-Z/space in the plaintext vocabulary: {plain_singles}"
    print(f"  plaintext vocabulary is built from {len(PLAIN_ALPHABET)} characters only")

    lengths = TokenizedSeq2SeqDataset(chunked["train"], cipher_tok, plain_tok,
                                      meta["max_src_len"], meta["max_tgt_len"]).source_lengths()
    sampler = LengthGroupedBatchSampler(lengths, 32)
    batches = list(iter(sampler))
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert len(batches) == len(sampler)
    padded = np.mean([max(lengths[i] for i in b) for b in batches])
    print(f"  length grouping: mean padded {padded:.0f} vs mean real {np.mean(lengths):.0f} "
          f"({100 * (1 - np.mean(lengths) / padded):.0f}% padding)")
    print("dataset.py self-test passed")
