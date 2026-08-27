"""Corpus loading, BPE tokenizers, and the tokenized / token-free datasets.

The corpus is 5000 line-aligned pairs: line k of brown_cipher.txt is line k of
brown_plain.txt expanded to exactly 8 bits per character. One training example is one
complete line.

Splitting is over lines (4000/500/500, seed 42).

Two tokenizers are trained on the training split only. The cipher side has an alphabet of
exactly {"0","1"}, so its merges are learned groupings of bits; the plaintext side uses a
ByteLevel pre-tokenizer so `decode` reproduces whitespace byte-for-byte.

Two dataset classes consume the same pair list:
  * TokenizedSeq2SeqDataset -- C1-C4, BPE ids on both sides.
  * ByteSeq2SeqDataset      -- C5, raw bytes, no vocabulary.
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
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent / "models"))
from blt import BYTE_EOS_ID, BYTE_PAD_ID, SRC_PATCH_SIZE, TGT_PATCH_SIZE

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

# --- subword vocabularies (C1-C4) --------------------------------------------------------
# Two separate tokenizers; a shared vocabulary would be meaningless here since the source
# alphabet is {"0","1"} and the target is English -- the overlap is empty.
CIPHER_VOCAB_SIZE = 1024
PLAIN_VOCAB_SIZE = 4096
SPECIAL_TOKENS = ["<pad>", "<unk>", "<bos>", "<eos>"]
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3


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


# --- tokenizers --------------------------------------------------------------------------


def _train_bpe(corpus, vocab_size, initial_alphabet, byte_level: bool) -> Tokenizer:
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    if byte_level:
        tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tok.decoder = decoders.ByteLevel()
    tok.train_from_iterator(corpus, trainer=trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=list(SPECIAL_TOKENS),
        initial_alphabet=initial_alphabet,
        show_progress=False,
        min_frequency=2,
    ))
    return tok


def build_tokenizers(cipher_texts, plain_texts, force: bool = False):
    """Train (or reload) both tokenizers and derive the sequence-length caps.

    Inputs must come from the training split only. Returns (cipher_tok, plain_tok, meta).
    Caps sit at the observed maximum: an example is a whole line, so truncating an outlier
    would delete the end of a document.
    """
    TOKENIZER_DIR.mkdir(parents=True, exist_ok=True)
    if not force and all(p.exists() for p in
                         (CIPHER_TOKENIZER_PATH, PLAIN_TOKENIZER_PATH, TOKENIZER_META_PATH)):
        return (Tokenizer.from_file(str(CIPHER_TOKENIZER_PATH)),
                Tokenizer.from_file(str(PLAIN_TOKENIZER_PATH)),
                json.loads(TOKENIZER_META_PATH.read_text(encoding="utf-8")))

    print(f"[tokenizer] training BPE on {len(cipher_texts):,} training examples ...")
    # No pre-tokenizer on the cipher: it has no word boundaries, so BPE may segment anywhere.
    cipher_tok = _train_bpe(cipher_texts, CIPHER_VOCAB_SIZE, ["0", "1"], byte_level=False)
    plain_tok = _train_bpe(plain_texts, PLAIN_VOCAB_SIZE,
                           pre_tokenizers.ByteLevel.alphabet(), byte_level=True)

    src_lengths = [len(e.ids) for e in cipher_tok.encode_batch(list(cipher_texts))]
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
                        for e in cipher_tok.encode_batch([p.cipher for p in pairs])]
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
    """Raw bytes on both sides -- no vocabulary. Source bytes are the literal characters of
    the bit string, ord('0')=48 and ord('1')=49, so C5 consumes the same input file as C1-C4
    with only the tokenizer removed."""

    def __init__(self, pairs):
        self.pairs = list(pairs)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, i):
        pair = self.pairs[i]
        return (torch.tensor(list(pair.cipher.encode("latin-1")), dtype=torch.long),
                torch.tensor(list(pair.plain.encode("latin-1")) + [BYTE_EOS_ID],
                             dtype=torch.long))

    def source_lengths(self) -> list[int]:
        return [len(p.cipher) for p in self.pairs]


class CollateBytes:
    """Pad to a whole number of patches so the patch grid is rectangular. Class-based to
    stay picklable for DataLoader workers."""

    def __init__(self, src_patch=SRC_PATCH_SIZE, tgt_patch=TGT_PATCH_SIZE):
        self.src_patch, self.tgt_patch = src_patch, tgt_patch

    @staticmethod
    def _round_up(n, multiple):
        return int(-(-n // multiple) * multiple)

    def __call__(self, batch):
        srcs, tgts = zip(*batch)
        max_src = self._round_up(max(len(s) for s in srcs), self.src_patch)
        max_tgt = self._round_up(max(len(t) for t in tgts), self.tgt_patch)
        src = torch.full((len(batch), max_src), BYTE_PAD_ID, dtype=torch.long)
        tgt = torch.full((len(batch), max_tgt), BYTE_PAD_ID, dtype=torch.long)
        for i, (s, t) in enumerate(zip(srcs, tgts)):
            src[i, :len(s)] = s
            tgt[i, :len(t)] = t
        return {"src": src, "tgt": tgt}


class LengthGroupedBatchSampler(torch.utils.data.Sampler):
    """Batch similar-length lines together.

    Lines run from 21 to 2670 characters, so uniform random batching pads to a mean of ~1113
    source tokens against a mean real length of 432 -- 61% of attention compute on padding.
    Grouping cuts that to 5%. Randomness is kept by shuffling first and sorting only within
    megabatches, so batch membership still changes every epoch.

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
    """Build train/val/test loaders plus the Pair lists the evaluation needs."""
    if splits is None:
        splits, _ = build_splits()
    info: dict = {"pairs": splits}

    if model_cfg.is_blt:
        collate = CollateBytes()
        datasets = {name: ByteSeq2SeqDataset(p) for name, p in splits.items()}
        info["meta"] = {"max_src_len": MAX_LINE_BITS, "max_tgt_len": MAX_LINE_CHARS + 1}
    else:
        if tokenizers is None:
            tokenizers = build_tokenizers([p.cipher for p in splits["train"]],
                                          [p.plain for p in splits["train"]])
        cipher_tok, plain_tok, meta = tokenizers
        collate = collate_tokenized
        datasets = {name: TokenizedSeq2SeqDataset(p, cipher_tok, plain_tok,
                                                  meta["max_src_len"], meta["max_tgt_len"])
                    for name, p in splits.items()}
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
        print(f"  {name:<6} {len(pairs):>5} examples")
        assert all(p.plain == plain_lines[p.line_id] for p in pairs)
        assert all(p.cipher == cipher_lines[p.line_id] for p in pairs)
    print("  every example is one whole corpus line")

    rng = random.Random(0)
    test = splits["test"]
    cipher_tok, plain_tok, meta = build_tokenizers([p.cipher for p in splits["train"]],
                                                   [p.plain for p in splits["train"]])
    for k, v in meta.items():
        print(f"  {k:<20} {v}")
    for pair in rng.sample(test, k=200):
        assert decode_plain(plain_tok, plain_tok.encode(pair.plain).ids) == pair.plain
    print("  plaintext BPE round-trip is lossless (200 examples)")

    lengths = TokenizedSeq2SeqDataset(splits["train"], cipher_tok, plain_tok,
                                      meta["max_src_len"], meta["max_tgt_len"]).source_lengths()
    sampler = LengthGroupedBatchSampler(lengths, 32)
    batches = list(iter(sampler))
    assert sorted(i for b in batches for i in b) == list(range(len(lengths)))
    assert len(batches) == len(sampler)
    padded = np.mean([max(lengths[i] for i in b) for b in batches])
    print(f"  length grouping: mean padded {padded:.0f} vs mean real {np.mean(lengths):.0f} "
          f"({100 * (1 - np.mean(lengths) / padded):.0f}% padding)")
    print("dataset.py self-test passed")
