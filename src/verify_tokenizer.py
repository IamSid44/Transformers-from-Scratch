"""Differential test: `src/bpe.py` against HuggingFace `tokenizers`.

Run this once, with the library still installed, to establish that the from-scratch
implementation is a true replacement rather than an approximation:

    python src/verify_tokenizer.py            # the corpus checks (a few minutes)
    python src/verify_tokenizer.py --quick    # skip retraining on the full corpus

Four things are compared, in the order that matters:

  1. pre-tokenization   -- every splitter in `bpe.py`: the GPT-2 byte-level regex, the
                           ` ?[A-Za-z]+` `Split` the plaintext uses, and `FixedLength` (a
                           general component `dataset.py` does not currently use). Over the
                           corpus, and over unicode and whitespace edge cases it does not
                           contain;
  2. training           -- both tokenizers retrained from scratch on the same training split,
                           with the same pre-tokenizers `dataset.py` configures -- the cipher
                           side over `dataset.cipher_to_symbols`-transformed text, so a merge
                           is free to span several original chunks -- compared to the
                           library's saved JSON byte for byte;
  3. encoding           -- token ids for all 5000 lines of both sides, with the library's own
                           tokenizer files loaded into this implementation;
  4. decoding           -- `decode`, and `dataset.decode_plain` / `dataset.decode_cipher` on
                           top of it.

The byte-level path is checked too even though `dataset.py` no longer trains on it: it is
still what `bpe.py` offers for general text, so it still has to be right.

Anything that differs is printed with the first offending example.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import bpe
import dataset as ds

import tokenizers as hf

EDGE_CASES = [
    "", " ", "  ", "\n", "\t", "a", "Hello, world!", "  leading and trailing  ",
    "don't you're I'll we've he'd it's THAT'S", "'s'S 'T 't", "a  b   c    d",
    "line\nbreak\r\nhere", "tabs\tand\vvertical\fform",
    "123 4.5 6,7 -8 1e9 007", "MiXeD CaSe", "punctuation!?!?...---___",
    "emoji \U0001F600\U0001F44D here", "accents: naïve café résumé",
    "cyrillic: привет мир", "cjk: 你好世界 こんにちは",
    "nbsp split", "  ", "   x", "zero​width", " line sep",
    "　ideographic", "arabic-indic ٠١", "roman ⅠⅡ",
    "combining áb", "﻿bom", "control\x1c\x1dchars",
    "<pad> <unk> <bos> <eos>", "no<pad>spaces<eos>", "<pa <pad", "<pad><pad>",
    "0101010101", "1" * 64, "0" * 7 + "1",
]


def check(name: str, ours, theirs, sample=None) -> bool:
    """Report one comparison; on failure show where it first diverges."""
    if ours == theirs:
        print(f"  ok    {name}")
        return True
    print(f"  FAIL  {name}")
    if isinstance(ours, list) and isinstance(theirs, list):
        for i, (a, b) in enumerate(zip(ours, theirs)):
            if a != b:
                print(f"        first difference at index {i}")
                if sample is not None:
                    print(f"        input:   {sample[i]!r}")
                print(f"        ours:    {a!r}")
                print(f"        library: {b!r}")
                break
        else:
            print(f"        lengths differ: {len(ours)} vs {len(theirs)}")
    else:
        print(f"        ours:    {str(ours)[:200]!r}")
        print(f"        library: {str(theirs)[:200]!r}")
    return False


# --- 1. pre-tokenization --------------------------------------------------------------------


def compare_split(name, ours, theirs, texts) -> bool:
    """Same pieces and same offsets from both implementations of one pre-tokenizer."""
    a = [ours.pre_tokenize_str(t) for t in texts]
    b = [[(s, tuple(o)) for s, o in theirs.pre_tokenize_str(t)] for t in texts]
    return check(name, a, b, texts)


def check_pretokenizer(texts, cipher_texts) -> bool:
    ok = True
    for add_prefix_space in (False, True):
        ok &= compare_split(
            f"byte-level split (add_prefix_space={add_prefix_space})",
            bpe.pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space),
            hf.pre_tokenizers.ByteLevel(add_prefix_space=add_prefix_space), texts)
    # The alphabet feeds a set (the trainer's `initial_alphabet`), and the library returns
    # it in hash order, so only the contents are meaningful.
    ok &= check("byte-level alphabet",
                sorted(bpe.pre_tokenizers.ByteLevel.alphabet()),
                sorted(hf.pre_tokenizers.ByteLevel.alphabet()))

    # The plaintext splitter, on text it will see and on text it will not: `re` and the
    # library's Rust engine have to agree on this pattern for the ASCII class to be safe.
    ok &= compare_split("plain split (%s)" % ds.PLAIN_SPLIT_PATTERN,
                        bpe.pre_tokenizers.Split(bpe.Regex(ds.PLAIN_SPLIT_PATTERN),
                                                 behavior="isolated"),
                        hf.pre_tokenizers.Split(hf.Regex(ds.PLAIN_SPLIT_PATTERN),
                                                behavior="isolated"), texts)

    # FixedLength -- not used on the cipher any more (that side has no pre-tokenizer, see
    # SCHEMES below), but still a general component worth keeping correct. Ragged inputs
    # matter: the last chunk of a text whose length is not a multiple of the chunk size is
    # short, and both sides must keep it rather than drop or pad it.
    chunky = cipher_texts[:500] + [t[:-3] for t in cipher_texts[:200]] + EDGE_CASES
    for length in (8, 3):
        ok &= compare_split(f"fixed-length split ({length})",
                            bpe.pre_tokenizers.FixedLength(length=length),
                            hf.pre_tokenizers.FixedLength(length=length), chunky)
    return ok


# --- 2. training ----------------------------------------------------------------------------


# The three (pre-tokenizer, decoder) pairings, built from either module. "cipher" and "plain"
# are exactly what `dataset.py` configures -- the cipher side has no pre-tokenizer at all, so
# a whole line is one "word" and merges are free to span several chunks -- "bytelevel" is the
# general-text path bpe.py still offers, kept under test even though nothing here trains on
# it any more.
SCHEMES = {
    "cipher": lambda m: (None, m.decoders.Fuse()),
    "plain": lambda m: (m.pre_tokenizers.Split(m.Regex(ds.PLAIN_SPLIT_PATTERN),
                                               behavior="isolated"),
                        m.decoders.Fuse()),
    "bytelevel": lambda m: (m.pre_tokenizers.ByteLevel(add_prefix_space=False),
                            m.decoders.ByteLevel()),
}


def build(module, scheme: str):
    """An untrained tokenizer wired up the same way in `bpe` and in `tokenizers`."""
    tok = module.Tokenizer(module.models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer, tok.decoder = SCHEMES[scheme](module)
    return tok


def train(module, corpus, vocab_size, initial_alphabet, scheme):
    tok = build(module, scheme)
    tok.train_from_iterator(corpus, trainer=module.trainers.BpeTrainer(
        vocab_size=vocab_size, special_tokens=list(ds.SPECIAL_TOKENS),
        initial_alphabet=list(initial_alphabet), show_progress=False, min_frequency=2))
    return tok


def check_training(name, corpus, vocab_size, initial_alphabet, scheme,
                   reference: Path | None = None) -> bool:
    ours = train(bpe, corpus, vocab_size, initial_alphabet, scheme)
    theirs = train(hf, corpus, vocab_size, initial_alphabet, scheme)
    ok = check(f"{name}: trained JSON", ours.to_str(pretty=True), theirs.to_str(pretty=True))
    if not ok:                                      # narrow it down: vocabulary or merges?
        a, b = json.loads(ours.to_str())["model"], json.loads(theirs.to_str())["model"]
        check(f"{name}: vocab", list(a["vocab"].items()), list(b["vocab"].items()))
        check(f"{name}: merges", a["merges"], b["merges"])
    if reference is not None and reference.exists():
        ok &= check(f"{name}: matches the checked-in {reference.name}",
                    ours.to_str(pretty=True), reference.read_text(encoding="utf-8"))
    return ok


# --- 3 & 4. encoding and decoding -----------------------------------------------------------


def check_encoding(name, path: Path, texts) -> bool:
    """Load the library's own tokenizer file into both and compare ids over `texts`."""
    ours = bpe.Tokenizer.from_file(path)
    theirs = hf.Tokenizer.from_file(str(path))

    ok = check(f"{name}: vocab size", ours.get_vocab_size(), theirs.get_vocab_size())
    ok &= check(f"{name}: vocab", ours.get_vocab(), theirs.get_vocab())

    a = [e.ids for e in ours.encode_batch(texts)]
    b = [e.ids for e in theirs.encode_batch(texts)]
    ok &= check(f"{name}: encode_batch ids ({len(texts):,} texts)", a, b, texts)

    sample = texts[:150] + texts[-50:]                  # corpus lines and the edge cases
    a = [ours.encode(t).tokens for t in sample]
    b = [theirs.encode(t).tokens for t in sample]
    ok &= check(f"{name}: encode tokens", a, b, sample)

    a = [ours.encode(t).offsets for t in sample]
    b = [theirs.encode(t).offsets for t in sample]
    ok &= check(f"{name}: encode offsets", a, b, sample)

    ids = b_ids(theirs, sample)
    for skip in (True, False):
        a = [ours.decode(i, skip_special_tokens=skip) for i in ids]
        c = [theirs.decode(i, skip_special_tokens=skip) for i in ids]
        ok &= check(f"{name}: decode (skip_special_tokens={skip})", a, c)
    return ok


def b_ids(tokenizer, texts):
    """Ids to decode: every encoding, plus the specials mixed in to exercise skipping."""
    out = []
    for e in tokenizer.encode_batch(texts):
        out.append(e.ids)
        out.append([ds.BOS_ID] + e.ids + [ds.EOS_ID, ds.PAD_ID, ds.UNK_ID])
    return out


def check_decode_plain(path: Path, pairs) -> bool:
    ours = bpe.Tokenizer.from_file(path)
    theirs = hf.Tokenizer.from_file(str(path))
    a, b = [], []
    for p in pairs:
        ids = [ds.BOS_ID] + theirs.encode(p.plain).ids + [ds.EOS_ID, ds.PAD_ID]
        a.append(ds.decode_plain(ours, ids))
        b.append(ds.decode_plain(theirs, ids))
    ok = check(f"decode_plain over {len(pairs):,} lines", a, b)
    ok &= check("decode_plain round-trips the plaintext", a, [p.plain for p in pairs])
    return ok


def check_decode_cipher(path: Path, pairs) -> bool:
    ours = bpe.Tokenizer.from_file(path)
    theirs = hf.Tokenizer.from_file(str(path))
    a, b = [], []
    for p in pairs:
        ids = theirs.encode(ds.cipher_to_symbols(p.cipher)).ids
        a.append(ds.decode_cipher(ours, ids))
        b.append(ds.decode_cipher(theirs, ids))
    ok = check(f"decode_cipher over {len(pairs):,} lines", a, b)
    ok &= check("decode_cipher round-trips the cipher bits", a, [p.cipher for p in pairs])
    return ok


# --- main -----------------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true",
                    help="skip retraining on the full corpus (uses 300 lines instead)")
    args = ap.parse_args()

    print(f"tokenizers {hf.__version__}\n")
    cipher_lines, plain_lines = ds.read_corpus()
    splits, _ = ds.build_splits()
    train_cipher = [p.cipher for p in splits["train"]]
    train_plain = [p.plain for p in splits["train"]]
    train_cipher_symbols = [ds.cipher_to_symbols(c) for c in train_cipher]
    cipher_symbols = [ds.cipher_to_symbols(c) for c in cipher_lines]
    ok = True

    print("pre-tokenization")
    ok &= check_pretokenizer(EDGE_CASES + plain_lines[:2000], cipher_lines)

    print("\ntraining")
    if args.quick:
        ok &= check_training("cipher (300 lines)", train_cipher_symbols[:300], 512,
                             bpe.pre_tokenizers.ByteLevel.alphabet(), "cipher")
        ok &= check_training("plain (300 lines)", train_plain[:300], 1024,
                             ds.PLAIN_ALPHABET, "plain")
        ok &= check_training("byte-level (300 lines)", train_plain[:300], 1024,
                             bpe.pre_tokenizers.ByteLevel.alphabet(), "bytelevel")
    else:
        ok &= check_training("cipher", train_cipher_symbols, ds.CIPHER_VOCAB_SIZE,
                             bpe.pre_tokenizers.ByteLevel.alphabet(), "cipher",
                             ds.CIPHER_TOKENIZER_PATH)
        ok &= check_training("plain", train_plain, ds.PLAIN_VOCAB_SIZE,
                             ds.PLAIN_ALPHABET, "plain", ds.PLAIN_TOKENIZER_PATH)
        ok &= check_training("byte-level", train_plain, ds.PLAIN_VOCAB_SIZE,
                             bpe.pre_tokenizers.ByteLevel.alphabet(), "bytelevel")

    print("\nencoding and decoding")
    ok &= check_encoding("cipher", ds.CIPHER_TOKENIZER_PATH, cipher_symbols)
    ok &= check_encoding("plain", ds.PLAIN_TOKENIZER_PATH, plain_lines + EDGE_CASES)
    ok &= check_decode_plain(ds.PLAIN_TOKENIZER_PATH, splits["test"])
    ok &= check_decode_cipher(ds.CIPHER_TOKENIZER_PATH, splits["test"])

    print("\nsequence-length metadata")
    ours = bpe.Tokenizer.from_file(ds.CIPHER_TOKENIZER_PATH)
    ours_plain = bpe.Tokenizer.from_file(ds.PLAIN_TOKENIZER_PATH)
    meta = json.loads(ds.TOKENIZER_META_PATH.read_text(encoding="utf-8"))
    src = [len(e.ids) for e in ours.encode_batch(train_cipher_symbols)]
    tgt = [len(e.ids) + 2 for e in ours_plain.encode_batch(train_plain)]
    ok &= check("max_src_len", int(-(-max(src) // 8) * 8), meta["max_src_len"])
    ok &= check("max_tgt_len", int(-(-max(tgt) // 8) * 8), meta["max_tgt_len"])
    ok &= check("src_len_mean", round(sum(src) / len(src), 9), round(meta["src_len_mean"], 9))
    ok &= check("tgt_len_mean", round(sum(tgt) / len(tgt), 9), round(meta["tgt_len_mean"], 9))

    print("\nPASS -- the implementation matches the library" if ok else
          "\nFAIL -- see the differences above")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
