"""Byte-Pair Encoding, implemented from scratch -- no `tokenizers` dependency.

This is a faithful reimplementation of the part of HuggingFace `tokenizers` 0.23.1 that this
project uses: the byte-level pre-tokenizer, the BPE model, and the BPE trainer. It mirrors
the Rust originals (`models/bpe/{model,word,trainer}.rs`, `pre_tokenizers/byte_level.rs`)
closely enough to be bit-for-bit identical: same learned vocabulary and merge order, same
token ids, same decoded text, same serialized JSON. `src/verify_tokenizer.py` checks all of
that against the library on this corpus.

The API is the subset `dataset.py` needs, with the library's own names, so swapping the two
is a one-line import change:

    from bpe import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers

Two pieces do the real work, and the rest decide what BPE gets to see:

  * `BPE`          -- encodes one pre-token by merging the lowest-ranked pair it contains,
                      over and over, until no merge applies.
  * `BpeTrainer`   -- learns those merges: count adjacent symbol pairs, merge the most
                      frequent pair, repeat until the vocabulary is full.

  * `ByteLevel`    -- splits text with the GPT-2 regex (hand-rolled here, since `re` has no
                      `\\p{L}`) and maps each UTF-8 byte to a printable character, so BPE
                      never sees a byte it has no symbol for and `decode` is lossless.
  * `FixedLength`  -- cuts the text into fixed-width chunks. The cipher side uses 8, one
                      chunk per plaintext character, so no token can straddle two characters.
  * `Split`        -- splits on a regex. The plaintext side uses ` ?[A-Za-z]+`: one pre-token
                      per word, each carrying its own leading space.
  * `Fuse`         -- the decoder for both of those: the vocabulary is made of the text's own
                      characters, so concatenating the tokens is already the inverse.

Not implemented (unused here, and rejected loudly rather than silently ignored): dropout,
byte_fallback, ignore_merges, max_token_length, and truncation/padding/post-processors.

Internally a word is a `str` whose characters are `chr(token_id)`; that makes merging a
pair a C-level `str.replace`, which is what keeps training on 19M cipher symbols tractable
in Python. It caps the vocabulary at 0x110000 ids, which is not a limit anyone will hit.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from heapq import heappop, heappush
from itertools import count as _counter
from operator import itemgetter
from pathlib import Path
from typing import Iterable, Sequence

_LAST = itemgetter(slice(-1, None))    # p[-1:] without a Python-level call: "" when empty
_FIRST = itemgetter(slice(0, 1))

# --- byte-level alphabet ------------------------------------------------------------------
# GPT-2's byte<->character table: every one of the 256 byte values gets a printable,
# non-whitespace character, so a pre-token is always representable as text and the mapping
# is reversible. The 188 bytes that are already printable keep their own character; the
# other 68 are shifted into the U+0100.. range (so 0x20 -> 'Ġ', 0x0A -> 'Ċ').


def bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1))
    bs += list(range(0xA1, 0xAC + 1)) + list(range(0xAE, 0xFF + 1))
    cs = list(bs)
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


BYTE_TO_CHAR = bytes_to_unicode()
CHAR_TO_BYTE = {c: b for b, c in BYTE_TO_CHAR.items()}

# --- GPT-2 pre-tokenization ---------------------------------------------------------------
# The library splits on this regex:
#
#     's|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
#
# It is matched by a backtracking engine, so at every position the *first* alternative that
# matches wins, and the alternatives cover every character -- which makes the regex a plain
# left-to-right scanner, written out below. `\s` there is the Unicode White_Space property
# (checked against the library, which is why `str.isspace()` -- it also counts U+001C-U+001F
# -- is not used).

_CONTRACTIONS = ("'s", "'t", "'re", "'ve", "'m", "'ll", "'d")

_WHITESPACE = frozenset(                                # Unicode White_Space, in full
    "\t\n\v\f\r \x85\xa0\u1680\u2028\u2029\u202f\u205f\u3000"
    + "".join(chr(c) for c in range(0x2000, 0x200B))    # EN QUAD .. HAIR SPACE
)


# `str.isalpha()` is exactly \p{L} (categories Lu Ll Lt Lm Lo). `str.isnumeric()` is \p{N}
# plus the numeric ideographs, which are Lo -- so it is exact here, where letters have
# already been ruled out. Both are C methods, and the split is the hot loop of encoding.


def _is_other(ch: str) -> bool:                         # [^\s\p{L}\p{N}]
    return ch not in _WHITESPACE and not ch.isalpha() and not ch.isnumeric()


def _run(text: str, i: int, n: int, keep) -> int:
    while i < n and keep(text[i]):
        i += 1
    return i


def _match_one(text: str, i: int, n: int) -> int:
    """End offset of the single regex alternative that matches at `i`. Always advances."""
    ch = text[i]

    if ch == "'":                                       # 's | 't | 're | 've | 'm | 'll | 'd
        for c in _CONTRACTIONS:
            if text.startswith(c, i):
                return i + len(c)

    # ` ?\p{L}+` / ` ?\p{N}+` / ` ?[^\s\p{L}\p{N}]+`: an optional single leading space, then
    # a run of one kind of character. The optional space is greedy but backtracks, so a lone
    # space only joins the run when the run is non-empty.
    start = i + 1 if ch == " " and i + 1 < n else i
    head = text[start]
    if head.isalpha():
        return _run(text, start + 1, n, str.isalpha)
    if head.isnumeric():
        return _run(text, start + 1, n, str.isnumeric)
    if head not in _WHITESPACE:
        return _run(text, start + 1, n, _is_other)

    # `\s+(?!\S)`: the whitespace run, minus its last character when a non-space follows
    # (that last space belongs to the next word). Then `\s+` as the fallback.
    end = _run(text, i, n, lambda c: c in _WHITESPACE)
    if end == n:
        return end
    return end - 1 if end - 1 > i else end


def gpt2_split(text: str) -> list[tuple[int, int]]:
    """Split `text` into the GPT-2 pre-tokens, as (start, end) character offsets."""
    spans, i, n = [], 0, len(text)
    while i < n:
        j = _match_one(text, i, n)
        spans.append((i, j))
        i = j
    return spans


class ByteLevel:
    """Byte-level pre-tokenizer and decoder (`pre_tokenizers.ByteLevel` / `decoders.ByteLevel`).

    As a pre-tokenizer it splits on the GPT-2 regex and re-encodes each piece byte by byte;
    as a decoder it maps those characters back to bytes and reassembles the text.
    """

    def __init__(self, add_prefix_space: bool = True, trim_offsets: bool = True,
                 use_regex: bool = True):
        self.add_prefix_space = add_prefix_space
        self.trim_offsets = trim_offsets
        self.use_regex = use_regex

    @staticmethod
    def alphabet() -> list[str]:
        """The 256 characters a byte can map to -- the trainer's initial alphabet."""
        return list(BYTE_TO_CHAR.values())

    def pre_tokenize_str(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        """Pieces of `text`, byte-level encoded, with their offsets in the original text."""
        if not text:
            return []
        if self.add_prefix_space and not text.startswith(" "):
            # The offset shift this implies is why the library treats it as a normalization.
            # The inserted space has no character of its own to point at, so it borrows the
            # first one: a piece that is nothing but that space still reports (0, 1) rather
            # than collapsing to the empty (0, 0). That only comes up when the text opens
            # with whitespace, which is when the space is left as a piece by itself.
            pieces = self._encode(" " + text)
            return [(s, (max(a - 1, 0), max(b - 1, 1))) for s, (a, b) in pieces]
        return self._encode(text)

    def _encode(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        # `.decode("latin-1")` turns the UTF-8 bytes into one character each, so the
        # byte -> character mapping is a `str.translate` rather than a Python loop.
        spans = gpt2_split(text) if self.use_regex else [(0, len(text))]
        return [(text[a:b].encode("utf-8").decode("latin-1").translate(BYTE_TO_CHAR), (a, b))
                for a, b in spans if b > a]

    def decode(self, tokens: Sequence[str]) -> str:
        """Tokens back to text. Bytes are pooled across tokens first, because a single token
        can hold a partial UTF-8 sequence that is only valid once its neighbours join it."""
        out = bytearray()
        for token in tokens:
            try:
                out += bytes(CHAR_TO_BYTE[c] for c in token)
            except KeyError:                            # not byte-level: pass it through
                out += token.encode("utf-8")
        return out.decode("utf-8", errors="replace")

    def to_dict(self) -> dict:
        return {"type": "ByteLevel", "add_prefix_space": self.add_prefix_space,
                "trim_offsets": self.trim_offsets, "use_regex": self.use_regex}

    @classmethod
    def from_dict(cls, d: dict) -> "ByteLevel":
        return cls(d.get("add_prefix_space", True), d.get("trim_offsets", True),
                   d.get("use_regex", True))


# --- fixed-width and regex pre-tokenization ------------------------------------------------
# Two more pre-tokenizers, one per side of this corpus, plus the decoder they share. Neither
# needs a byte mapping: the cipher is written in "0"/"1" and the plaintext in ASCII letters
# and spaces, so the characters already *are* the alphabet and concatenating tokens (`Fuse`)
# puts a line back together exactly.


class FixedLength:
    """Cut the text into fixed-width chunks (`pre_tokenizers.FixedLength`).

    The cipher spends exactly 8 bits on each plaintext character, so chunking at 8 puts one
    pre-token on each character. BPE then merges *inside* a chunk and never across one, which
    is the whole point: no token may straddle two characters, and the source sequence ends up
    one token per character, aligned with the target it has to produce.
    """

    def __init__(self, length: int = 8):
        if length < 1:
            raise ValueError(f"length must be positive, not {length}")
        self.length = length

    def pre_tokenize_str(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        n, size = len(text), self.length
        return [(text[i:i + size], (i, min(i + size, n))) for i in range(0, n, size)]

    def to_dict(self) -> dict:
        return {"type": "FixedLength", "length": self.length}

    @classmethod
    def from_dict(cls, d: dict) -> "FixedLength":
        return cls(d.get("length", 8))


class Regex:
    """Marks a pattern as a regex for `Split`; a bare `str` there is a literal, as in
    `tokenizers.Regex`."""

    def __init__(self, pattern: str):
        self.pattern = pattern


class Split:
    """Split on a pattern (`pre_tokenizers.Split`).

    The plaintext uses ` ?[A-Za-z]+`, which is the ` ?\\p{L}+` arm of the GPT-2 regex narrowed
    to this corpus's alphabet: every pre-token is a word carrying its own leading space, so
    the space is an ordinary vocabulary character rather than a marker, and `Fuse` decodes.

    `behavior="isolated"` emits the matches and the gaps between them; `"removed"` drops the
    matches and keeps the gaps. The pattern is handed to Python's `re`, which agrees with the
    library's Rust engine on ASCII patterns like this one -- `verify_tokenizer.py` checks that
    over the whole corpus rather than taking it on trust.
    """

    _BEHAVIORS = {"isolated": "Isolated", "removed": "Removed"}

    def __init__(self, pattern: "str | Regex", behavior: str = "isolated",
                 invert: bool = False):
        if invert:
            raise NotImplementedError("Split(invert=True) is not implemented")
        if behavior.lower() not in self._BEHAVIORS:
            raise NotImplementedError(f"Split behavior {behavior!r} is not implemented")
        self.is_literal = isinstance(pattern, str)
        self.pattern = pattern if self.is_literal else pattern.pattern
        self.behavior = behavior.lower()
        self.invert = False
        self._re = re.compile(re.escape(self.pattern) if self.is_literal else self.pattern)

    def pre_tokenize_str(self, text: str) -> list[tuple[str, tuple[int, int]]]:
        pieces, at = [], 0
        keep = self.behavior == "isolated"
        for match in self._re.finditer(text):
            a, b = match.span()
            if a > at:                                  # the gap before this match
                pieces.append((text[at:a], (at, a)))
            if keep and b > a:                          # zero-width matches split nothing
                pieces.append((text[a:b], (a, b)))
            at = b
        if at < len(text):
            pieces.append((text[at:], (at, len(text))))
        return pieces

    def to_dict(self) -> dict:
        key = "String" if self.is_literal else "Regex"
        return {"type": "Split", "pattern": {key: self.pattern},
                "behavior": self._BEHAVIORS[self.behavior], "invert": self.invert}

    @classmethod
    def from_dict(cls, d: dict) -> "Split":
        pattern = d["pattern"]
        raw = pattern.get("Regex")
        return cls(pattern["String"] if raw is None else Regex(raw),
                   d.get("behavior", "Isolated"), d.get("invert", False))


class Fuse:
    """Concatenate the tokens (`decoders.Fuse`).

    Lossless whenever the vocabulary is built from the text's own characters, which is the
    case for both tokenizers here -- unlike `ByteLevel`, which has a mapping to undo first.
    """

    def decode(self, tokens: Sequence[str]) -> str:
        return "".join(tokens)

    def to_dict(self) -> dict:
        return {"type": "Fuse"}

    @classmethod
    def from_dict(cls, d: dict) -> "Fuse":
        return cls()


# --- the BPE model ------------------------------------------------------------------------


class BPE:
    """Learned vocabulary plus ranked merges; turns one pre-token into token ids."""

    def __init__(self, vocab: dict[str, int] | None = None,
                 merges: Sequence[tuple[str, str]] | None = None,
                 unk_token: str | None = None, dropout: float | None = None,
                 continuing_subword_prefix: str | None = None,
                 end_of_word_suffix: str | None = None, fuse_unk: bool = False,
                 byte_fallback: bool = False, ignore_merges: bool = False,
                 cache_capacity: int | None = None):
        if dropout:
            raise NotImplementedError("BPE dropout is not implemented")
        if byte_fallback or ignore_merges:
            raise NotImplementedError("byte_fallback / ignore_merges are not implemented")
        self.unk_token = unk_token
        self.fuse_unk = fuse_unk
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.dropout, self.byte_fallback, self.ignore_merges = None, False, False
        self._set_vocab(dict(vocab or {}), list(merges or []))

    def _set_vocab(self, vocab: dict[str, int], merges: Sequence[Sequence[str]]) -> None:
        self.vocab = vocab
        self.vocab_r = {i: t for t, i in vocab.items()}
        # (left id, right id) -> (rank, merged id). A pair listed twice keeps its last rank,
        # exactly as the Rust map does.
        cut = len(self.continuing_subword_prefix or "")
        self.merges: dict[tuple[int, int], tuple[int, int]] = {}
        for rank, (a, b) in enumerate(merges):
            self.merges[(vocab[a], vocab[b])] = (rank, vocab[a + b[cut:]])
        self._check_well_founded()

        # Encoding works on `chr(id)` strings, so applying a merge is one `str.replace`.
        self._ranked = {chr(a) + chr(b): (rank, chr(new))
                        for (a, b), (rank, new) in self.merges.items()}
        self._by_rank = [(pair, found[1])
                         for pair, found in sorted(self._ranked.items(),
                                                   key=lambda kv: kv[1][0])]
        self._to_ids = {ord(t): i for t, i in vocab.items() if len(t) == 1}
        self._drop_known = dict.fromkeys(self._to_ids)   # deletes every known character
        self._cache: dict[str, list[int]] = {}

    def _check_well_founded(self) -> None:
        """A merge only ever uses tokens that earlier merges created -- the token a merge
        builds is brand new, so nothing before it can mention it. Both encoding paths below
        rely on that, so check it rather than assume it: a hand-written merge table where
        two different pairs spell the same token can break it."""
        produced: dict[int, int] = {}
        for rank, new in self.merges.values():
            produced[new] = max(produced.get(new, -1), rank)
        for (a, b), (rank, _) in self.merges.items():
            if produced.get(a, -1) >= rank or produced.get(b, -1) >= rank:
                raise NotImplementedError(
                    f"merge {self.vocab_r[a]!r} + {self.vocab_r[b]!r} (rank {rank}) uses a "
                    "token that a later merge produces; this table cannot be applied in "
                    "rank order")

    # -- encoding --

    def _symbols(self, word: str) -> tuple[str, list[int]]:
        """One symbol per character of the pre-token, as `chr(id)`.

        Also returns how many characters each `<unk>` symbol stands for, in order -- one
        each, or the whole run when `fuse_unk`. Every other symbol covers exactly one
        character, so that list is all the caller needs to place the tokens back in the text.
        """
        if not (self.continuing_subword_prefix or self.end_of_word_suffix):
            if not word.translate(self._drop_known):    # nothing unknown is left over
                return word.translate(self._to_ids), []
        symbols, unk_widths, unk, run, last = [], [], None, 0, len(word) - 1
        for k, ch in enumerate(word):
            s = ch
            if k and self.continuing_subword_prefix:
                s = self.continuing_subword_prefix + s
            if k == last and self.end_of_word_suffix:
                s = s + self.end_of_word_suffix
            known = self.vocab.get(s)
            if known is not None:
                if unk is not None:
                    symbols.append(unk)
                    unk_widths.append(run)
                    unk, run = None, 0
                symbols.append(chr(known))
            elif self.unk_token is not None:
                if unk is not None and not self.fuse_unk:
                    symbols.append(unk)                 # one <unk> per unknown character
                    unk_widths.append(run)
                    run = 0
                unk = chr(self.vocab[self.unk_token])
                run += 1
        if unk is not None:
            symbols.append(unk)
            unk_widths.append(run)
        return "".join(symbols), unk_widths

    def _merge_lowest_first(self, word: str) -> str:
        """Merge the lowest-ranked pair the word still has, and repeat: the definition of
        BPE encoding. Scanning the word for its pairs costs a Python loop, so this is the
        path for short pre-tokens."""
        ranked = self._ranked
        while True:
            best, target = None, None
            for i in range(len(word) - 1):
                pair = word[i:i + 2]
                found = ranked.get(pair)
                if found is not None and (best is None or found[0] < best[0]):
                    best, target = found, pair
            if best is None:
                return word
            word = word.replace(target, best[1])

    def _merge_by_rank(self, word: str) -> str:
        """The same merges in the same order, driven by the table instead of the word.

        Walking the table from rank 0 reaches the pairs in exactly the order the loop above
        would pick them (a merge cannot create a lower-ranked pair, which is what
        `_check_well_founded` guarantees), and `str.replace` does the scanning in C. Worth
        it once a pre-token is long enough that scanning it in Python costs more than
        stepping over the merges that do not apply -- a whole cipher line, say.
        """
        for target, replacement in self._by_rank:
            word = word.replace(target, replacement)    # a miss returns the word untouched
        return word

    def _merge_word(self, word: str) -> tuple[list[int], list[int]]:
        symbols, unk_widths = self._symbols(word)
        if len(symbols) > 1 and self._ranked:
            symbols = (self._merge_by_rank(symbols) if 40 * len(symbols) > len(self._by_rank)
                       else self._merge_lowest_first(symbols))
        ids = list(map(ord, symbols))
        return ids, self._widths(ids, unk_widths)

    def _widths(self, ids: list[int], unk_widths: list[int]) -> list[int]:
        """How many characters of the pre-token each id covers.

        A token spells out the characters it was merged from, so its own length is its width
        -- except `<unk>`, which stands for characters it does not spell, and whose widths
        `_symbols` counted on the way past.
        """
        tokens = self.vocab_r
        if not unk_widths:
            return [len(tokens[i]) for i in ids]
        unk_id = self.vocab[self.unk_token]
        remaining = iter(unk_widths)
        return [next(remaining) if i == unk_id else len(tokens[i]) for i in ids]

    def _tokenize_cached(self, word: str) -> tuple[list[int], list[int]]:
        """(ids, widths) for one pre-token, memoised: the same word recurs constantly."""
        hit = self._cache.get(word)
        if hit is None:
            hit = self._cache[word] = self._merge_word(word)
        return hit

    def tokenize(self, word: str) -> list[int]:
        """Token ids for one pre-token."""
        return self._tokenize_cached(word)[0] if word else []

    # -- serialization --

    def to_dict(self) -> dict:
        by_rank = {rank: [self.vocab_r[a], self.vocab_r[b]]
                   for (a, b), (rank, _) in self.merges.items()}
        return {
            "type": "BPE",
            "dropout": None,
            "unk_token": self.unk_token,
            "continuing_subword_prefix": self.continuing_subword_prefix,
            "end_of_word_suffix": self.end_of_word_suffix,
            "fuse_unk": self.fuse_unk,
            "byte_fallback": False,
            "ignore_merges": False,
            "vocab": dict(sorted(self.vocab.items(), key=lambda kv: kv[1])),
            # A pair merged twice survives only at its last rank, as in the Rust map.
            "merges": [by_rank[rank] for rank in sorted(by_rank)],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "BPE":
        model = cls(unk_token=d.get("unk_token"), dropout=d.get("dropout"),
                    continuing_subword_prefix=d.get("continuing_subword_prefix"),
                    end_of_word_suffix=d.get("end_of_word_suffix"),
                    fuse_unk=d.get("fuse_unk", False),
                    byte_fallback=d.get("byte_fallback", False),
                    ignore_merges=d.get("ignore_merges", False))
        model._set_vocab(dict(d["vocab"]), [tuple(m) for m in d["merges"]])
        return model


# --- the trainer ---------------------------------------------------------------------------


class BpeTrainer:
    """Learns a BPE vocabulary: repeatedly merge the most frequent adjacent pair.

    The bookkeeping is the fiddly part and is kept identical to the library's:

      * ids are handed out in a fixed order -- special tokens, then the alphabet sorted by
        code point, then one per merge -- so the vocabulary is reproducible;
      * pair counts are maintained incrementally. Merging a pair only changes the counts of
        the pairs straddling each merge site, so each merge touches the words that contain
        it rather than the whole corpus;
      * the queue is a max-heap on count, ties broken by the smaller (left id, right id).
        An entry whose count is out of date is refreshed and pushed back rather than
        trusted, since counts fall as other merges eat into them.
    """

    def __init__(self, vocab_size: int = 30000, min_frequency: int = 0,
                 special_tokens: Sequence[str] = (), limit_alphabet: int | None = None,
                 initial_alphabet: Sequence[str] = (),
                 continuing_subword_prefix: str | None = None,
                 end_of_word_suffix: str | None = None, max_token_length: int | None = None,
                 show_progress: bool = True):
        if max_token_length is not None:
            raise NotImplementedError("max_token_length is not implemented")
        self.vocab_size = vocab_size
        self.min_frequency = min_frequency
        self.special_tokens = [str(t) for t in special_tokens]
        self.limit_alphabet = limit_alphabet
        self.initial_alphabet = list(dict.fromkeys(initial_alphabet))
        self.continuing_subword_prefix = continuing_subword_prefix
        self.end_of_word_suffix = end_of_word_suffix
        self.show_progress = show_progress

    def train(self, word_counts: dict[str, int], model: BPE) -> list[str]:
        """Fill `model` from `{pre-token: occurrences}`. Returns the special tokens added."""
        vocab, id_to_token, alphabet = self._initial_vocab(word_counts)
        words, counts = self._encode_words(word_counts, vocab, id_to_token, alphabet)
        merges = self._learn_merges(words, counts, vocab, id_to_token)

        model.continuing_subword_prefix = self.continuing_subword_prefix
        model.end_of_word_suffix = self.end_of_word_suffix
        model._set_vocab(vocab, [(id_to_token[ord(a)], id_to_token[ord(b)])
                                 for a, b in merges])
        return list(self.special_tokens)

    def _initial_vocab(self, word_counts: dict[str, int]
                       ) -> tuple[dict[str, int], list[str], dict[str, float]]:
        """Special tokens first, then every character seen, sorted by code point."""
        vocab: dict[str, int] = {}
        id_to_token: list[str] = []

        def add(token: str) -> None:
            if token not in vocab:
                vocab[token] = len(id_to_token)
                id_to_token.append(token)

        for token in self.special_tokens:
            add(token)

        alphabet: dict[str, float] = {}
        for word, count in word_counts.items():
            for ch, seen in Counter(word).items():
                alphabet[ch] = alphabet.get(ch, 0) + seen * count
        for ch in self.initial_alphabet:                # kept whatever its frequency
            alphabet[ch] = float("inf")

        kept = list(alphabet.items())
        if self.limit_alphabet is not None and len(kept) > self.limit_alphabet:
            kept.sort(key=lambda kv: kv[1])             # drop the rarest characters
            kept = kept[len(kept) - self.limit_alphabet:]
        for ch, _ in sorted(kept, key=lambda kv: ord(kv[0])):
            add(ch)
        return vocab, id_to_token, alphabet

    def _encode_words(self, word_counts: dict[str, int], vocab: dict[str, int],
                      id_to_token: list[str],
                      alphabet: dict[str, float]) -> tuple[list[str], list[int]]:
        """Each word becomes a string of `chr(id)`, one id per character it kept."""
        words, counts = [], []
        prefix, suffix = self.continuing_subword_prefix, self.end_of_word_suffix
        # Characters dropped by `limit_alphabet` map to None, which deletes them.
        table = {ord(ch): vocab.get(ch) for ch in alphabet}
        for word, count in word_counts.items():
            counts.append(count)
            if not (prefix or suffix):
                words.append(word.translate(table))
                continue
            symbols, last = [], len(word) - 1
            for k, ch in enumerate(word):
                if ch not in vocab:                     # dropped by limit_alphabet
                    continue
                s = (prefix + ch if k and prefix else ch) + (suffix if k == last and suffix else "")
                if s not in vocab:
                    vocab[s] = len(id_to_token)
                    id_to_token.append(s)
                symbols.append(chr(vocab[s]))
            words.append("".join(symbols))
        return words, counts

    def _learn_merges(self, words: list[str], counts: list[int], vocab: dict[str, int],
                      id_to_token: list[str]) -> list[tuple[str, str]]:
        pair_counts: dict[tuple[str, str], int] = {}
        pending: dict[tuple[str, str], set[int]] = {}   # pairs whose count just went up
        for i, word in enumerate(words):
            weight = counts[i]
            for pair, seen in Counter(zip(word, word[1:])).items():
                pair_counts[pair] = pair_counts.get(pair, 0) + seen * weight
                pending.setdefault(pair, set()).add(i)

        # Heap entries are (-count, pair, tiebreak, words holding the pair): most frequent
        # first, then the smaller pair, which is the library's ordering.
        tiebreak = _counter()
        queue: list[tuple[int, tuple[str, str], int, set[int]]] = []

        def flush() -> None:
            for pair, where in pending.items():
                total = pair_counts[pair]
                if total > 0:
                    heappush(queue, (-total, pair, next(tiebreak), where))
            pending.clear()

        flush()
        merges: list[tuple[str, str]] = []
        prefix = self.continuing_subword_prefix
        while len(vocab) < self.vocab_size and queue:
            negated, pair, _, where = heappop(queue)
            count, current = -negated, pair_counts[pair]
            if count != current:                        # stale: re-file it at its real count
                heappush(queue, (-current, pair, next(tiebreak), where))
                continue
            if count < 1 or count < self.min_frequency:
                break                                   # nothing frequent enough is left

            left, right = id_to_token[ord(pair[0])], id_to_token[ord(pair[1])]
            if prefix and right.startswith(prefix):
                right = right[len(prefix):]
            token = left + right
            if token in vocab:
                new_id = vocab[token]                   # two pairs can spell the same token
            else:
                new_id = vocab[token] = len(id_to_token)
                id_to_token.append(token)
            merges.append(pair)

            self._apply(words, counts, where, pair, chr(new_id), pair_counts, pending)
            flush()

        if self.show_progress:
            print(f"[bpe] {len(merges):,} merges, vocab {len(vocab):,}")
        return merges

    @staticmethod
    def _apply(words: list[str], counts: list[int], where: Iterable[int],
               pair: tuple[str, str], new_char: str, pair_counts: dict[tuple[str, str], int],
               pending: dict[tuple[str, str], set[int]]) -> None:
        """Merge `pair` everywhere it occurs in the given words and adjust the pair counts.

        Only the two pairs straddling a merge site change: (left, c1) and (c2, right) lose an
        occurrence, (left, new) and (new, right) gain one. Occurrences are taken left to
        right and never overlap, which is exactly what `str.split` finds -- so the word is
        cut at its merge sites once, and the neighbouring symbols are read off the ends of
        the pieces rather than by walking the word in Python.
        """
        c1, c2 = pair
        target = c1 + c2
        for i in where:
            parts = words[i].split(target)
            sites = len(parts) - 1
            if not sites:                               # eaten by an earlier merge
                continue
            words[i] = new_char.join(parts)

            # Left neighbours: the last character of the piece before each site. An empty
            # piece means the previous site is adjacent, so the neighbour is the symbol this
            # merge just made -- unless the site is at the very start, which has no
            # neighbour and is the one empty piece to discount.
            left = Counter(map(_LAST, parts[:sites]))
            adjacent = left.pop("", 0) - (0 if parts[0] else 1)
            if adjacent:
                left[new_char] = left.get(new_char, 0) + adjacent
            # Right neighbours, symmetrically. An empty piece after a site means the next
            # site follows immediately, and its left half is still unmerged at this point.
            right = Counter(map(_FIRST, parts[1:]))
            adjacent = right.pop("", 0) - (0 if parts[-1] else 1)
            if adjacent:
                right[c1] = right.get(c1, 0) + adjacent

            weight = counts[i]
            for neighbour, seen in left.items():
                delta = seen * weight
                key = (neighbour, c1)
                pair_counts[key] = pair_counts.get(key, 0) - delta
                key = (neighbour, new_char)
                pair_counts[key] = pair_counts.get(key, 0) + delta
                holders = pending.get(key)
                if holders is None:
                    pending[key] = {i}
                else:
                    holders.add(i)
            for neighbour, seen in right.items():
                delta = seen * weight
                key = (c2, neighbour)
                pair_counts[key] = pair_counts.get(key, 0) - delta
                key = (new_char, neighbour)
                pair_counts[key] = pair_counts.get(key, 0) + delta
                holders = pending.get(key)
                if holders is None:
                    pending[key] = {i}
                else:
                    holders.add(i)


# --- the tokenizer -------------------------------------------------------------------------


class AddedToken:
    """A token added outside the model -- here only the specials, which are matched in the
    raw text before the model ever sees it."""

    def __init__(self, content: str, special: bool = False, single_word: bool = False,
                 lstrip: bool = False, rstrip: bool = False, normalized: bool | None = None):
        self.content = content
        self.special = special
        self.single_word = single_word
        self.lstrip = lstrip
        self.rstrip = rstrip
        self.normalized = (not special) if normalized is None else normalized

    def to_dict(self, token_id: int) -> dict:
        return {"id": token_id, "content": self.content, "single_word": self.single_word,
                "lstrip": self.lstrip, "rstrip": self.rstrip, "normalized": self.normalized,
                "special": self.special}


class Encoding:
    """What `encode` returns. `ids` is what the model consumes; the rest is for inspection."""

    __slots__ = ("ids", "tokens", "offsets")

    def __init__(self, ids: list[int], tokens: list[str],
                 offsets: list[tuple[int, int]]):
        self.ids, self.tokens, self.offsets = ids, tokens, offsets

    def __len__(self) -> int:
        return len(self.ids)

    def __repr__(self) -> str:
        return f"Encoding(len={len(self.ids)}, tokens={self.tokens[:8]}...)"


# Which classes a saved file's `"type"` may name, per slot.
PRE_TOKENIZER_TYPES = {"ByteLevel": ByteLevel, "FixedLength": FixedLength, "Split": Split}
DECODER_TYPES = {"ByteLevel": ByteLevel, "Fuse": Fuse}


def _build(types: dict, spec: dict | None, slot: str):
    """One component from its saved form; unknown kinds are refused, not ignored."""
    if not spec:
        return None
    kind = spec.get("type")
    if kind not in types:
        raise NotImplementedError(f"{slot} {kind!r} is not implemented")
    return types[kind].from_dict(spec)


class Tokenizer:
    """Text <-> ids: added tokens, then the pre-tokenizer, then the BPE model."""

    def __init__(self, model: BPE):
        self.model = model
        self.pre_tokenizer: ByteLevel | None = None
        self.decoder: ByteLevel | None = None
        self.normalizer = None
        self.added_tokens: list[AddedToken] = []
        self._added_by_content: dict[str, int] = {}
        self._added_by_id: dict[int, AddedToken] = {}

    # -- vocabulary --

    def add_special_tokens(self, tokens: Sequence[str | AddedToken]) -> int:
        added = 0
        for token in tokens:
            token = token if isinstance(token, AddedToken) else AddedToken(token, special=True)
            if token.content in self._added_by_content:
                continue
            token_id = self.model.vocab.get(token.content)
            if token_id is None:                        # not learned: append to the vocabulary
                token_id = len(self.model.vocab)
                self.model.vocab[token.content] = token_id
                self.model.vocab_r[token_id] = token.content
            self.added_tokens.append(token)
            self._added_by_content[token.content] = token_id
            self._added_by_id[token_id] = token
            added += 1
        return added

    def get_vocab(self, with_added_tokens: bool = True) -> dict[str, int]:
        vocab = dict(self.model.vocab)
        if with_added_tokens:
            vocab.update(self._added_by_content)
        return vocab

    def get_vocab_size(self, with_added_tokens: bool = True) -> int:
        return len(self.get_vocab(with_added_tokens))

    def token_to_id(self, token: str) -> int | None:
        return self.get_vocab().get(token)

    def id_to_token(self, token_id: int) -> str | None:
        added = self._added_by_id.get(token_id)
        return added.content if added is not None else self.model.vocab_r.get(token_id)

    # -- encoding --

    def _split_added(self, text: str) -> list[tuple[str, int | None]]:
        """Cut the added tokens out of `text`; the pieces between them go to the model."""
        contents = self._added_by_content
        if not contents or not any(c in text for c in contents):
            return [(text, None)]
        pieces: list[tuple[str, int | None]] = []
        i, start, n = 0, 0, len(text)
        while i < n:
            hit = max((c for c in contents if text.startswith(c, i)), key=len, default=None)
            if hit is None:
                i += 1
                continue
            if i > start:
                pieces.append((text[start:i], None))
            pieces.append((hit, contents[hit]))
            i = start = i + len(hit)
        if start < n:
            pieces.append((text[start:], None))
        return pieces

    def encode(self, sequence: str, add_special_tokens: bool = True) -> Encoding:
        ids: list[int] = []
        tokens: list[str] = []
        offsets: list[tuple[int, int]] = []
        base = 0
        for piece, added_id in self._split_added(sequence):
            if added_id is not None:
                ids.append(added_id)
                tokens.append(piece)
                offsets.append((base, base + len(piece)))
            else:
                self._encode_piece(piece, base, ids, tokens, offsets)
            base += len(piece)
        return Encoding(ids, tokens, offsets)

    def _encode_piece(self, text: str, base: int, ids: list[int], tokens: list[str],
                      offsets: list[tuple[int, int]]) -> None:
        if self.pre_tokenizer is None:
            pre_tokens = [(text, (0, len(text)))] if text else []
        else:
            pre_tokens = self.pre_tokenizer.pre_tokenize_str(text)
        for word, (start, end) in pre_tokens:
            if not word:
                continue
            word_ids, widths = self.model._tokenize_cached(word)
            ids += word_ids
            tokens += [self.model.vocab_r[i] for i in word_ids]
            offsets += self._offsets(text[start:end], word, widths, base + start)

    @staticmethod
    def _offsets(source: str, word: str, widths: list[int],
                 start: int) -> list[tuple[int, int]]:
        """Map each token back to a span of the original text, given how many characters of
        the pre-token each one covers. Byte-level encoding makes one pre-token character per
        UTF-8 byte, so there the span is found by walking the source's byte lengths."""
        if len(word) == len(source):                    # no expansion: one character each
            spans, at = [], start
            for width in widths:
                spans.append((at, at + width))
                at += width
            return spans
        positions, at = [], start                       # pre-token character -> source character
        for ch in source:
            positions += [at] * len(ch.encode("utf-8"))
            at += 1
        # A token can end inside a multi-byte character; the library reports the whole
        # character, so the span runs to one past the character its last byte belongs to.
        spans, at = [], 0
        for width in widths:
            spans.append((positions[at], positions[at + width - 1] + 1))
            at += width
        return spans

    def encode_batch(self, sequences: Sequence[str],
                     add_special_tokens: bool = True) -> list[Encoding]:
        return [self.encode(s, add_special_tokens) for s in sequences]

    # -- decoding --

    def decode(self, ids: Iterable[int], skip_special_tokens: bool = True) -> str:
        tokens = []
        for token_id in ids:
            token = self.id_to_token(int(token_id))
            if token is None:
                continue
            added = self._added_by_content.get(token)
            if skip_special_tokens and added is not None and self._added_by_id[added].special:
                continue
            tokens.append(token)
        if self.decoder is None:
            return " ".join(tokens)
        return self.decoder.decode(tokens)

    def decode_batch(self, sequences: Iterable[Iterable[int]],
                     skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids, skip_special_tokens) for ids in sequences]

    # -- training --

    def train_from_iterator(self, iterator: Iterable[str], trainer: BpeTrainer,
                            length: int | None = None) -> None:
        """Count pre-tokens over the corpus, learn the merges, install the result."""
        word_counts: Counter[str] = Counter()
        for sequence in iterator:
            if self.pre_tokenizer is None:
                word_counts[sequence] += 1
            else:
                word_counts.update(w for w, _ in self.pre_tokenizer.pre_tokenize_str(sequence))
        specials = trainer.train(dict(word_counts), self.model)
        self.added_tokens.clear()
        self._added_by_content.clear()
        self._added_by_id.clear()
        self.add_special_tokens([AddedToken(t, special=True) for t in specials])

    # -- serialization --

    def to_dict(self) -> dict:
        return {
            "version": "1.0",
            "truncation": None,
            "padding": None,
            "added_tokens": [t.to_dict(self._added_by_content[t.content])
                             for t in self.added_tokens],
            "normalizer": None,
            "pre_tokenizer": self.pre_tokenizer.to_dict() if self.pre_tokenizer else None,
            "post_processor": None,
            "decoder": self.decoder.to_dict() if self.decoder else None,
            "model": self.model.to_dict(),
        }

    def to_str(self, pretty: bool = False) -> str:
        """The same JSON the library writes: 2-space indent, raw UTF-8, no trailing newline."""
        if pretty:
            return json.dumps(self.to_dict(), indent=2, ensure_ascii=False)
        return json.dumps(self.to_dict(), separators=(",", ":"), ensure_ascii=False)

    def save(self, path: str | Path, pretty: bool = True) -> None:
        Path(path).write_text(self.to_str(pretty), encoding="utf-8")

    @classmethod
    def from_dict(cls, d: dict) -> "Tokenizer":
        if d.get("normalizer") or d.get("post_processor"):
            raise NotImplementedError("normalizers and post-processors are not implemented")
        if d.get("truncation") or d.get("padding"):
            raise NotImplementedError("truncation and padding are not implemented")
        tok = cls(BPE.from_dict(d["model"]))
        tok.pre_tokenizer = _build(PRE_TOKENIZER_TYPES, d.get("pre_tokenizer"), "pre-tokenizer")
        tok.decoder = _build(DECODER_TYPES, d.get("decoder"), "decoder")
        for entry in d.get("added_tokens", []):
            tok.add_special_tokens([AddedToken(
                entry["content"], special=entry.get("special", False),
                single_word=entry.get("single_word", False),
                lstrip=entry.get("lstrip", False), rstrip=entry.get("rstrip", False),
                normalized=entry.get("normalized", False))])
            got = tok._added_by_content[entry["content"]]
            if got != entry["id"]:
                raise ValueError(f"added token {entry['content']!r} is id {got}, "
                                 f"not {entry['id']} as the file says")
        return tok

    @classmethod
    def from_file(cls, path: str | Path) -> "Tokenizer":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    @classmethod
    def from_str(cls, text: str) -> "Tokenizer":
        return cls.from_dict(json.loads(text))


# --- drop-in namespaces --------------------------------------------------------------------
# So that swapping this module in is one import line, these mirror the library's layout:
#     from tokenizers import Regex, Tokenizer, decoders, models, pre_tokenizers, trainers


class models:
    BPE = BPE


class pre_tokenizers:
    ByteLevel = ByteLevel
    FixedLength = FixedLength
    Split = Split


class decoders:
    ByteLevel = ByteLevel
    Fuse = Fuse


class trainers:
    BpeTrainer = BpeTrainer


# --- self-test -------------------------------------------------------------------------------
# `src/verify_tokenizer.py` is the real check, but it needs the library installed. This one
# stands on its own: the invariants below are what the library would otherwise be asserting.


if __name__ == "__main__":
    assert len(BYTE_TO_CHAR) == 256 and len(CHAR_TO_BYTE) == 256, "byte table is not a bijection"
    assert BYTE_TO_CHAR[0x20] == "Ġ" and BYTE_TO_CHAR[0x0A] == "Ċ"

    splits = {
        "Hello, world!": ["Hello", ",", "Ġworld", "!"],
        "don't stop": ["don", "'t", "Ġstop"],
        "a   b": ["a", "ĠĠ", "Ġb"],
        "3 apples, 42 pears": ["3", "Ġapples", ",", "Ġ42", "Ġpears"],
        "  trailing  ": ["Ġ", "Ġtrailing", "ĠĠ"],
    }
    byte_level = ByteLevel(add_prefix_space=False)
    for text, expected in splits.items():
        got = [s for s, _ in byte_level.pre_tokenize_str(text)]
        assert got == expected, f"{text!r}: {got} != {expected}"
    print(f"  GPT-2 pre-tokenization matches on {len(splits)} cases")

    corpus = ["the cat sat on the mat", "the cat ate the rat", "a cat and a rat"] * 20
    tok = Tokenizer(models.BPE(unk_token="<unk>"))
    tok.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tok.decoder = ByteLevel()
    tok.train_from_iterator(corpus, trainer=BpeTrainer(
        vocab_size=300, special_tokens=["<pad>", "<unk>"], min_frequency=2,
        initial_alphabet=ByteLevel.alphabet(), show_progress=False))
    # 2 specials + 256 bytes + merges. The cap is a ceiling, not a target: on a corpus this
    # small `min_frequency` runs out of pairs worth merging long before 300 is reached.
    assert 258 < tok.get_vocab_size() <= 300, tok.get_vocab_size()
    assert tok.token_to_id("<pad>") == 0 and tok.token_to_id("<unk>") == 1
    assert "Ġcat" in tok.get_vocab(), "the most frequent word should have been merged"
    for text in corpus[:3] + ["unseen words entirely", "<pad> and <unk>", ""]:
        assert tok.decode(tok.encode(text).ids, skip_special_tokens=False) == text, text
    print(f"  trained a {tok.get_vocab_size()}-token vocabulary and round-tripped it")

    # Both encoding paths implement the same rule and must agree, whatever the input.
    model = tok.model
    for text in corpus + ["a rat sat on a cat" * 30]:
        for word, _ in tok.pre_tokenizer.pre_tokenize_str(text):
            symbols, _ = model._symbols(word)
            assert model._merge_lowest_first(symbols) == model._merge_by_rank(symbols), word
    print("  the two merge paths agree")

    # A token's span is the characters it stands for, not the characters it spells: <unk> is
    # five characters wide and covers one.
    unknown = tok.encode("caté", add_special_tokens=False)
    assert unknown.offsets[-1] == (3, 4), unknown.offsets
    assert [t for t in unknown.tokens if t == "<unk>"] == [], "byte-level has no unknowns"
    letters = Tokenizer(models.BPE(unk_token="<unk>"))
    letters.pre_tokenizer, letters.decoder = Split(Regex("[a-z]+"), "isolated"), Fuse()
    letters.train_from_iterator(["abc"] * 4, trainer=BpeTrainer(
        vocab_size=32, special_tokens=["<pad>", "<unk>"], min_frequency=2,
        initial_alphabet=list("abc"), show_progress=False))
    got = letters.encode("ab9c")
    assert got.tokens == ["ab", "<unk>", "c"], got.tokens
    assert got.offsets == [(0, 2), (2, 3), (3, 4)], got.offsets
    print("  <unk> spans the characters it replaced")

    reloaded = Tokenizer.from_str(tok.to_str(pretty=True))
    assert reloaded.to_str() == tok.to_str(), "save/load is not a round-trip"
    assert [reloaded.encode(t).ids for t in corpus] == [tok.encode(t).ids for t in corpus]
    print("  JSON round-trip is lossless")

    # -- the two pre-tokenizers this project actually trains on --

    assert [s for s, _ in FixedLength(8).pre_tokenize_str("0" * 8 + "1" * 8 + "01")] == \
        ["00000000", "11111111", "01"], "fixed-width chunking"
    assert FixedLength(8).pre_tokenize_str("")== [] and FixedLength(3).pre_tokenize_str(
        "abcdef") == [("abc", (0, 3)), ("def", (3, 6))]

    words = Split(Regex(" ?[A-Za-z]+"), behavior="isolated")
    assert [s for s, _ in words.pre_tokenize_str("Robert Boulter is an actor")] == \
        ["Robert", " Boulter", " is", " an", " actor"], "word split"
    assert [s for s, _ in words.pre_tokenize_str("  a  b ")] == \
        [" ", " a", " ", " b", " "], "the gaps between matches are kept too"
    assert [s for s, _ in Split(Regex("[a-z]+"), behavior="removed").pre_tokenize_str(
        "a1bc2d")] == ["1", "2"], "removed behavior"
    for bad in (lambda: Split("x", behavior="contiguous"), lambda: Split("x", invert=True)):
        try:
            bad()
        except NotImplementedError:
            pass
        else:
            raise AssertionError("an unsupported Split option was accepted")
    print("  FixedLength and Split split as the library does")

    # A cipher line: 8 bits per character, so chunking at 8 and merging inside each chunk
    # must give exactly one token per character, and Fuse must give the line back.
    bits = {c: f"{i + 33:08b}" for i, c in enumerate("abcdefgh ")}
    lines = ["".join(bits[c] for c in text)
             for text in ("abc def", "dead beef", "cabbage", "a bad deed") * 25]
    cipher = Tokenizer(models.BPE(unk_token="<unk>"))
    cipher.pre_tokenizer, cipher.decoder = FixedLength(8), Fuse()
    cipher.train_from_iterator(lines, trainer=BpeTrainer(
        vocab_size=1024, special_tokens=["<pad>", "<unk>"], min_frequency=2,
        initial_alphabet=["0", "1"], show_progress=False))
    for line in lines[:8]:
        assert len(cipher.encode(line).ids) == len(line) // 8, "a chunk did not merge whole"
        assert cipher.decode(cipher.encode(line).ids) == line, "Fuse did not round-trip"
    print(f"  8-bit chunks collapse to one token each ({cipher.get_vocab_size()} tokens)")

    # The plaintext side: the vocabulary is built from the corpus's own 53 characters.
    alphabet = [" "] + [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
    text = ["the cat sat on the mat", "The Cat ate the Rat", "a cat and a rat"] * 20
    plain = Tokenizer(models.BPE(unk_token="<unk>"))
    plain.pre_tokenizer, plain.decoder = words, Fuse()
    plain.train_from_iterator(text, trainer=BpeTrainer(
        vocab_size=512, special_tokens=["<pad>", "<unk>"], min_frequency=2,
        initial_alphabet=alphabet, show_progress=False))
    learned = set(plain.get_vocab())
    assert set(alphabet) <= learned, "the alphabet is not fully in the vocabulary"
    assert not {t for t in learned if len(t) == 1} - set(alphabet) - {"<pad>", "<unk>"}, \
        "a character outside a-z, A-Z and space reached the vocabulary"
    assert " cat" in learned, "the most frequent word should have been merged"
    for line in text[:3] + ["unseen words entirely", ""]:
        assert plain.decode(plain.encode(line).ids) == line, line
    assert Tokenizer.from_str(plain.to_str()).to_str() == plain.to_str()
    assert Tokenizer.from_str(cipher.to_str()).to_str() == cipher.to_str()
    print(f"  plaintext vocabulary is {len(alphabet)} characters + merges "
          f"({plain.get_vocab_size()} tokens), and round-trips")
    print("bpe.py self-test passed")
