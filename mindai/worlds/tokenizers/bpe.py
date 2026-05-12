"""BPETokenizer — tiktoken cl100k_base wrapper.

cl100k_base is the tokenizer used by GPT-4 — proven coverage for English
plus reasonable handling of other languages and code.

Vocab cap: by default truncated to `max_vocab=8192` to keep TOKEN_NEURONS
patterns memory-bounded. Truncation drops the highest token ids; the
remaining set still covers all common English. Out-of-vocab tokens map
to UNK.
"""

from __future__ import annotations

import tiktoken


class BPETokenizer:
    """tiktoken cl100k_base tokenizer with vocab truncation."""

    def __init__(self, encoding: str = 'cl100k_base', max_vocab: int = 8192):
        self._enc       = tiktoken.get_encoding(encoding)
        self._max_vocab = max_vocab
        self.eos_id     = max_vocab - 1
        self.unk_id     = max_vocab - 2
        self.vocab_size = max_vocab

    def encode(self, text: str) -> list[int]:
        ids = self._enc.encode(text, disallowed_special=())
        clean = [i if i < self.unk_id else self.unk_id for i in ids]
        clean.append(self.eos_id)
        return clean

    def decode(self, ids: list[int]) -> str:
        usable = [i for i in ids if i < self.unk_id]
        try:
            return self._enc.decode(usable)
        except Exception:
            return ''


def get_tokenizer(name: str = 'auto', **kwargs) -> BPETokenizer:
    """Return a BPETokenizer. `name` is accepted for backward compatibility."""
    return BPETokenizer(**kwargs)
