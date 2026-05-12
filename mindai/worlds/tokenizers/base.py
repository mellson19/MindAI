"""Tokenizer base class — minimal interface used by AgentWorld."""

from __future__ import annotations
from typing import Protocol, runtime_checkable


@runtime_checkable
class Tokenizer(Protocol):
    vocab_size: int
    eos_id:     int
    unk_id:     int

    def encode(self, text: str) -> list[int]: ...
    def decode(self, ids: list[int]) -> str: ...
