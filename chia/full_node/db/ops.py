from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class Put:
    cf: str
    key: bytes
    value: bytes


@dataclass(frozen=True)
class Delete:
    cf: str
    key: bytes


@dataclass(frozen=True)
class DeleteRange:
    """Delete keys `start <= key < end` in one column family."""

    cf: str
    start: bytes
    end: bytes


Op = Put | Delete | DeleteRange


class KvBackend(Protocol):
    async def get(self, cf: str, key: bytes) -> bytes | None: ...

    async def get_many(self, cf: str, keys: list[bytes]) -> dict[bytes, bytes]: ...

    async def scan(self, cf: str, start: bytes, end: bytes | None) -> list[tuple[bytes, bytes]]: ...

    async def apply(self, ops: list[Op]) -> None: ...

    async def close(self) -> None: ...


def key_in_range(key: bytes, start: bytes, end: bytes | None) -> bool:
    if key < start:
        return False
    if end is not None and key >= end:
        return False
    return True
