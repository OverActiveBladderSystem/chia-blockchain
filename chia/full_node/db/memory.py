from __future__ import annotations

from collections import defaultdict

from chia.full_node.db.ops import Delete, DeleteRange, Op, Put, key_in_range


class MemoryBackend:
    """In-process column families. Used by tests and as the session-semantics reference."""

    def __init__(self) -> None:
        self.data: dict[str, dict[bytes, bytes]] = defaultdict(dict)

    async def get(self, cf: str, key: bytes) -> bytes | None:
        return self.data[cf].get(key)

    async def get_many(self, cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
        family = self.data[cf]
        return {key: family[key] for key in keys if key in family}

    async def scan(self, cf: str, start: bytes, end: bytes | None) -> list[tuple[bytes, bytes]]:
        rows = [(key, value) for key, value in self.data[cf].items() if key_in_range(key, start, end)]
        rows.sort(key=lambda item: item[0])
        return rows

    async def apply(self, ops: list[Op]) -> None:
        for op in ops:
            family = self.data[op.cf]
            if isinstance(op, Put):
                family[op.key] = op.value
            elif isinstance(op, Delete):
                family.pop(op.key, None)
            elif isinstance(op, DeleteRange):
                stale = [key for key in family if key_in_range(key, op.start, op.end)]
                for key in stale:
                    del family[key]
            else:  # pragma: no cover
                raise TypeError(f"unknown op {op!r}")

    async def close(self) -> None:
        return None
