from __future__ import annotations

import logging
from collections.abc import Collection

from chia_rs.sized_bytes import bytes32

from chia.full_node.db.chain_db import ChainDB, _View
from chia.full_node.db.keys import (
    CF_HINTS_BY_COIN,
    CF_HINTS_BY_HINT,
    CF_META,
    INDEX_SCAN_CHUNK,
    META_FORMAT,
    META_FORMAT_VALUE,
    META_HINT_COUNT,
    META_SCHEMA_VERSION,
    key_after,
    prefix_end,
)

log = logging.getLogger(__name__)


async def _coins_for_hints(
    view: _View, hints: list[bytes], *, max_items: int | None = None
) -> list[bytes32]:
    """Coin ids in hint order. A cap stops the walk once that many ids are collected."""
    found: list[bytes32] = []
    for hint in hints:
        if max_items is not None and len(found) >= max_items:
            break
        start = hint
        end = prefix_end(hint)
        width = len(hint) + 32
        while end is None or start < end:
            room = None if max_items is None else max_items - len(found)
            if room is not None and room <= 0:
                break
            limit = INDEX_SCAN_CHUNK if room is None or room > INDEX_SCAN_CHUNK else room
            rows = await view.scan(CF_HINTS_BY_HINT, start, end, limit=limit)
            if not rows:
                break
            for key, _value in rows:
                if len(key) != width:
                    continue
                found.append(bytes32(key[-32:]))
                if max_items is not None and len(found) >= max_items:
                    return found
            if len(rows) < limit:
                break
            start = key_after(rows[-1][0])
    return found


class RocksHintStore:
    """Coin-id / hint pairs in the chain RocksDB. Hints are not rolled back with blocks."""

    def __init__(self, db: ChainDB) -> None:
        self.db = db

    @classmethod
    async def create(cls, db: ChainDB) -> RocksHintStore:
        self = cls(db)
        async with db.writer_maybe_transaction() as session:
            if await session.get(CF_META, META_FORMAT) is None:
                session.put(CF_META, META_FORMAT, META_FORMAT_VALUE)
                session.put(CF_META, META_SCHEMA_VERSION, (2).to_bytes(4, "little", signed=False))
        return self

    async def get_coin_ids(self, hint: bytes, *, max_items: int = 50000) -> list[bytes32]:
        return await self.get_coin_ids_multi({hint}, max_items=max_items)

    async def get_coin_ids_multi(self, hints: set[bytes], *, max_items: int = 50000) -> list[bytes32]:
        if max_items <= 0:
            return []
        async with self.db.reader_no_transaction() as view:
            return await _coins_for_hints(view, list(hints), max_items=max_items)

    async def get_coin_ids_by_hints(self, hints: Collection[bytes]) -> set[bytes32]:
        if len(hints) == 0:
            return set()
        async with self.db.reader_no_transaction() as view:
            found = await _coins_for_hints(view, list(hints))
        return set(found)

    async def get_hints(self, coin_ids: list[bytes32]) -> list[bytes32]:
        if len(coin_ids) == 0:
            return []
        ranges = [(bytes(coin_id), prefix_end(bytes(coin_id))) for coin_id in coin_ids]
        hints: list[bytes32] = []
        async with self.db.reader_no_transaction() as view:
            grouped = await view.scan_prefixes(CF_HINTS_BY_COIN, ranges)
        for rows in grouped:
            for key, _value in rows:
                hint = key[32:]
                if len(hint) == 32:
                    hints.append(bytes32(hint))
        return hints

    async def add_hints(self, coin_hint_list: list[tuple[bytes32, bytes]], *, assume_new: bool = False) -> None:
        if len(coin_hint_list) == 0:
            return None
        async with self.db.writer_maybe_transaction() as session:
            if assume_new:
                existing: dict[bytes, bytes] = {}
            else:
                keys = [bytes(coin_id) + hint for coin_id, hint in coin_hint_list]
                existing = await session.get_many(CF_HINTS_BY_COIN, keys)
            seen = set(existing)
            added = 0
            for coin_id, hint in coin_hint_list:
                by_coin = bytes(coin_id) + hint
                if by_coin in seen:
                    continue
                seen.add(by_coin)
                session.put(CF_HINTS_BY_COIN, by_coin, b"")
                session.put(CF_HINTS_BY_HINT, hint + bytes(coin_id), b"")
                added += 1
            if added == 0:
                return
            raw = await session.get(CF_META, META_HINT_COUNT)
            current = 0 if raw is None else int.from_bytes(raw, "little", signed=False)
            session.put(CF_META, META_HINT_COUNT, (current + added).to_bytes(8, "little", signed=False))

    async def clear_all(self) -> None:
        async with self.db.writer_maybe_transaction() as session:
            session.delete_range(CF_HINTS_BY_COIN, b"", b"\xff" * 80)
            session.delete_range(CF_HINTS_BY_HINT, b"", b"\xff" * 80)
            session.put(CF_META, META_HINT_COUNT, (0).to_bytes(8, "little", signed=False))

    async def count_hints(self) -> int:
        async with self.db.reader_no_transaction() as view:
            raw = await view.get(CF_META, META_HINT_COUNT)
        if raw is None:
            return 0
        return int.from_bytes(raw, "little", signed=False)
