from __future__ import annotations

import logging
from collections.abc import Collection

from chia_rs.sized_bytes import bytes32

from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.keys import (
    CF_HINTS_BY_COIN,
    CF_HINTS_BY_HINT,
    CF_META,
    META_FORMAT,
    META_FORMAT_VALUE,
    META_HINT_COUNT,
    META_SCHEMA_VERSION,
    prefix_end,
)

log = logging.getLogger(__name__)


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
        coin_ids: list[bytes32] = []
        async with self.db.reader_no_transaction() as view:
            for hint in hints:
                rows = await view.scan(CF_HINTS_BY_HINT, hint, prefix_end(hint))
                for key, _value in rows:
                    if len(key) != len(hint) + 32:
                        continue
                    coin_ids.append(bytes32(key[-32:]))
                    if len(coin_ids) >= max_items:
                        return coin_ids
        return coin_ids

    async def get_coin_ids_by_hints(self, hints: Collection[bytes]) -> set[bytes32]:
        if len(hints) == 0:
            return set()
        found: set[bytes32] = set()
        async with self.db.reader_no_transaction() as view:
            for hint in hints:
                rows = await view.scan(CF_HINTS_BY_HINT, hint, prefix_end(hint))
                for key, _value in rows:
                    if len(key) != len(hint) + 32:
                        continue
                    found.add(bytes32(key[-32:]))
        return found

    async def get_hints(self, coin_ids: list[bytes32]) -> list[bytes32]:
        hints: list[bytes32] = []
        async with self.db.reader_no_transaction() as view:
            for coin_id in coin_ids:
                prefix = bytes(coin_id)
                rows = await view.scan(CF_HINTS_BY_COIN, prefix, prefix_end(prefix))
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
