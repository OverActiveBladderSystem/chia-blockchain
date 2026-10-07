from __future__ import annotations

import logging
import time
from collections.abc import Collection

from chia_rs import CoinRecord, CoinState
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32, uint64

from chia.full_node.db.chain_db import ChainDB, WriteSession, _View
from chia.full_node.db.coin_codec import (
    StoredCoin,
    added_lookup_keys,
    decode_coin,
    decode_delta,
    delta_removals,
    encode_coin,
    encode_delta,
    index_entries,
    lookup_entries,
)
from chia.full_node.db.keys import (
    CF_COIN_DELTA,
    CF_COINS,
    CF_COINS_BY_CONFIRMED,
    CF_COINS_BY_PUZZLE_CONFIRMED,
    CF_COINS_BY_PUZZLE_SPENT,
    CF_COINS_BY_SPENT,
    CF_FF_UNSPENT,
    CF_HINTS_BY_HINT,
    CF_META,
    META_COIN_INDEXED,
    META_UNSPENT,
    prefix_end,
    u32_be,
    u32_from_be,
)
from chia.types.blockchain_format.coin import Coin
from chia.types.mempool_item import UnspentLineageInfo

log = logging.getLogger(__name__)

# One long-sync batch is at most 32 blocks, and each block journals at most one row.
# Indexing that many releases the chain writer before the next batch needs it.
COIN_INDEX_CHUNK = 32


class RocksCoinStore:
    """Current-peak UTXO set stored in the chain RocksDB."""

    def __init__(self, db: ChainDB) -> None:
        self.db = db

    @classmethod
    async def create(cls, db: ChainDB) -> RocksCoinStore:
        return cls(db)

    async def import_new_rows(self, rows: list[tuple[object, ...]]) -> None:
        """Insert coins that are known to be absent. One sorted write, with the WAL off.

        A crash during the coin snapshot recopies every coin, so this batch does not need
        to survive a power loss by itself. `rows` are the SQLite coin_record columns:
        name, confirmed, spent, coinbase, puzzle hash, parent, amount, timestamp.
        """
        if len(rows) == 0:
            return
        coins: list[tuple[bytes, bytes]] = []
        by_confirmed: list[tuple[bytes, bytes]] = []
        by_spent: list[tuple[bytes, bytes]] = []
        by_puzzle_confirmed: list[tuple[bytes, bytes]] = []
        by_puzzle_spent: list[tuple[bytes, bytes]] = []
        fast_forward: list[tuple[bytes, bytes]] = []
        unspent_delta = 0
        empty = b""
        for row in rows:
            name = bytes(row[0])  # type: ignore[arg-type]
            confirmed_index = int(row[1])  # type: ignore[arg-type]
            spent_index = int(row[2])  # type: ignore[arg-type]
            puzzle_hash = bytes(row[4])  # type: ignore[arg-type]
            parent = bytes(row[5])  # type: ignore[arg-type]
            amount = row[6]
            amount_int = int.from_bytes(amount, "big") if isinstance(amount, bytes) else int(amount)
            confirmed = u32_be(confirmed_index)
            value = (
                confirmed_index.to_bytes(4, "little", signed=False)
                + spent_index.to_bytes(8, "little", signed=True)
                + bytes([1 if row[3] else 0])
                + puzzle_hash
                + parent
                + amount_int.to_bytes(8, "big", signed=False)
                + int(row[7]).to_bytes(8, "little", signed=False)  # type: ignore[arg-type]
            )
            coins.append((name, value))
            by_confirmed.append((confirmed + name, empty))
            by_puzzle_confirmed.append((puzzle_hash + confirmed + name, empty))
            if spent_index > 0:
                spent = u32_be(spent_index)
                by_spent.append((spent + name, empty))
                by_puzzle_spent.append((puzzle_hash + spent + name, empty))
            elif spent_index == -1:
                fast_forward.append((puzzle_hash + name, empty))
            if spent_index <= 0:
                unspent_delta += 1
        coins.sort()
        by_confirmed.sort()
        by_spent.sort()
        by_puzzle_confirmed.sort()
        by_puzzle_spent.sort()
        fast_forward.sort()
        await self.db.write_column_batches(
            [
                (CF_COINS, coins),
                (CF_COINS_BY_CONFIRMED, by_confirmed),
                (CF_COINS_BY_SPENT, by_spent),
                (CF_COINS_BY_PUZZLE_CONFIRMED, by_puzzle_confirmed),
                (CF_COINS_BY_PUZZLE_SPENT, by_puzzle_spent),
                (CF_FF_UNSPENT, fast_forward),
            ],
            disable_wal=True,
        )
        if unspent_delta != 0:
            async with self.db.writer_maybe_transaction() as session:
                await self._add_unspent(session, unspent_delta)

    async def import_coins(self, records: list[StoredCoin], *, assume_new: bool = False) -> None:
        if len(records) == 0:
            return
        async with self.db.writer_maybe_transaction() as session:
            if assume_new:
                existing: dict[bytes, bytes] = {}
            else:
                existing = await session.get_many(CF_COINS, [bytes(record.coin_name) for record in records])
            staged: dict[bytes, StoredCoin] = {}
            unspent_delta = 0
            for record in records:
                key = bytes(record.coin_name)
                if key in staged:
                    previous: StoredCoin | None = staged[key]
                elif key in existing:
                    previous = decode_coin(record.coin_name, existing[key])
                else:
                    previous = None
                if previous is not None:
                    self._drop_indexes_now(session, previous)
                    if previous.spent_index <= 0:
                        unspent_delta -= 1
                session.put(CF_COINS, key, encode_coin(record))
                for family, index_key in index_entries(record):
                    if family == CF_COINS:
                        continue
                    session.put(family, index_key, b"")
                if record.spent_index <= 0:
                    unspent_delta += 1
                staged[key] = record
            if unspent_delta != 0:
                await self._add_unspent(session, unspent_delta)

    async def clear_all(self) -> None:
        async with self.db.writer_maybe_transaction() as session:
            for family in (
                CF_COINS,
                CF_COINS_BY_CONFIRMED,
                CF_COINS_BY_SPENT,
                CF_COINS_BY_PUZZLE_CONFIRMED,
                CF_COINS_BY_PUZZLE_SPENT,
                CF_COIN_DELTA,
                CF_FF_UNSPENT,
            ):
                session.delete_range(family, b"", b"\xff" * 80)
            session.put(CF_META, META_UNSPENT, (0).to_bytes(8, "little", signed=False))
            session.delete(CF_META, META_COIN_INDEXED)

    async def num_unspent(self) -> int:
        async with self.db.reader_no_transaction() as view:
            raw = await view.get(CF_META, META_UNSPENT)
        if raw is None:
            return 0
        return int.from_bytes(raw, "little", signed=False)

    async def new_block(
        self,
        height: uint32,
        timestamp: uint64,
        included_reward_coins: Collection[Coin],
        tx_additions: Collection[tuple[bytes32, Coin, bool]],
        tx_removals: list[bytes32],
        *,
        assume_additions_are_new: bool = False,
    ) -> None:
        started = time.monotonic()
        additions = [
            StoredCoin(
                coin_id,
                int(height),
                -1 if same_as_parent else 0,
                False,
                coin.puzzle_hash,
                coin.parent_coin_info,
                coin.amount,
                int(timestamp),
            )
            for coin_id, coin, same_as_parent in tx_additions
        ]
        additions.extend(
            StoredCoin(
                coin.name(),
                int(height),
                0,
                True,
                coin.puzzle_hash,
                coin.parent_coin_info,
                coin.amount,
                int(timestamp),
            )
            for coin in included_reward_coins
        )
        async with self.db.writer_maybe_transaction() as session:
            # Block acceptance has already rejected a repeated output and a repeated
            # spend. Probing the historical coin table for every new id is a miss
            # per coin and dominates db-write. Spends are few and still loaded.
            seen_additions: set[bytes] = set()
            for record in additions:
                name = bytes(record.coin_name)
                if name in seen_additions:
                    raise ValueError(f"Coin {record.coin_name.hex()} already exists")
                seen_additions.add(name)
            lookup_keys = [bytes(name) for name in tx_removals]
            if not assume_additions_are_new:
                lookup_keys.extend(seen_additions)
            existing = await session.get_many(CF_COINS, lookup_keys) if lookup_keys else {}
            counter_raw = await session.get(CF_META, META_UNSPENT)
            counter = 0 if counter_raw is None else int.from_bytes(counter_raw, "little", signed=False)
            created: list[StoredCoin] = []
            removed: list[tuple[bytes32, int]] = []
            for record in additions:
                name = bytes(record.coin_name)
                if not assume_additions_are_new and existing.get(name) is not None:
                    raise ValueError(f"Coin {record.coin_name.hex()} already exists")
                encoded = encode_coin(record)
                session.put(CF_COINS, name, encoded)
                counter += 1
                created.append(record)
                existing[name] = encoded
            if tx_removals:
                if height <= 0:
                    raise ValueError(f"spent height must be positive, got {int(height)}")
                changed = 0
                for name in tx_removals:
                    previous_raw = existing.get(bytes(name))
                    if previous_raw is None:
                        continue
                    previous = decode_coin(name, previous_raw)
                    if previous.spent_index > 0:
                        continue
                    spent = StoredCoin(
                        previous.coin_name,
                        previous.confirmed_index,
                        int(height),
                        previous.coinbase,
                        previous.puzzle_hash,
                        previous.parent,
                        previous.amount,
                        previous.timestamp,
                    )
                    session.put(CF_COINS, bytes(name), encode_coin(spent))
                    counter -= 1
                    removed.append((name, previous.spent_index))
                    changed += 1
                if changed != len(tx_removals):
                    raise ValueError(
                        f"Invalid operation to set spent, total updates {changed} expected {len(tx_removals)}"
                    )
            if counter < 0:
                raise ValueError(f"unspent counter went negative ({counter})")
            session.put(CF_META, META_UNSPENT, counter.to_bytes(8, "little", signed=False))
            # The coin records commit with the block. Puzzle and parent lookup keys are
            # written later by index_pending so this batch stays one key per coin.
            if created or removed:
                await self._merge_delta(session, int(height), created, removed)
        elapsed = time.monotonic() - started
        message = (
            f"Height {height}: It took {elapsed:0.2f}s to apply {len(tx_additions)} additions and "
            f"{len(tx_removals)} removals to the coin store."
        )
        log.log(logging.WARNING if elapsed > 10 else logging.DEBUG, message)

    def _drop_indexes_now(self, session: WriteSession, record: StoredCoin) -> None:
        for family, key in index_entries(record):
            if family == CF_COINS:
                continue
            session.delete(family, key)

    async def _insert(self, session: WriteSession, record: StoredCoin) -> None:
        previous = await self._load(session, record.coin_name)
        if previous is not None:
            await self._drop_indexes(session, previous)
            if previous.spent_index <= 0:
                await self._add_unspent(session, -1)
        session.put(CF_COINS, bytes(record.coin_name), encode_coin(record))
        for cf, key in index_entries(record):
            if cf == CF_COINS:
                continue
            session.put(cf, key, b"")
        if record.spent_index <= 0:
            await self._add_unspent(session, 1)

    async def _load(self, session: WriteSession, coin_name: bytes32) -> StoredCoin | None:
        raw = await session.get(CF_COINS, bytes(coin_name))
        if raw is None:
            return None
        return decode_coin(coin_name, raw)

    async def _load_many(self, view: _View, names: list[bytes32]) -> dict[bytes32, StoredCoin]:
        if len(names) == 0:
            return {}
        found = await view.get_many(CF_COINS, [bytes(name) for name in names])
        loaded: dict[bytes32, StoredCoin] = {}
        for name in names:
            raw = found.get(bytes(name))
            if raw is not None:
                loaded[name] = decode_coin(name, raw)
        return loaded

    async def _drop_indexes(self, session: WriteSession, record: StoredCoin) -> None:
        for cf, key in index_entries(record):
            if cf == CF_COINS:
                continue
            session.delete(cf, key)

    async def _add_unspent(self, session: WriteSession, delta: int) -> None:
        raw = await session.get(CF_META, META_UNSPENT)
        current = 0 if raw is None else int.from_bytes(raw, "little", signed=False)
        updated = current + delta
        if updated < 0:
            raise ValueError(f"unspent counter went negative ({updated})")
        session.put(CF_META, META_UNSPENT, updated.to_bytes(8, "little", signed=False))

    async def _merge_delta(
        self,
        session: WriteSession,
        height: int,
        added: list[StoredCoin],
        removed: list[tuple[bytes32, int]],
    ) -> None:
        key = u32_be(height)
        existing = await session.get(CF_COIN_DELTA, key)
        if existing is None:
            session.put(CF_COIN_DELTA, key, encode_delta(added, removed))
            if await session.get(CF_META, META_COIN_INDEXED) is None:
                session.put(CF_META, META_COIN_INDEXED, int(height - 1).to_bytes(4, "big", signed=True))
            return
        old_added, old_removed = decode_delta(existing)
        session.put(CF_COIN_DELTA, key, encode_delta([*old_added, *added], [*old_removed, *removed]))

    def _put_lookup(self, session: WriteSession, record: StoredCoin) -> None:
        for family, key in lookup_entries(record):
            session.put(family, key, b"")

    def _drop_lookup(self, session: WriteSession, record: StoredCoin) -> None:
        for family, key in lookup_entries(record):
            session.delete(family, key)

    async def index_pending(self, *, limit: int | None = None) -> bool:
        """Write puzzle, parent, and fast-forward lookup keys for one chunk of journals.

        The block commit already stored the coin records. This second write is what wallet
        peers search. One call commits at most one chunk and releases the chain writer.
        Returns True when journals are still waiting.
        """
        chunk = COIN_INDEX_CHUNK if limit is None else limit
        if chunk <= 0:
            raise ValueError(f"coin index chunk must be positive, got {chunk}")
        started = time.monotonic()
        more = False
        added_count = 0
        last = 0
        async with self.db.writer() as session:
            raw = await session.get(CF_META, META_COIN_INDEXED)
            if raw is None:
                return False
            indexed = int.from_bytes(raw, "big", signed=True)
            start = b"\x00\x00\x00\x00" if indexed < 0 else u32_be(indexed + 1)
            # One extra row tells us whether another journal follows, without reading the tail.
            rows = await session.scan(CF_COIN_DELTA, start, None, limit=chunk + 1)
            if len(rows) == 0:
                return False
            more = len(rows) > chunk
            if more:
                rows = rows[:chunk]
            # Load every spend in one read before queuing lookup keys. A coin read scans
            # every key already queued in this session, and those keys are not coin records.
            # Fast-forward deletes are applied after the new keys so a coin created and
            # spent inside this batch still drops its unspent key.
            removals: list[tuple[bytes32, int]] = []
            for _key, value in rows:
                removals.extend(delta_removals(value))
            removal_names = [bytes(name) for name, _previous in removals]
            raw_by_name = await session.get_many(CF_COINS, removal_names) if removal_names else {}
            spent_records = [(name, previous, raw_by_name.get(bytes(name))) for name, previous in removals]
            last = indexed
            for key, value in rows:
                added_here, lookup_keys = added_lookup_keys(value)
                for family, lookup_key in lookup_keys:
                    session.put(family, lookup_key, b"")
                added_count += added_here
                last = u32_from_be(key)
            self._apply_removal_lookups(session, spent_records)
            session.put(CF_META, META_COIN_INDEXED, int(last).to_bytes(4, "big", signed=True))
        elapsed = time.monotonic() - started
        if more or elapsed >= 0.2 or added_count >= 1000:
            log.info(
                "coin lookups indexed through height %s in %.2fs (%s coins)",
                last,
                elapsed,
                added_count,
            )
        return more

    async def index_pending_all(self) -> None:
        """Index every waiting journal, releasing the chain writer between chunks."""
        while await self.index_pending():
            pass

    def _apply_removal_lookups(
        self,
        session: WriteSession,
        spent_records: list[tuple[bytes32, int, bytes | None]],
    ) -> None:
        for name, previous, raw in spent_records:
            if raw is None:
                continue
            current = decode_coin(name, raw)
            if previous == -1:
                session.delete(CF_FF_UNSPENT, bytes(current.puzzle_hash) + bytes(name))
            if current.spent_index > 0:
                spent_key = bytes(current.puzzle_hash) + u32_be(current.spent_index) + bytes(name)
                session.put(CF_COINS_BY_PUZZLE_SPENT, spent_key, b"")

    async def _set_spent(self, coin_names: list[bytes32], index: int) -> None:
        """Mark coins spent at `index`. Opens its own write, matching the SQLite coin store."""
        if len(coin_names) == 0:
            return
        if index <= 0:
            raise ValueError(f"spent height must be positive, got {index}")
        async with self.db.writer_maybe_transaction() as session:
            await self._set_spent_in_session(session, coin_names, index)

    async def _set_spent_in_session(self, session: WriteSession, coin_names: list[bytes32], index: int) -> None:
        updated = 0
        removed: list[tuple[bytes32, int]] = []
        for name in coin_names:
            record = await self._load(session, name)
            if record is None or record.spent_index > 0:
                continue
            spent = StoredCoin(
                record.coin_name,
                record.confirmed_index,
                index,
                record.coinbase,
                record.puzzle_hash,
                record.parent,
                record.amount,
                record.timestamp,
            )
            session.put(CF_COINS, bytes(name), encode_coin(spent))
            await self._add_unspent(session, -1)
            removed.append((name, record.spent_index))
            updated += 1
        if updated != len(coin_names):
            raise ValueError(f"Invalid operation to set spent, total updates {updated} expected {len(coin_names)}")
        if removed:
            await self._merge_delta(session, index, [], removed)

    async def get_coin_record(self, coin_name: bytes32) -> CoinRecord | None:
        async with self.db.reader_no_transaction() as view:
            raw = await view.get(CF_COINS, bytes(coin_name))
        if raw is None:
            return None
        return _to_record(decode_coin(coin_name, raw))

    async def get_coin_records_by_names(
        self,
        include_spent_coins: bool,
        names: list[bytes32],
        start_height: uint32 = uint32(0),
        end_height: uint32 = uint32((2**32) - 1),
    ) -> list[CoinRecord]:
        if len(names) == 0:
            return []
        found: list[CoinRecord] = []
        async with self.db.reader_no_transaction() as view:
            for name in names:
                raw = await view.get(CF_COINS, bytes(name))
                if raw is None:
                    continue
                stored = decode_coin(name, raw)
                if stored.confirmed_index < int(start_height) or stored.confirmed_index >= int(end_height):
                    continue
                if not include_spent_coins and stored.spent_index > 0:
                    continue
                found.append(_to_record(stored))
        return found

    async def get_coin_records(self, names: Collection[bytes32]) -> list[CoinRecord]:
        # One read for the whole request. Keep the asked-for order, repeat a name
        # that was asked for twice, and leave out a name that is not stored.
        ordered = list(names)
        if len(ordered) == 0:
            return []
        async with self.db.reader_no_transaction() as view:
            loaded = await self._load_many(view, ordered)
        return [_to_record(loaded[name]) for name in ordered if name in loaded]

    async def get_coins_added_at_height(self, height: uint32) -> list[CoinRecord]:
        async with self.db.reader_no_transaction() as view:
            stored = await self._stored_coins_added_at_height(view, int(height))
        return [_to_record(record) for record in stored]

    async def _stored_coins_added_at_height(self, view: _View, height: int) -> list[StoredCoin]:
        raw = await view.get(CF_COIN_DELTA, u32_be(height))
        if raw is not None:
            added, _removed = decode_delta(raw)
            loaded = await self._load_many(view, [created.coin_name for created in added])
            return [loaded[created.coin_name] for created in added if created.coin_name in loaded]
        prefix = u32_be(height)
        rows = await view.scan(CF_COINS_BY_CONFIRMED, prefix, prefix_end(prefix))
        names = [bytes32(key[-32:]) for key, _value in rows]
        loaded = await self._load_many(view, names)
        return [loaded[name] for name in names if name in loaded]

    async def get_coins_removed_at_height(self, height: uint32) -> list[CoinRecord]:
        if height == 0:
            return []
        async with self.db.reader_no_transaction() as view:
            raw = await view.get(CF_COIN_DELTA, u32_be(int(height)))
            if raw is not None:
                _added, removed = decode_delta(raw)
                loaded = await self._load_many(view, [name for name, _previous in removed])
                return [_to_record(loaded[name]) for name, _previous in removed if name in loaded]
        return await self._records_by_height_index(CF_COINS_BY_SPENT, int(height))

    async def _records_by_height_index(self, cf: str, height: int) -> list[CoinRecord]:
        prefix = u32_be(height)
        end = prefix_end(prefix)
        async with self.db.reader_no_transaction() as view:
            rows = await view.scan(cf, prefix, end)
            names = [bytes32(key[-32:]) for key, _value in rows]
            loaded = await self._load_many(view, names)
        return [_to_record(loaded[name]) for name in names if name in loaded]

    async def rollback_to_block(self, block_index: int) -> dict[bytes32, CoinRecord]:
        changes: dict[bytes32, CoinRecord] = {}
        async with self.db.writer_maybe_transaction() as session:
            handled: set[bytes32] = set()
            # Restored records stay in memory so undoing the block that created
            # them does not read each coin back from the table.
            restored_records: dict[bytes32, StoredCoin] = {}
            unspent_delta = 0
            start = b"\x00\x00\x00\x00" if block_index < 0 else u32_be(block_index + 1)
            journals = await session.scan(CF_COIN_DELTA, start, None)
            for key, value in reversed(journals):
                added_records, removed = decode_delta(value)
                missing = [name for name, _previous in removed if name not in restored_records and name not in handled]
                loaded = await self._load_many(session, missing)
                for name, previous in removed:
                    record = restored_records.get(name)
                    if record is None:
                        record = loaded.get(name)
                    if record is None or name in handled:
                        continue
                    self._drop_lookup(session, record)
                    restored = StoredCoin(
                        record.coin_name,
                        record.confirmed_index,
                        previous,
                        record.coinbase,
                        record.puzzle_hash,
                        record.parent,
                        record.amount,
                        record.timestamp,
                    )
                    session.put(CF_COINS, bytes(name), encode_coin(restored))
                    self._put_lookup(session, restored)
                    if record.spent_index > 0 and previous <= 0:
                        unspent_delta += 1
                    restored_records[name] = restored
                    changes[name] = CoinRecord(
                        record.coin(),
                        uint32(record.confirmed_index),
                        uint32(0),
                        record.coinbase,
                        uint64(record.timestamp),
                    )
                for created in added_records:
                    record = restored_records.pop(created.coin_name, created)
                    self._drop_lookup(session, record)
                    session.delete(CF_COINS, bytes(created.coin_name))
                    if record.spent_index <= 0:
                        unspent_delta -= 1
                    handled.add(created.coin_name)
                    changes[created.coin_name] = CoinRecord(
                        record.coin(),
                        uint32(0),
                        uint32(0) if record.spent_index <= 0 else uint32(record.spent_index),
                        record.coinbase,
                        uint64(0),
                    )
                session.delete(CF_COIN_DELTA, key)
            if unspent_delta != 0:
                await self._add_unspent(session, unspent_delta)
            indexed_raw = await session.get(CF_META, META_COIN_INDEXED)
            if indexed_raw is not None:
                indexed = int.from_bytes(indexed_raw, "big", signed=True)
                if indexed > block_index:
                    session.put(CF_META, META_COIN_INDEXED, int(block_index).to_bytes(4, "big", signed=True))

            confirmed_start = b"\x00\x00\x00\x00" if block_index < 0 else u32_be(block_index + 1)
            added = await session.scan(CF_COINS_BY_CONFIRMED, confirmed_start, None)
            for key, _value in added:
                name = bytes32(key[-32:])
                if name in handled:
                    continue
                record = await self._load(session, name)
                if record is None:
                    continue
                changes[name] = CoinRecord(
                    record.coin(),
                    uint32(0),
                    uint32(0) if record.spent_index <= 0 else uint32(record.spent_index),
                    record.coinbase,
                    uint64(0),
                )
                await self._delete_record(session, record)

            spent_start = b"\x00\x00\x00\x00" if block_index < 0 else u32_be(block_index + 1)
            spent = await session.scan(CF_COINS_BY_SPENT, spent_start, None)
            for key, _value in spent:
                name = bytes32(key[-32:])
                if name in changes or name in handled:
                    continue
                record = await self._load(session, name)
                if record is None or record.spent_index <= block_index:
                    continue
                restored_spent = await self._unspent_sentinel(session, record)
                restored = StoredCoin(
                    record.coin_name,
                    record.confirmed_index,
                    restored_spent,
                    record.coinbase,
                    record.puzzle_hash,
                    record.parent,
                    record.amount,
                    record.timestamp,
                )
                await self._drop_indexes(session, record)
                session.put(CF_COINS, bytes(name), encode_coin(restored))
                for cf, index_key in index_entries(restored):
                    if cf != CF_COINS:
                        session.put(cf, index_key, b"")
                await self._add_unspent(session, 1)
                changes[name] = CoinRecord(
                    record.coin(),
                    uint32(record.confirmed_index),
                    uint32(0),
                    record.coinbase,
                    uint64(record.timestamp),
                )
        return changes

    async def _unspent_sentinel(self, session: WriteSession, record: StoredCoin) -> int:
        if record.coinbase:
            return 0
        parent = await self._load(session, record.parent)
        if (
            parent is not None
            and parent.spent_index > 0
            and parent.puzzle_hash == record.puzzle_hash
            and parent.amount == record.amount
        ):
            return -1
        return 0

    async def _delete_record(self, session: WriteSession, record: StoredCoin) -> None:
        await self._drop_indexes(session, record)
        session.delete(CF_COINS, bytes(record.coin_name))
        if record.spent_index <= 0:
            await self._add_unspent(session, -1)

    async def _pending_journals(self, view: _View) -> list[tuple[list[StoredCoin], list[tuple[bytes32, int]]]]:
        raw = await view.get(CF_META, META_COIN_INDEXED)
        if raw is None:
            return []
        indexed = int.from_bytes(raw, "big", signed=True)
        start = b"\x00\x00\x00\x00" if indexed < 0 else u32_be(indexed + 1)
        rows = await view.scan(CF_COIN_DELTA, start, None)
        return [decode_delta(value) for _key, value in rows]

    async def _pending_coins(
        self,
        view: _View,
        journals: list[tuple[list[StoredCoin], list[tuple[bytes32, int]]]],
        *,
        puzzle_hash: bytes32 | None = None,
        parent: bytes32 | None = None,
        include_removals: bool = False,
    ) -> list[StoredCoin]:
        """Current records for coins whose lookup keys are not written yet."""
        ordered: list[bytes32] = []
        seen: set[bytes32] = set()
        for added, removed in journals:
            for record in added:
                if puzzle_hash is not None and record.puzzle_hash != puzzle_hash:
                    continue
                if parent is not None and record.parent != parent:
                    continue
                if record.coin_name in seen:
                    continue
                seen.add(record.coin_name)
                ordered.append(record.coin_name)
            if not include_removals or parent is not None:
                continue
            for name, _previous in removed:
                if name in seen:
                    continue
                seen.add(name)
                ordered.append(name)
        found: list[StoredCoin] = []
        for name in ordered:
            raw = await view.get(CF_COINS, bytes(name))
            if raw is None:
                continue
            stored = decode_coin(name, raw)
            if puzzle_hash is not None and stored.puzzle_hash != puzzle_hash:
                continue
            if parent is not None and stored.parent != parent:
                continue
            found.append(stored)
        return found

    async def get_coin_records_by_puzzle_hash(
        self,
        include_spent_coins: bool,
        puzzle_hash: bytes32,
        start_height: uint32 = uint32(0),
        end_height: uint32 = uint32((2**32) - 1),
    ) -> list[CoinRecord]:
        return await self.get_coin_records_by_puzzle_hashes(
            include_spent_coins, [puzzle_hash], start_height, end_height
        )

    async def get_coin_records_by_puzzle_hashes(
        self,
        include_spent_coins: bool,
        puzzle_hashes: list[bytes32],
        start_height: uint32 = uint32(0),
        end_height: uint32 = uint32((2**32) - 1),
    ) -> list[CoinRecord]:
        found: dict[bytes32, CoinRecord] = {}
        async with self.db.reader_no_transaction() as view:
            journals = await self._pending_journals(view)
            for puzzle_hash in puzzle_hashes:
                start = bytes(puzzle_hash) + u32_be(int(start_height))
                end = bytes(puzzle_hash) + u32_be(int(end_height))
                rows = await view.scan(CF_COINS_BY_PUZZLE_CONFIRMED, start, end)
                for key, _value in rows:
                    name = bytes32(key[-32:])
                    raw = await view.get(CF_COINS, bytes(name))
                    if raw is None:
                        continue
                    stored = decode_coin(name, raw)
                    if not include_spent_coins and stored.spent_index > 0:
                        continue
                    found[name] = _to_record(stored)
                pending = await self._pending_coins(view, journals, puzzle_hash=puzzle_hash)
                for stored in pending:
                    if stored.confirmed_index < int(start_height) or stored.confirmed_index >= int(end_height):
                        continue
                    if not include_spent_coins and stored.spent_index > 0:
                        continue
                    found[stored.coin_name] = _to_record(stored)
        return list(found.values())

    async def get_coin_records_by_parent_ids(
        self,
        include_spent_coins: bool,
        parent_ids: list[bytes32],
        start_height: uint32 = uint32(0),
        end_height: uint32 = uint32((2**32) - 1),
        *,
        max_items: int = 50000,
    ) -> list[CoinRecord]:
        if max_items <= 0 or len(parent_ids) == 0:
            return []
        # A child is created in the block that spends its parent. The parent record
        # has that height, and the coins added there already carry the parent id.
        found: list[CoinRecord] = []
        seen: set[bytes32] = set()
        added_at: dict[int, list[StoredCoin]] = {}
        async with self.db.reader_no_transaction() as view:
            for parent_id in parent_ids:
                raw = await view.get(CF_COINS, bytes(parent_id))
                if raw is None:
                    continue
                parent = decode_coin(parent_id, raw)
                if parent.spent_index <= 0:
                    continue
                height = parent.spent_index
                if height not in added_at:
                    added_at[height] = await self._stored_coins_added_at_height(view, height)
                matches = [stored for stored in added_at[height] if stored.parent == parent_id]
                matches.sort(key=lambda stored: (stored.confirmed_index, bytes(stored.coin_name)))
                for stored in matches:
                    if stored.coin_name in seen:
                        continue
                    if stored.confirmed_index < int(start_height) or stored.confirmed_index >= int(end_height):
                        continue
                    if not include_spent_coins and stored.spent_index > 0:
                        continue
                    seen.add(stored.coin_name)
                    found.append(_to_record(stored))
                    if len(found) >= max_items:
                        return found
        return found

    async def get_coin_states_by_puzzle_hashes(
        self,
        include_spent_coins: bool,
        puzzle_hashes: set[bytes32],
        min_height: uint32 = uint32(0),
        *,
        max_items: int = 50000,
    ) -> set[CoinState]:
        if max_items <= 0 or len(puzzle_hashes) == 0:
            return set()
        states: set[CoinState] = set()
        async with self.db.reader_no_transaction() as view:
            journals = await self._pending_journals(view)
            for puzzle_hash in puzzle_hashes:
                await self._collect_puzzle_states(
                    view, puzzle_hash, int(min_height), include_spent_coins, states, max_items, journals
                )
                if len(states) >= max_items:
                    break
        return states

    async def _collect_puzzle_states(
        self,
        view: _View,
        puzzle_hash: bytes32,
        min_height: int,
        include_spent_coins: bool,
        states: set[CoinState],
        max_items: int,
        journals: list[tuple[list[StoredCoin], list[tuple[bytes32, int]]]] | None = None,
    ) -> None:
        start = bytes(puzzle_hash) + u32_be(min_height)
        end = prefix_end(bytes(puzzle_hash))
        rows = await view.scan(CF_COINS_BY_PUZZLE_CONFIRMED, start, end)
        if include_spent_coins:
            rows.extend(await view.scan(CF_COINS_BY_PUZZLE_SPENT, start, end))
        seen: set[bytes32] = set()
        for key, _value in rows:
            name = bytes32(key[-32:])
            if name in seen:
                continue
            seen.add(name)
            raw = await view.get(CF_COINS, bytes(name))
            if raw is None:
                continue
            stored = decode_coin(name, raw)
            if not self._state_visible(stored, min_height, include_spent_coins):
                continue
            states.add(_to_state(stored))
            if len(states) >= max_items:
                return
        if not journals:
            return
        pending = await self._pending_coins(
            view, journals, puzzle_hash=puzzle_hash, include_removals=include_spent_coins
        )
        for stored in pending:
            if stored.coin_name in seen:
                continue
            if not self._state_visible(stored, min_height, include_spent_coins):
                continue
            states.add(_to_state(stored))
            if len(states) >= max_items:
                return

    async def get_coin_states_by_ids(
        self,
        include_spent_coins: bool,
        coin_ids: Collection[bytes32],
        min_height: uint32 = uint32(0),
        *,
        max_height: uint32 = uint32.MAXIMUM,
        max_items: int = 50000,
    ) -> list[CoinState]:
        if max_items <= 0 or len(coin_ids) == 0:
            return []
        states: list[CoinState] = []
        limit_height = max_height != uint32.MAXIMUM
        async with self.db.reader_no_transaction() as view:
            for coin_id in coin_ids:
                raw = await view.get(CF_COINS, bytes(coin_id))
                if raw is None:
                    continue
                stored = decode_coin(coin_id, raw)
                if stored.confirmed_index < int(min_height) and stored.spent_index < int(min_height):
                    continue
                if limit_height and (stored.confirmed_index > int(max_height) or stored.spent_index > int(max_height)):
                    continue
                if not include_spent_coins and stored.spent_index > 0:
                    continue
                states.append(_to_state(stored))
                if len(states) >= max_items:
                    break
        return states

    async def get_unspent_lineage_info_for_puzzle_hash(self, puzzle_hash: bytes32) -> UnspentLineageInfo | None:
        end = prefix_end(bytes(puzzle_hash))
        async with self.db.reader_no_transaction() as view:
            rows = await view.scan(CF_FF_UNSPENT, bytes(puzzle_hash), end)
            matches: dict[bytes32, UnspentLineageInfo] = {}
            for key, _value in rows:
                name = bytes32(key[-32:])
                raw = await view.get(CF_COINS, bytes(name))
                if raw is None:
                    continue
                unspent = decode_coin(name, raw)
                if unspent.spent_index != -1 or unspent.puzzle_hash != puzzle_hash:
                    continue
                parent_raw = await view.get(CF_COINS, bytes(unspent.parent))
                if parent_raw is None:
                    continue
                parent = decode_coin(unspent.parent, parent_raw)
                if (
                    parent.spent_index > 0
                    and parent.puzzle_hash == unspent.puzzle_hash
                    and parent.amount == unspent.amount
                ):
                    matches[unspent.coin_name] = UnspentLineageInfo(
                        coin_id=unspent.coin_name,
                        parent_id=unspent.parent,
                        parent_parent_id=parent.parent,
                    )
            journals = await self._pending_journals(view)
            pending = await self._pending_coins(view, journals, puzzle_hash=puzzle_hash)
            for unspent in pending:
                if unspent.spent_index != -1 or unspent.coin_name in matches:
                    continue
                parent_raw = await view.get(CF_COINS, bytes(unspent.parent))
                if parent_raw is None:
                    continue
                parent = decode_coin(unspent.parent, parent_raw)
                if (
                    parent.spent_index > 0
                    and parent.puzzle_hash == unspent.puzzle_hash
                    and parent.amount == unspent.amount
                ):
                    matches[unspent.coin_name] = UnspentLineageInfo(
                        coin_id=unspent.coin_name,
                        parent_id=unspent.parent,
                        parent_parent_id=parent.parent,
                    )
        found = list(matches.values())
        if len(found) != 1:
            log.debug("Expected 1 unspent with puzzle hash %s, but found %s", puzzle_hash.hex(), len(found))
            return None
        return found[0]

    @staticmethod
    def _state_visible(stored: StoredCoin, min_height: int, include_spent_coins: bool) -> bool:
        if stored.confirmed_index < min_height and stored.spent_index < min_height:
            return False
        return include_spent_coins or stored.spent_index <= 0

    async def is_empty(self) -> bool:
        async with self.db.reader_no_transaction() as view:
            rows = await view.scan(CF_COINS, b"", None)
        return len(rows) == 0

    async def batch_coin_states_by_puzzle_hashes(
        self,
        puzzle_hashes: list[bytes32],
        *,
        min_height: uint32 = uint32(0),
        include_spent: bool = True,
        include_unspent: bool = True,
        include_hinted: bool = True,
        min_amount: uint64 = uint64(0),
        max_items: int = 50000,
    ) -> tuple[list[CoinState], uint32 | None]:
        if len(puzzle_hashes) == 0 or (not include_spent and not include_unspent):
            return [], None
        by_name: dict[bytes32, CoinState] = {}
        async with self.db.reader_no_transaction() as view:
            journals = await self._pending_journals(view)
            for puzzle_hash in puzzle_hashes:
                await _collect_ordered(
                    view,
                    puzzle_hash,
                    int(min_height),
                    include_spent,
                    include_unspent,
                    int(min_amount),
                    by_name,
                )
                pending = await self._pending_coins(
                    view, journals, puzzle_hash=puzzle_hash, include_removals=include_spent
                )
                for stored in pending:
                    if stored.coin_name in by_name:
                        continue
                    if not _batch_accepts(stored, int(min_height), include_spent, include_unspent, int(min_amount)):
                        continue
                    by_name[stored.coin_name] = _to_state(stored)
            if include_hinted:
                for puzzle_hash in puzzle_hashes:
                    hint_end = prefix_end(bytes(puzzle_hash))
                    hint_rows = await view.scan(CF_HINTS_BY_HINT, bytes(puzzle_hash), hint_end)
                    for key, _value in hint_rows:
                        name = bytes32(key[-32:])
                        if name in by_name:
                            continue
                        raw = await view.get(CF_COINS, bytes(name))
                        if raw is None:
                            continue
                        stored = decode_coin(name, raw)
                        if not _batch_accepts(stored, int(min_height), include_spent, include_unspent, int(min_amount)):
                            continue
                        by_name[name] = _to_state(stored)
        ordered = sorted(by_name.values(), key=_order_height)
        return _paginate(ordered, max_items)


def _order_height(state: CoinState) -> int:
    created = int(state.created_height or 0)
    spent = int(state.spent_height or 0)
    return max(created, spent)


def _batch_accepts(
    stored: StoredCoin, min_height: int, include_spent: bool, include_unspent: bool, min_amount: int
) -> bool:
    if stored.amount < min_amount:
        return False
    spent = stored.spent_index > 0
    if spent and not include_spent:
        return False
    if not spent and not include_unspent:
        return False
    if stored.confirmed_index < min_height and stored.spent_index < min_height:
        return False
    return True


async def _collect_ordered(
    view: _View,
    puzzle_hash: bytes32,
    min_height: int,
    include_spent: bool,
    include_unspent: bool,
    min_amount: int,
    by_name: dict[bytes32, CoinState],
) -> None:
    end = prefix_end(bytes(puzzle_hash))
    start = bytes(puzzle_hash) + u32_be(0)
    rows = await view.scan(CF_COINS_BY_PUZZLE_CONFIRMED, start, end)
    for key, _value in rows:
        name = bytes32(key[-32:])
        raw = await view.get(CF_COINS, bytes(name))
        if raw is None:
            continue
        stored = decode_coin(name, raw)
        if not _batch_accepts(stored, min_height, include_spent, include_unspent, min_amount):
            continue
        by_name[name] = _to_state(stored)


def _paginate(ordered: list[CoinState], max_items: int) -> tuple[list[CoinState], uint32 | None]:
    if len(ordered) <= max_items:
        return ordered, None
    ordered = ordered[: max_items + 1]
    next_state = ordered.pop()
    next_height = uint32(_order_height(next_state))
    while ordered:
        if uint32(_order_height(ordered[-1])) != next_height:
            break
        ordered.pop()
    return ordered, next_height


def _to_record(stored: StoredCoin) -> CoinRecord:
    spent = uint32(0) if stored.spent_index <= 0 else uint32(stored.spent_index)
    return CoinRecord(stored.coin(), uint32(stored.confirmed_index), spent, stored.coinbase, uint64(stored.timestamp))


def _to_state(stored: StoredCoin) -> CoinState:
    spent_h = None if stored.spent_index <= 0 else uint32(stored.spent_index)
    return CoinState(stored.coin(), spent_h, uint32(stored.confirmed_index))
