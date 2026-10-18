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
    INDEX_SCAN_CHUNK,
    META_COIN_INDEXED,
    META_UNSPENT,
    key_after,
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


# A reorg can commit between two reads. Retry until the journal and the coin rows agree.
_HEIGHT_READ_ATTEMPTS = 4


class RocksCoinStore:
    """Current-peak UTXO set stored in the chain RocksDB."""

    def __init__(self, db: ChainDB, reward_parent_prefixes: tuple[bytes, bytes] | None = None) -> None:
        self.db = db
        # Farmer and pool reward parents are not coins. Each prefix is 16 bytes of
        # the genesis challenge, and the other 16 bytes are the block height.
        self._reward_parent_prefixes = reward_parent_prefixes

    @classmethod
    async def create(
        cls, db: ChainDB, reward_parent_prefixes: tuple[bytes, bytes] | None = None
    ) -> RocksCoinStore:
        return cls(db, reward_parent_prefixes)

    def _reward_parent_height(self, parent_id: bytes32) -> int | None:
        prefixes = self._reward_parent_prefixes
        if prefixes is None:
            return None
        raw = bytes(parent_id)
        if raw[:16] not in prefixes:
            return None
        height = int.from_bytes(raw[16:], "big")
        if height > 0xFFFFFFFF:
            return None
        return height

    async def import_new_rows(self, rows: list[tuple[object, ...]]) -> None:
        """Insert coins that are known to be absent. One sorted write, with the WAL off.

        A crash while the copy is still in the coin step recopies every coin, so this
        batch does not need its own log. The copy flushes these families before it
        records that the coin step is finished. `rows` are the SQLite coin_record columns:
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
        async with self.db.reader_no_transaction() as view:
            loaded = await self._load_many(view, names)
        found: list[CoinRecord] = []
        for name in names:
            stored = loaded.get(name)
            if stored is None:
                continue
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
        added, _removed = await self._stable_height_coins(int(height))
        return [_to_record(record) for record in added]

    async def get_coins_removed_at_height(self, height: uint32) -> list[CoinRecord]:
        if height == 0:
            return []
        _added, removed = await self._stable_height_coins(int(height))
        return [_to_record(record) for record in removed]

    async def get_additions_and_removals_at_height(
        self, height: uint32
    ) -> tuple[list[CoinRecord], list[CoinRecord]]:
        added, removed = await self._stable_height_coins(int(height))
        if height == 0:
            removed = []
        return [_to_record(record) for record in added], [_to_record(record) for record in removed]

    async def _stable_height_coins(self, height: int) -> tuple[list[StoredCoin], list[StoredCoin]]:
        """Additions and removals at one height, from one journal value.

        The journal and the coin rows are separate reads. A reorg can land between
        them and make every old name miss, which is an empty list that was never
        the chain. The read is repeated until both sides show the same journal.
        """
        key = u32_be(height)
        async with self.db.reader_no_transaction() as view:
            for _attempt in range(_HEIGHT_READ_ATTEMPTS):
                raw = await view.get(CF_COIN_DELTA, key)
                if raw is not None:
                    added_records, removed = decode_delta(raw)
                    names = [record.coin_name for record in added_records]
                    names.extend(name for name, _previous in removed)
                    loaded = await self._load_many(view, names)
                    if await view.get(CF_COIN_DELTA, key) != raw:
                        continue
                    return (
                        [loaded[record.coin_name] for record in added_records if record.coin_name in loaded],
                        [loaded[name] for name, _previous in removed if name in loaded],
                    )
                added_names = await self._height_index_names(view, CF_COINS_BY_CONFIRMED, height)
                removed_names = [] if height == 0 else await self._height_index_names(view, CF_COINS_BY_SPENT, height)
                loaded = await self._load_many(view, [*added_names, *removed_names])
                added_again = await self._height_index_names(view, CF_COINS_BY_CONFIRMED, height)
                removed_again = [] if height == 0 else await self._height_index_names(view, CF_COINS_BY_SPENT, height)
                if added_names == added_again and removed_names == removed_again:
                    return (
                        [loaded[name] for name in added_names if name in loaded],
                        [loaded[name] for name in removed_names if name in loaded],
                    )
        raise RuntimeError("coin records changed while a height was read")

    async def _height_index_names(self, view: _View, cf: str, height: int) -> list[bytes32]:
        prefix = u32_be(height)
        rows = await view.scan(cf, prefix, prefix_end(prefix))
        return [bytes32(key[-32:]) for key, _value in rows]

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
        loaded = await self._load_many(view, ordered)
        found: list[StoredCoin] = []
        for name in ordered:
            stored = loaded.get(name)
            if stored is None:
                continue
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
        ranges = [
            (bytes(puzzle_hash) + u32_be(int(start_height)), bytes(puzzle_hash) + u32_be(int(end_height)))
            for puzzle_hash in puzzle_hashes
        ]
        async with self.db.reader_no_transaction() as view:
            journals = await self._pending_journals(view)
            grouped = await view.scan_prefixes(CF_COINS_BY_PUZZLE_CONFIRMED, ranges)
            names = _unique_names(grouped)
            loaded = await self._load_many(view, names)
            for puzzle_hash, rows in zip(puzzle_hashes, grouped, strict=True):
                for name in _coin_names(rows):
                    stored = loaded.get(name)
                    if stored is None:
                        continue
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
            parents = await view.get_many(CF_COINS, [bytes(parent_id) for parent_id in parent_ids])
            for parent_id in parent_ids:
                raw = parents.get(bytes(parent_id))
                if raw is None:
                    height = self._reward_parent_height(parent_id)
                    if height is None:
                        continue
                else:
                    parent = decode_coin(parent_id, raw)
                    if parent.spent_index <= 0:
                        continue
                    height = parent.spent_index
                if height < int(start_height) or height >= int(end_height):
                    continue
                if height not in added_at:
                    added_at[height] = (await self._stable_height_coins(height))[0]
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

    async def _append_capped_states(
        self,
        view: _View,
        names: list[bytes32],
        states: set[CoinState],
        min_height: int,
        include_spent_coins: bool,
        max_items: int,
    ) -> None:
        """Load coin records in cap-sized windows and stop once the cap is full."""
        cursor = 0
        while cursor < len(names) and len(states) < max_items:
            room = max_items - len(states)
            window = names[cursor : cursor + room]
            cursor += len(window)
            loaded = await self._load_many(view, window)
            for name in window:
                stored = loaded.get(name)
                if stored is None:
                    continue
                if not self._state_visible(stored, min_height, include_spent_coins):
                    continue
                states.add(_to_state(stored))
                if len(states) >= max_items:
                    return

    async def _scan_capped_index(
        self,
        view: _View,
        family: str,
        puzzle_hash: bytes32,
        min_height: int,
        include_spent_coins: bool,
        states: set[CoinState],
        seen: set[bytes32],
        max_items: int,
    ) -> None:
        """Walk one puzzle index until the cap is full or the prefix ends."""
        start = bytes(puzzle_hash) + u32_be(min_height)
        end = prefix_end(bytes(puzzle_hash))
        while len(states) < max_items and (end is None or start < end):
            room = max_items - len(states)
            limit = room if room < INDEX_SCAN_CHUNK else INDEX_SCAN_CHUNK
            rows = await view.scan(family, start, end, limit=limit)
            if not rows:
                return
            names: list[bytes32] = []
            for key, _value in rows:
                name = bytes32(key[-32:])
                if name in seen:
                    continue
                seen.add(name)
                names.append(name)
            await self._append_capped_states(view, names, states, min_height, include_spent_coins, max_items)
            if len(states) >= max_items or len(rows) < limit:
                return
            start = key_after(rows[-1][0])

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
                seen: set[bytes32] = set()
                await self._scan_capped_index(
                    view,
                    CF_COINS_BY_PUZZLE_CONFIRMED,
                    puzzle_hash,
                    int(min_height),
                    include_spent_coins,
                    states,
                    seen,
                    max_items,
                )
                if len(states) >= max_items:
                    return states
                if include_spent_coins:
                    await self._scan_capped_index(
                        view,
                        CF_COINS_BY_PUZZLE_SPENT,
                        puzzle_hash,
                        int(min_height),
                        include_spent_coins,
                        states,
                        seen,
                        max_items,
                    )
                    if len(states) >= max_items:
                        return states
                if not journals:
                    continue
                pending = await self._pending_coins(
                    view, journals, puzzle_hash=puzzle_hash, include_removals=include_spent_coins
                )
                for stored in pending:
                    if stored.coin_name in seen:
                        continue
                    if not self._state_visible(stored, int(min_height), include_spent_coins):
                        continue
                    states.add(_to_state(stored))
                    if len(states) >= max_items:
                        return states
        return states

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
        ordered_ids = list(coin_ids)
        states: list[CoinState] = []
        limit_height = max_height != uint32.MAXIMUM
        cursor = 0
        async with self.db.reader_no_transaction() as view:
            while cursor < len(ordered_ids) and len(states) < max_items:
                room = max_items - len(states)
                window = ordered_ids[cursor : cursor + room]
                cursor += len(window)
                loaded = await self._load_many(view, window)
                for coin_id in window:
                    stored = loaded.get(coin_id)
                    if stored is None:
                        continue
                    if stored.confirmed_index < int(min_height) and stored.spent_index < int(min_height):
                        continue
                    if limit_height and (
                        stored.confirmed_index > int(max_height) or stored.spent_index > int(max_height)
                    ):
                        continue
                    if not include_spent_coins and stored.spent_index > 0:
                        continue
                    states.append(_to_state(stored))
                    if len(states) >= max_items:
                        return states
        return states

    async def get_unspent_lineage_info_for_puzzle_hash(self, puzzle_hash: bytes32) -> UnspentLineageInfo | None:
        end = prefix_end(bytes(puzzle_hash))
        async with self.db.reader_no_transaction() as view:
            rows = await view.scan(CF_FF_UNSPENT, bytes(puzzle_hash), end)
            names = _coin_names(rows)
            loaded = await self._load_many(view, names)
            parent_ids = list(dict.fromkeys(loaded[name].parent for name in names if name in loaded))
            parents = await self._load_many(view, parent_ids)
            matches: dict[bytes32, UnspentLineageInfo] = {}
            for name in names:
                unspent = loaded.get(name)
                if unspent is None:
                    continue
                if unspent.spent_index != -1 or unspent.puzzle_hash != puzzle_hash:
                    continue
                parent = parents.get(unspent.parent)
                if parent is None:
                    continue
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
            pending_parents = await self._load_many(
                view, list(dict.fromkeys(unspent.parent for unspent in pending))
            )
            for unspent in pending:
                if unspent.spent_index != -1 or unspent.coin_name in matches:
                    continue
                parent = pending_parents.get(unspent.parent)
                if parent is None:
                    continue
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
            closed_before = await _fill_batch_page(
                self,
                view,
                puzzle_hashes,
                by_name,
                int(min_height),
                include_spent,
                include_unspent,
                int(min_amount),
                max_items,
            )
            if journals:
                for puzzle_hash in puzzle_hashes:
                    pending = await self._pending_coins(
                        view, journals, puzzle_hash=puzzle_hash, include_removals=include_spent
                    )
                    _keep_batch_coins(
                        pending,
                        by_name,
                        int(min_height),
                        include_spent,
                        include_unspent,
                        int(min_amount),
                        closed_before,
                    )
            if include_hinted:
                await _fill_batch_hints(
                    self,
                    view,
                    puzzle_hashes,
                    by_name,
                    int(min_height),
                    include_spent,
                    include_unspent,
                    int(min_amount),
                    closed_before,
                )
        ordered = sorted(by_name.values(), key=_order_height)
        return _paginate(ordered, max_items)


class _HeightCursor:
    """Next unread confirmed-index keys for one puzzle hash."""

    def __init__(self, puzzle_hash: bytes32) -> None:
        self.start: bytes | None = bytes(puzzle_hash) + u32_be(0)
        self.end = prefix_end(bytes(puzzle_hash))
        self.rows: list[tuple[bytes, bytes]] = []
        self.pos = 0
        self.no_more = False

    def has_row(self) -> bool:
        return self.pos < len(self.rows)

    def head(self) -> bytes:
        return self.rows[self.pos][0]

    async def fill(self, view: _View) -> None:
        if self.has_row() or self.no_more or self.start is None:
            return
        if self.end is not None and self.start >= self.end:
            self.no_more = True
            return
        rows = await view.scan(CF_COINS_BY_PUZZLE_CONFIRMED, self.start, self.end, limit=INDEX_SCAN_CHUNK)
        if len(rows) < INDEX_SCAN_CHUNK:
            self.no_more = True
        if not rows:
            return
        self.rows = rows
        self.pos = 0
        self.start = key_after(rows[-1][0])


def _confirmed_height(key: bytes) -> int:
    return u32_from_be(key[32:36])


def _next_cursor(cursors: list[_HeightCursor]) -> _HeightCursor | None:
    chosen: _HeightCursor | None = None
    chosen_key = b""
    chosen_height = 0
    for cursor in cursors:
        if not cursor.has_row():
            continue
        key = cursor.head()
        height = _confirmed_height(key)
        if chosen is None or height < chosen_height or (height == chosen_height and key < chosen_key):
            chosen = cursor
            chosen_key = key
            chosen_height = height
    return chosen


async def _refill(view: _View, cursors: list[_HeightCursor]) -> None:
    for cursor in cursors:
        if not cursor.has_row():
            await cursor.fill(view)


def _page_closed(states: dict[bytes32, CoinState], before_height: int, max_items: int) -> bool:
    """True when every unseen coin sorts at or after the page SQLite would return."""
    low = 0
    for state in states.values():
        if _order_height(state) < before_height:
            low += 1
            if low >= max_items + 1:
                return True
    return False


def _keep_batch_coins(
    records: list[StoredCoin],
    by_name: dict[bytes32, CoinState],
    min_height: int,
    include_spent: bool,
    include_unspent: bool,
    min_amount: int,
    closed_before: int | None,
) -> None:
    for stored in records:
        if stored.coin_name in by_name:
            continue
        if not _batch_accepts(stored, min_height, include_spent, include_unspent, min_amount):
            continue
        state = _to_state(stored)
        if closed_before is not None and _order_height(state) >= closed_before:
            continue
        by_name[stored.coin_name] = state


async def _fill_batch_page(
    store: RocksCoinStore,
    view: _View,
    puzzle_hashes: list[bytes32],
    by_name: dict[bytes32, CoinState],
    min_height: int,
    include_spent: bool,
    include_unspent: bool,
    min_amount: int,
    max_items: int,
) -> int | None:
    """Load confirmed coins in height order and stop once a later height cannot change the page."""
    cursors = [_HeightCursor(puzzle_hash) for puzzle_hash in puzzle_hashes]
    await _refill(view, cursors)
    while True:
        await _refill(view, cursors)
        first = _next_cursor(cursors)
        if first is None:
            return None
        height = _confirmed_height(first.head())
        names: list[bytes32] = []
        while True:
            await _refill(view, cursors)
            current = _next_cursor(cursors)
            if current is None:
                next_height: int | None = None
                break
            key = current.head()
            key_height = _confirmed_height(key)
            if key_height != height:
                next_height = key_height
                break
            current.pos += 1
            names.append(bytes32(key[-32:]))
        fresh = [name for name in dict.fromkeys(names) if name not in by_name]
        loaded = await store._load_many(view, fresh)
        _keep_batch_coins(
            [loaded[name] for name in fresh if name in loaded],
            by_name,
            min_height,
            include_spent,
            include_unspent,
            min_amount,
            None,
        )
        if next_height is not None and _page_closed(by_name, next_height, max_items):
            return next_height


async def _fill_batch_hints(
    store: RocksCoinStore,
    view: _View,
    puzzle_hashes: list[bytes32],
    by_name: dict[bytes32, CoinState],
    min_height: int,
    include_spent: bool,
    include_unspent: bool,
    min_amount: int,
    closed_before: int | None,
) -> None:
    """Hint keys are not height-ordered, so each hint is read in short pieces."""
    for puzzle_hash in puzzle_hashes:
        start = bytes(puzzle_hash)
        end = prefix_end(start)
        while end is None or start < end:
            rows = await view.scan(CF_HINTS_BY_HINT, start, end, limit=INDEX_SCAN_CHUNK)
            if not rows:
                break
            fresh = [name for name in dict.fromkeys(_coin_names(rows)) if name not in by_name]
            loaded = await store._load_many(view, fresh)
            _keep_batch_coins(
                [loaded[name] for name in fresh if name in loaded],
                by_name,
                min_height,
                include_spent,
                include_unspent,
                min_amount,
                closed_before,
            )
            if len(rows) < INDEX_SCAN_CHUNK:
                break
            start = key_after(rows[-1][0])


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


def _coin_names(rows: list[tuple[bytes, bytes]]) -> list[bytes32]:
    return [bytes32(key[-32:]) for key, _value in rows]


def _unique_names(groups: list[list[tuple[bytes, bytes]]]) -> list[bytes32]:
    names: list[bytes32] = []
    seen: set[bytes32] = set()
    for rows in groups:
        for name in _coin_names(rows):
            if name in seen:
                continue
            seen.add(name)
            names.append(name)
    return names


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
