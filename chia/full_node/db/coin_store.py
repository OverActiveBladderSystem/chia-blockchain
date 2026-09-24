from __future__ import annotations

import logging
import time
from collections.abc import Collection

from chia_rs import CoinRecord, CoinState
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32, uint64

from chia.full_node.db.chain_db import ChainDB, WriteSession, _View
from chia.full_node.db.coin_codec import StoredCoin, decode_coin, encode_coin, index_entries
from chia.full_node.db.keys import (
    CF_COINS,
    CF_COINS_BY_CONFIRMED,
    CF_COINS_BY_PARENT,
    CF_COINS_BY_PUZZLE_CONFIRMED,
    CF_COINS_BY_PUZZLE_SPENT,
    CF_COINS_BY_SPENT,
    CF_FF_UNSPENT,
    CF_HINTS_BY_HINT,
    CF_META,
    META_UNSPENT,
    prefix_end,
    u32_be,
)
from chia.types.blockchain_format.coin import Coin
from chia.types.mempool_item import UnspentLineageInfo

log = logging.getLogger(__name__)


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
        by_parent: list[tuple[bytes, bytes]] = []
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
            by_parent.append((parent + confirmed + name, empty))
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
        by_parent.sort()
        fast_forward.sort()
        await self.db.write_column_batches(
            [
                (CF_COINS, coins),
                (CF_COINS_BY_CONFIRMED, by_confirmed),
                (CF_COINS_BY_SPENT, by_spent),
                (CF_COINS_BY_PUZZLE_CONFIRMED, by_puzzle_confirmed),
                (CF_COINS_BY_PUZZLE_SPENT, by_puzzle_spent),
                (CF_COINS_BY_PARENT, by_parent),
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
                CF_COINS_BY_PARENT,
                CF_FF_UNSPENT,
            ):
                session.delete_range(family, b"", b"\xff" * 80)
            session.put(CF_META, META_UNSPENT, (0).to_bytes(8, "little", signed=False))

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
            keys = [bytes(record.coin_name) for record in additions]
            keys.extend(bytes(name) for name in tx_removals)
            existing = await session.get_many(CF_COINS, keys)
            counter_raw = await session.get(CF_META, META_UNSPENT)
            counter = 0 if counter_raw is None else int.from_bytes(counter_raw, "little", signed=False)
            for record in additions:
                previous_raw = existing.get(bytes(record.coin_name))
                if previous_raw is not None:
                    raise ValueError(f"Coin {record.coin_name.hex()} already exists")
                counter = self._write_coin(session, record, None, counter)
                existing[bytes(record.coin_name)] = encode_coin(record)
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
                    counter = self._spend_coin(session, previous, int(height), counter)
                    changed += 1
                if changed != len(tx_removals):
                    raise ValueError(
                        f"Invalid operation to set spent, total updates {changed} expected {len(tx_removals)}"
                    )
            session.put(CF_META, META_UNSPENT, counter.to_bytes(8, "little", signed=False))
        elapsed = time.monotonic() - started
        message = (
            f"Height {height}: It took {elapsed:0.2f}s to apply {len(tx_additions)} additions and "
            f"{len(tx_removals)} removals to the coin store."
        )
        log.log(logging.WARNING if elapsed > 10 else logging.DEBUG, message)

    def _write_coin(
        self, session: WriteSession, record: StoredCoin, previous: StoredCoin | None, counter: int
    ) -> int:
        if previous is not None:
            self._drop_indexes_now(session, previous)
            if previous.spent_index <= 0:
                counter -= 1
        session.put(CF_COINS, bytes(record.coin_name), encode_coin(record))
        for family, key in index_entries(record):
            if family == CF_COINS:
                continue
            session.put(family, key, b"")
        if record.spent_index <= 0:
            counter += 1
        if counter < 0:
            raise ValueError(f"unspent counter went negative ({counter})")
        return counter

    def _spend_coin(self, session: WriteSession, record: StoredCoin, index: int, counter: int) -> int:
        self._drop_indexes_now(session, record)
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
        session.put(CF_COINS, bytes(record.coin_name), encode_coin(spent))
        for family, key in index_entries(spent):
            if family == CF_COINS:
                continue
            session.put(family, key, b"")
        counter -= 1
        if counter < 0:
            raise ValueError(f"unspent counter went negative ({counter})")
        return counter

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
        for name in coin_names:
            record = await self._load(session, name)
            if record is None or record.spent_index > 0:
                continue
            await self._drop_indexes(session, record)
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
            for cf, key in index_entries(spent):
                if cf == CF_COINS:
                    continue
                session.put(cf, key, b"")
            await self._add_unspent(session, -1)
            updated += 1
        if updated != len(coin_names):
            raise ValueError(f"Invalid operation to set spent, total updates {updated} expected {len(coin_names)}")

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
        records: list[CoinRecord] = []
        async with self.db.reader_no_transaction() as view:
            for name in names:
                raw = await view.get(CF_COINS, bytes(name))
                if raw is not None:
                    records.append(_to_record(decode_coin(name, raw)))
        return records

    async def get_coins_added_at_height(self, height: uint32) -> list[CoinRecord]:
        return await self._records_by_height_index(CF_COINS_BY_CONFIRMED, int(height))

    async def get_coins_removed_at_height(self, height: uint32) -> list[CoinRecord]:
        if height == 0:
            return []
        return await self._records_by_height_index(CF_COINS_BY_SPENT, int(height))

    async def _records_by_height_index(self, cf: str, height: int) -> list[CoinRecord]:
        prefix = u32_be(height)
        end = prefix_end(prefix)
        async with self.db.reader_no_transaction() as view:
            rows = await view.scan(cf, prefix, end)
            records: list[CoinRecord] = []
            for key, _value in rows:
                name = bytes32(key[-32:])
                raw = await view.get(CF_COINS, bytes(name))
                if raw is None:
                    continue
                records.append(_to_record(decode_coin(name, raw)))
        return records

    async def rollback_to_block(self, block_index: int) -> dict[bytes32, CoinRecord]:
        changes: dict[bytes32, CoinRecord] = {}
        async with self.db.writer_maybe_transaction() as session:
            added = await session.scan(CF_COINS_BY_CONFIRMED, u32_be(block_index + 1), None)
            for key, _value in added:
                name = bytes32(key[-32:])
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

            spent = await session.scan(CF_COINS_BY_SPENT, u32_be(block_index + 1), None)
            for key, _value in spent:
                name = bytes32(key[-32:])
                if name in changes:
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
        found: list[CoinRecord] = []
        async with self.db.reader_no_transaction() as view:
            for parent_id in parent_ids:
                start = bytes(parent_id) + u32_be(int(start_height))
                end = bytes(parent_id) + u32_be(int(end_height))
                rows = await view.scan(CF_COINS_BY_PARENT, start, end)
                for key, _value in rows:
                    name = bytes32(key[-32:])
                    raw = await view.get(CF_COINS, bytes(name))
                    if raw is None:
                        continue
                    stored = decode_coin(name, raw)
                    if not include_spent_coins and stored.spent_index > 0:
                        continue
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
            for puzzle_hash in puzzle_hashes:
                await self._collect_puzzle_states(
                    view, puzzle_hash, int(min_height), include_spent_coins, states, max_items
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
            if stored.confirmed_index < min_height and not (stored.spent_index >= min_height):
                continue
            if not include_spent_coins and stored.spent_index > 0:
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
                if limit_height and (
                    stored.confirmed_index > int(max_height) or stored.spent_index > int(max_height)
                ):
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
            matches: list[UnspentLineageInfo] = []
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
                    matches.append(
                        UnspentLineageInfo(
                            coin_id=unspent.coin_name,
                            parent_id=unspent.parent,
                            parent_parent_id=parent.parent,
                        )
                    )
        if len(matches) != 1:
            log.debug("Expected 1 unspent with puzzle hash %s, but found %s", puzzle_hash.hex(), len(matches))
            return None
        return matches[0]

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
