from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from chia_rs import CoinRecord
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint64

from chia._tests.util.db_connection import DBConnection
from chia.consensus.coin_store_protocol import CoinStoreProtocol
from chia.full_node.coin_store import CoinStore
from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.coin_codec import StoredCoin, decode_coin
from chia.full_node.db.coin_store import RocksCoinStore
from chia.full_node.db.hint_store import RocksHintStore
from chia.full_node.db.keys import CF_COINS
from chia.full_node.db.rocks import RocksBackend
from chia.full_node.hint_store import HintStore
from chia.util.casts import int_from_bytes
from chia.util.db_wrapper import DBWrapper2


async def add_coin_records_to_db(coin_store: CoinStoreProtocol, records: list[CoinRecord]) -> None:
    if len(records) == 0:
        return
    if isinstance(coin_store, RocksCoinStore):
        await coin_store.import_coins(
            [
                StoredCoin(
                    record.coin.name(),
                    int(record.confirmed_block_index),
                    int(record.spent_block_index),
                    record.coinbase,
                    record.coin.puzzle_hash,
                    record.coin.parent_coin_info,
                    record.coin.amount,
                    int(record.timestamp),
                )
                for record in records
            ]
        )
        return
    db_wrapper = getattr(coin_store, "db_wrapper", None)
    assert isinstance(db_wrapper, DBWrapper2), "CoinStore must use DBWrapper2"
    async with db_wrapper.writer_maybe_transaction() as conn:
        await conn.executemany(
            "INSERT INTO coin_record VALUES(?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (
                    record.coin.name(),
                    record.confirmed_block_index,
                    record.spent_block_index,
                    int(record.coinbase),
                    record.coin.puzzle_hash,
                    record.coin.parent_coin_info,
                    record.coin.amount.stream_to_bytes(),
                    record.timestamp,
                )
                for record in records
            ),
        )


CoinAndHint = tuple[CoinStore | RocksCoinStore, HintStore | RocksHintStore]


@asynccontextmanager
async def open_coin_stores(root: Path, engine: str) -> AsyncIterator[CoinAndHint]:
    """One coin store and one hint store. SQLite is in memory. RocksDB is one directory."""
    if engine == "sqlite":
        async with DBConnection(2) as wrapper:
            yield await CoinStore.create(wrapper), await HintStore.create(wrapper)
        return
    if engine != "rocksdb":
        raise ValueError(f"unknown chain database engine {engine}")
    root.mkdir(parents=True, exist_ok=True)
    chain = ChainDB(RocksBackend(root / "coins.rocksdb", sync="OFF"))
    try:
        yield await RocksCoinStore.create(chain), await RocksHintStore.create(chain)
    finally:
        await chain.close()


async def insert_coin_rows(
    coin_store: CoinStore | RocksCoinStore,
    rows: list[tuple[bytes, int, int, int, bytes, bytes, bytes, int]],
) -> None:
    """Insert coin rows. `spent_index` may be -1. The amount field is the SQLite blob."""
    if isinstance(coin_store, RocksCoinStore):
        await coin_store.import_coins(
            [
                StoredCoin(
                    bytes32(name),
                    int(confirmed),
                    int(spent),
                    bool(coinbase),
                    bytes32(puzzle),
                    bytes32(parent),
                    uint64(int_from_bytes(amount)),
                    int(timestamp),
                )
                for name, confirmed, spent, coinbase, puzzle, parent, amount, timestamp in rows
            ]
        )
        return
    async with coin_store.db_wrapper.writer() as conn:
        await conn.executemany("INSERT INTO coin_record VALUES(?, ?, ?, ?, ?, ?, ?, ?)", rows)


async def stored_spent_index(coin_store: CoinStore | RocksCoinStore, coin_name: bytes32) -> int:
    """The spent index as stored, including the fast-forward sentinel -1."""
    if isinstance(coin_store, RocksCoinStore):
        async with coin_store.db.reader_no_transaction() as view:
            raw = await view.get(CF_COINS, bytes(coin_name))
        assert raw is not None
        return decode_coin(coin_name, raw).spent_index
    async with coin_store.db_wrapper.reader_no_transaction() as conn:
        cursor = await conn.execute("SELECT spent_index FROM coin_record WHERE coin_name = ?", (coin_name,))
        row = await cursor.fetchone()
        await cursor.close()
    assert row is not None
    return int(row[0])
