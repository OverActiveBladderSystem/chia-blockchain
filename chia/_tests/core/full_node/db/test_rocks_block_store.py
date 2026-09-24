from __future__ import annotations

from pathlib import Path

import pytest
from chia_rs.sized_ints import uint32

from chia._tests.blockchain.blockchain_test_utils import _validate_and_add_block
from chia._tests.util.db_connection import DBConnection
from chia.consensus.block_height_map import BlockHeightMap
from chia.consensus.blockchain import Blockchain
from chia.full_node.block_store import BlockStore
from chia.full_node.coin_store import CoinStore
from chia.full_node.db.block_store import RocksBlockStore
from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.memory import MemoryBackend
from chia.full_node.db.rocks import RocksBackend
from chia.simulator.block_tools import BlockTools
from chia.util.inline_executor import InlineExecutor


@pytest.mark.limit_consensus_modes(reason="save time")
@pytest.mark.anyio
async def test_rocks_block_store_round_trip(tmp_path: Path, bt: BlockTools) -> None:
    blocks = bt.get_consecutive_blocks(3, guarantee_transaction_block=True)
    async with DBConnection(2) as wrapper:
        coin_store = await CoinStore.create(wrapper)
        sqlite_store = await BlockStore.create(wrapper)
        height_map = await BlockHeightMap.create(tmp_path, wrapper)
        blockchain = await Blockchain.create(coin_store, sqlite_store, height_map, bt.constants, InlineExecutor())
        records = []
        for block in blocks:
            await _validate_and_add_block(blockchain, block)
            records.append(blockchain.block_record(block.header_hash))

    store = await RocksBlockStore.create(ChainDB(MemoryBackend()))
    async with store.transaction():
        for block, record in zip(blocks, records, strict=True):
            await store.add_full_block(block.header_hash, block, record)
            await store.set_in_chain([(record.header_hash,)])
            await store.set_peak(record.header_hash)

    for block, record in zip(blocks, records, strict=True):
        assert await store.get_full_block(block.header_hash) == block
        assert await store.get_block_record(block.header_hash) == record
        assert await store.get_prev_hash(block.header_hash) == block.prev_header_hash

    peak = await store.get_peak()
    assert peak == (blocks[-1].header_hash, uint32(blocks[-1].height))
    assert await store.count_compactified_blocks() + await store.count_uncompactified_blocks() == len(blocks)
    in_range = await store.get_block_records_in_range(0, blocks[-1].height)
    assert len(in_range) == len(blocks)

    await store.rollback(0)
    remaining = await store.get_block_records_in_range(0, blocks[-1].height)
    assert list(remaining) == [blocks[0].header_hash]
    assert await store.count_compactified_blocks() + await store.count_uncompactified_blocks() == 1


@pytest.mark.anyio
async def test_rocks_backend_reopens(tmp_path: Path) -> None:
    path = tmp_path / "chain.rocksdb"
    database = ChainDB(RocksBackend(path, sync="OFF"))
    async with database.writer() as session:
        session.put("meta", b"peak", b"abc")
    await database.close()

    reopened = ChainDB(RocksBackend(path, sync="OFF"))
    async with reopened.reader_no_transaction() as view:
        assert await view.get("meta", b"peak") == b"abc"
    await reopened.close()
