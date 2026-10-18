from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from chia._tests.blockchain.blockchain_test_utils import _validate_and_add_block_no_error
from chia._tests.conftest import ConsensusMode
from chia._tests.util.blockchain import create_blockchain
from chia.full_node.db.block_store import RocksBlockStore
from chia.full_node.db.keys import CF_BLOCK_META
from chia.full_node.db.rocks import RocksBackend
from chia.simulator.block_tools import BlockTools
from chia.util.task_referencer import create_referenced_task


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_a_block_save_finishes_during_a_block_record_read(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
) -> None:
    del consensus_mode
    blocks = bt.get_consecutive_blocks(4)
    release = threading.Event()
    started = threading.Event()
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        for block in blocks[:3]:
            await _validate_and_add_block_no_error(blockchain, block)
        store = blockchain.block_store
        assert isinstance(store, RocksBlockStore)
        backend = store.db._backend
        assert isinstance(backend, RocksBackend)
        real = backend._multiget
        first = True

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            nonlocal first
            if first and cf == CF_BLOCK_META and len(keys) == 3:
                first = False
                started.set()
                assert release.wait(30)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        hashes = [block.header_hash for block in blocks[:3]]
        try:
            read = create_referenced_task(store.get_block_records_by_hash(hashes))
            assert await asyncio.to_thread(started.wait, 30)
            await asyncio.wait_for(_validate_and_add_block_no_error(blockchain, blocks[3]), timeout=30)
            assert not read.done()
            release.set()
            records = await read
            peak = blockchain.get_peak()
            assert peak is not None
            assert peak.height == 3
            assert [record.height for record in records] == [0, 1, 2]
        finally:
            release.set()
