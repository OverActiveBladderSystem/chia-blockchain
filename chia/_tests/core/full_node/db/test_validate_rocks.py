from __future__ import annotations

from pathlib import Path

import pytest

from chia._tests.blockchain.blockchain_test_utils import _validate_and_add_block
from chia._tests.conftest import ConsensusMode
from chia._tests.util.blockchain import open_v2_stores
from chia.consensus.blockchain import Blockchain
from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.keys import CF_BLOCK_BLOBS
from chia.full_node.db.rocks import RocksBackend
from chia.full_node.db.validate import validate_rocks
from chia.simulator.block_tools import BlockTools
from chia.util.inline_executor import InlineExecutor


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="one short chain is enough to parse stored blocks")
async def test_validate_rocks_parses_full_block_bytes(
    tmp_path: Path, bt: BlockTools, consensus_mode: ConsensusMode
) -> None:
    del consensus_mode
    blocks = bt.get_consecutive_blocks(3, guarantee_transaction_block=True)
    async with open_v2_stores(tmp_path, 2, engine="rocksdb") as (coin_store, block_store, height_map):
        blockchain = await Blockchain.create(coin_store, block_store, height_map, bt.constants, InlineExecutor())
        for block in blocks:
            await _validate_and_add_block(blockchain, block)
    path = tmp_path / "blockchain_v2_unit.rocksdb"
    peak, height, _unspent = await validate_rocks(
        path, genesis=bt.constants.GENESIS_CHALLENGE, validate_blocks=True
    )
    assert peak == blocks[-1].header_hash
    assert height == 2

    chain = ChainDB(RocksBackend(path, sync="OFF"))
    async with chain.writer() as session:
        session.put(CF_BLOCK_BLOBS, bytes(blocks[-1].header_hash), b"garbage")
    await chain.close()
    with pytest.raises(RuntimeError, match="could not be parsed"):
        await validate_rocks(path, genesis=bt.constants.GENESIS_CHALLENGE, validate_blocks=True)
