from __future__ import annotations

import pytest
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32, uint64

from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.coin_store import RocksCoinStore
from chia.full_node.db.memory import MemoryBackend
from chia.types.blockchain_format.coin import Coin

PARENT = bytes32(b"\x11" * 32)
PUZZLE = bytes32(b"\x22" * 32)
OTHER = bytes32(b"\x33" * 32)


def _coin(parent: bytes32, puzzle: bytes32, amount: int) -> Coin:
    return Coin(parent, puzzle, uint64(amount))


@pytest.mark.anyio
async def test_new_block_spend_and_unspent_count() -> None:
    store = RocksCoinStore(ChainDB(MemoryBackend()))
    created = _coin(PARENT, PUZZLE, 100)
    reward = _coin(PARENT, OTHER, 1_750_000_000_000)
    await store.new_block(uint32(1), uint64(10), [reward], [(created.name(), created, False)], [])
    assert await store.num_unspent() == 2
    assert (await store.get_coin_record(created.name())).spent_block_index == 0

    await store.new_block(uint32(2), uint64(20), [], [], [created.name()])
    spent = await store.get_coin_record(created.name())
    assert spent is not None
    assert spent.spent_block_index == 2
    assert await store.num_unspent() == 1


@pytest.mark.anyio
async def test_set_spent_mismatch_rolls_back_the_insert() -> None:
    store = RocksCoinStore(ChainDB(MemoryBackend()))
    created = _coin(PARENT, PUZZLE, 50)
    missing = bytes32(b"\x44" * 32)
    with pytest.raises(ValueError, match="Invalid operation to set spent"):
        await store.new_block(uint32(1), uint64(1), [], [(created.name(), created, True)], [missing])
    assert await store.get_coin_record(created.name()) is None
    assert await store.num_unspent() == 0


@pytest.mark.anyio
async def test_rollback_restores_fast_forward_sentinel() -> None:
    store = RocksCoinStore(ChainDB(MemoryBackend()))
    parent = _coin(bytes32(b"\x01" * 32), PUZZLE, 100)
    child = _coin(parent.name(), PUZZLE, 100)
    extra = _coin(bytes32(b"\x02" * 32), OTHER, 5)
    await store.new_block(uint32(1), uint64(1), [], [(parent.name(), parent, False)], [])
    await store.new_block(uint32(2), uint64(2), [], [], [parent.name()])
    await store.new_block(uint32(3), uint64(3), [], [(child.name(), child, True), (extra.name(), extra, False)], [])

    changes = await store.rollback_to_block(2)
    assert extra.name() in changes
    assert await store.get_coin_record(extra.name()) is None
    child_record = await store.get_coin_record(child.name())
    assert child_record is None or child.name() in changes

    # child was created at height 3, so it is gone. Parent spend at height 2 remains.
    parent_record = await store.get_coin_record(parent.name())
    assert parent_record is not None
    assert parent_record.spent_block_index == 2

    # A coin spent above the fork, whose parent is the same puzzle and amount and already spent,
    # is restored with spent index -1 in storage and reported as unspent (0) to the caller.
    kept = _coin(parent.name(), PUZZLE, 100)
    await store.new_block(uint32(3), uint64(3), [], [(kept.name(), kept, True)], [])
    await store.new_block(uint32(4), uint64(4), [], [], [kept.name()])
    reported = await store.rollback_to_block(3)
    assert reported[kept.name()].spent_block_index == 0
    lineage = await store.get_unspent_lineage_info_for_puzzle_hash(PUZZLE)
    assert lineage is not None
    assert lineage.coin_id == kept.name()
    assert lineage.parent_id == parent.name()


@pytest.mark.anyio
async def test_puzzle_hash_query_filters_spent_coins() -> None:
    store = RocksCoinStore(ChainDB(MemoryBackend()))
    first = _coin(PARENT, PUZZLE, 1)
    second = _coin(PARENT, PUZZLE, 2)
    await store.new_block(
        uint32(5),
        uint64(5),
        [],
        [(first.name(), first, False), (second.name(), second, False)],
        [],
    )
    await store.new_block(uint32(6), uint64(6), [], [], [first.name()])
    unspent = await store.get_coin_records_by_puzzle_hash(False, PUZZLE)
    assert {record.name for record in unspent} == {second.name()}
    added = await store.get_coins_added_at_height(uint32(5))
    assert {record.name for record in added} == {first.name(), second.name()}
    removed = await store.get_coins_removed_at_height(uint32(6))
    assert [record.name for record in removed] == [first.name()]
