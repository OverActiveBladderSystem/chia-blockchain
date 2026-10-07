from __future__ import annotations

import pytest
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32, uint64

from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.coin_store import COIN_INDEX_CHUNK, RocksCoinStore
from chia.full_node.db.keys import (
    CF_COINS,
    CF_COINS_BY_PUZZLE_CONFIRMED,
    CF_COINS_BY_PUZZLE_SPENT,
    CF_FF_UNSPENT,
    CF_META,
    META_COIN_INDEXED,
)
from chia.full_node.db.memory import MemoryBackend
from chia.full_node.db.ops import Op
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


@pytest.mark.anyio
async def test_one_index_batch_drops_a_fast_forward_key_spent_later_in_the_batch() -> None:
    store = RocksCoinStore(ChainDB(MemoryBackend()))
    parent = _coin(bytes32(b"\x01" * 32), PUZZLE, 100)
    child = _coin(parent.name(), PUZZLE, 100)
    await store.new_block(uint32(1), uint64(1), [], [(parent.name(), parent, False)], [])
    await store.new_block(uint32(2), uint64(2), [], [], [parent.name()])
    await store.new_block(uint32(3), uint64(3), [], [(child.name(), child, True)], [])
    await store.new_block(uint32(4), uint64(4), [], [], [child.name()])

    await store.index_pending_all()

    async with store.db.reader_no_transaction() as view:
        fast_forward = await view.scan(CF_FF_UNSPENT, b"", None)
        spent_keys = await view.scan(CF_COINS_BY_PUZZLE_SPENT, b"", None)
        confirmed_keys = await view.scan(CF_COINS_BY_PUZZLE_CONFIRMED, b"", None)
    assert fast_forward == []
    assert any(key.endswith(bytes(child.name())) for key, _value in spent_keys)
    assert any(key.endswith(bytes(child.name())) for key, _value in confirmed_keys)


@pytest.mark.anyio
async def test_block_save_defers_lookup_keys_until_index_pending() -> None:
    store = RocksCoinStore(ChainDB(MemoryBackend()))
    created = _coin(PARENT, PUZZLE, 100)
    reward = _coin(PARENT, OTHER, 1_750_000_000_000)
    await store.new_block(uint32(1), uint64(10), [reward], [(created.name(), created, False)], [])

    async with store.db.reader_no_transaction() as view:
        puzzle_keys = await view.scan(CF_COINS_BY_PUZZLE_CONFIRMED, b"", None)
    assert puzzle_keys == []

    by_puzzle = await store.get_coin_records_by_puzzle_hash(True, PUZZLE)
    assert {record.name for record in by_puzzle} == {created.name()}
    # PARENT is not a spent coin, so it has no children to report.
    assert await store.get_coin_records_by_parent_ids(True, [PARENT]) == []

    await store.index_pending()
    async with store.db.reader_no_transaction() as view:
        puzzle_keys = await view.scan(CF_COINS_BY_PUZZLE_CONFIRMED, b"", None)
    assert len(puzzle_keys) == 2
    by_puzzle_after = await store.get_coin_records_by_puzzle_hash(False, PUZZLE)
    assert {record.name for record in by_puzzle_after} == {created.name()}

    await store.new_block(uint32(2), uint64(20), [], [], [created.name()])
    hidden = await store.get_coin_records_by_puzzle_hash(False, PUZZLE)
    assert {record.name for record in hidden} == set()
    spent_states = await store.get_coin_states_by_puzzle_hashes(True, {PUZZLE}, uint32(2))
    assert {state.coin.name() for state in spent_states} == {created.name()}
    await store.index_pending()
    spent_again = await store.get_coin_states_by_puzzle_hashes(True, {PUZZLE}, uint32(2))
    assert {state.coin.name() for state in spent_again} == {created.name()}


class _ApplyCounter(MemoryBackend):
    def __init__(self) -> None:
        super().__init__()
        self.applies = 0

    async def apply(self, ops: list[Op]) -> None:
        self.applies += 1
        await super().apply(ops)


class _CountingBackend(MemoryBackend):
    def __init__(self) -> None:
        super().__init__()
        self.coin_lookups: list[list[bytes]] = []

    async def get_many(self, cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
        if cf == CF_COINS:
            self.coin_lookups.append(list(keys))
        return await super().get_many(cf, keys)


@pytest.mark.anyio
async def test_block_save_does_not_probe_new_coin_ids() -> None:
    backend = _CountingBackend()
    store = RocksCoinStore(ChainDB(backend))
    created = _coin(PARENT, PUZZLE, 100)
    await store.new_block(
        uint32(1),
        uint64(10),
        [],
        [(created.name(), created, False)],
        [],
        assume_additions_are_new=True,
    )
    assert backend.coin_lookups == []
    assert (await store.get_coin_record(created.name())) is not None

    await store.new_block(uint32(2), uint64(20), [], [], [created.name()], assume_additions_are_new=True)
    assert backend.coin_lookups == [[bytes(created.name())]]
    spent = await store.get_coin_record(created.name())
    assert spent is not None
    assert spent.spent_block_index == 2

    repeated = _coin(PARENT, OTHER, 5)
    with pytest.raises(ValueError, match="already exists"):
        await store.new_block(
            uint32(3),
            uint64(30),
            [],
            [(repeated.name(), repeated, False), (repeated.name(), repeated, False)],
            [],
            assume_additions_are_new=True,
        )


@pytest.mark.anyio
async def test_children_come_from_the_block_that_spent_the_parent() -> None:
    store = RocksCoinStore(ChainDB(MemoryBackend()))
    parent = _coin(PARENT, PUZZLE, 50)
    other = _coin(PARENT, OTHER, 7)
    await store.new_block(uint32(1), uint64(10), [], [(parent.name(), parent, False), (other.name(), other, False)], [])
    assert await store.get_coin_records_by_parent_ids(True, [parent.name()]) == []

    first = _coin(parent.name(), PUZZLE, 20)
    second = _coin(parent.name(), OTHER, 30)
    unrelated = _coin(other.name(), PUZZLE, 1)
    await store.new_block(
        uint32(2),
        uint64(20),
        [],
        [(first.name(), first, False), (second.name(), second, False), (unrelated.name(), unrelated, False)],
        [parent.name(), other.name()],
    )
    children = await store.get_coin_records_by_parent_ids(True, [parent.name(), other.name()])
    assert {record.name for record in children} == {first.name(), second.name(), unrelated.name()}
    assert await store.get_coin_records_by_parent_ids(True, [bytes32([9] * 32)]) == []


@pytest.mark.anyio
async def test_new_block_still_rejects_a_coin_already_in_the_store() -> None:
    store = RocksCoinStore(ChainDB(MemoryBackend()))
    created = _coin(PARENT, PUZZLE, 100)
    await store.new_block(uint32(1), uint64(10), [], [(created.name(), created, False)], [])
    with pytest.raises(ValueError, match="already exists"):
        await store.new_block(uint32(2), uint64(20), [], [(created.name(), created, False)], [])


async def _indexed_height(store: RocksCoinStore) -> int:
    async with store.db.reader_no_transaction() as view:
        raw = await view.get(CF_META, META_COIN_INDEXED)
    assert raw is not None
    return int.from_bytes(raw, "big", signed=True)


async def _puzzle_key_count(store: RocksCoinStore) -> int:
    async with store.db.reader_no_transaction() as view:
        rows = await view.scan(CF_COINS_BY_PUZZLE_CONFIRMED, b"", None)
    return len(rows)


@pytest.mark.anyio
async def test_lookup_index_catches_up_in_chunks_and_releases_the_writer() -> None:
    backend = _ApplyCounter()
    store = RocksCoinStore(ChainDB(backend))
    total = COIN_INDEX_CHUNK * 2 + 1
    for height in range(1, total + 1):
        coin = _coin(PARENT, PUZZLE, height)
        await store.new_block(uint32(height), uint64(height), [], [(coin.name(), coin, False)], [])
    assert await _puzzle_key_count(store) == 0

    # One chunk commits and returns the writer. The rest of the backlog stays unindexed.
    saved_applies = backend.applies
    assert await store.index_pending() is True
    assert backend.applies == saved_applies + 1
    async with store.db.writer() as session:
        raw = await session.get(CF_META, META_COIN_INDEXED)
    assert raw is not None
    assert int.from_bytes(raw, "big", signed=True) == COIN_INDEX_CHUNK
    assert await _puzzle_key_count(store) == COIN_INDEX_CHUNK

    # This is the write the next short-sync block needs. It must not wait on the remaining index.
    follower = _coin(PARENT, OTHER, 1)
    await store.new_block(uint32(total + 1), uint64(total + 1), [], [(follower.name(), follower, False)], [])
    assert await _puzzle_key_count(store) == COIN_INDEX_CHUNK
    hidden = await store.get_coin_records_by_puzzle_hash(True, OTHER)
    assert {record.name for record in hidden} == {follower.name()}

    # The remainder is more than one chunk, so catching up commits more than once.
    before_rest = backend.applies
    await store.index_pending_all()
    assert backend.applies == before_rest + 2
    assert await _indexed_height(store) == total + 1
    assert await _puzzle_key_count(store) == total + 1
    visible = await store.get_coin_records_by_puzzle_hash(False, PUZZLE)
    assert len(visible) == total


@pytest.mark.anyio
async def test_index_pending_follows_the_next_journal_across_a_height_gap() -> None:
    store = RocksCoinStore(ChainDB(MemoryBackend()))
    early = _coin(PARENT, PUZZLE, 1)
    late = _coin(PARENT, PUZZLE, 2)
    await store.new_block(uint32(1), uint64(1), [], [(early.name(), early, False)], [])
    await store.new_block(uint32(50), uint64(50), [], [(late.name(), late, False)], [])

    assert await store.index_pending(limit=1) is True
    assert await _indexed_height(store) == 1
    both = await store.get_coin_records_by_puzzle_hash(True, PUZZLE)
    assert {record.name for record in both} == {early.name(), late.name()}

    assert await store.index_pending(limit=1) is False
    assert await _indexed_height(store) == 50
    assert await _puzzle_key_count(store) == 2


@pytest.mark.anyio
async def test_get_coin_records_keeps_request_order_and_skips_missing() -> None:
    backend = _CountingBackend()
    store = RocksCoinStore(ChainDB(backend))
    first = _coin(PARENT, PUZZLE, 1)
    second = _coin(PARENT, PUZZLE, 2)
    await store.new_block(
        uint32(1),
        uint64(1),
        [],
        [(first.name(), first, False), (second.name(), second, False)],
        [],
    )
    backend.coin_lookups.clear()
    missing = bytes32(b"\x55" * 32)
    assert await store.get_coin_records([]) == []
    assert backend.coin_lookups == []

    records = await store.get_coin_records([second.name(), missing, first.name(), second.name()])
    assert [record.name for record in records] == [second.name(), first.name(), second.name()]
    assert [record.coin.amount for record in records] == [uint64(2), uint64(1), uint64(2)]
    assert len(backend.coin_lookups) == 1
    assert set(backend.coin_lookups[0]) == {bytes(second.name()), bytes(missing), bytes(first.name())}


@pytest.mark.anyio
async def test_index_pending_reads_every_spend_once_and_writes_puzzle_spent_keys() -> None:
    backend = _CountingBackend()
    store = RocksCoinStore(ChainDB(backend))
    first = _coin(PARENT, PUZZLE, 1)
    second = _coin(PARENT, OTHER, 2)
    third = _coin(bytes32(b"\x44" * 32), PUZZLE, 3)
    await store.new_block(
        uint32(1),
        uint64(1),
        [],
        [
            (first.name(), first, False),
            (second.name(), second, False),
            (third.name(), third, False),
        ],
        [],
    )
    await store.new_block(uint32(2), uint64(2), [], [], [second.name(), first.name()])
    await store.new_block(uint32(3), uint64(3), [], [], [third.name()])
    backend.coin_lookups.clear()

    assert await store.index_pending() is False
    assert len(backend.coin_lookups) == 1
    assert set(backend.coin_lookups[0]) == {bytes(first.name()), bytes(second.name()), bytes(third.name())}
    async with store.db.reader_no_transaction() as view:
        spent_keys = await view.scan(CF_COINS_BY_PUZZLE_SPENT, b"", None)
    assert {key[-32:] for key, _value in spent_keys} == {
        bytes(first.name()),
        bytes(second.name()),
        bytes(third.name()),
    }
