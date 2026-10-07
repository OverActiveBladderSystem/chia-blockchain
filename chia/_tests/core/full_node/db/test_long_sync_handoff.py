from __future__ import annotations

import logging
from typing import Any

import pytest
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32, uint64

from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.coin_store import COIN_INDEX_CHUNK, RocksCoinStore
from chia.full_node.db.keys import CF_COINS_BY_PUZZLE_CONFIRMED, CF_META, META_COIN_INDEXED
from chia.full_node.db.memory import MemoryBackend
from chia.full_node.db.ops import Op, Put
from chia.full_node.full_node import FullNode
from chia.full_node.sync_store import SyncStore
from chia.types.blockchain_format.coin import Coin

PARENT = bytes32(b"\x11" * 32)
PUZZLE = bytes32(b"\x22" * 32)

# One index commit, then the sync-mode notification. Each entry is the event name
# plus whether sync mode and long sync were still on when it happened.
_Trace = list[tuple[str, bool, bool]]


class _IndexOrderBackend(MemoryBackend):
    """Records each lookup-index commit against the sync flags at that moment."""

    def __init__(self) -> None:
        super().__init__()
        self.trace: _Trace = []
        self.sync_store: SyncStore | None = None
        self.record = False

    async def apply(self, ops: list[Op]) -> None:
        sync_store = self.sync_store
        if self.record and sync_store is not None and _writes_coin_index(ops):
            self.trace.append(("index", sync_store.get_sync_mode(), sync_store.get_long_sync()))
        await super().apply(ops)


class _SqliteCoinStore:
    """Stand-in for the SQLite coin store, which has no lookup-key backlog."""


def _writes_coin_index(ops: list[Op]) -> bool:
    return any(isinstance(op, Put) and op.cf == CF_META and op.key == META_COIN_INDEXED for op in ops)


def _node_at_sync_handoff(coin_store: object, sync_store: SyncStore, trace: _Trace) -> FullNode:
    node = FullNode.__new__(FullNode)
    node.log = logging.getLogger(__name__)
    node._coin_store = coin_store  # type: ignore[assignment]
    node.sync_store = sync_store
    node._server = None

    def state_changed(change: str, change_data: dict[str, Any] | None) -> None:
        trace.append((change, sync_store.get_sync_mode(), sync_store.get_long_sync()))

    node.state_changed_callback = state_changed
    return node


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
async def test_long_sync_finishes_lookup_keys_before_leaving_sync_mode() -> None:
    """A backlog larger than one chunk is fully indexed while sync mode is still on.

    Short sync uses the same chain writer. Leaving sync mode first lets the next block
    wait behind the rest of the backlog, which is the stall this order exists to prevent.
    Server is left unset so _finish_sync returns at the handoff and cannot hide the index
    write on the far side of that return.
    """
    backend = _IndexOrderBackend()
    store = RocksCoinStore(ChainDB(backend))
    total = COIN_INDEX_CHUNK + 1
    for height in range(1, total + 1):
        coin = Coin(PARENT, PUZZLE, uint64(height))
        await store.new_block(uint32(height), uint64(height), [], [(coin.name(), coin, False)], [])
    assert await _puzzle_key_count(store) == 0

    sync_store = SyncStore()
    sync_store.set_sync_mode(True)
    sync_store.set_long_sync(True)
    backend.sync_store = sync_store
    backend.record = True
    node = _node_at_sync_handoff(store, sync_store, backend.trace)

    await FullNode._finish_sync(node, None)

    assert backend.trace[:-1]
    assert set(backend.trace[:-1]) == {("index", True, True)}
    assert backend.trace[-1] == ("sync_mode", False, False)
    assert sync_store.get_sync_mode() is False
    assert sync_store.get_long_sync() is False
    assert await _indexed_height(store) == total
    assert await _puzzle_key_count(store) == total


@pytest.mark.anyio
async def test_long_sync_leaves_sync_mode_without_a_rocks_coin_store() -> None:
    """SQLite has no deferred lookup keys, and long sync still turns sync mode off."""
    sync_store = SyncStore()
    sync_store.set_sync_mode(True)
    sync_store.set_long_sync(True)
    trace: _Trace = []
    node = _node_at_sync_handoff(_SqliteCoinStore(), sync_store, trace)

    await FullNode._finish_sync(node, None)

    assert trace == [("sync_mode", False, False)]
    assert sync_store.get_sync_mode() is False
    assert sync_store.get_long_sync() is False
