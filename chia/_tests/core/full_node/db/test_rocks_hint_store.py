from __future__ import annotations

import pytest
from chia_rs.sized_bytes import bytes32

from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.hint_store import RocksHintStore
from chia.full_node.db.memory import MemoryBackend


@pytest.mark.anyio
async def test_add_and_lookup_hints() -> None:
    store = await RocksHintStore.create(ChainDB(MemoryBackend()))
    hint_0 = 32 * b"\x00"
    hint_1 = 32 * b"\x01"
    coin_0 = bytes32(32 * b"\x04")
    coin_1 = bytes32(32 * b"\x05")
    coin_2 = bytes32(32 * b"\x06")

    await store.add_hints([(coin_0, hint_0), (coin_1, hint_0), (coin_2, hint_1)])
    await store.add_hints([(coin_0, hint_0)])

    assert set(await store.get_coin_ids(hint_0)) == {coin_0, coin_1}
    assert await store.get_coin_ids(hint_1) == [coin_2]
    assert await store.get_coin_ids(32 * b"\x03") == []
    assert await store.count_hints() == 3
    assert set(await store.get_hints([coin_0, coin_1])) == {bytes32(hint_0)}


@pytest.mark.anyio
async def test_one_coin_can_have_two_hints() -> None:
    store = await RocksHintStore.create(ChainDB(MemoryBackend()))
    hint_0 = 32 * b"\x00"
    hint_1 = 32 * b"\x01"
    coin_0 = bytes32(32 * b"\x04")
    await store.add_hints([(coin_0, hint_0), (coin_0, hint_1)])
    assert await store.get_coin_ids(hint_0) == [coin_0]
    assert await store.get_coin_ids(hint_1) == [coin_0]
    assert set(await store.get_hints([coin_0])) == {bytes32(hint_0), bytes32(hint_1)}
    assert await store.count_hints() == 2
