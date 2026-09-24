from __future__ import annotations

import asyncio
import random
import shutil
import sys
import tempfile
from pathlib import Path
from time import monotonic

from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32, uint64

from chia._tests.util.benchmarks import rand_hash, rewards
from chia.full_node.coin_store import CoinStore
from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.coin_store import RocksCoinStore
from chia.full_node.db.rocks import RocksBackend
from chia.types.blockchain_format.coin import Coin
from chia.util.db_wrapper import DBWrapper2

# One full block: 4000 additions and a handful of spends.
# python -m benchmarks.full_block_coins

ADDITIONS = 4000
REMOVALS = 8
# Time one full block after the coin set reaches each of these sizes.
MILESTONES = (0, 100_000, 400_000, 1_000_000)
random.seed(123456789)


def make_coins(count: int) -> list[tuple[bytes32, Coin, bool]]:
    additions: list[tuple[bytes32, Coin, bool]] = []
    for _ in range(count):
        coin = Coin(rand_hash(), rand_hash(), uint64(1))
        additions.append((coin.name(), coin, False))
    return additions


async def apply_block(
    store: CoinStore | RocksCoinStore,
    height: int,
    additions: list[tuple[bytes32, Coin, bool]],
    removals: list[bytes32],
) -> float:
    farmer, pool = rewards(uint32(height))
    started = monotonic()
    await store.new_block(uint32(height), uint64(1_700_000_000 + height), [farmer, pool], additions, removals)
    return monotonic() - started


async def run_curve(
    store: CoinStore | RocksCoinStore,
    batches: list[list[tuple[bytes32, Coin, bool]]],
) -> list[tuple[int, float, int]]:
    """Apply batches. Time the block that starts at each milestone. Return (coins_before, seconds, unspent)."""
    unspent_ids: list[bytes32] = []
    height = 1
    coins_before = 0
    timed: list[tuple[int, float, int]] = []
    milestones = list(MILESTONES)
    batch_index = 0
    while milestones and batch_index < len(batches):
        while batch_index < len(batches) and coins_before < milestones[0]:
            await apply_block(store, height, batches[batch_index], [])
            unspent_ids.extend(coin_id for coin_id, _, _ in batches[batch_index])
            coins_before += len(batches[batch_index]) + 2
            height += 1
            batch_index += 1
        if batch_index >= len(batches):
            break
        mark = milestones.pop(0)
        removals = unspent_ids[:REMOVALS] if mark > 0 else []
        elapsed = await apply_block(store, height, batches[batch_index], removals)
        timed.append((coins_before, elapsed, await store.num_unspent()))
        print(f"    timed block after {coins_before:,} coins in {elapsed:.3f}s", flush=True)
        if removals:
            del unspent_ids[:REMOVALS]
        unspent_ids.extend(coin_id for coin_id, _, _ in batches[batch_index])
        coins_before += len(batches[batch_index]) + 2
        height += 1
        batch_index += 1
    return timed


def batches_for_curve() -> list[list[tuple[bytes32, Coin, bool]]]:
    # One extra block past the last milestone so the timed block is the one that crosses it.
    blocks = (max(MILESTONES) // ADDITIONS) + len(MILESTONES) + 2
    return [make_coins(ADDITIONS) for _ in range(blocks)]


async def time_sqlite(root: Path, batches: list[list[tuple[bytes32, Coin, bool]]]) -> list[tuple[int, float, int]]:
    path = root / "coin-store.sqlite"
    async with DBWrapper2.managed(
        database=path,
        db_version=2,
        reader_count=1,
        journal_mode="wal",
        synchronous="full",
    ) as wrapper:
        store = await CoinStore.create(wrapper)
        return await run_curve(store, batches)


async def time_rocks(root: Path, batches: list[list[tuple[bytes32, Coin, bool]]]) -> list[tuple[int, float, int]]:
    path = root / "coin-store.rocksdb"
    chain = ChainDB(RocksBackend(path, sync="FULL"))
    try:
        store = await RocksCoinStore.create(chain)
        return await run_curve(store, batches)
    finally:
        await chain.close()


def print_curve(label: str, rows: list[tuple[int, float, int]]) -> None:
    print(label, flush=True)
    for coins_before, seconds, unspent in rows:
        print(f"  after {coins_before:>9,} coins  new_block {seconds:.3f}s  unspent {unspent:,}", flush=True)


# Timed after the database already holds this many coins, so the comparison is not an empty file.
SHAPE_PREFILL = 1_000_000
SHAPES = ((4000, 8), (2000, 2000), (8, 4000))


async def run_shapes(
    store: CoinStore | RocksCoinStore,
    prefill: list[list[tuple[bytes32, Coin, bool]]],
    shape_additions: list[list[tuple[bytes32, Coin, bool]]],
) -> list[tuple[int, int, float, int]]:
    unspent_ids: list[bytes32] = []
    height = 1
    for batch in prefill:
        await apply_block(store, height, batch, [])
        unspent_ids.extend(coin_id for coin_id, _, _ in batch)
        height += 1
    timed: list[tuple[int, int, float, int]] = []
    for (add_count, remove_count), additions in zip(SHAPES, shape_additions, strict=True):
        removals = unspent_ids[:remove_count]
        del unspent_ids[:remove_count]
        elapsed = await apply_block(store, height, additions, removals)
        unspent = await store.num_unspent()
        timed.append((add_count, remove_count, elapsed, unspent))
        print(
            f"    {add_count:>5,} additions  {remove_count:>5,} removals  {elapsed:.3f}s  unspent {unspent:,}",
            flush=True,
        )
        unspent_ids.extend(coin_id for coin_id, _, _ in additions)
        height += 1
    return timed


async def shapes_sqlite(root: Path, prefill: list, shape_additions: list) -> list[tuple[int, int, float, int]]:
    async with DBWrapper2.managed(
        database=root / "shapes.sqlite",
        db_version=2,
        reader_count=1,
        journal_mode="wal",
        synchronous="full",
    ) as wrapper:
        return await run_shapes(await CoinStore.create(wrapper), prefill, shape_additions)


async def shapes_rocks(root: Path, prefill: list, shape_additions: list) -> list[tuple[int, int, float, int]]:
    chain = ChainDB(RocksBackend(root / "shapes.rocksdb", sync="FULL"))
    try:
        return await run_shapes(await RocksCoinStore.create(chain), prefill, shape_additions)
    finally:
        await chain.close()


async def main_shapes() -> None:
    print(f"Prefilling {SHAPE_PREFILL:,} coins, then timing three block shapes", flush=True)
    prefill_blocks = (SHAPE_PREFILL // ADDITIONS) + 1
    prefill = [make_coins(ADDITIONS) for _ in range(prefill_blocks)]
    shape_additions = [make_coins(add_count) for add_count, _remove_count in SHAPES]
    root = Path(tempfile.mkdtemp(prefix="chia-shapes-"))
    try:
        print("SQLite, synchronous=full", flush=True)
        sqlite_rows = await shapes_sqlite(root, prefill, shape_additions)
        print("RocksDB, WAL fsync per block", flush=True)
        rocks_rows = await shapes_rocks(root, prefill, shape_additions)
        if [row[3] for row in sqlite_rows] != [row[3] for row in rocks_rows]:
            raise SystemExit(f"unspent counts differ: sqlite {sqlite_rows} rocks {rocks_rows}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


async def main() -> None:
    print(
        f"Building up to {max(MILESTONES):,} coins, timing a {ADDITIONS}-coin block at {MILESTONES}",
        flush=True,
    )
    batches = batches_for_curve()
    root = Path(tempfile.mkdtemp(prefix="chia-full-block-"))
    try:
        print("SQLite, synchronous=full", flush=True)
        sqlite_rows = await time_sqlite(root, batches)
        print_curve("SQLite results", sqlite_rows)
        print("RocksDB, WAL fsync per block", flush=True)
        rocks_rows = await time_rocks(root, batches)
        print_curve("RocksDB results", rocks_rows)
        if [row[2] for row in sqlite_rows] != [row[2] for row in rocks_rows]:
            raise SystemExit(f"unspent counts differ: sqlite {sqlite_rows} rocks {rocks_rows}")
    finally:
        shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    asyncio.run(main_shapes() if "--shapes" in sys.argv else main())
