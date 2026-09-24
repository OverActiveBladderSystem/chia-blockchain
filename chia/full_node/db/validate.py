from __future__ import annotations

from pathlib import Path

from chia_rs.sized_bytes import bytes32

from chia.full_node.db.block_store import RocksBlockStore
from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.coin_store import RocksCoinStore
from chia.full_node.db.rocks import RocksBackend
from chia.full_node.db.startup import unfinished_migration_reason


async def validate_rocks(
    path: Path,
    *,
    genesis: bytes32 | None = None,
    validate_blocks: bool = False,
) -> tuple[bytes32, int, int]:
    """Walk the main chain from the peak down to height 0. Return peak hash, height, and unspent count.

    `validate_blocks` also parses every stored full block and block record.
    """
    if not path.is_dir():
        raise RuntimeError(f"RocksDB database is not a directory: {path}")
    reason = await unfinished_migration_reason(path)
    if reason is not None:
        raise RuntimeError(reason)
    chain = ChainDB(RocksBackend(path, sync="OFF"))
    try:
        block_store = await RocksBlockStore.create(chain, use_cache=False)
        coin_store = await RocksCoinStore.create(chain)
        peak = await block_store.get_peak()
        if peak is None:
            raise RuntimeError(f"{path} has no peak")
        expect = peak[0]
        height = int(peak[1])
        while height >= 0:
            found = await block_store.main_chain_hash_at(height)
            if found != expect:
                raise RuntimeError(f"height {height} has {found}, expected {expect}")
            expect = await block_store.get_prev_hash(found)
            height -= 1
        if genesis is not None and expect != genesis:
            raise RuntimeError(f"genesis prev hash {expect} does not match {genesis}")
        if validate_blocks:
            await _validate_block_bytes(block_store)
        unspent = await coin_store.num_unspent()
        return peak[0], int(peak[1]), unspent
    finally:
        await chain.close()


async def _validate_block_bytes(block_store: RocksBlockStore) -> None:
    for header_hash, meta in await block_store.stored_block_metas():
        try:
            block = await block_store.get_full_block(header_hash)
        except Exception as exc:
            raise RuntimeError(f"Block {header_hash.hex()} blob could not be parsed") from exc
        if block is None:
            raise RuntimeError(f"Block {header_hash.hex()} is missing its blob")
        try:
            record = await block_store.get_block_record(header_hash)
        except Exception as exc:
            raise RuntimeError(f"Block {header_hash.hex()} block record could not be parsed") from exc
        if record is None:
            raise RuntimeError(f"Block {header_hash.hex()} is missing its block record")
        if block.header_hash != header_hash:
            raise RuntimeError(
                f"Block {header_hash.hex()} has a blob with mismatching hash: {block.header_hash.hex()}"
            )
        if record.header_hash != header_hash:
            raise RuntimeError(
                f"Block {header_hash.hex()} has a block record with mismatching hash: {record.header_hash.hex()}"
            )
        if record.total_iters != block.total_iters:
            raise RuntimeError(
                f"Block {header_hash.hex()} has a block record with mismatching total "
                f"iters: {record.total_iters} expected {block.total_iters}"
            )
        if record.prev_hash != block.prev_header_hash or meta.prev_hash != block.prev_header_hash:
            raise RuntimeError(f"Block {header_hash.hex()} has a mismatching prev hash")
        if block.height != meta.height:
            raise RuntimeError(
                f"Block {header_hash.hex()} has a mismatching height: {block.height} expected {meta.height}"
            )
        main_hash = await block_store.main_chain_hash_at(meta.height)
        if meta.in_main_chain != (main_hash == header_hash):
            raise RuntimeError(
                f"block {header_hash.hex()} (height: {meta.height}) has in_main_chain={meta.in_main_chain}"
            )


async def backup_rocks(source: Path, destination: Path) -> None:
    """Write a RocksDB checkpoint. The source stays in place."""
    if not source.is_dir() or not (source / "CURRENT").exists():
        raise RuntimeError(f"RocksDB database was not found at {source}")
    if destination.exists():
        raise RuntimeError(f"backup destination already exists: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    chain = ChainDB(RocksBackend(source, sync="OFF"))
    try:
        await chain.checkpoint(destination)
    finally:
        await chain.close()
