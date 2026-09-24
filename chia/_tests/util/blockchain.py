from __future__ import annotations

import contextlib
import os
import pickle  # ruff: ignore[suspicious-pickle-import]  # TODO: use explicit serialization instead of pickle
import tempfile
from collections.abc import AsyncIterator
from pathlib import Path

from chia_rs import ConsensusConstants, FullBlock
from chia_rs.sized_ints import uint64
from filelock import FileLock

from chia._tests.util.db_connection import DBConnection
from chia.consensus.block_height_map import BlockHeightMap
from chia.consensus.blockchain import Blockchain
from chia.full_node.block_store import BlockStore
from chia.full_node.coin_store import CoinStore
from chia.full_node.db.block_store import RocksBlockStore
from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.coin_store import RocksCoinStore
from chia.full_node.db.hint_store import RocksHintStore
from chia.full_node.db.rocks import RocksBackend
from chia.full_node.hint_store import HintStore
from chia.simulator.block_tools import BlockTools
from chia.util.db_wrapper import DBWrapper2, generate_in_memory_db_uri
from chia.util.default_root import DEFAULT_ROOT_PATH
from chia.util.inline_executor import InlineExecutor


@contextlib.asynccontextmanager
async def create_blockchain(
    constants: ConsensusConstants,
    db_version: int,
    *,
    engine: str = "sqlite",
    root_path: Path | None = None,
) -> AsyncIterator[tuple[Blockchain, DBWrapper2 | None]]:
    """Open a Blockchain. `engine="rocksdb"` uses the full-node RocksDB stores."""
    if engine == "sqlite":
        db_uri = generate_in_memory_db_uri()
        async with DBWrapper2.managed(database=db_uri, uri=True, reader_count=1, db_version=db_version) as wrapper:
            coin_store = await CoinStore.create(wrapper)
            store = await BlockStore.create(wrapper)
            path = root_path if root_path is not None else Path(".")
            height_map = await BlockHeightMap.create(path, wrapper)
            bc1 = await Blockchain.create(coin_store, store, height_map, constants, InlineExecutor(), log_coins=True)
            try:
                assert bc1.get_peak() is None
                yield bc1, wrapper
            finally:
                bc1.shut_down()
        return
    if engine != "rocksdb":
        raise ValueError(f"unknown chain database engine {engine}")
    if db_version != 2:
        raise RuntimeError("RocksDB full-node stores require database version 2")

    owns_directory = root_path is None
    directory_context = tempfile.TemporaryDirectory() if owns_directory else contextlib.nullcontext(str(root_path))
    with directory_context as directory_name:
        directory = Path(directory_name)
        chain = ChainDB(RocksBackend(directory / "blockchain_v2_unit.rocksdb", sync="OFF"))
        try:
            coin_store = await RocksCoinStore.create(chain)
            store = await RocksBlockStore.create(chain)
            height_map = await BlockHeightMap.create_for_rocks(directory, store)
            bc1 = await Blockchain.create(coin_store, store, height_map, constants, InlineExecutor(), log_coins=True)
            try:
                assert bc1.get_peak() is None
                yield bc1, None
            finally:
                bc1.shut_down()
        finally:
            await chain.close()


@contextlib.asynccontextmanager
async def open_v2_stores(
    root_path: Path,
    db_version: int,
    *,
    engine: str = "sqlite",
    use_cache: bool = True,
) -> AsyncIterator[tuple[CoinStore | RocksCoinStore, BlockStore | RocksBlockStore, BlockHeightMap]]:
    """Coin store, block store, and height map for one engine. SQLite stays in memory."""
    root_path.mkdir(parents=True, exist_ok=True)
    if engine == "sqlite":
        async with DBConnection(db_version) as wrapper:
            coin_store = await CoinStore.create(wrapper)
            block_store = await BlockStore.create(wrapper, use_cache=use_cache)
            height_map = await BlockHeightMap.create(root_path, wrapper)
            yield coin_store, block_store, height_map
        return
    if engine != "rocksdb":
        raise ValueError(f"unknown chain database engine {engine}")
    if db_version != 2:
        raise RuntimeError("RocksDB full-node stores require database version 2")
    chain = ChainDB(RocksBackend(root_path / "blockchain_v2_unit.rocksdb", sync="OFF"))
    try:
        coin_store = await RocksCoinStore.create(chain)
        block_store = await RocksBlockStore.create(chain, use_cache=use_cache)
        height_map = await BlockHeightMap.create_for_rocks(root_path, block_store)
        yield coin_store, block_store, height_map
    finally:
        await chain.close()


@contextlib.asynccontextmanager
async def open_hint_store(
    root_path: Path,
    db_version: int,
    *,
    engine: str,
) -> AsyncIterator[HintStore | RocksHintStore]:
    if engine == "sqlite":
        async with DBConnection(db_version) as wrapper:
            yield await HintStore.create(wrapper)
        return
    if engine != "rocksdb":
        raise ValueError(f"unknown chain database engine {engine}")
    if db_version != 2:
        raise RuntimeError("RocksDB full-node stores require database version 2")
    chain = ChainDB(RocksBackend(root_path / "hints.rocksdb", sync="OFF"))
    try:
        yield await RocksHintStore.create(chain)
    finally:
        await chain.close()


def persistent_blocks(
    num_of_blocks: int,
    db_name: str,
    bt: BlockTools,
    seed: bytes = b"",
    empty_sub_slots: int = 0,
    *,
    normalized_to_identity_cc_eos: bool = False,
    normalized_to_identity_icc_eos: bool = False,
    normalized_to_identity_cc_sp: bool = False,
    normalized_to_identity_cc_ip: bool = False,
    block_list_input: list[FullBlock] | None = None,
    time_per_block: float | None = None,
    dummy_block_references: bool = False,
    include_transactions: bool = False,
) -> list[FullBlock]:
    # try loading from disc, if not create new blocks.db file
    # TODO hash fixtures.py and blocktool.py, add to path, delete if the files changed
    if block_list_input is None:
        block_list_input = []
    block_path_dir = DEFAULT_ROOT_PATH.parent.joinpath("blocks")
    file_path = block_path_dir.joinpath(db_name)
    lock_file_path = block_path_dir / (db_name + ".lockfile")

    ci = os.environ.get("CI")
    if ci is not None and not file_path.exists():
        raise Exception(f"Running in CI and expected path not found: {file_path!r}")

    block_path_dir.mkdir(parents=True, exist_ok=True)

    with FileLock(lock_file_path):
        if file_path.exists():
            print(f"File found at: {file_path}")
            try:
                bytes_list = file_path.read_bytes()
                # TODO: use explicit serialization instead of pickle
                block_bytes_list: list[bytes] = pickle.loads(bytes_list)  # ruff: ignore[suspicious-pickle-usage]
                blocks: list[FullBlock] = []
                for block_bytes in block_bytes_list:
                    blocks.append(FullBlock.from_bytes_unchecked(block_bytes))
                if len(blocks) == num_of_blocks + len(block_list_input):
                    print(f"\n loaded {file_path} with {len(blocks)} blocks")

                    return blocks
            except EOFError:
                print("\n error reading db file")
        else:
            print(f"File not found at: {file_path}")

        print("Creating a new test db")
        return new_test_db(
            file_path,
            num_of_blocks,
            seed,
            empty_sub_slots,
            bt,
            block_list_input,
            time_per_block,
            normalized_to_identity_cc_eos=normalized_to_identity_cc_eos,
            normalized_to_identity_icc_eos=normalized_to_identity_icc_eos,
            normalized_to_identity_cc_sp=normalized_to_identity_cc_sp,
            normalized_to_identity_cc_ip=normalized_to_identity_cc_ip,
            dummy_block_references=dummy_block_references,
            include_transactions=include_transactions,
        )


def new_test_db(
    path: Path,
    num_of_blocks: int,
    seed: bytes,
    empty_sub_slots: int,
    bt: BlockTools,
    block_list_input: list[FullBlock],
    time_per_block: float | None,
    *,
    normalized_to_identity_cc_eos: bool = False,  # CC_EOS,
    normalized_to_identity_icc_eos: bool = False,  # ICC_EOS
    normalized_to_identity_cc_sp: bool = False,  # CC_SP,
    normalized_to_identity_cc_ip: bool = False,  # CC_IP
    dummy_block_references: bool = False,
    include_transactions: bool = False,
) -> list[FullBlock]:
    print(f"create {path} with {num_of_blocks} blocks with ")
    blocks: list[FullBlock] = bt.get_consecutive_blocks(
        num_of_blocks,
        block_list_input=block_list_input,
        time_per_block=time_per_block,
        seed=seed,
        skip_slots=empty_sub_slots,
        normalized_to_identity_cc_eos=normalized_to_identity_cc_eos,
        normalized_to_identity_icc_eos=normalized_to_identity_icc_eos,
        normalized_to_identity_cc_sp=normalized_to_identity_cc_sp,
        normalized_to_identity_cc_ip=normalized_to_identity_cc_ip,
        dummy_block_references=dummy_block_references,
        include_transactions=include_transactions,
        genesis_timestamp=uint64(1234567890),
    )
    block_bytes_list: list[bytes] = []
    for block in blocks:
        block_bytes_list.append(bytes(block))
    bytes_fn = pickle.dumps(block_bytes_list)
    path.write_bytes(bytes_fn)
    return blocks
