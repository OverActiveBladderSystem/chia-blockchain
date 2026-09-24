from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.keys import ROCKS_SUFFIX, SQLITE_SUFFIX
from chia.full_node.db.memory import MemoryBackend
from chia.full_node.db.path import choose_full_node_database
from chia.util.task_referencer import create_referenced_task


@pytest.fixture
def chain_db() -> ChainDB:
    return ChainDB(MemoryBackend())


@pytest.mark.anyio
async def test_old_diagnostic_logs_are_capped(tmp_path: Path) -> None:
    from chia.full_node.db.rocks import RocksBackend

    path = tmp_path / "chain.rocksdb"
    chain = ChainDB(RocksBackend(path, sync="OFF"))
    await chain.close()
    for index in range(10):
        (path / f"LOG.old.{index}").write_bytes(b"x" * 64)
    chain = ChainDB(RocksBackend(path, sync="OFF"))
    await chain.close()
    assert len(list(path.glob("LOG.old.*"))) <= 7


@pytest.mark.anyio
async def test_commit_is_visible_to_a_later_reader(chain_db: ChainDB) -> None:
    async with chain_db.writer() as session:
        session.put("meta", b"a", b"1")
    async with chain_db.reader_no_transaction() as view:
        assert await view.get("meta", b"a") == b"1"


@pytest.mark.anyio
async def test_uncommitted_write_is_hidden_from_other_tasks(chain_db: ChainDB) -> None:
    started = asyncio.Event()
    release = asyncio.Event()

    async def writer() -> None:
        async with chain_db.writer() as session:
            session.put("meta", b"a", b"1")
            started.set()
            await release.wait()

    task = create_referenced_task(writer())
    await started.wait()
    async with chain_db.reader_no_transaction() as view:
        assert await view.get("meta", b"a") is None
    release.set()
    await task
    async with chain_db.reader_no_transaction() as view:
        assert await view.get("meta", b"a") == b"1"


@pytest.mark.anyio
async def test_same_task_reads_its_own_writes(chain_db: ChainDB) -> None:
    async with chain_db.writer() as session:
        session.put("coins", b"aa", b"one")
        async with chain_db.reader_no_transaction() as view:
            assert await view.get("coins", b"aa") == b"one"
        async with chain_db.writer_maybe_transaction() as joined:
            joined.put("coins", b"bb", b"two")
            assert await joined.get("coins", b"bb") == b"two"


@pytest.mark.anyio
async def test_nested_savepoint_rolls_back_without_losing_the_outer_put(chain_db: ChainDB) -> None:
    async with chain_db.writer() as session:
        session.put("meta", b"keep", b"yes")
        try:
            async with chain_db.writer() as inner:
                inner.put("meta", b"drop", b"no")
                raise RuntimeError("inner")
        except RuntimeError:
            assert await session.get("meta", b"drop") is None
            assert await session.get("meta", b"keep") == b"yes"
    async with chain_db.reader_no_transaction() as view:
        assert await view.get("meta", b"keep") == b"yes"
        assert await view.get("meta", b"drop") is None


@pytest.mark.anyio
async def test_outer_exception_discards_the_batch(chain_db: ChainDB) -> None:
    with pytest.raises(RuntimeError):
        async with chain_db.writer() as session:
            session.put("meta", b"a", b"1")
            raise RuntimeError("fail")
    async with chain_db.reader_no_transaction() as view:
        assert await view.get("meta", b"a") is None


@pytest.mark.anyio
async def test_delete_range_then_put_is_visible_in_scan(chain_db: ChainDB) -> None:
    async with chain_db.writer() as session:
        session.put("coins", b"\x01a", b"old")
        session.put("coins", b"\x01b", b"old")
        session.put("coins", b"\x02c", b"stay")
    async with chain_db.writer() as session:
        session.delete_range("coins", b"\x01", b"\x02")
        session.put("coins", b"\x01z", b"new")
        rows = await session.scan("coins", b"", None)
    assert rows == [(b"\x01z", b"new"), (b"\x02c", b"stay")]


def test_existing_sqlite_file_stays_sqlite(tmp_path: Path) -> None:
    db = tmp_path / "db"
    db.mkdir()
    sqlite_file = db / "blockchain_v2_mainnet.sqlite"
    sqlite_file.write_bytes(b"")
    choice = choose_full_node_database(tmp_path, f"db/blockchain_v2_CHALLENGE{SQLITE_SUFFIX}", "mainnet")
    assert choice.engine == "sqlite"
    assert choice.path == sqlite_file.resolve()
    assert choice.config_database_path is None
    assert choice.suggest_migrate is True


def test_missing_sqlite_with_no_siblings_rewrites_to_rocksdb(tmp_path: Path) -> None:
    choice = choose_full_node_database(tmp_path, f"db/blockchain_v2_CHALLENGE{SQLITE_SUFFIX}", "mainnet")
    assert choice.engine == "rocksdb"
    assert choice.path.name == f"blockchain_v2_mainnet{ROCKS_SUFFIX}"
    assert choice.config_database_path == f"db/blockchain_v2_CHALLENGE{ROCKS_SUFFIX}"


def test_missing_network_sqlite_does_not_redirect_other_networks(tmp_path: Path) -> None:
    db = tmp_path / "db"
    db.mkdir()
    (db / "blockchain_v2_mainnet.sqlite").write_bytes(b"")
    choice = choose_full_node_database(tmp_path, f"db/blockchain_v2_CHALLENGE{SQLITE_SUFFIX}", "testnet11")
    assert choice.engine == "rocksdb"
    assert choice.path.name == f"blockchain_v2_testnet11{ROCKS_SUFFIX}"
    assert choice.config_database_path is None


def test_rocksdb_suffix_is_used_even_when_the_directory_is_absent(tmp_path: Path) -> None:
    choice = choose_full_node_database(tmp_path, f"db/blockchain_v2_CHALLENGE{ROCKS_SUFFIX}", "mainnet")
    assert choice.engine == "rocksdb"
    assert choice.path.name == f"blockchain_v2_mainnet{ROCKS_SUFFIX}"
    assert not choice.path.exists()


def test_unknown_suffix_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must end in"):
        choose_full_node_database(tmp_path, "db/blockchain_v2_CHALLENGE.db", "mainnet")
