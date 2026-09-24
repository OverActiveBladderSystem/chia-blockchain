from __future__ import annotations

import asyncio
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import pytest
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32, uint64

from chia._tests.util.benchmarks import rewards
from chia.consensus.block_height_map import BlockHeightMap
from chia.consensus.default_constants import DEFAULT_CONSTANTS
from chia.full_node.block_store import BlockStore
from chia.full_node.coin_store import CoinStore
from chia.full_node.db import migrate as migrate_module
from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.coin_store import RocksCoinStore
from chia.full_node.db.keys import CF_SES
from chia.full_node.db.migrate import (
    MigrationPaused,
    abort_migration,
    migrate_database,
    offered_database_path,
    require_destination_space,
)
from chia.full_node.db.rocks import RocksBackend
from chia.full_node.db.startup import _sqlite_path_to_restore, unfinished_migration_reason
from chia.full_node.db.validate import backup_rocks, validate_rocks
from chia.full_node.hint_store import HintStore
from chia.types.blockchain_format.coin import Coin
from chia.util.db_wrapper import DBWrapper2


@pytest.mark.anyio
async def test_quiet_batch_redraws_the_progress_line() -> None:
    paints: list[float] = []
    stop = asyncio.Event()
    update = migrate_module.MigrateProgress("blocks", 1, 10, "rows 1 / 10", work_done=1, work_total=10)

    async def stop_soon() -> None:
        await asyncio.sleep(0.2)
        stop.set()

    await asyncio.gather(
        migrate_module.replay_progress(
            stop,
            lambda: update,
            lambda _update, now: paints.append(now),
            interval=0.05,
        ),
        stop_soon(),
    )
    assert len(paints) >= 2


def test_progress_line_stays_narrower_than_the_terminal() -> None:
    wide = "x" * 200
    fitted = migrate_module.fit_terminal_line(wide, 80)
    assert len(fitted) == 79
    assert fitted == "x" * 79
    short = migrate_module.fit_terminal_line("blocks 10", 20)
    assert len(short) == 19
    assert short.startswith("blocks 10")
    assert short.endswith(" ")


def test_progress_line_shows_height_rate_and_eta() -> None:
    progress = migrate_module.MigrateProgress(
        "blocks",
        1_204_331,
        9_298_314,
        "rows 1,300,000 / 9,500,000",
        work_done=1_300_000,
        work_total=9_500_000,
    )
    line = migrate_module.format_progress(progress, elapsed_seconds=(42 * 60) + 11, done_this_run=130_000)
    assert "[1/9]" in line
    assert "1,204,331 / 9,298,314" in line
    assert "13.0%" in line
    assert "elapsed 00:42:11" in line
    assert "rows/s" in line
    assert "ETA" in line
    resumed = migrate_module.format_progress(progress, elapsed_seconds=10, done_this_run=0)
    assert "ETA" not in resumed
    clock = migrate_module.ProgressClock(started=0, phase_started=0)
    first = migrate_module.MigrateProgress("blocks", 1, 10, work_done=500, work_total=1_000)
    assert "ETA" not in clock.line(first, now=0)
    second = migrate_module.MigrateProgress("blocks", 2, 10, work_done=600, work_total=1_000)
    assert "ETA 40s" in clock.line(second, now=10)
    coins = migrate_module.MigrateProgress("coins", 50, 200, work_done=50, work_total=200)
    assert "ETA" not in clock.line(coins, now=1_000)
    coins_later = migrate_module.MigrateProgress("coins", 100, 200, work_done=100, work_total=200)
    caught_up = clock.line(coins_later, now=1_010)
    assert "[3/9]" in caught_up
    assert "ETA 20s" in caught_up
    assert "elapsed 00:16:50" in caught_up


def test_restore_path_uses_the_sqlite_file_the_copy_started_from() -> None:
    rocks = Path("D:/copy/blockchain_v2_mainnet.rocksdb")
    source = Path("C:/chia/db/blockchain_v2_mainnet.sqlite")
    assert _sqlite_path_to_restore(rocks, str(source).encode()) == source
    assert _sqlite_path_to_restore(rocks, None) == Path("D:/copy/blockchain_v2_mainnet.sqlite")


def test_offered_path_keeps_challenge_token(tmp_path: Path) -> None:
    offered = offered_database_path(
        "db/blockchain_v2_CHALLENGE.sqlite",
        tmp_path,
        tmp_path / "blockchain_v2_mainnet.rocksdb",
    )
    assert offered.endswith("blockchain_v2_CHALLENGE.rocksdb")


def _write_sqlite(path: Path) -> tuple[bytes32, int]:
    """A two-block chain with three coin rows. Block blobs are not parsed when the peaks already match."""
    genesis = bytes32(b"\x01" * 32)
    peak = bytes32(b"\x02" * 32)
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE full_blocks("
            "header_hash blob PRIMARY KEY, prev_hash blob, height bigint, sub_epoch_summary blob,"
            "is_fully_compactified tinyint, in_main_chain tinyint, block blob, block_record blob)"
        )
        db.execute("CREATE TABLE current_peak(key int PRIMARY KEY, hash blob)")
        db.execute(
            "CREATE TABLE coin_record("
            "coin_name blob PRIMARY KEY, confirmed_index bigint, spent_index bigint, coinbase int,"
            "puzzle_hash blob, coin_parent blob, amount blob, timestamp bigint)"
        )
        db.execute("CREATE TABLE hints(coin_id blob, hint blob, UNIQUE (coin_id, hint))")
        db.execute("CREATE TABLE sub_epoch_segments_v3(ses_block_hash blob PRIMARY KEY, challenge_segments blob)")
        db.execute("CREATE TABLE database_version(version int)")
        db.execute("INSERT INTO database_version VALUES (2)")
        db.executemany(
            "INSERT INTO full_blocks VALUES (?,?,?,?,?,?,?,?)",
            [
                (genesis, bytes(32), 0, None, 0, 1, b"block-0", b"record-0"),
                (peak, genesis, 1, None, 0, 1, b"block-1", b"record-1"),
            ],
        )
        db.execute("INSERT INTO current_peak VALUES (0, ?)", (peak,))
        puzzle = b"\x11" * 32
        parent = b"\x22" * 32
        amount = (100).to_bytes(8, "big")
        db.executemany(
            "INSERT INTO coin_record VALUES (?,?,?,?,?,?,?,?)",
            [
                (b"\x31" * 32, 0, 0, 1, puzzle, parent, amount, 10),
                (b"\x32" * 32, 1, 1, 0, puzzle, parent, amount, 11),
                (b"\x33" * 32, 1, -1, 0, puzzle, parent, amount, 11),
            ],
        )
        db.execute("INSERT INTO hints VALUES (?, ?)", (b"\x33" * 32, b"\x44" * 32))
        db.execute("INSERT INTO sub_epoch_segments_v3 VALUES (?, ?)", (genesis, b"ses-blob"))
        db.commit()
    return peak, 2


@pytest.mark.anyio
async def test_full_node_opens_an_empty_rocks_database(tmp_path: Path) -> None:
    from chia.consensus.block_height_map import BlockHeightMap
    from chia.full_node.full_node import FullNode

    node = await FullNode.create(
        {
            "database_path": "db/blockchain_v2_CHALLENGE.rocksdb",
            "selected_network": "mainnet",
            "db_readers": 4,
        },
        tmp_path,
        DEFAULT_CONSTANTS,
    )
    assert node.db_path.name == "blockchain_v2_mainnet.rocksdb"
    assert node._database_engine == "rocksdb"
    async with node._open_chain_db(None, "OFF", 2):
        assert await node.coin_store.is_empty()
        assert await node.block_store.get_peak() is None
        await BlockHeightMap.create_for_rocks(node.db_path.parent, node.block_store, "mainnet")
    assert await unfinished_migration_reason(node.db_path) is None


@pytest.mark.anyio
async def test_testing_flag_keeps_a_legacy_db_name_on_sqlite(tmp_path: Path) -> None:
    from chia.full_node.full_node import FullNode

    node = await FullNode.create(
        {
            "database_path": "db/blockchain_test_0_sim.db",
            "selected_network": "mainnet",
            "testing": True,
            "db_readers": 1,
        },
        tmp_path,
        DEFAULT_CONSTANTS,
    )
    assert node._database_engine == "sqlite"
    assert node.db_path.name == "blockchain_test_0_sim.db"
    async with node._open_chain_db(None, "OFF", 2):
        assert await node.coin_store.is_empty()
    assert node.db_path.is_file()

    rocks = await FullNode.create(
        {
            "database_path": "db/blockchain_v2_CHALLENGE.rocksdb",
            "selected_network": "mainnet",
            "testing": True,
        },
        tmp_path,
        DEFAULT_CONSTANTS,
    )
    assert rocks._database_engine == "rocksdb"
    assert rocks.db_path.name == "blockchain_v2_mainnet.rocksdb"


@pytest.mark.anyio
async def test_block_copy_progress_reports_chain_height(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "chain.sqlite"
    _write_sqlite(sqlite_path)
    seen: list[migrate_module.MigrateProgress] = []
    await migrate_database(
        sqlite_path,
        tmp_path / "out",
        selected_network="mainnet",
        constants=DEFAULT_CONSTANTS,
        progress=seen.append,
        assume_source_headroom=True,
    )
    blocks = [item for item in seen if item.phase == "blocks"]
    height = [item for item in seen if item.phase == "height"]
    epochs = [item for item in seen if item.phase == "epochs"]
    assert height[-1].detail == "writing height-to-hash"
    assert epochs[-1].detail == "writing sub-epoch-summaries"
    assert (tmp_path / "out" / "height-to-hash").is_file()
    assert (tmp_path / "out" / "sub-epoch-summaries").is_file()
    assert (tmp_path / "out" / "height-to-hash").stat().st_size == 64
    assert blocks[-1].current == 1
    assert blocks[-1].target == 1
    assert blocks[-1].detail == "rows 2 / 2"
    assert "1 / 1" in migrate_module.format_progress(blocks[-1])


@pytest.mark.anyio
async def test_migrate_matches_sqlite_peak_and_coins(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "chain.sqlite"
    peak, unspent = _write_sqlite(sqlite_path)
    output = tmp_path / "out"
    stop = asyncio.Event()
    stop.set()
    with pytest.raises(MigrationPaused):
        await migrate_database(
            sqlite_path,
            output,
            selected_network="mainnet",
            constants=DEFAULT_CONSTANTS,
            stop=stop,
            assume_source_headroom=True,
        )
    partial = output / "blockchain_v2_mainnet.rocksdb"
    reason = await unfinished_migration_reason(partial)
    assert reason is not None
    assert str(sqlite_path) in reason

    rocks_path = await migrate_database(
        sqlite_path,
        output,
        selected_network="mainnet",
        constants=DEFAULT_CONSTANTS,
        assume_source_headroom=True,
    )
    assert await unfinished_migration_reason(rocks_path) is None
    chain = ChainDB(RocksBackend(rocks_path, sync="OFF"))
    try:
        from chia.full_node.db.block_store import RocksBlockStore

        copied_blocks = await RocksBlockStore.create(chain, use_cache=False)
        copied_coins = await RocksCoinStore.create(chain)
        copied_peak = await copied_blocks.get_peak()
        assert copied_peak is not None
        assert copied_peak[0] == peak
        assert copied_peak[1] == 1
        assert await copied_coins.num_unspent() == unspent
        from chia.full_node.db.hint_store import RocksHintStore

        hints = await RocksHintStore.create(chain)
        assert await hints.get_hints([bytes32(b"\x33" * 32)]) == [bytes32(b"\x44" * 32)]
        async with chain.reader_no_transaction() as view:
            assert await view.get(CF_SES, b"\x01" * 32) == b"ses-blob"
    finally:
        await chain.close()
    assert sqlite_path.is_file()


@pytest.mark.anyio
async def test_follow_rolls_back_a_reorg_that_happens_during_the_copy(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "chain.sqlite"
    block0 = bytes32(b"\x01" * 32)
    block1 = bytes32(b"\x02" * 32)
    abandoned = bytes32(b"\x03" * 32)
    replacement = bytes32(b"\x04" * 32)
    coin_a = bytes32(b"\x11" * 32)
    coin_b = bytes32(b"\x12" * 32)
    coin_c = bytes32(b"\x13" * 32)
    coin_d = bytes32(b"\x14" * 32)
    amount = (100).to_bytes(8, "big")
    puzzle = b"\x21" * 32
    parent = b"\x22" * 32
    with sqlite3.connect(sqlite_path) as db:
        db.execute(
            "CREATE TABLE full_blocks("
            "header_hash blob PRIMARY KEY, prev_hash blob, height bigint, sub_epoch_summary blob,"
            "is_fully_compactified tinyint, in_main_chain tinyint, block blob, block_record blob)"
        )
        db.execute("CREATE TABLE current_peak(key int PRIMARY KEY, hash blob)")
        db.execute(
            "CREATE TABLE coin_record("
            "coin_name blob PRIMARY KEY, confirmed_index bigint, spent_index bigint, coinbase int,"
            "puzzle_hash blob, coin_parent blob, amount blob, timestamp bigint)"
        )
        db.execute("CREATE TABLE hints(coin_id blob, hint blob, UNIQUE (coin_id, hint))")
        db.execute("CREATE TABLE sub_epoch_segments_v3(ses_block_hash blob PRIMARY KEY, challenge_segments blob)")
        db.execute("CREATE TABLE database_version(version int)")
        db.execute("INSERT INTO database_version VALUES (2)")
        db.executemany(
            "INSERT INTO full_blocks VALUES (?,?,?,?,?,?,?,?)",
            [
                (block0, bytes(32), 0, None, 0, 1, b"block-0", b"record-0"),
                (block1, block0, 1, None, 0, 1, b"block-1", b"record-1"),
                (abandoned, block1, 2, None, 0, 1, b"block-2", b"record-2"),
            ],
        )
        db.execute("INSERT INTO current_peak VALUES (0, ?)", (abandoned,))
        db.executemany(
            "INSERT INTO coin_record VALUES (?,?,?,?,?,?,?,?)",
            [
                (coin_a, 0, 0, 0, puzzle, parent, amount, 1),
                (coin_b, 1, 2, 0, puzzle, parent, amount, 2),
                (coin_c, 2, 0, 0, puzzle, parent, amount, 3),
            ],
        )
        db.commit()

    async def reorg_before_follow() -> None:
        with sqlite3.connect(sqlite_path) as db:
            db.execute("UPDATE full_blocks SET in_main_chain=0 WHERE header_hash=?", (abandoned,))
            db.execute(
                "INSERT INTO full_blocks VALUES (?,?,?,?,?,?,?,?)",
                (replacement, block1, 2, None, 0, 1, b"block-2b", b"record-2b"),
            )
            db.execute("UPDATE current_peak SET hash=? WHERE key=0", (replacement,))
            db.execute("DELETE FROM coin_record WHERE coin_name=?", (coin_c,))
            db.execute("UPDATE coin_record SET spent_index=0 WHERE coin_name=?", (coin_b,))
            db.execute(
                "INSERT INTO coin_record VALUES (?,?,?,?,?,?,?,?)",
                (coin_d, 2, -1, 0, puzzle, parent, amount, 4),
            )
            db.commit()

    rocks_path = await migrate_database(
        sqlite_path,
        tmp_path / "out",
        selected_network="mainnet",
        constants=DEFAULT_CONSTANTS,
        assume_source_headroom=True,
        before_follow=reorg_before_follow,
    )
    chain = ChainDB(RocksBackend(rocks_path, sync="OFF"))
    try:
        from chia.full_node.db.block_store import RocksBlockStore

        blocks = await RocksBlockStore.create(chain, use_cache=False)
        coins = await RocksCoinStore.create(chain)
        assert await blocks.get_peak() == (replacement, 2)
        assert await blocks.main_chain_hash_at(2) == replacement
        assert await coins.get_coin_record(coin_c) is None
        kept = await coins.get_coin_record(coin_b)
        assert kept is not None
        assert kept.spent_block_index == 0
        added = await coins.get_coin_record(coin_d)
        assert added is not None
        assert added.spent_block_index == 0
        # spent_index -1 is stored for the fast-forward case and reported as unspent.
        assert await coins.num_unspent() == 3
    finally:
        await chain.close()


@pytest.mark.anyio
async def test_follow_picks_up_a_block_added_during_the_copy(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "chain.sqlite"
    peak, _unspent = _write_sqlite(sqlite_path)
    output = tmp_path / "out"
    new_peak = bytes32(b"\x03" * 32)
    new_coin = b"\x35" * 32

    async def add_block_before_follow() -> None:
        with sqlite3.connect(sqlite_path) as db:
            db.execute(
                "INSERT INTO full_blocks VALUES (?,?,?,?,?,?,?,?)",
                (new_peak, peak, 2, None, 0, 1, b"block-2", b"record-2"),
            )
            db.execute("UPDATE current_peak SET hash=? WHERE key=0", (new_peak,))
            db.execute(
                "INSERT INTO coin_record VALUES (?,?,?,?,?,?,?,?)",
                (new_coin, 2, 0, 0, b"\x11" * 32, b"\x22" * 32, (50).to_bytes(8, "big"), 12),
            )
            db.execute("UPDATE coin_record SET spent_index=2 WHERE coin_name=?", (b"\x31" * 32,))
            db.commit()

    rocks_path = await migrate_database(
        sqlite_path,
        output,
        selected_network="mainnet",
        constants=DEFAULT_CONSTANTS,
        assume_source_headroom=True,
        before_follow=add_block_before_follow,
    )
    chain = ChainDB(RocksBackend(rocks_path, sync="OFF"))
    try:
        from chia.full_node.db.block_store import RocksBlockStore

        copied_blocks = await RocksBlockStore.create(chain, use_cache=False)
        copied_coins = await RocksCoinStore.create(chain)
        copied_peak = await copied_blocks.get_peak()
        assert copied_peak == (new_peak, 2)
        # The original unspent coin was spent at height 2 and one new unspent coin was added.
        assert await copied_coins.num_unspent() == 2
        spent = await copied_coins.get_coin_record(bytes32(b"\x31" * 32))
        assert spent is not None
        assert spent.spent_block_index == 2
    finally:
        await chain.close()

    backup = tmp_path / "backup.rocksdb"
    await backup_rocks(rocks_path, backup)
    assert await validate_rocks(backup) == (new_peak, 2, 2)
    with pytest.raises(RuntimeError, match="--no_indexes"):
        from chia.cmds.db_backup_func import backup_db

        backup_db(rocks_path, tmp_path / "ignored.rocksdb", no_indexes=True)


@pytest.mark.anyio
async def test_v1_database_is_refused(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "old.sqlite"
    with sqlite3.connect(sqlite_path) as db:
        db.execute("CREATE TABLE database_version(version int)")
        db.execute("INSERT INTO database_version VALUES (1)")
        db.commit()
    with pytest.raises(RuntimeError, match="chia db upgrade"):
        await migrate_database(
            sqlite_path,
            tmp_path / "out",
            selected_network="mainnet",
            constants=DEFAULT_CONSTANTS,
            assume_source_headroom=True,
        )
    assert sqlite_path.is_file()
    assert not (tmp_path / "out" / "blockchain_v2_mainnet.rocksdb").exists()


@pytest.mark.anyio
async def test_abort_deletes_unfinished_copy_only(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "chain.sqlite"
    _write_sqlite(sqlite_path)
    output = tmp_path / "out"
    stop = asyncio.Event()
    stop.set()
    with pytest.raises(MigrationPaused):
        await migrate_database(
            sqlite_path,
            output,
            selected_network="mainnet",
            constants=DEFAULT_CONSTANTS,
            stop=stop,
            assume_source_headroom=True,
        )
    partial = output / "blockchain_v2_mainnet.rocksdb"
    assert partial.is_dir()
    message = await abort_migration(output, "mainnet")
    assert "Deleted the unfinished copy" in message
    assert not partial.exists()
    assert sqlite_path.is_file()
    rocks_path = await migrate_database(
        sqlite_path,
        output,
        selected_network="mainnet",
        constants=DEFAULT_CONSTANTS,
        assume_source_headroom=True,
    )
    with pytest.raises(RuntimeError, match="finished RocksDB"):
        await abort_migration(output, "mainnet")
    assert rocks_path.is_dir()


def _write_many_coins(path: Path, count: int) -> None:
    genesis = b"\x01" * 32
    with sqlite3.connect(path) as db:
        db.execute(
            "CREATE TABLE full_blocks("
            "header_hash blob PRIMARY KEY, prev_hash blob, height bigint, sub_epoch_summary blob,"
            "is_fully_compactified tinyint, in_main_chain tinyint, block blob, block_record blob)"
        )
        db.execute("CREATE TABLE current_peak(key int PRIMARY KEY, hash blob)")
        db.execute(
            "CREATE TABLE coin_record("
            "coin_name blob PRIMARY KEY, confirmed_index bigint, spent_index bigint, coinbase int,"
            "puzzle_hash blob, coin_parent blob, amount blob, timestamp bigint)"
        )
        db.execute("CREATE TABLE hints(coin_id blob, hint blob)")
        db.execute("CREATE TABLE sub_epoch_segments_v3(ses_block_hash blob PRIMARY KEY, challenge_segments blob)")
        db.execute("CREATE TABLE database_version(version int)")
        db.execute("INSERT INTO database_version VALUES (2)")
        db.execute(
            "INSERT INTO full_blocks VALUES (?,?,?,?,?,?,?,?)",
            (genesis, bytes(32), 0, None, 0, 1, b"block-0", b"record-0"),
        )
        db.execute("INSERT INTO current_peak VALUES (0, ?)", (genesis,))
        amount = (1).to_bytes(8, "big")
        db.executemany(
            "INSERT INTO coin_record VALUES (?,?,?,?,?,?,?,?)",
            [
                (i.to_bytes(32, "big"), 0, 0, 0, b"\x11" * 32, b"\x22" * 32, amount, 1)
                for i in range(1, count + 1)
            ],
        )
        db.commit()


def test_space_check_refuses_a_small_destination(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    sqlite_path = tmp_path / "chain.sqlite"
    sqlite_path.write_bytes(b"x" * 1000)
    monkeypatch.setattr(migrate_module.shutil, "disk_usage", lambda _path: SimpleNamespace(free=100))
    with pytest.raises(RuntimeError, match=r"1\.5x"):
        require_destination_space(sqlite_path, tmp_path)


@pytest.mark.anyio
async def test_stopping_during_the_coin_copy_resumes_without_mixing_rows(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "chain.sqlite"
    _write_many_coins(sqlite_path, 600)
    output = tmp_path / "out"
    stop = asyncio.Event()

    def pause_after_first_coin_batch(update: migrate_module.MigrateProgress) -> None:
        if update.phase == "coins" and update.current > 0:
            stop.set()

    with pytest.raises(MigrationPaused):
        await migrate_database(
            sqlite_path,
            output,
            selected_network="mainnet",
            constants=DEFAULT_CONSTANTS,
            progress=pause_after_first_coin_batch,
            stop=stop,
            assume_source_headroom=True,
        )
    rocks_path = await migrate_database(
        sqlite_path,
        output,
        selected_network="mainnet",
        constants=DEFAULT_CONSTANTS,
        assume_source_headroom=True,
    )
    chain = ChainDB(RocksBackend(rocks_path, sync="OFF"))
    try:
        coins = await RocksCoinStore.create(chain)
        assert await coins.num_unspent() == 600
        first = await coins.get_coin_record(bytes32((1).to_bytes(32, "big")))
        last = await coins.get_coin_record(bytes32((600).to_bytes(32, "big")))
        assert first is not None
        assert last is not None
    finally:
        await chain.close()
    assert sqlite_path.is_file()


@pytest.mark.anyio
async def test_height_map_loads_from_a_migrated_database(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "chain.sqlite"
    peak, _unspent = _write_sqlite(sqlite_path)
    genesis = bytes32(b"\x01" * 32)
    rocks_path = await migrate_database(
        sqlite_path,
        tmp_path / "out",
        selected_network="mainnet",
        constants=DEFAULT_CONSTANTS,
        assume_source_headroom=True,
    )
    chain = ChainDB(RocksBackend(rocks_path, sync="OFF"))
    try:
        from chia.full_node.db.block_store import RocksBlockStore

        blocks = await RocksBlockStore.create(chain, use_cache=False)
        height_map = await BlockHeightMap.create_for_rocks(tmp_path / "height", blocks, "mainnet")
        assert height_map.get_hash(uint32(0)) == genesis
        assert height_map.get_hash(uint32(1)) == peak
    finally:
        await chain.close()


@pytest.mark.anyio
async def test_migrate_matches_coins_written_by_the_real_coin_store(tmp_path: Path) -> None:
    sqlite_path = tmp_path / "chain.sqlite"
    farmer, pool = rewards(uint32(1))
    puzzle = bytes32(b"\x42" * 32)
    child = Coin(bytes32(b"\x41" * 32), puzzle, uint64(100))
    hint = b"\x43" * 32
    async with DBWrapper2.managed(database=sqlite_path, reader_count=1, db_version=2) as wrapper:
        coin_store = await CoinStore.create(wrapper)
        hint_store = await HintStore.create(wrapper)
        await BlockStore.create(wrapper)
        await coin_store.new_block(
            uint32(1),
            uint64(1_000),
            [farmer, pool],
            [(child.name(), child, True)],
            [],
        )
        await coin_store.new_block(uint32(2), uint64(2_000), list(rewards(uint32(2))), [], [child.name()])
        await hint_store.add_hints([(child.name(), hint)])
        sqlite_unspent = await coin_store.num_unspent()
        sqlite_child = await coin_store.get_coin_record(child.name())
        sqlite_farmer = await coin_store.get_coin_record(farmer.name())
    from chia.util.db_version import set_db_version

    with sqlite3.connect(sqlite_path) as connection:
        set_db_version(connection, 2)

    rocks_path = await migrate_database(
        sqlite_path,
        tmp_path / "out",
        selected_network="mainnet",
        constants=DEFAULT_CONSTANTS,
        assume_source_headroom=True,
    )
    chain = ChainDB(RocksBackend(rocks_path, sync="OFF"))
    try:
        from chia.full_node.db.coin_codec import decode_coin
        from chia.full_node.db.hint_store import RocksHintStore
        from chia.full_node.db.keys import CF_COINS

        copied = await RocksCoinStore.create(chain)
        hints = await RocksHintStore.create(chain)
        assert await copied.num_unspent() == sqlite_unspent
        copied_child = await copied.get_coin_record(child.name())
        copied_farmer = await copied.get_coin_record(farmer.name())
        assert copied_child is not None and sqlite_child is not None
        assert copied_child.coin.amount == sqlite_child.coin.amount
        assert copied_child.spent_block_index == sqlite_child.spent_block_index
        assert copied_farmer is not None and sqlite_farmer is not None
        assert copied_farmer.coinbase == sqlite_farmer.coinbase
        assert copied_farmer.coin.amount == sqlite_farmer.coin.amount
        async with chain.reader_no_transaction() as view:
            raw = await view.get(CF_COINS, bytes(child.name()))
        assert raw is not None
        assert decode_coin(child.name(), raw).spent_index == 2
        assert await hints.get_hints([child.name()]) == [bytes32(hint)]
    finally:
        await chain.close()
