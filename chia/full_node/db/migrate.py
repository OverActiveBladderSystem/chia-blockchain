from __future__ import annotations

import asyncio
import logging
import shutil
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from pathlib import Path

import aiosqlite
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32, uint64

from chia.consensus.block_height_map import BlockHeightMap
from chia.consensus.constants import ConsensusConstants
from chia.full_node.db.block_store import RocksBlockStore
from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.coin_store import RocksCoinStore
from chia.full_node.db.hint_store import RocksHintStore
from chia.full_node.db.keys import (
    CF_BLOCK_BLOBS,
    CF_BLOCK_META,
    CF_BLOCKS_AT_HEIGHT,
    CF_COINS,
    CF_COINS_BY_CONFIRMED,
    CF_COINS_BY_PARENT,
    CF_COINS_BY_PUZZLE_CONFIRMED,
    CF_COINS_BY_PUZZLE_SPENT,
    CF_COINS_BY_SPENT,
    CF_FF_UNSPENT,
    CF_HINTS_BY_COIN,
    CF_HINTS_BY_HINT,
    CF_MAIN_CHAIN,
    CF_META,
    CF_SES,
    CF_UNCOMPACTIFIED,
    META_COMPLETE,
    META_FORMAT,
    META_FORMAT_VALUE,
    META_MIGRATE_HEIGHT,
    META_MIGRATE_PHASE,
    META_MIGRATE_ROWID,
    META_MIGRATE_SOURCE,
    META_SCHEMA_VERSION,
    ROCKS_SUFFIX,
)
from chia.full_node.db.rocks import RocksBackend
from chia.full_node.db.startup import unfinished_migration_reason
from chia.types.blockchain_format.coin import Coin

log = logging.getLogger(__name__)

_GIB = 1024 * 1024 * 1024
_SOURCE_HEADROOM = 10 * _GIB
_BLOCK_BATCH = 2000
_CHAIN_BATCH = 16000
_COIN_BATCH = 50000
_HINT_BATCH = 20000
_SUMMARY_BATCH = 8
_SUMMARY_BYTES = 32 * 1024 * 1024
_MERGE_FAMILIES: tuple[tuple[str, str], ...] = (
    (CF_BLOCK_BLOBS, "block files"),
    (CF_BLOCK_META, "block records"),
    (CF_BLOCKS_AT_HEIGHT, "blocks by height"),
    (CF_MAIN_CHAIN, "chain index"),
    (CF_UNCOMPACTIFIED, "compact index"),
    (CF_COINS, "coin records"),
    (CF_COINS_BY_CONFIRMED, "coins by height"),
    (CF_COINS_BY_SPENT, "spent coins"),
    (CF_COINS_BY_PUZZLE_CONFIRMED, "coins by puzzle"),
    (CF_COINS_BY_PUZZLE_SPENT, "spent coins by puzzle"),
    (CF_COINS_BY_PARENT, "coins by parent"),
    (CF_FF_UNSPENT, "fast-forward coins"),
    (CF_HINTS_BY_COIN, "hints by coin"),
    (CF_HINTS_BY_HINT, "hints by hint"),
    (CF_SES, "sub-epoch summaries"),
    (CF_META, "database info"),
)


class MigrationPaused(Exception):
    """The copy stopped and can be resumed. The SQLite file was not changed."""


@dataclass
class MigrateProgress:
    phase: str
    current: int
    target: int
    detail: str = ""
    # Monotonic work counter for rate and ETA. Block copy uses SQLite rowid, not height.
    work_done: int = 0
    work_total: int = 0

    @property
    def percent(self) -> float:
        if self.target <= 0:
            return 100.0
        return min(100.0, 100.0 * self.current / self.target)


def fit_terminal_line(line: str, width: int) -> str:
    """Stay on one row. A line as wide as the terminal wraps, and the next update starts below it."""
    usable = max(1, width - 1)
    if len(line) >= usable:
        return line[:usable]
    return line + (" " * (usable - len(line)))


def format_progress(
    progress: MigrateProgress,
    *,
    elapsed_seconds: float | None = None,
    done_this_run: int | None = None,
    rate_seconds: float | None = None,
) -> str:
    step = {
        "blocks": "1/9",
        "chain": "2/9",
        "coins": "3/9",
        "hints": "4/9",
        "summary": "5/9",
        "pack": "6/9",
        "follow": "7/9",
        "height": "8/9",
        "epochs": "9/9",
        "complete": "9/9",
    }.get(progress.phase, "1/9")
    line = (
        f"[{step}] {progress.phase:<8} {progress.current:,} / {progress.target:,}  "
        f"{progress.percent:5.1f}%  {progress.detail}"
    )
    if elapsed_seconds is None:
        return line
    elapsed = max(0.0, elapsed_seconds)
    line += f"  elapsed {_format_elapsed(elapsed)}"
    window = elapsed if rate_seconds is None else max(0.0, rate_seconds)
    if (
        done_this_run is None
        or done_this_run <= 0
        or window < 1
        or progress.work_total <= 0
        or progress.work_done >= progress.work_total
    ):
        return line
    rate = done_this_run / window
    remaining = progress.work_total - progress.work_done
    unit = {
        "blocks": "rows/s",
        "chain": "rows/s",
        "coins": "coins/s",
        "hints": "rows/s",
        "summary": "rows/s",
        "height": "blocks/s",
        "epochs": "rows/s",
        "follow": "blocks/s",
    }.get(progress.phase, "rows/s")
    return f"{line}  {rate:,.1f} {unit}  ETA {_format_eta(remaining / rate)}"


@dataclass
class ProgressClock:
    """Elapsed time for the whole command, and a rate window that restarts each phase."""

    started: float
    phase_started: float
    baseline_phase: str | None = None
    baseline: int = 0

    def line(self, update: MigrateProgress, now: float) -> str:
        if self.baseline_phase != update.phase:
            self.baseline_phase = update.phase
            self.baseline = update.work_done
            self.phase_started = now
        return format_progress(
            update,
            elapsed_seconds=max(0.0, now - self.started),
            done_this_run=max(0, update.work_done - self.baseline),
            rate_seconds=max(0.0, now - self.phase_started),
        )


async def replay_progress(
    stop: asyncio.Event,
    latest: Callable[[], MigrateProgress | None],
    paint: Callable[[MigrateProgress, float], None],
    *,
    interval: float = 1.0,
) -> None:
    """Redraw the last progress line on a timer so a slow batch does not look frozen."""
    while not stop.is_set():
        try:
            await asyncio.wait_for(stop.wait(), interval)
        except TimeoutError:
            update = latest()
            if update is not None and update.phase != "complete":
                paint(update, time.monotonic())


def _format_elapsed(seconds: float) -> str:
    total = int(seconds)
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def _format_eta(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def sqlite_file_size(path: Path) -> int:
    total = path.stat().st_size
    for suffix in ("-wal", "-shm"):
        extra = Path(str(path) + suffix)
        if extra.exists():
            total += extra.stat().st_size
    return total


def require_destination_space(sqlite_path: Path, output_parent: Path) -> tuple[int, int]:
    """Return (sqlite bytes, required free bytes). Raise when the destination is short of 1.5x."""
    size = sqlite_file_size(sqlite_path)
    needed = int(size * 1.5)
    parent = output_parent if output_parent.exists() else output_parent.parent
    free = shutil.disk_usage(parent).free
    if free < needed:
        raise RuntimeError(
            f"Need {needed:,} bytes free in {parent} (1.5x the {size:,} byte SQLite file) but only {free:,} are free."
        )
    return size, needed


def source_headroom_warning(sqlite_path: Path) -> str | None:
    free = shutil.disk_usage(sqlite_path.parent).free
    if free >= _SOURCE_HEADROOM:
        return None
    return (
        f"The disk holding {sqlite_path} has {free:,} bytes free. Keep about 10 GB free. "
        "The coin snapshot holds a SQLite read view, so the WAL cannot shrink until that phase ends."
    )


def rocks_directory_for(output_parent: Path, selected_network: str) -> Path:
    return output_parent / f"blockchain_v2_{selected_network}{ROCKS_SUFFIX}"


def offered_database_path(current_database_path: str, output_parent: Path, rocks_dir: Path) -> str:
    if "CHALLENGE" in current_database_path:
        return str(output_parent / f"blockchain_v2_CHALLENGE{ROCKS_SUFFIX}")
    return str(rocks_dir)


async def migrate_database(
    sqlite_path: Path,
    output_parent: Path,
    *,
    selected_network: str,
    constants: ConsensusConstants,
    progress: Callable[[MigrateProgress], None] | None = None,
    stop: asyncio.Event | None = None,
    assume_source_headroom: bool = False,
    before_follow: Callable[[], Awaitable[None]] | None = None,
) -> Path:
    """Copy one network's SQLite chain DB into a RocksDB directory. The SQLite file is only read."""
    if not sqlite_path.is_file():
        raise RuntimeError(f"SQLite database does not exist: {sqlite_path}")
    await _require_v2(sqlite_path)
    output_parent.mkdir(parents=True, exist_ok=True)
    require_destination_space(sqlite_path, output_parent)
    warning = source_headroom_warning(sqlite_path)
    if warning is not None and not assume_source_headroom:
        raise RuntimeError(warning + " Rerun with --yes to continue anyway.")

    rocks_path = rocks_directory_for(output_parent, selected_network)
    log.info("opening RocksDB at %s", rocks_path)
    # WAL stays on so closing the window keeps every finished batch.
    chain = ChainDB(RocksBackend(rocks_path, sync="NORMAL", bulk=True))
    log.info("RocksDB open")
    try:
        await _ensure_migration_marker(chain)
        await _put_meta(chain, META_MIGRATE_SOURCE, str(sqlite_path).encode())
        block_store = await RocksBlockStore.create(chain, use_cache=False)
        coin_store = await RocksCoinStore.create(chain)
        hint_store = await RocksHintStore.create(chain)
        phase = await _meta(chain, META_MIGRATE_PHASE) or b"blocks"
        log.info("migration phase %s", phase)
        if phase == b"complete" or await _meta(chain, META_COMPLETE) == b"1":
            _report(progress, MigrateProgress("complete", 1, 1, "already complete"))
            return rocks_path
        if phase == b"blocks":
            await _copy_blocks(sqlite_path, chain, block_store, progress, stop)
            phase = b"coins"
        if phase == b"coins":
            await _copy_snapshot(
                sqlite_path,
                chain,
                block_store,
                coin_store,
                hint_store,
                progress,
                stop,
                assume_source_headroom=assume_source_headroom,
            )
            phase = b"merge"
        if phase == b"merge":
            await _merge_stored_files(chain, progress, stop)
            await _put_meta(chain, META_MIGRATE_PHASE, b"follow")
            phase = b"follow"
        if phase in {b"follow", b"hints"}:
            if before_follow is not None:
                await before_follow()
            log.debug("follow genesis %s", constants.GENESIS_CHALLENGE)
            await _follow(sqlite_path, chain, block_store, coin_store, hint_store, progress, stop)
            await _put_meta(chain, META_MIGRATE_PHASE, b"startup")
            phase = b"startup"
        if phase == b"startup":
            await _write_startup_cache(rocks_path.parent, block_store, selected_network, progress, stop)
        await _put_meta(chain, META_MIGRATE_PHASE, b"complete")
        await _put_meta(chain, META_COMPLETE, b"1")
        _report(progress, MigrateProgress("complete", 1, 1, str(rocks_path)))
        return rocks_path
    except MigrationPaused:
        raise
    finally:
        await chain.close()


async def _require_v2(sqlite_path: Path) -> None:
    async with aiosqlite.connect(sqlite_path) as db:
        try:
            cursor = await db.execute("SELECT version FROM database_version LIMIT 1")
            row = await cursor.fetchone()
            await cursor.close()
        except aiosqlite.OperationalError as exc:
            raise RuntimeError(
                f"{sqlite_path} is not a v2 full node database. Run `chia db upgrade` before migrating."
            ) from exc
    if row is None or int(row[0]) != 2:
        found = "missing" if row is None else str(row[0])
        raise RuntimeError(
            f"{sqlite_path} is database version {found}. Run `chia db upgrade` so it is version 2 before migrating."
        )


async def abort_migration(output_parent: Path, selected_network: str) -> str:
    """Delete an unfinished RocksDB copy. A finished database and every SQLite file are left in place."""
    rocks_path = rocks_directory_for(output_parent, selected_network)
    if not rocks_path.exists():
        return f"No migration directory at {rocks_path}. Nothing was deleted."
    if (rocks_path / "CURRENT").exists():
        reason = await unfinished_migration_reason(rocks_path)
        if reason is None:
            raise RuntimeError(
                f"{rocks_path} is a finished RocksDB database. It was not deleted. "
                "Delete that folder yourself only after you are sure you do not want it."
            )
    shutil.rmtree(rocks_path)
    return f"Deleted the unfinished copy at {rocks_path}. The SQLite database was not changed."


async def _ensure_migration_marker(chain: ChainDB) -> None:
    async with chain.writer() as session:
        if await session.get(CF_META, META_FORMAT) is None:
            session.put(CF_META, META_FORMAT, META_FORMAT_VALUE)
            session.put(CF_META, META_SCHEMA_VERSION, (2).to_bytes(4, "little", signed=False))
        phase = await session.get(CF_META, META_MIGRATE_PHASE)
        if phase is None:
            session.put(CF_META, META_MIGRATE_PHASE, b"blocks")


async def _copy_blocks(
    sqlite_path: Path,
    chain: ChainDB,
    block_store: RocksBlockStore,
    progress: Callable[[MigrateProgress], None] | None,
    stop: asyncio.Event | None,
) -> None:
    log.info("copying blocks from %s", sqlite_path)
    async with aiosqlite.connect(sqlite_path) as db:
        total = await _count(db, "SELECT COUNT(*) FROM full_blocks")
        log.info("full_blocks count %s", total)
        peak_height = await _peak_height(db)
        rowid = _u64(await _meta(chain, META_MIGRATE_ROWID))
        highest = _u64(await _meta(chain, META_MIGRATE_HEIGHT))
        while True:
            _check_stop(stop)
            cursor = await db.execute(
                "SELECT rowid, header_hash, prev_hash, height, sub_epoch_summary, is_fully_compactified, "
                "in_main_chain, block, block_record FROM full_blocks WHERE rowid>? ORDER BY rowid LIMIT ?",
                (rowid, _BLOCK_BATCH),
            )
            rows = await cursor.fetchall()
            await cursor.close()
            if len(rows) == 0:
                break
            pending: list[tuple[bytes32, bytes32, int, bytes | None, bool, bool, bytes, bytes]] = []
            for row in rows:
                rowid = int(row[0])
                height = int(row[3])
                highest = max(highest, height)
                ses = row[4]
                pending.append(
                    (
                        bytes32(row[1]),
                        bytes32(row[2]),
                        height,
                        None if ses is None else bytes(ses),
                        bool(row[5]),
                        bool(row[6]),
                        bytes(row[7]),
                        bytes(row[8]),
                    )
                )
            await block_store.import_block_rows(pending)
            await _put_meta_many(
                chain,
                [
                    (META_MIGRATE_ROWID, rowid.to_bytes(8, "little", signed=False)),
                    (META_MIGRATE_HEIGHT, highest.to_bytes(4, "little", signed=False)),
                ],
            )
            copied_rows = max(total, rowid)
            _report(
                progress,
                MigrateProgress(
                    "blocks",
                    highest,
                    max(peak_height, highest),
                    f"rows {rowid:,} / {total:,}",
                    work_done=rowid,
                    work_total=copied_rows,
                ),
            )
    await _put_meta(chain, META_MIGRATE_PHASE, b"coins")
    await _put_meta(chain, META_MIGRATE_ROWID, (0).to_bytes(8, "little", signed=False))


async def _copy_snapshot(
    sqlite_path: Path,
    chain: ChainDB,
    block_store: RocksBlockStore,
    coin_store: RocksCoinStore,
    hint_store: RocksHintStore,
    progress: Callable[[MigrateProgress], None] | None,
    stop: asyncio.Event | None,
    *,
    assume_source_headroom: bool,
) -> None:
    log.info("copying coin snapshot")
    _report(progress, MigrateProgress("chain", 0, 1, "preparing the coin copy"))
    if not assume_source_headroom:
        _check_source_space(sqlite_path)
    await coin_store.clear_all()
    await hint_store.clear_all()
    async with aiosqlite.connect(sqlite_path) as db:
        await db.execute("BEGIN")
        try:
            peak_hash, peak_height = await _peak(db)
            block_total = await _count(db, "SELECT COUNT(*) FROM full_blocks")
            _report(
                progress,
                MigrateProgress(
                    "chain",
                    0,
                    max(block_total, 1),
                    "reading the old database",
                    work_total=max(block_total, 1),
                ),
            )
            await block_store.clear_main_chain_index()
            cursor = await db.execute("SELECT header_hash, in_main_chain, is_fully_compactified FROM full_blocks")
            seen = 0
            while True:
                _check_stop(stop)
                fetched = await cursor.fetchmany(_CHAIN_BATCH)
                if len(fetched) == 0:
                    break
                membership = [
                    (bytes32(header_hash), bool(in_chain), bool(compact)) for header_hash, in_chain, compact in fetched
                ]
                await block_store.apply_membership(membership)
                seen += len(membership)
                _report(
                    progress,
                    MigrateProgress(
                        "chain",
                        seen,
                        max(block_total, seen),
                        "copying the chain index",
                        work_done=seen,
                        work_total=max(block_total, seen),
                    ),
                )
            await cursor.close()

            total = await _count(db, "SELECT COUNT(*) FROM coin_record")
            _report(
                progress,
                MigrateProgress("coins", 0, max(total, 1), "copying coins", work_total=max(total, 1)),
            )
            copied = 0
            rowid = 0
            while True:
                _check_stop(stop)
                if copied and copied % 10000 == 0:
                    _check_source_space(sqlite_path)
                cursor = await db.execute(
                    "SELECT rowid, coin_name, confirmed_index, spent_index, coinbase, puzzle_hash, "
                    "coin_parent, amount, timestamp FROM coin_record WHERE rowid>? ORDER BY rowid LIMIT ?",
                    (rowid, _COIN_BATCH),
                )
                rows = await cursor.fetchall()
                await cursor.close()
                if len(rows) == 0:
                    break
                pending_rows: list[tuple[object, ...]] = []
                for row in rows:
                    rowid = int(row[0])
                    pending_rows.append((row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]))
                await coin_store.import_new_rows(pending_rows)
                copied += len(pending_rows)
                _report(
                    progress,
                    MigrateProgress(
                        "coins",
                        copied,
                        total,
                        f"snapshot height {peak_height}",
                        work_done=copied,
                        work_total=max(total, copied),
                    ),
                )
            await _copy_hints_and_ses(db, chain, hint_store, progress, stop)
            if peak_hash is not None:
                await block_store.set_peak(peak_hash)
            await _put_meta(
                chain,
                META_MIGRATE_HEIGHT,
                int(peak_height).to_bytes(4, "little", signed=False),
            )
        finally:
            await db.execute("ROLLBACK")
    await _put_meta(chain, META_MIGRATE_PHASE, b"merge")


async def _merge_stored_files(
    chain: ChainDB,
    progress: Callable[[MigrateProgress], None] | None,
    stop: asyncio.Event | None,
) -> None:
    """One merge after the copy, instead of merging continuously while coins are written."""
    total = len(_MERGE_FAMILIES)
    for index, (family, label) in enumerate(_MERGE_FAMILIES, start=1):
        _check_stop(stop)
        _report(
            progress,
            MigrateProgress(
                "pack",
                index - 1,
                total,
                f"packing {label}",
                work_done=index - 1,
                work_total=total,
            ),
        )
        await chain.merge_families([family])
    _report(progress, MigrateProgress("pack", total, total, "packed the database files", work_done=total, work_total=total))


async def _copy_hints_and_ses(
    db: aiosqlite.Connection,
    chain: ChainDB,
    hint_store: RocksHintStore,
    progress: Callable[[MigrateProgress], None] | None,
    stop: asyncio.Event | None,
) -> None:
    try:
        hint_total = await _count(db, "SELECT COUNT(*) FROM hints")
        cursor = await db.execute("SELECT coin_id, hint FROM hints")
    except aiosqlite.OperationalError:
        cursor = None
        hint_total = 0
    if cursor is not None:
        copied = 0
        _report(
            progress,
            MigrateProgress("hints", 0, max(hint_total, 1), "copying hints", work_total=max(hint_total, 1)),
        )
        while True:
            _check_stop(stop)
            fetched = await cursor.fetchmany(_HINT_BATCH)
            if len(fetched) == 0:
                break
            batch = [(bytes32(coin_id), bytes(hint)) for coin_id, hint in fetched]
            await hint_store.add_hints(batch, assume_new=True)
            copied += len(batch)
            _report(
                progress,
                MigrateProgress(
                    "hints",
                    copied,
                    max(hint_total, copied),
                    "copying hints",
                    work_done=copied,
                    work_total=max(hint_total, copied),
                ),
            )
        await cursor.close()
    await _copy_sub_epoch_segments(db, chain, progress, stop)


async def _copy_sub_epoch_segments(
    db: aiosqlite.Connection,
    chain: ChainDB,
    progress: Callable[[MigrateProgress], None] | None,
    stop: asyncio.Event | None,
) -> None:
    """Copy sub-epoch segments in small batches.

    One segment can be megabytes, so the whole table must not be loaded at once.
    """
    try:
        total = await _count(db, "SELECT COUNT(*) FROM sub_epoch_segments_v3")
        cursor = await db.execute("SELECT ses_block_hash, challenge_segments FROM sub_epoch_segments_v3")
    except aiosqlite.OperationalError:
        return
    _report(
        progress,
        MigrateProgress(
            "summary",
            0,
            max(total, 1),
            "saving sub-epoch summaries",
            work_total=max(total, 1),
        ),
    )
    async with chain.writer() as session:
        session.delete_range(CF_SES, b"", b"\xff" * 80)
    copied = 0
    pending: list[tuple[bytes, bytes]] = []
    pending_bytes = 0

    async def flush() -> None:
        nonlocal pending, pending_bytes, copied
        if len(pending) == 0:
            return
        async with chain.writer() as session:
            for ses_hash, blob in pending:
                session.put(CF_SES, ses_hash, blob)
        copied += len(pending)
        pending = []
        pending_bytes = 0
        _report(
            progress,
            MigrateProgress(
                "summary",
                copied,
                max(total, copied),
                "saving sub-epoch summaries",
                work_done=copied,
                work_total=max(total, copied),
            ),
        )

    while True:
        _check_stop(stop)
        fetched = await cursor.fetchmany(_SUMMARY_BATCH)
        if len(fetched) == 0:
            break
        for ses_hash, blob in fetched:
            raw = bytes(blob)
            pending.append((bytes(ses_hash), raw))
            pending_bytes += len(raw)
            if pending_bytes >= _SUMMARY_BYTES:
                await flush()
        if len(pending) >= _SUMMARY_BATCH:
            await flush()
    await flush()
    await cursor.close()


async def _write_startup_cache(
    blockchain_dir: Path,
    block_store: RocksBlockStore,
    selected_network: str,
    progress: Callable[[MigrateProgress], None] | None,
    stop: asyncio.Event | None,
) -> None:
    """Write height-to-hash and sub-epoch-summaries beside the RocksDB directory, at its current peak."""
    row = await block_store.peak_height_map_row()
    if row is None:
        return
    peak_height = int(row[2])
    target = max(peak_height, 1)

    def on_height(height: int) -> None:
        _check_stop(stop)
        done = max(0, peak_height - height)
        _report(
            progress,
            MigrateProgress(
                "height",
                done,
                target,
                "writing height-to-hash",
                work_done=done,
                work_total=target,
            ),
        )

    _report(
        progress,
        MigrateProgress("height", 0, target, "writing height-to-hash", work_total=target),
    )
    height_map = await BlockHeightMap.create_for_rocks(
        blockchain_dir,
        block_store,
        selected_network,
        on_height=on_height,
    )
    await height_map.write_height_file()
    _report(
        progress,
        MigrateProgress(
            "height",
            target,
            target,
            "writing height-to-hash",
            work_done=target,
            work_total=target,
        ),
    )
    summary_count = len(height_map.get_ses_heights())
    summary_target = max(summary_count, 1)
    _report(
        progress,
        MigrateProgress(
            "epochs",
            0,
            summary_target,
            "writing sub-epoch-summaries",
            work_total=summary_target,
        ),
    )
    await height_map.write_ses_file()
    _report(
        progress,
        MigrateProgress(
            "epochs",
            summary_count,
            summary_target,
            "writing sub-epoch-summaries",
            work_done=summary_count,
            work_total=summary_target,
        ),
    )


async def _follow(
    sqlite_path: Path,
    chain: ChainDB,
    block_store: RocksBlockStore,
    coin_store: RocksCoinStore,
    hint_store: RocksHintStore,
    progress: Callable[[MigrateProgress], None] | None,
    stop: asyncio.Event | None,
) -> None:
    previous_height = -1
    while True:
        _check_stop(stop)
        async with aiosqlite.connect(sqlite_path) as db:
            sqlite_peak, sqlite_height = await _peak(db)
            rocks_peak = await block_store.get_peak()
            if sqlite_peak is None or rocks_peak is None:
                return
            if sqlite_peak == rocks_peak[0]:
                _report(
                    progress,
                    MigrateProgress(
                        "follow",
                        int(sqlite_height),
                        int(sqlite_height),
                        "caught up",
                        work_done=int(sqlite_height),
                        work_total=int(sqlite_height),
                    ),
                )
                return
            rocks_at_height = await _main_hash(db, int(rocks_peak[1]))
            if rocks_at_height != rocks_peak[0]:
                await _rollback_to_common(db, chain, block_store, coin_store, int(rocks_peak[1]))
                previous_height = -1
                continue
            nxt = int(rocks_peak[1]) + 1
            if nxt == previous_height:
                raise RuntimeError(f"follow did not advance past height {nxt}")
            row = await _main_block_row(db, nxt)
        if row is None:
            return
        log.info("follow height %s toward %s", nxt, sqlite_height)
        await _apply_followed_block(sqlite_path, row, block_store, coin_store, hint_store)
        previous_height = nxt
        target_height = int(sqlite_height)
        _report(
            progress,
            MigrateProgress(
                "follow",
                nxt,
                target_height,
                f"{target_height - nxt} behind",
                work_done=nxt,
                work_total=max(target_height, nxt),
            ),
        )
        await _put_meta(chain, META_MIGRATE_HEIGHT, nxt.to_bytes(4, "little", signed=False))


async def _apply_followed_block(
    sqlite_path: Path,
    row: aiosqlite.Row,
    block_store: RocksBlockStore,
    coin_store: RocksCoinStore,
    hint_store: RocksHintStore,
) -> None:
    """Apply one new main-chain height from the SQLite coin rows, without running the generator."""
    header_hash = bytes32(row[0])
    height = int(row[2])
    await block_store.import_block_row(
        header_hash,
        bytes32(row[1]),
        height,
        row[3],
        bool(row[4]),
        True,
        bytes(row[6]),
        bytes(row[7]),
    )
    await block_store.set_in_chain([(header_hash,)])
    additions: list[tuple[bytes32, Coin, bool]] = []
    rewards: list[Coin] = []
    removals: list[bytes32] = []
    hints: list[tuple[bytes32, bytes]] = []
    timestamp = 0
    async with aiosqlite.connect(sqlite_path) as db:
        cursor = await db.execute(
            "SELECT coin_name, confirmed_index, spent_index, coinbase, puzzle_hash, coin_parent, amount, timestamp "
            "FROM coin_record WHERE confirmed_index=? OR spent_index=?",
            (height, height),
        )
        coin_rows = await cursor.fetchall()
        await cursor.close()
        for coin_row in coin_rows:
            name = bytes32(coin_row[0])
            confirmed = int(coin_row[1])
            spent = int(coin_row[2])
            amount_raw = coin_row[6]
            amount = int.from_bytes(amount_raw, "big") if isinstance(amount_raw, bytes) else int(amount_raw)
            coin = Coin(bytes32(coin_row[5]), bytes32(coin_row[4]), uint64(amount))
            timestamp = max(timestamp, int(coin_row[7]))
            if confirmed == height:
                if bool(coin_row[3]):
                    rewards.append(coin)
                else:
                    additions.append((name, coin, spent == -1))
            if spent == height:
                removals.append(name)
        if additions or rewards:
            names = [name for name, _, _ in additions]
            names.extend(coin.name() for coin in rewards)
            try:
                hint_cursor = await db.execute(
                    f"SELECT coin_id, hint FROM hints WHERE coin_id IN ({','.join('?' for _ in names)})",
                    names,
                )
            except aiosqlite.OperationalError:
                hint_cursor = None
            if hint_cursor is not None:
                for coin_id, hint in await hint_cursor.fetchall():
                    if isinstance(hint, bytes) and len(hint) > 0:
                        hints.append((bytes32(coin_id), bytes(hint)))
                await hint_cursor.close()
    if additions or rewards or removals:
        await coin_store.new_block(uint32(height), uint64(timestamp), rewards, additions, removals)
    if hints:
        await hint_store.add_hints(hints)
    await block_store.set_peak(header_hash)


async def _rollback_to_common(
    db: aiosqlite.Connection,
    chain: ChainDB,
    block_store: RocksBlockStore,
    coin_store: RocksCoinStore,
    height: int,
) -> None:
    fork = height
    fork_hash: bytes32 | None = None
    while fork >= 0:
        sqlite_hash = await _main_hash(db, fork)
        rocks_hash = await block_store.main_chain_hash_at(fork)
        if sqlite_hash is not None and sqlite_hash == rocks_hash:
            fork_hash = sqlite_hash
            break
        fork -= 1
    if fork_hash is None:
        raise RuntimeError("SQLite and the RocksDB copy have no common block. Abort and start the copy again.")
    async with chain.writer():
        await coin_store.rollback_to_block(fork)
        await block_store.rollback(fork)
        await block_store.set_peak(fork_hash)


async def _peak(db: aiosqlite.Connection) -> tuple[bytes32 | None, int]:
    cursor = await db.execute("SELECT hash FROM current_peak WHERE key=0")
    row = await cursor.fetchone()
    await cursor.close()
    if row is None or row[0] is None:
        return None, 0
    header_hash = bytes32(row[0])
    cursor = await db.execute("SELECT height FROM full_blocks WHERE header_hash=?", (header_hash,))
    height_row = await cursor.fetchone()
    await cursor.close()
    if height_row is None:
        return header_hash, 0
    return header_hash, int(height_row[0])


async def _peak_height(db: aiosqlite.Connection) -> int:
    _hash, height = await _peak(db)
    return height


async def _main_hash(db: aiosqlite.Connection, height: int) -> bytes32 | None:
    cursor = await db.execute(
        "SELECT header_hash FROM full_blocks WHERE height=? AND in_main_chain=1",
        (height,),
    )
    row = await cursor.fetchone()
    await cursor.close()
    if row is None:
        return None
    return bytes32(row[0])


async def _main_block_row(db: aiosqlite.Connection, height: int) -> aiosqlite.Row | None:
    cursor = await db.execute(
        "SELECT header_hash, prev_hash, height, sub_epoch_summary, is_fully_compactified, in_main_chain, "
        "block, block_record FROM full_blocks WHERE height=? AND in_main_chain=1",
        (height,),
    )
    row = await cursor.fetchone()
    await cursor.close()
    return row


async def _count(db: aiosqlite.Connection, sql: str) -> int:
    cursor = await db.execute(sql)
    row = await cursor.fetchone()
    await cursor.close()
    return 0 if row is None else int(row[0])


async def _meta(chain: ChainDB, key: bytes) -> bytes | None:
    async with chain.reader_no_transaction() as view:
        return await view.get(CF_META, key)


async def _put_meta(chain: ChainDB, key: bytes, value: bytes) -> None:
    await _put_meta_many(chain, [(key, value)])


async def _put_meta_many(chain: ChainDB, items: list[tuple[bytes, bytes]]) -> None:
    async with chain.writer() as session:
        for key, value in items:
            session.put(CF_META, key, value)


def _u64(raw: bytes | None) -> int:
    if raw is None:
        return 0
    return int.from_bytes(raw, "little", signed=False)


def _check_stop(stop: asyncio.Event | None) -> None:
    if stop is not None and stop.is_set():
        raise MigrationPaused(
            "Migration paused. Run chia db migrate again to continue. The SQLite file was not changed."
        )


def _check_source_space(sqlite_path: Path) -> None:
    free = shutil.disk_usage(sqlite_path.parent).free
    if free < _SOURCE_HEADROOM:
        raise MigrationPaused(
            f"Stopped the coin snapshot because only {free:,} bytes are free next to the SQLite file. "
            "Free about 10 GB and run chia db migrate again. Copied blocks are kept. Coins will be recopied."
        )


def _report(progress: Callable[[MigrateProgress], None] | None, update: MigrateProgress) -> None:
    if progress is not None:
        progress(update)
