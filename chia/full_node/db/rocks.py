from __future__ import annotations

import asyncio
import logging
import os
import queue
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from chia.full_node.db.file_chip import (
    CHIP_FAMILIES,
    CHUNK_BYTES,
    FILE_CAP,
    REORG_HEADROOM_BLOCKS,
    SMALL_BYTES,
    ChipChoice,
    SstSpan,
    families_over_cap,
    parse_sstables,
    select_chip_range,
)
from chia.full_node.db.keys import CF_COINS_BY_PARENT, COLUMN_FAMILIES, UNCOMPRESSED_FAMILIES
from chia.full_node.db.ops import Delete, DeleteRange, Op, Put, key_in_range

log = logging.getLogger(__name__)

_T = TypeVar("_T")


def _deliver_call_result(future: asyncio.Future[object], outcome: tuple[object] | BaseException) -> None:
    """Hand a database-thread result back to its caller.

    Shutdown can cancel that caller while the thread is still inside the call.
    ``Future.set_result`` on an already finished future raises, and the node
    then sits at zero CPU with the database still open.
    """
    if future.done():
        return
    if isinstance(outcome, BaseException):
        future.set_exception(outcome)
        return
    (result,) = outcome
    future.set_result(result)


class RocksBackend:
    """
    rocksdict access serialized on one thread.

    Reads and writes share that thread so a committed scan cannot observe a
    partial batch. The event loop only waits on the result.
    """

    def __init__(self, path: Path, *, sync: str = "NORMAL", bulk: bool = False) -> None:
        self._path = path
        self._sync = sync
        self._bulk = bulk
        self._calls: queue.Queue[tuple[Callable[[], object], asyncio.Future[object]] | None] = queue.Queue()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread = threading.Thread(target=self._run, name="rocksdb", daemon=True)
        self._ready = threading.Event()
        self._open_error: BaseException | None = None
        self._db: object | None = None
        self._cache: object | None = None
        self._memtables: object | None = None
        self._families: dict[str, object] = {}
        self._chip_stop = threading.Event()
        self._chip_pause = threading.Event()
        self._chip_lock = threading.Lock()
        self._chip_thread: threading.Thread | None = None
        # File numbers whose last pack did not reduce the count. A reorg mints
        # new file numbers, so those heights are packed again. This is not a
        # remembered height.
        self._chip_stuck: set[int] = set()
        self._chip_file_cap = FILE_CAP
        self._chip_chunk_bytes = CHUNK_BYTES
        self._chip_small_bytes = SMALL_BYTES
        self._chip_reorg_headroom = REORG_HEADROOM_BLOCKS
        # Test hook. The running node leaves this unset and compacts for real.
        self._chip_compact: Callable[[ChipChoice], None] | None = None
        self._thread.start()
        self._ready.wait()
        if self._open_error is not None:
            raise self._open_error

    def _run(self) -> None:
        try:
            self._db, self._cache, self._memtables = _open_db(self._path, self._sync, bulk=self._bulk)
            # rocksdict flushes a column family when its handle is discarded.
            # A handle per read was flushing on every lookup and leaving a new
            # write-ahead file behind. Keep one handle per family until close.
            self._families = {name: self._db.get_column_family(name) for name in COLUMN_FAMILIES}  # type: ignore[attr-defined]
        except BaseException as exc:  # pragma: no cover
            self._open_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._calls.get()
            if item is None:
                # Dropping the handles flushes each family once, so the
                # write-ahead files can be removed before the process exits.
                self._families = {}
                db = self._db
                if db is not None:
                    db.close()  # type: ignore[attr-defined]
                self._db = None
                self._cache = None
                self._memtables = None
                return
            fn, future = item
            loop = self._loop
            assert loop is not None
            try:
                outcome: tuple[object] | BaseException = (fn(),)
            except BaseException as exc:
                outcome = exc
            # Checked on the loop thread: the future may already be cancelled.
            loop.call_soon_threadsafe(_deliver_call_result, future, outcome)

    async def _call(self, fn: Callable[[], _T]) -> _T:
        loop = asyncio.get_running_loop()
        self._loop = loop
        future: asyncio.Future[object] = loop.create_future()
        self._calls.put((fn, future))
        return await future  # type: ignore[return-value]

    async def get(self, cf: str, key: bytes) -> bytes | None:
        def read() -> bytes | None:
            family = self._families[cf]
            try:
                value = family[key]
            except KeyError:
                return None
            if value is None:
                return None
            return bytes(value)

        return await self._call(read)

    async def get_many(self, cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
        def read() -> dict[bytes, bytes]:
            family = self._families[cf]
            found: dict[bytes, bytes] = {}
            for key in keys:
                try:
                    value = family[key]
                except KeyError:
                    continue
                if value is not None:
                    found[key] = bytes(value)
            return found

        return await self._call(read)

    async def scan(
        self, cf: str, start: bytes, end: bytes | None, *, limit: int | None = None
    ) -> list[tuple[bytes, bytes]]:
        def read() -> list[tuple[bytes, bytes]]:
            family = self._families[cf]
            iterator = family.iter()
            iterator.seek(start)
            rows: list[tuple[bytes, bytes]] = []
            while iterator.valid():
                # A coin-index catch-up must not load every pending journal into one list.
                if limit is not None and len(rows) >= limit:
                    break
                key = bytes(iterator.key())
                if not key_in_range(key, start, end):
                    break
                value = iterator.value()
                rows.append((key, b"" if value is None else bytes(value)))
                iterator.next()
            return rows

        return await self._call(read)

    async def apply(self, ops: list[Op]) -> None:
        def write() -> None:
            db = self._db
            assert db is not None
            ordered, sorted_keys = _ordered_write_ops(ops)
            batch = _new_batch()
            handles: dict[str, object] = {}
            for op in ordered:
                handle = handles.get(op.cf)
                if handle is None:
                    handle = db.get_column_family_handle(op.cf)  # type: ignore[attr-defined]
                    handles[op.cf] = handle
                if isinstance(op, Put):
                    batch.put(op.key, op.value, handle)
                elif isinstance(op, Delete):
                    batch.delete(op.key, handle)
                elif isinstance(op, DeleteRange):
                    if hasattr(batch, "delete_range"):
                        batch.delete_range(op.start, op.end, handle)
                    else:  # pragma: no cover
                        family = self._families[op.cf]
                        iterator = family.iter()
                        iterator.seek(op.start)
                        while iterator.valid():
                            key = bytes(iterator.key())
                            if not key_in_range(key, op.start, op.end):
                                break
                            batch.delete(key, handle)
                            iterator.next()
            db.write(batch, self._write_options(ordered=sorted_keys))  # type: ignore[attr-defined]

        await self._call(write)

    async def write_column_batches(
        self,
        groups: list[tuple[str, list[tuple[bytes, bytes]]]],
        *,
        disable_wal: bool = False,
    ) -> None:
        """Write many keys per column family in one batch. Keys in each group must be sorted."""

        def write() -> None:
            db = self._db
            assert db is not None
            batch = _new_batch()
            for cf, pairs in groups:
                if len(pairs) == 0:
                    continue
                handle = db.get_column_family_handle(cf)  # type: ignore[attr-defined]
                for key, value in pairs:
                    batch.put(key, value, handle)
            db.write(batch, self._write_options(ordered=True, disable_wal=disable_wal))  # type: ignore[attr-defined]

        await self._call(write)

    async def checkpoint(self, destination: Path) -> None:
        def write() -> None:
            from rocksdict import Checkpoint

            db = self._db
            assert db is not None
            Checkpoint(db).create_checkpoint(str(destination))

        await self._call(write)

    def directory(self) -> Path:
        return self._path

    async def stored_family_bytes(self, names: list[str]) -> dict[str, int]:
        """On-disk size of each column family, used to pace the pack progress line."""

        def read() -> dict[str, int]:
            sizes: dict[str, int] = {}
            for name in names:
                family = self._families[name]
                value = family.property_int_value("rocksdb.total-sst-files-size")  # type: ignore[attr-defined]
                sizes[name] = 0 if value is None else int(value)
            return sizes

        return await self._call(read)

    async def merge_families(self, names: list[str]) -> None:
        """Merge each column family once, then turn background merges back on."""

        def merge() -> None:
            from rocksdict import BottommostLevelCompaction, CompactOptions

            db = self._db
            assert db is not None
            compact_opt = CompactOptions()
            # Files already on the bottom level are skipped unless this is forced.
            # The chain index can be tens of thousands of tiny files there.
            compact_opt.set_bottommost_level_compaction(BottommostLevelCompaction.force())
            for name in names:
                family = self._families[name]
                family.compact_range(None, None, compact_opt)
                # The copy opens with compaction held off and the file-count limits
                # pushed out of the way. Put the normal limits back or a finished
                # database can pile up files again.
                family.set_options(
                    {
                        "disable_auto_compactions": "false",
                        "level0_file_num_compaction_trigger": "8",
                        "level0_slowdown_writes_trigger": "32",
                        "level0_stop_writes_trigger": "64",
                    }
                )

        await self._call(merge)

    def set_file_chip_limits(
        self,
        *,
        file_cap: int | None = None,
        chunk_bytes: int | None = None,
        small_bytes: int | None = None,
        reorg_headroom: int | None = None,
    ) -> None:
        """Test hook. The running node keeps the defaults."""
        if file_cap is not None:
            self._chip_file_cap = file_cap
        if chunk_bytes is not None:
            self._chip_chunk_bytes = chunk_bytes
        if small_bytes is not None:
            self._chip_small_bytes = small_bytes
        if reorg_headroom is not None:
            self._chip_reorg_headroom = reorg_headroom

    def pause_file_chip(self, paused: bool) -> None:
        """Stop a pack from starting. A pack already rewriting a slice finishes that slice."""
        if paused:
            self._chip_pause.set()
            return
        self._chip_pause.clear()

    def request_file_chip(self, *, long_sync: bool, save_waiting: bool, last_save_slow: bool) -> None:
        """Start one pack if this is a quiet moment. Returns without waiting for it."""
        if long_sync or save_waiting or last_save_slow or self._chip_stop.is_set() or self._chip_pause.is_set():
            return
        # A block save asks for this while it holds the chain lock. The rewrite
        # holds `_chip_lock` until it finishes, so waiting here would hold the
        # chain lock for the whole slice.
        if not self._chip_lock.acquire(blocking=False):
            return
        try:
            if self._chip_thread is not None and self._chip_thread.is_alive():
                return
            self._chip_thread = threading.Thread(target=self._chip_thread_main, name="rocksdb-file-chip", daemon=True)
            self._chip_thread.start()
        finally:
            self._chip_lock.release()

    async def pack_file_chunk(
        self,
        *,
        long_sync: bool,
        save_waiting: bool,
        last_save_slow: bool,
    ) -> tuple[str, int, int] | None:
        """Pack one slice and wait for it. The node uses `request_file_chip` instead."""
        if long_sync or save_waiting or last_save_slow or self._chip_stop.is_set() or self._chip_pause.is_set():
            return None

        def run() -> tuple[str, int, int] | None:
            with self._chip_lock:
                if self._chip_thread is not None and self._chip_thread.is_alive():
                    return None
                return self._chip_once()

        return await asyncio.to_thread(run)

    async def flush(self, cf: str) -> None:
        """Make this family's memtable into a file. The pack only sees files."""

        def write() -> None:
            family = self._families[cf]
            family.flush(True)  # type: ignore[attr-defined]

        await self._call(write)

    def _chip_thread_main(self) -> None:
        with self._chip_lock:
            try:
                self._chip_once()
            except Exception:
                log.exception("File pack failed")

    def _chip_once(self) -> tuple[str, int, int] | None:
        if self._chip_stop.is_set() or self._chip_pause.is_set():
            return None
        counts: dict[str, int] = {}
        listings: dict[str, list[SstSpan]] = {}
        for name in CHIP_FAMILIES:
            listed = self._list_family_files(name)
            if listed is None:
                continue
            count, spans = listed
            counts[name] = count
            listings[name] = spans
        for name in families_over_cap(counts, file_cap=self._chip_file_cap):
            if self._chip_stop.is_set() or self._chip_pause.is_set():
                return None
            choice = select_chip_range(
                listings[name],
                file_count=counts[name],
                file_cap=self._chip_file_cap,
                chunk_bytes=self._chip_chunk_bytes,
                small_bytes=self._chip_small_bytes,
                reorg_headroom=self._chip_reorg_headroom,
            )
            if choice is None or len(choice.start) == 0 or len(choice.end) == 0:
                continue
            if choice.file_numbers and all(number in self._chip_stuck for number in choice.file_numbers):
                continue
            packed = self._compact_choice(name, counts[name], choice)
            if packed is not None:
                return packed
        return None

    def _list_family_files(self, name: str) -> tuple[int, list[SstSpan]] | None:
        family = self._families.get(name)
        if family is None:
            return None
        text = family.property_value("rocksdb.sstables")  # type: ignore[attr-defined]
        if not isinstance(text, str):
            return None
        parsed = parse_sstables(text)
        if parsed is None:
            return None
        level_count = _family_file_count(family)
        if level_count is None:
            level_count = len(parsed)
        # A flush can land between the two property reads. A large gap means
        # the file list was not understood, so this round packs nothing.
        if abs(level_count - len(parsed)) > 2:
            return None
        return max(level_count, len(parsed)), parsed

    def _compact_choice(self, name: str, before: int, choice: ChipChoice) -> tuple[str, int, int] | None:
        if self._chip_stop.is_set() or self._chip_pause.is_set():
            return None
        from rocksdict import BottommostLevelCompaction, CompactOptions

        family = self._families[name]
        compact_opt = CompactOptions()
        # Bottom-level files are skipped unless this is forced. That is the
        # whole reason these height-ordered families never merge on their own.
        compact_opt.set_bottommost_level_compaction(BottommostLevelCompaction.force())
        started = time.monotonic()
        if self._chip_compact is not None:
            self._chip_compact(choice)
        else:
            family.compact_range(choice.start, choice.end, compact_opt)  # type: ignore[attr-defined]
        listed = self._list_family_files(name)
        after = before if listed is None else listed[0]
        elapsed = time.monotonic() - started
        if after >= before:
            self._chip_stuck.update(choice.file_numbers)
        else:
            self._chip_stuck.difference_update(choice.file_numbers)
        log.info(f"Packed {name} files {before} -> {after} in {elapsed:.1f}s")
        return name, before, after

    def _write_options(self, *, ordered: bool, disable_wal: bool = False) -> object:
        from rocksdict import WriteOptions

        options = WriteOptions()
        if disable_wal or self._sync.upper() == "OFF":
            options.disable_wal = True
        elif self._sync.upper() == "FULL":
            options.sync = True
        else:
            options.sync = False
        if ordered:
            options.memtable_insert_hint_per_batch = True
        return options

    async def close(self) -> None:
        self._loop = asyncio.get_running_loop()
        # The pack thread is the only other user of the column-family handles.
        # Wait for it before those handles are dropped.
        self._chip_stop.set()
        chip = self._chip_thread
        if chip is not None and chip.is_alive() and chip is not threading.current_thread():
            await asyncio.to_thread(chip.join)

        def _shutdown() -> None:
            self._calls.put(None)
            self._thread.join(timeout=30)

        await asyncio.to_thread(_shutdown)


def _family_file_count(family: object) -> int | None:
    total = 0
    seen = False
    for level in range(7):
        value = family.property_int_value(f"rocksdb.num-files-at-level{level}")  # type: ignore[attr-defined]
        if value is None:
            continue
        seen = True
        total += int(value)
    if not seen:
        return None
    return total


def _new_batch() -> object:
    from rocksdict import WriteBatch

    try:
        return WriteBatch(raw_mode=True)
    except TypeError:  # pragma: no cover
        return WriteBatch()


def _cf_options(name: str) -> object:
    from rocksdict import DBCompressionType, Options

    options = Options(raw_mode=True)
    if name in UNCOMPRESSED_FAMILIES:
        options.set_compression_type(DBCompressionType.none())
    else:
        options.set_compression_type(DBCompressionType.lz4())
    return options


# Same shape as the full node debug.log: log_maxfilesrotation and log_maxbytesrotation.
# Seven archived LOG.old files, each rolled at 50 MB. The live LOG is separate.
_KEEP_DIAGNOSTIC_LOGS = 7
_MAX_DIAGNOSTIC_LOG_BYTES = 50 * 1024 * 1024


def _cap_diagnostic_logs(options: object) -> None:
    options.set_keep_log_file_num(_KEEP_DIAGNOSTIC_LOGS)  # type: ignore[attr-defined]
    options.set_max_log_file_size(_MAX_DIAGNOSTIC_LOG_BYTES)  # type: ignore[attr-defined]


# A write-ahead file is deleted only after every column family has flushed past it.
# Keep the total small so a long run folds them into the data files itself.
_MAX_WAL_BYTES = 64 * 1024 * 1024
# One cache for every column family. 64 MB is enough to hold recent index blocks
# and still leaves room on a 4 GB machine. The library default is about 9 MB.
_BLOCK_CACHE_BYTES = 64 * 1024 * 1024
# All column-family memtables share this. The copy uses 64 MB times 3 per family,
# which can reach several GB. 256 MB is the steady-state budget.
_MEMTABLE_BUDGET_BYTES = 256 * 1024 * 1024
_RUNTIME_WRITE_BUFFER_BYTES = 16 * 1024 * 1024
_RUNTIME_WRITE_BUFFERS = 2


def _cap_wal(options: object) -> None:
    options.set_max_total_wal_size(_MAX_WAL_BYTES)  # type: ignore[attr-defined]
    options.set_atomic_flush(True)  # type: ignore[attr-defined]


def _relax_wal_for_copy(options: object) -> None:
    # The 64 MB cap makes a long-running node fold its logs. During the copy it
    # stalls every large batch to flush. Leave the copy uncapped.
    options.set_max_total_wal_size(0)  # type: ignore[attr-defined]
    options.set_atomic_flush(False)  # type: ignore[attr-defined]


def _share_block_cache(options: object, cache: object) -> None:
    from rocksdict import BlockBasedOptions

    table = BlockBasedOptions()
    table.set_block_cache(cache)  # type: ignore[arg-type]
    options.set_block_based_table_factory(table)  # type: ignore[attr-defined]


def _cap_memtables(options: object, manager: object) -> None:
    options.set_write_buffer_manager(manager)  # type: ignore[attr-defined]
    options.set_write_buffer_size(_RUNTIME_WRITE_BUFFER_BYTES)  # type: ignore[attr-defined]
    options.set_max_write_buffer_number(_RUNTIME_WRITE_BUFFERS)  # type: ignore[attr-defined]


def _db_options() -> object:
    from rocksdict import Options

    options = Options(raw_mode=True)
    options.create_if_missing(True)
    options.create_missing_column_families(True)
    _cap_diagnostic_logs(options)
    # 16 MB is the running node. The copy replaces this with 64 MB buffers.
    options.set_write_buffer_size(16 * 1024 * 1024)
    options.set_max_write_buffer_number(2)
    options.set_level_zero_file_num_compaction_trigger(8)
    options.set_level_zero_slowdown_writes_trigger(32)
    options.set_level_zero_stop_writes_trigger(64)
    cpus = os.cpu_count() or 2
    options.set_max_background_jobs(max(cpus, 2))
    return options


def _ordered_write_ops(ops: list[Op]) -> tuple[list[Op], bool]:
    """Sort puts by column family and key so RocksDB can insert them in order.

    A range delete stays in the original order. For a single key, the last put or delete wins.
    """
    if any(isinstance(op, DeleteRange) for op in ops):
        return ops, False
    final: dict[tuple[str, bytes], Put | Delete] = {}
    for op in ops:
        if isinstance(op, Put | Delete):
            final[op.cf, op.key] = op
    deletes = sorted((op for op in final.values() if isinstance(op, Delete)), key=lambda op: (op.cf, op.key))
    puts = sorted((op for op in final.values() if isinstance(op, Put)), key=lambda op: (op.cf, op.key))
    return [*deletes, *puts], True


def _apply_bulk_options(options: object) -> None:
    # Hold the merge until the copy finishes. Merging while the coins are still
    # being written makes the SSD bounce between the two jobs.
    options.set_write_buffer_size(64 * 1024 * 1024)  # type: ignore[attr-defined]
    options.set_max_write_buffer_number(3)  # type: ignore[attr-defined]
    options.set_disable_auto_compactions(True)  # type: ignore[attr-defined]
    options.set_level_zero_file_num_compaction_trigger(1_000_000)  # type: ignore[attr-defined]
    options.set_level_zero_slowdown_writes_trigger(1_000_000)  # type: ignore[attr-defined]
    options.set_level_zero_stop_writes_trigger(1_000_000)  # type: ignore[attr-defined]


def _open_db(path: Path, sync: str, *, bulk: bool = False) -> tuple[object, object, object | None]:
    from rocksdict import Cache, Rdict, WriteBufferManager, WriteOptions

    path.mkdir(parents=True, exist_ok=True)
    location = str(path)
    # Held for the life of the database. Dropping them drops the shared cache.
    cache = Cache(_BLOCK_CACHE_BYTES)
    memtables = None if bulk else WriteBufferManager(_MEMTABLE_BUDGET_BYTES, False)
    options = _db_options()
    families = {name: _cf_options(name) for name in COLUMN_FAMILIES}
    if bulk:
        _apply_bulk_options(options)
        _relax_wal_for_copy(options)
        for family_options in families.values():
            _apply_bulk_options(family_options)
    else:
        _cap_wal(options)
        _share_block_cache(options, cache)
        _cap_memtables(options, memtables)
        for family_options in families.values():
            _share_block_cache(family_options, cache)
            _cap_memtables(family_options, memtables)
    if (path / "CURRENT").exists():
        from rocksdict import Options

        loaded, existing = Options.load_latest(location, cache=cache)
        loaded.create_missing_column_families(True)
        _cap_diagnostic_logs(loaded)
        if bulk:
            _apply_bulk_options(loaded)
            _relax_wal_for_copy(loaded)
            for family_options in existing.values():
                _apply_bulk_options(family_options)
        else:
            _cap_wal(loaded)
            _cap_memtables(loaded, memtables)
            for family_options in existing.values():
                _cap_memtables(family_options, memtables)
        for name, family_options in families.items():
            existing.setdefault(name, family_options)
        drop_parent = CF_COINS_BY_PARENT in existing
        db = Rdict(location, options=loaded, column_families=existing)
        if drop_parent:
            # Children are read from the block that spent the parent. This family
            # is a second copy of that fact and is safe to delete.
            db.drop_column_family(CF_COINS_BY_PARENT)
    else:
        db = Rdict(location, options=options, column_families=families)
    write_options = WriteOptions()
    if sync.upper() == "FULL":
        write_options.sync = True
    elif sync.upper() == "OFF":
        write_options.disable_wal = True
    else:
        write_options.sync = False
    db.set_write_options(write_options)
    return db, cache, memtables
