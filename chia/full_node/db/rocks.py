from __future__ import annotations

import asyncio
import os
import queue
import threading
from collections.abc import Callable
from pathlib import Path
from typing import TypeVar

from chia.full_node.db.keys import COLUMN_FAMILIES, UNCOMPRESSED_FAMILIES
from chia.full_node.db.ops import Delete, DeleteRange, Op, Put, key_in_range

_T = TypeVar("_T")


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
        self._thread.start()
        self._ready.wait()
        if self._open_error is not None:
            raise self._open_error

    def _run(self) -> None:
        try:
            self._db = _open_db(self._path, self._sync, bulk=self._bulk)
        except BaseException as exc:  # pragma: no cover
            self._open_error = exc
            self._ready.set()
            return
        self._ready.set()
        while True:
            item = self._calls.get()
            if item is None:
                db = self._db
                if db is not None:
                    db.close()  # type: ignore[attr-defined]
                return
            fn, future = item
            loop = self._loop
            assert loop is not None
            try:
                result = fn()
            except BaseException as exc:
                loop.call_soon_threadsafe(future.set_exception, exc)
            else:
                loop.call_soon_threadsafe(future.set_result, result)

    async def _call(self, fn: Callable[[], _T]) -> _T:
        loop = asyncio.get_running_loop()
        self._loop = loop
        future: asyncio.Future[object] = loop.create_future()
        self._calls.put((fn, future))
        return await future  # type: ignore[return-value]

    async def get(self, cf: str, key: bytes) -> bytes | None:
        def read() -> bytes | None:
            family = self._db.get_column_family(cf)  # type: ignore[attr-defined]
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
            family = self._db.get_column_family(cf)  # type: ignore[attr-defined]
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

    async def scan(self, cf: str, start: bytes, end: bytes | None) -> list[tuple[bytes, bytes]]:
        def read() -> list[tuple[bytes, bytes]]:
            family = self._db.get_column_family(cf)  # type: ignore[attr-defined]
            iterator = family.iter()
            iterator.seek(start)
            rows: list[tuple[bytes, bytes]] = []
            while iterator.valid():
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
            batch = _new_batch()
            for op in ops:
                handle = db.get_column_family_handle(op.cf)  # type: ignore[attr-defined]
                if isinstance(op, Put):
                    batch.put(op.key, op.value, handle)
                elif isinstance(op, Delete):
                    batch.delete(op.key, handle)
                elif isinstance(op, DeleteRange):
                    if hasattr(batch, "delete_range"):
                        batch.delete_range(op.start, op.end, handle)
                    else:  # pragma: no cover
                        family = db.get_column_family(op.cf)  # type: ignore[attr-defined]
                        iterator = family.iter()
                        iterator.seek(op.start)
                        while iterator.valid():
                            key = bytes(iterator.key())
                            if not key_in_range(key, op.start, op.end):
                                break
                            batch.delete(key, handle)
                            iterator.next()
            db.write(batch)  # type: ignore[attr-defined]

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
            if not disable_wal:
                db.write(batch)  # type: ignore[attr-defined]
                return
            from rocksdict import WriteOptions

            options = WriteOptions()
            options.disable_wal = True
            options.memtable_insert_hint_per_batch = True
            db.write(batch, options)  # type: ignore[attr-defined]

        await self._call(write)

    async def checkpoint(self, destination: Path) -> None:
        def write() -> None:
            from rocksdict import Checkpoint

            db = self._db
            assert db is not None
            Checkpoint(db).create_checkpoint(str(destination))

        await self._call(write)

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
                family = db.get_column_family(name)  # type: ignore[attr-defined]
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

    async def close(self) -> None:
        self._loop = asyncio.get_running_loop()

        def _shutdown() -> None:
            self._calls.put(None)
            self._thread.join(timeout=30)

        await asyncio.to_thread(_shutdown)


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


def _db_options() -> object:
    from rocksdict import Options

    options = Options(raw_mode=True)
    options.create_if_missing(True)
    options.create_missing_column_families(True)
    # 256 MB times several buffers times every column family reserves tens of GB
    # before any block is stored. 16 MB is enough for the copy batches.
    options.set_write_buffer_size(16 * 1024 * 1024)
    options.set_max_write_buffer_number(2)
    options.set_level_zero_file_num_compaction_trigger(8)
    options.set_level_zero_slowdown_writes_trigger(32)
    options.set_level_zero_stop_writes_trigger(64)
    cpus = os.cpu_count() or 2
    options.set_max_background_jobs(max(cpus, 2))
    return options


def _apply_bulk_options(options: object) -> None:
    # Hold the merge until the copy finishes. Merging while the coins are still
    # being written makes the SSD bounce between the two jobs.
    options.set_write_buffer_size(64 * 1024 * 1024)  # type: ignore[attr-defined]
    options.set_max_write_buffer_number(3)  # type: ignore[attr-defined]
    options.set_disable_auto_compactions(True)  # type: ignore[attr-defined]
    options.set_level_zero_file_num_compaction_trigger(1_000_000)  # type: ignore[attr-defined]
    options.set_level_zero_slowdown_writes_trigger(1_000_000)  # type: ignore[attr-defined]
    options.set_level_zero_stop_writes_trigger(1_000_000)  # type: ignore[attr-defined]


def _open_db(path: Path, sync: str, *, bulk: bool = False) -> object:
    from rocksdict import Rdict, WriteOptions

    path.mkdir(parents=True, exist_ok=True)
    location = str(path)
    options = _db_options()
    families = {name: _cf_options(name) for name in COLUMN_FAMILIES}
    if bulk:
        _apply_bulk_options(options)
        for family_options in families.values():
            _apply_bulk_options(family_options)
    if (path / "CURRENT").exists():
        from rocksdict import Options

        loaded, existing = Options.load_latest(location)
        loaded.create_missing_column_families(True)
        if bulk:
            _apply_bulk_options(loaded)
            for family_options in existing.values():
                _apply_bulk_options(family_options)
        for name, family_options in families.items():
            existing.setdefault(name, family_options)
        db = Rdict(location, options=loaded, column_families=existing)
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
    return db
