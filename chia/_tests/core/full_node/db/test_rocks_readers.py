from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest

from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.keys import CF_BLOCK_BLOBS, CF_META
from chia.full_node.db.ops import DeleteRange, Put
from chia.full_node.db.rocks import RocksBackend
from chia.util.task_referencer import create_referenced_task


class _CountFamily:
    def __init__(self, family: object) -> None:
        self._family = family
        self.calls: list[object] = []

    def get(self, key: object, *args: object, **kwargs: object) -> object:
        self.calls.append(key)
        return self._family.get(key, *args, **kwargs)  # type: ignore[attr-defined]

    def __getitem__(self, key: object) -> object:
        return self._family[key]  # type: ignore[index]


class _HoldIter:
    def __init__(self, inner: object, entered: threading.Event, release: threading.Event) -> None:
        self._inner = inner
        self._entered = entered
        self._release = release
        self._held = False

    def seek(self, key: bytes) -> None:
        self._inner.seek(key)  # type: ignore[attr-defined]

    def valid(self) -> bool:
        return bool(self._inner.valid())  # type: ignore[attr-defined]

    def key(self) -> object:
        return self._inner.key()  # type: ignore[attr-defined]

    def value(self) -> object:
        return self._inner.value()  # type: ignore[attr-defined]

    def next(self) -> None:
        if not self._held:
            self._held = True
            self._entered.set()
            assert self._release.wait(30)
        self._inner.next()  # type: ignore[attr-defined]


class _HoldFamily:
    def __init__(self, family: object, entered: threading.Event, release: threading.Event) -> None:
        self._family = family
        self._entered = entered
        self._release = release

    def iter(self) -> _HoldIter:
        return _HoldIter(self._family.iter(), self._entered, self._release)  # type: ignore[attr-defined]


def test_db_readers_below_one_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "rejected.rocksdb"
    for reader_count in (0, -3, True):
        with pytest.raises(ValueError, match="db_readers"):
            RocksBackend(path, sync="OFF", reader_count=reader_count)
    assert not path.exists()


@pytest.mark.anyio
async def test_get_many_reads_the_list_once_and_skips_missing_keys(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "bytes.rocksdb", sync="OFF", reader_count=2)
    try:
        await backend.apply([Put(CF_META, b"a", b"one"), Put(CF_META, b"c", b"three")])
        counter = _CountFamily(backend._families[CF_META])
        backend._families[CF_META] = counter
        assert await backend.get_many(CF_META, []) == {}
        found = await backend.get_many(CF_META, [b"a", b"missing", b"c"])
        assert found == {b"a": b"one", b"c": b"three"}
        assert counter.calls == [[b"a", b"missing", b"c"]]
        assert await backend.get(CF_META, b"a") == b"one"
        assert counter.calls == [[b"a", b"missing", b"c"]]
    finally:
        await backend.close()


@pytest.mark.anyio
async def test_reader_count_caps_the_multiget_threads(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "eight.rocksdb", sync="OFF", reader_count=8)
    try:
        assert [thread.name for thread in backend._reader_threads] == [f"rocksdb-read-{index}" for index in range(8)]
        names: list[str] = []
        real = backend._multiget

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            names.append(threading.current_thread().name)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        await backend.apply([Put(CF_META, b"a", b"v")])
        assert await backend.get_many(CF_META, [b"a"]) == {b"a": b"v"}
        assert names == ["rocksdb-read-0"]

        seen: list[str] = []
        family = backend._families[CF_META]

        class _Point:
            def __getitem__(self, key: object) -> object:
                seen.append(threading.current_thread().name)
                return family[key]  # type: ignore[index]

        backend._families[CF_META] = _Point()
        assert await backend.get(CF_META, b"a") == b"v"
        assert seen == ["rocksdb"]
    finally:
        await backend.close()


@pytest.mark.anyio
async def test_a_bulk_copy_keeps_multiget_on_the_writer(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "bulk.rocksdb", sync="OFF", bulk=True, reader_count=8)
    try:
        assert backend._reader_threads == []
        names: list[str] = []
        real = backend._multiget

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            names.append(threading.current_thread().name)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        await backend.apply([Put(CF_META, b"a", b"v")])
        assert await backend.get_many(CF_META, [b"a", b"missing"]) == {b"a": b"v"}
        assert names == ["rocksdb"]
    finally:
        await backend.close()


@pytest.mark.anyio
async def test_a_save_commits_during_one_multiget(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "overlap.rocksdb", sync="OFF", reader_count=1)
    release = threading.Event()
    entered = threading.Event()
    try:
        await backend.apply([Put(CF_META, b"a", b"old"), Put(CF_META, b"b", b"old")])
        real = backend._multiget

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            entered.set()
            assert release.wait(30)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        read = create_referenced_task(backend.get_many(CF_META, [b"a", b"b"]))
        assert await asyncio.to_thread(entered.wait, 30)
        await asyncio.wait_for(backend.apply([Put(CF_META, b"a", b"new")]), timeout=2)
        assert not read.done()
        release.set()
        assert await read == {b"a": b"new", b"b": b"old"}
    finally:
        release.set()
        await backend.close()


@pytest.mark.anyio
async def test_a_small_multiget_runs_beside_a_large_one_when_the_cap_allows(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "beside.rocksdb", sync="OFF", reader_count=4)
    release = threading.Event()
    large_entered = threading.Event()
    try:
        await backend.apply([Put(CF_META, b"a", b"v"), Put(CF_META, b"b", b"v")])
        real = backend._multiget

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            if len(keys) > 1:
                large_entered.set()
                assert release.wait(30)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        large = create_referenced_task(backend.get_many(CF_META, [b"a", b"b"]))
        assert await asyncio.to_thread(large_entered.wait, 30)
        small = create_referenced_task(backend.get_many(CF_META, [b"a"]))
        assert await asyncio.wait_for(small, timeout=2) == {b"a": b"v"}
        assert not large.done()
        release.set()
        assert await large == {b"a": b"v", b"b": b"v"}
    finally:
        release.set()
        await backend.close()


@pytest.mark.anyio
async def test_one_reader_holds_a_second_multiget_until_the_first_finishes(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "serial.rocksdb", sync="OFF", reader_count=1)
    release = threading.Event()
    entered = threading.Event()
    try:
        await backend.apply([Put(CF_META, b"a", b"v")])
        real = backend._multiget

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            if len(keys) > 1:
                entered.set()
                assert release.wait(30)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        large = create_referenced_task(backend.get_many(CF_META, [b"a", b"b"]))
        assert await asyncio.to_thread(entered.wait, 30)
        small = create_referenced_task(backend.get_many(CF_META, [b"a"]))
        await asyncio.sleep(0.05)
        assert not small.done()
        release.set()
        assert await large == {b"a": b"v"}
        assert await small == {b"a": b"v"}
    finally:
        release.set()
        await backend.close()


@pytest.mark.anyio
async def test_a_fifth_multiget_waits_for_a_free_reader(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "fifth.rocksdb", sync="OFF", reader_count=4)
    release = threading.Event()
    entered = 0
    lock = threading.Lock()
    try:
        await backend.apply([Put(CF_META, b"a", b"v")])
        real = backend._multiget

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            nonlocal entered
            with lock:
                entered += 1
            assert release.wait(30)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        tasks = [create_referenced_task(backend.get_many(CF_META, [b"a", bytes([index])])) for index in range(4)]
        for _ in range(50):
            if entered >= 4:
                break
            await asyncio.sleep(0.01)
        assert entered == 4
        fifth = create_referenced_task(backend.get_many(CF_META, [b"a"]))
        await asyncio.sleep(0.05)
        assert entered == 4
        assert not fifth.done()
        release.set()
        found = await asyncio.gather(*tasks, fifth)
        assert entered == 5
        assert all(item[b"a"] == b"v" for item in found)
    finally:
        release.set()
        await backend.close()


@pytest.mark.anyio
async def test_a_scan_does_not_see_the_commit_waiting_behind_it(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "scan.rocksdb", sync="OFF", reader_count=4)
    release = threading.Event()
    entered = threading.Event()
    try:
        keys = [bytes([index]) for index in range(3)]
        await backend.apply([Put(CF_META, key, b"old") for key in keys])
        family = backend._families[CF_META]
        backend._families[CF_META] = _HoldFamily(family, entered, release)
        scan = create_referenced_task(backend.scan(CF_META, b"", None))
        assert await asyncio.to_thread(entered.wait, 30)
        save = create_referenced_task(backend.apply([Put(CF_META, key, b"new") for key in keys]))
        queued = False
        for _ in range(50):
            if backend._calls.qsize() >= 1:
                queued = True
                break
            await asyncio.sleep(0.01)
        assert queued
        assert not save.done()
        release.set()
        assert await scan == [(key, b"old") for key in keys]
        await asyncio.wait_for(save, timeout=2)
        backend._families[CF_META] = family
        assert await backend.get_many(CF_META, keys) == {key: b"new" for key in keys}
    finally:
        release.set()
        await backend.close()


@pytest.mark.anyio
async def test_another_task_does_not_see_an_uncommitted_key(tmp_path: Path) -> None:
    chain = ChainDB(RocksBackend(tmp_path / "staged.rocksdb", sync="OFF", reader_count=2))
    release = asyncio.Event()
    try:
        started = asyncio.Event()

        async def writer() -> None:
            async with chain.writer() as session:
                session.put(CF_META, b"k", b"new")
                assert await session.get_many(CF_META, [b"k"]) == {b"k": b"new"}
                started.set()
                await release.wait()

        task = create_referenced_task(writer())
        await started.wait()
        async with chain.reader_no_transaction() as view:
            assert await view.get_many(CF_META, [b"k"]) == {}
            assert await view.get(CF_META, b"k") is None
        release.set()
        await task
        async with chain.reader_no_transaction() as view:
            assert await view.get_many(CF_META, [b"k"]) == {b"k": b"new"}
    finally:
        release.set()
        await chain.close()


@pytest.mark.anyio
async def test_close_waits_for_an_inflight_multiget(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "close.rocksdb", sync="OFF", reader_count=2)
    release = threading.Event()
    entered = threading.Event()
    closing: asyncio.Task[None] | None = None
    try:
        await backend.apply([Put(CF_META, b"a", b"v")])
        real = backend._multiget

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            entered.set()
            assert release.wait(30)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        read = create_referenced_task(backend.get_many(CF_META, [b"a"]))
        assert await asyncio.to_thread(entered.wait, 30)
        closing = create_referenced_task(backend.close())
        await asyncio.sleep(0.05)
        assert not closing.done()
        release.set()
        assert await read == {b"a": b"v"}
        await closing
    finally:
        release.set()
        if closing is None or not closing.done():
            await backend.close()


@pytest.mark.anyio
async def test_a_cancelled_multiget_can_close_and_reopen(tmp_path: Path) -> None:
    path = tmp_path / "reopen.rocksdb"
    backend = RocksBackend(path, sync="OFF", reader_count=2)
    release = threading.Event()
    entered = threading.Event()
    try:
        await backend.apply([Put(CF_META, b"a", b"v")])
        real = backend._multiget

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            entered.set()
            assert release.wait(30)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        read = create_referenced_task(backend.get_many(CF_META, [b"a"]))
        assert await asyncio.to_thread(entered.wait, 30)
        read.cancel()
        release.set()
        with pytest.raises(asyncio.CancelledError):
            await read
        await backend.close()
        reopened = RocksBackend(path, sync="OFF", reader_count=1)
        try:
            assert await reopened.get(CF_META, b"a") == b"v"
            assert await reopened.get_many(CF_META, [b"a", b"missing"]) == {b"a": b"v"}
        finally:
            await reopened.close()
    finally:
        release.set()
        if backend._thread.is_alive():
            await backend.close()


@pytest.mark.anyio
async def test_multiget_pairs_reversed_duplicate_and_empty_values(tmp_path: Path) -> None:
    """Request order is not key order. A sorted reply would attach the wrong bytes."""
    backend = RocksBackend(tmp_path / "order.rocksdb", sync="OFF", reader_count=2)
    try:
        await backend.apply(
            [
                Put(CF_META, b"\x00", b"low"),
                Put(CF_META, b"\x7f", b""),
                Put(CF_META, b"\xff", b"high"),
            ]
        )
        asked = [b"\xff", b"\x00", b"\x00", b"missing", b"\x7f"]
        found = await backend.get_many(CF_META, asked)
        assert asked == [b"\xff", b"\x00", b"\x00", b"missing", b"\x7f"]
        assert found == {b"\xff": b"high", b"\x00": b"low", b"\x7f": b""}
        assert all(type(value) is bytes for value in found.values())
        await backend.apply([Put(CF_META, b"\xff", b"replaced")])
        assert found[b"\xff"] == b"high"
    finally:
        await backend.close()


@pytest.mark.anyio
async def test_another_task_does_not_see_an_uncommitted_range_delete(tmp_path: Path) -> None:
    chain = ChainDB(RocksBackend(tmp_path / "range.rocksdb", sync="OFF", reader_count=2))
    release = asyncio.Event()
    try:
        async with chain.writer() as session:
            for key in (b"\x01", b"\x02", b"\x03"):
                session.put(CF_META, key, b"v")
        started = asyncio.Event()

        async def writer() -> None:
            async with chain.writer() as session:
                session.delete_range(CF_META, b"\x01", b"\x03")
                assert await session.get_many(CF_META, [b"\x01", b"\x02", b"\x03"]) == {b"\x03": b"v"}
                started.set()
                await release.wait()

        task = create_referenced_task(writer())
        await started.wait()
        async with chain.reader_no_transaction() as view:
            assert await view.get_many(CF_META, [b"\x01", b"\x02", b"\x03"]) == {
                b"\x01": b"v",
                b"\x02": b"v",
                b"\x03": b"v",
            }
        release.set()
        await task
        async with chain.reader_no_transaction() as view:
            assert await view.get_many(CF_META, [b"\x01", b"\x02", b"\x03"]) == {b"\x03": b"v"}
    finally:
        release.set()
        await chain.close()


@pytest.mark.anyio
async def test_a_multiget_started_during_close_fails_instead_of_waiting(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "closing.rocksdb", sync="OFF", reader_count=2)
    release = threading.Event()
    entered = threading.Event()
    closing: asyncio.Task[None] | None = None
    try:
        await backend.apply([Put(CF_META, b"a", b"v")])
        real = backend._multiget

        def wrapped(cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
            entered.set()
            assert release.wait(30)
            return real(cf, keys)

        backend._multiget = wrapped  # type: ignore[method-assign]
        read = create_referenced_task(backend.get_many(CF_META, [b"a"]))
        assert await asyncio.to_thread(entered.wait, 30)
        closing = create_referenced_task(backend.close())
        for _ in range(100):
            if backend._closing:
                break
            await asyncio.sleep(0.01)
        assert backend._closing
        with pytest.raises(RuntimeError, match="closing"):
            await asyncio.wait_for(backend.get_many(CF_META, [b"a"]), timeout=2)
        release.set()
        assert await read == {b"a": b"v"}
        await closing
    finally:
        release.set()
        if closing is None or not closing.done():
            await backend.close()


def _uniform(found: dict[bytes, bytes], keys: list[bytes]) -> bytes | None:
    values = {found.get(key) for key in keys}
    assert len(values) == 1
    return values.pop()


@pytest.mark.anyio
async def test_a_live_multiget_sees_one_whole_batch(tmp_path: Path) -> None:
    """The save starts after the engine call has entered, and one reply is one batch."""
    backend = RocksBackend(tmp_path / "live.rocksdb", sync="OFF", reader_count=4)
    started = threading.Event()
    try:
        count = 16_000
        keys = [index.to_bytes(4, "big") for index in range(count)]
        changed = keys[:48]
        await backend.apply([Put(CF_META, key, b"\x00\x00") for key in keys])
        await backend.apply([Put(CF_BLOCK_BLOBS, b"body", b"\x00\x00")])
        real = backend._multiget

        def wrapped(cf: str, batch: list[bytes]) -> dict[bytes, bytes]:
            if len(batch) > 1000:
                started.set()
            return real(cf, batch)

        backend._multiget = wrapped  # type: ignore[method-assign]
        saw_new = 0
        for generation in range(1, 21):
            stamp = generation.to_bytes(2, "big")
            started.clear()

            async def commit(stamp: bytes = stamp) -> None:
                assert await asyncio.to_thread(started.wait, 30)
                await backend.apply(
                    [Put(CF_META, key, stamp) for key in changed] + [Put(CF_BLOCK_BLOBS, b"body", stamp)]
                )

            read = create_referenced_task(backend.get_many(CF_META, keys))
            write = create_referenced_task(commit())
            found = await asyncio.wait_for(read, timeout=30)
            body = await asyncio.wait_for(backend.get(CF_BLOCK_BLOBS, b"body"), timeout=30)
            await asyncio.wait_for(write, timeout=30)
            seen = _uniform(found, changed)
            assert _uniform(found, keys[48:]) == b"\x00\x00"
            if seen == stamp:
                saw_new += 1
                assert body == stamp
        assert saw_new > 0
    finally:
        await backend.close()


@pytest.mark.anyio
async def test_a_live_multiget_sees_a_whole_range_delete(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "range-live.rocksdb", sync="OFF", reader_count=4)
    started = threading.Event()
    try:
        count = 16_000
        keys = [index.to_bytes(4, "big") for index in range(count)]
        inside = keys[400:800]
        await backend.apply([Put(CF_META, key, b"base") for key in [*keys, b"marker"]])
        real = backend._multiget

        def wrapped(cf: str, batch: list[bytes]) -> dict[bytes, bytes]:
            if len(batch) > 1000:
                started.set()
            return real(cf, batch)

        backend._multiget = wrapped  # type: ignore[method-assign]
        saw_new = 0
        asked = [*keys, b"marker"]
        for generation in range(1, 16):
            stamp = bytes([generation])
            started.clear()

            async def commit(stamp: bytes = stamp) -> None:
                assert await asyncio.to_thread(started.wait, 30)
                await backend.apply(
                    [
                        DeleteRange(CF_META, keys[400], keys[800]),
                        Put(CF_META, b"marker", stamp),
                        *[Put(CF_META, key, stamp) for key in inside],
                    ]
                )

            read = create_referenced_task(backend.get_many(CF_META, asked))
            write = create_referenced_task(commit())
            found = await asyncio.wait_for(read, timeout=30)
            await asyncio.wait_for(write, timeout=30)
            seen = _uniform(found, inside)
            assert found[b"marker"] == seen
            assert found[keys[800]] == b"base"
            assert _uniform(found, keys[:400]) == b"base"
            if seen == stamp:
                saw_new += 1
        assert saw_new > 0
    finally:
        await backend.close()
