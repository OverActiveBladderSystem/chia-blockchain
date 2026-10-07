from __future__ import annotations

import asyncio
import threading
from pathlib import Path

import pytest
from chia_rs.sized_bytes import bytes32

from chia.full_node.db.file_chip import (
    FILE_CAP,
    SLOW_SAVE_SECONDS,
    SstSpan,
    parse_sstables,
    select_chip_range,
)
from chia.full_node.db.keys import CF_BLOCKS_AT_HEIGHT, CF_COIN_DELTA, CF_MAIN_CHAIN, u32_be
from chia.full_node.db.ops import Delete, Put
from chia.full_node.db.rocks import RocksBackend
from chia.util.task_referencer import create_referenced_task


def _span(number: int, start: int, end: int, size: int, *, suffix: bytes = b"") -> SstSpan:
    return SstSpan(number, u32_be(start) + suffix, u32_be(end) + suffix, size)


def _gates(**overrides: bool | int) -> dict[str, bool | int]:
    values: dict[str, bool | int] = {
        "file_count": FILE_CAP + 1,
        "file_cap": FILE_CAP,
        "chunk_bytes": 64,
        "small_bytes": 32,
        "reorg_headroom": 0,
        "save_waiting": False,
        "last_save_slow": False,
        "chip_running": False,
        "long_sync": False,
    }
    values.update(overrides)
    return values


def test_parse_sstables_reads_inclusive_bounds() -> None:
    text = """
--- level 6 ---
  26:1086[0 .. 0]['00000000' seq:0, type:1 .. '00000002' seq:0, type:1](0)
  27:400[1 .. 1]['00000003' seq:0, type:1 .. '00000003' seq:0, type:1](0)
"""
    parsed = parse_sstables(text)
    assert parsed is not None
    assert parsed[0] == SstSpan(26, bytes.fromhex("00000000"), bytes.fromhex("00000002"), 1086)
    assert parsed[1].start == bytes.fromhex("00000003")
    assert parse_sstables("") == []
    assert parse_sstables("  1:5[0 .. 0]['zz' seq:0, type:1 .. 'aa' seq:0, type:1]") is None


def test_select_skips_when_the_chain_is_busy_or_under_the_cap() -> None:
    files = [_span(1, 0, 0, 10), _span(2, 1, 1, 10)]
    assert select_chip_range(files, **_gates(file_count=FILE_CAP)) is None
    assert select_chip_range(files, **_gates(long_sync=True)) is None
    assert select_chip_range(files, **_gates(save_waiting=True)) is None
    assert select_chip_range(files, **_gates(last_save_slow=True)) is None
    assert select_chip_range(files, **_gates(chip_running=True)) is None
    assert select_chip_range([_span(1, 0, 0, 10)], **_gates()) is None


def test_select_takes_the_oldest_small_run_and_stops_before_a_large_file() -> None:
    files = [
        _span(1, 0, 0, 10),
        _span(2, 1, 1, 10),
        _span(3, 2, 2, 10),
        _span(4, 3, 8, 100),
        _span(5, 9, 9, 10),
        _span(6, 10, 10, 10),
    ]
    choice = select_chip_range(files, **_gates(chunk_bytes=25))
    assert choice is not None
    assert choice.start == u32_be(0)
    assert choice.end == u32_be(1)
    assert choice.file_numbers == (1, 2)


def test_select_leaves_the_recent_tip_unpacked() -> None:
    files = [_span(1, 0, 0, 10), _span(2, 1, 1, 10), _span(3, 100, 100, 10), _span(4, 101, 101, 10)]
    choice = select_chip_range(files, **_gates(reorg_headroom=8, chunk_bytes=100))
    assert choice is not None
    assert choice.start == u32_be(0)
    assert choice.end == u32_be(1)
    assert select_chip_range(files[2:], **_gates(reorg_headroom=8)) is None


def test_select_rereads_a_reorg_instead_of_skipping_past_rewritten_heights() -> None:
    # A later listing: the packed file still covers 0..1, and the reorg wrote
    # two new small files on those same heights. The next slice includes them.
    packed = SstSpan(10, u32_be(0), u32_be(1), 100)
    rewritten = [
        _span(20, 0, 0, 10),
        _span(21, 1, 1, 10),
        packed,
        _span(4, 2, 5, 100),
        _span(5, 6, 6, 10),
        _span(6, 7, 7, 10),
    ]
    choice = select_chip_range(rewritten, **_gates(chunk_bytes=100, small_bytes=32))
    assert choice is not None
    assert choice.start == u32_be(0)
    assert choice.end == u32_be(1)
    assert choice.file_numbers == (20, 21)
    # The packed file is larger than one chunk, so that slice is left for later
    # and a run that does not touch it can still be packed.
    later = select_chip_range(
        [
            _span(20, 0, 0, 10),
            _span(21, 1, 1, 10),
            SstSpan(10, u32_be(0), u32_be(1), 500),
            _span(30, 5, 5, 10),
            _span(31, 6, 6, 10),
        ],
        **_gates(chunk_bytes=40, small_bytes=32),
    )
    assert later is not None
    assert later.start == u32_be(5)
    assert later.file_numbers == (30, 31)


def test_select_keeps_a_blocks_at_height_hash_suffix() -> None:
    suffix = b"\xab" * 32
    files = [
        _span(1, 0, 0, 10, suffix=suffix),
        _span(2, 1, 1, 10, suffix=suffix),
    ]
    choice = select_chip_range(files, **_gates(chunk_bytes=100))
    assert choice is not None
    assert choice.start == u32_be(0) + suffix
    assert choice.end == u32_be(1) + suffix


@pytest.mark.anyio
async def test_one_chunk_keeps_every_key(tmp_path: Path) -> None:
    path = tmp_path / "chip.rocksdb"
    backend = RocksBackend(path, sync="OFF")
    backend.set_file_chip_limits(file_cap=2, chunk_bytes=1, small_bytes=10_000_000, reorg_headroom=0)
    expected = {height: b"v" + bytes([height]) for height in range(6)}
    try:
        for height, value in expected.items():
            await backend.apply([Put(CF_MAIN_CHAIN, u32_be(height), value)])
            await backend.flush(CF_MAIN_CHAIN)
        assert await backend.pack_file_chunk(long_sync=True, save_waiting=False, last_save_slow=False) is None
        assert await backend.pack_file_chunk(long_sync=False, save_waiting=True, last_save_slow=False) is None
        assert await backend.pack_file_chunk(long_sync=False, save_waiting=False, last_save_slow=True) is None
        packed = await backend.pack_file_chunk(long_sync=False, save_waiting=False, last_save_slow=False)
        assert packed is not None
        family, before, after = packed
        assert family == CF_MAIN_CHAIN
        assert before > 2
        assert after < before
        assert after > 1
        for height, value in expected.items():
            assert await backend.get(CF_MAIN_CHAIN, u32_be(height)) == value
    finally:
        await backend.close()


@pytest.mark.anyio
async def test_reorg_rewrite_survives_later_packs(tmp_path: Path) -> None:
    path = tmp_path / "reorg-chip.rocksdb"
    backend = RocksBackend(path, sync="OFF")
    backend.set_file_chip_limits(file_cap=2, chunk_bytes=1, small_bytes=10_000_000, reorg_headroom=0)
    try:
        for height in range(6):
            await backend.apply([Put(CF_MAIN_CHAIN, u32_be(height), f"old-{height}".encode())])
            await backend.flush(CF_MAIN_CHAIN)
            await backend.apply([Put(CF_COIN_DELTA, u32_be(height), f"delta-{height}".encode())])
            await backend.flush(CF_COIN_DELTA)
        first = await backend.pack_file_chunk(long_sync=False, save_waiting=False, last_save_slow=False)
        assert first is not None

        # Roll the tip heights back the way a reorg does: drop one key, replace another.
        await backend.apply([Delete(CF_MAIN_CHAIN, u32_be(0)), Delete(CF_COIN_DELTA, u32_be(0))])
        await backend.flush(CF_MAIN_CHAIN)
        await backend.apply(
            [
                Put(CF_MAIN_CHAIN, u32_be(1), b"new-1"),
                Put(CF_COIN_DELTA, u32_be(1), b"new-delta-1"),
                Put(CF_BLOCKS_AT_HEIGHT, u32_be(1) + b"\x11" * 32, b"block-1"),
            ]
        )
        await backend.flush(CF_MAIN_CHAIN)

        for _ in range(8):
            await backend.pack_file_chunk(long_sync=False, save_waiting=False, last_save_slow=False)

        assert await backend.get(CF_MAIN_CHAIN, u32_be(0)) is None
        assert await backend.get(CF_COIN_DELTA, u32_be(0)) is None
        assert await backend.get(CF_MAIN_CHAIN, u32_be(1)) == b"new-1"
        assert await backend.get(CF_COIN_DELTA, u32_be(1)) == b"new-delta-1"
        assert await backend.get(CF_BLOCKS_AT_HEIGHT, u32_be(1) + b"\x11" * 32) == b"block-1"
        for height in range(2, 6):
            assert await backend.get(CF_MAIN_CHAIN, u32_be(height)) == f"old-{height}".encode()
            assert await backend.get(CF_COIN_DELTA, u32_be(height)) == f"delta-{height}".encode()
    finally:
        await backend.close()


class _LowLock:
    def __init__(self) -> None:
        self.calls = 0

    def has_waiters(self) -> bool:
        return False

    def acquire(self, *, priority: object) -> _LowLock:
        self.calls += 1
        return self

    async def __aenter__(self) -> None:
        return None

    async def __aexit__(self, *_exc: object) -> None:
        return None


class _Chain:
    def __init__(self) -> None:
        self.calls: list[dict[str, bool]] = []

    def request_file_chip(self, *, long_sync: bool, save_waiting: bool, last_save_slow: bool) -> None:
        self.calls.append({"long_sync": long_sync, "save_waiting": save_waiting, "last_save_slow": last_save_slow})


@pytest.mark.anyio
async def test_sqlite_offer_returns_without_the_chain_lock() -> None:
    from chia.full_node.full_node import FullNode

    node = FullNode.__new__(FullNode)
    node._chain_db = None
    await FullNode._offer_file_chip_when_idle(node)


@pytest.mark.anyio
async def test_rocks_idle_offer_takes_one_low_priority_lock() -> None:
    from chia.full_node.full_node import FullNode
    from chia.full_node.sync_store import SyncStore

    lock = _LowLock()
    chain = _Chain()

    class _Blockchain:
        priority_mutex = lock

    node = FullNode.__new__(FullNode)
    node._chain_db = chain  # type: ignore[assignment]
    node._blockchain = _Blockchain()  # type: ignore[assignment]
    node.sync_store = SyncStore()
    node._file_chip_save_noted = False
    node._last_save_had_waiters = False
    node._last_db_write_seconds = 0.0
    await FullNode._offer_file_chip_when_idle(node)
    assert lock.calls == 1
    assert chain.calls == [{"long_sync": False, "save_waiting": False, "last_save_slow": False}]


def test_select_leaves_full_size_files_alone() -> None:
    files = [_span(number, number, number, 64) for number in range(FILE_CAP + 10)]
    assert select_chip_range(files, **_gates(file_count=len(files), small_bytes=32, chunk_bytes=64)) is None


def test_offer_skips_sync_and_remembers_one_slow_save() -> None:
    from chia.full_node.full_node import FullNode
    from chia.full_node.sync_store import SyncStore

    peer = bytes32(b"\x22" * 32)
    lock = _LowLock()
    chain = _Chain()

    class _Blockchain:
        priority_mutex = lock

    node = FullNode.__new__(FullNode)
    node._chain_db = chain  # type: ignore[assignment]
    node._blockchain = _Blockchain()  # type: ignore[assignment]
    node.sync_store = SyncStore()
    node._file_chip_save_noted = False
    node._last_save_had_waiters = False
    node._last_db_write_seconds = 0.0

    node.sync_store.set_long_sync(True)
    node._file_chip_save_noted = True
    node._offer_file_chip()
    assert chain.calls == []
    assert node._file_chip_save_noted is True
    node.sync_store.set_long_sync(False)

    node.sync_store.set_sync_mode(True)
    node._offer_file_chip()
    assert chain.calls == []
    node.sync_store.set_sync_mode(False)

    node._last_db_write_seconds = SLOW_SAVE_SECONDS
    node.sync_store.increment_backtrack_syncing(peer)
    node._offer_file_chip()
    assert chain.calls == []
    assert node._file_chip_save_noted is True
    node.sync_store.decrement_backtrack_syncing(peer)
    node._offer_file_chip()
    assert chain.calls == [{"long_sync": False, "save_waiting": False, "last_save_slow": True}]
    assert node._file_chip_save_noted is False
    node._offer_file_chip()
    assert chain.calls[-1]["last_save_slow"] is False

    node._last_save_had_waiters = True
    node._offer_file_chip()
    assert chain.calls[-1]["save_waiting"] is True
    assert node._last_save_had_waiters is False
    node._offer_file_chip()
    assert chain.calls[-1]["save_waiting"] is False

    # A short batch does not record how long its saves took. Once the batch
    # finishes, one pack is allowed even if those saves would have been slow.
    node.sync_store.batch_syncing.add(peer)
    node._file_chip_save_noted = False
    node._last_db_write_seconds = SLOW_SAVE_SECONDS
    node._offer_file_chip()
    assert len(chain.calls) == 4
    node.sync_store.batch_syncing.discard(peer)
    node._offer_file_chip()
    assert chain.calls[-1] == {"long_sync": False, "save_waiting": False, "last_save_slow": False}


def _pack_limits(backend: RocksBackend, *, chunk_bytes: int) -> None:
    backend.set_file_chip_limits(file_cap=2, chunk_bytes=chunk_bytes, small_bytes=10_000_000, reorg_headroom=0)


async def _flush_main(backend: RocksBackend, heights: range) -> None:
    for height in heights:
        await backend.apply([Put(CF_MAIN_CHAIN, u32_be(height), b"v" + bytes([height]))])
        await backend.flush(CF_MAIN_CHAIN)


async def _join_chip(backend: RocksBackend) -> None:
    chip = backend._chip_thread
    assert chip is not None
    await asyncio.to_thread(chip.join, 30)
    assert not chip.is_alive()


@pytest.mark.anyio
async def test_one_pack_at_a_time_and_pause_blocks_the_next(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "one-pack.rocksdb", sync="OFF")
    _pack_limits(backend, chunk_bytes=1)
    entered = threading.Event()
    release = threading.Event()
    calls = 0
    try:
        await _flush_main(backend, range(4))
        backend.pause_file_chip(True)
        backend.request_file_chip(long_sync=False, save_waiting=False, last_save_slow=False)
        assert backend._chip_thread is None

        backend.pause_file_chip(False)
        original = backend._compact_choice

        def hold(name: str, before: int, choice: object) -> tuple[str, int, int] | None:
            nonlocal calls
            calls += 1
            entered.set()
            assert release.wait(timeout=30)
            return original(name, before, choice)

        backend._compact_choice = hold  # type: ignore[method-assign]
        backend.request_file_chip(long_sync=False, save_waiting=False, last_save_slow=False)
        assert await asyncio.to_thread(entered.wait, 30)
        first = backend._chip_thread
        assert first is not None and first.is_alive()
        await asyncio.wait_for(
            asyncio.to_thread(backend.request_file_chip, long_sync=False, save_waiting=False, last_save_slow=False),
            timeout=2,
        )
        assert calls == 1
        assert backend._chip_thread is first
        release.set()
        await _join_chip(backend)
        assert calls == 1
    finally:
        release.set()
        await backend.close()


@pytest.mark.anyio
async def test_writes_during_a_pack_survive(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "during-pack.rocksdb", sync="OFF")
    _pack_limits(backend, chunk_bytes=1)
    entered = threading.Event()
    release = threading.Event()
    stop = threading.Event()
    try:
        await _flush_main(backend, range(6))
        original = backend._compact_choice

        def hold(name: str, before: int, choice: object) -> tuple[str, int, int] | None:
            entered.set()
            assert release.wait(timeout=30)
            return original(name, before, choice)

        backend._compact_choice = hold  # type: ignore[method-assign]
        backend.request_file_chip(long_sync=False, save_waiting=False, last_save_slow=False)
        assert await asyncio.to_thread(entered.wait, 30)
        await backend.apply(
            [
                Delete(CF_MAIN_CHAIN, u32_be(0)),
                Put(CF_MAIN_CHAIN, u32_be(1), b"inside"),
                Delete(CF_MAIN_CHAIN, u32_be(4)),
                Put(CF_MAIN_CHAIN, u32_be(5), b"outside"),
            ]
        )

        async def hammer() -> None:
            while not stop.is_set():
                await backend.apply(
                    [
                        Delete(CF_MAIN_CHAIN, u32_be(0)),
                        Put(CF_MAIN_CHAIN, u32_be(1), b"inside"),
                        Delete(CF_MAIN_CHAIN, u32_be(4)),
                        Put(CF_MAIN_CHAIN, u32_be(5), b"outside"),
                    ]
                )

        writer = create_referenced_task(hammer())
        release.set()
        await _join_chip(backend)
        stop.set()
        await writer
        assert await backend.get(CF_MAIN_CHAIN, u32_be(0)) is None
        assert await backend.get(CF_MAIN_CHAIN, u32_be(1)) == b"inside"
        assert await backend.get(CF_MAIN_CHAIN, u32_be(2)) == b"v\x02"
        assert await backend.get(CF_MAIN_CHAIN, u32_be(3)) == b"v\x03"
        assert await backend.get(CF_MAIN_CHAIN, u32_be(4)) is None
        assert await backend.get(CF_MAIN_CHAIN, u32_be(5)) == b"outside"
    finally:
        release.set()
        stop.set()
        await backend.close()


@pytest.mark.anyio
async def test_close_waits_for_the_running_pack(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "close-pack.rocksdb", sync="OFF")
    _pack_limits(backend, chunk_bytes=1)
    entered = threading.Event()
    release = threading.Event()
    closing: asyncio.Task[None] | None = None
    try:
        await _flush_main(backend, range(4))
        original = backend._compact_choice

        def hold(name: str, before: int, choice: object) -> tuple[str, int, int] | None:
            entered.set()
            assert release.wait(timeout=30)
            return original(name, before, choice)

        backend._compact_choice = hold  # type: ignore[method-assign]
        backend.request_file_chip(long_sync=False, save_waiting=False, last_save_slow=False)
        assert await asyncio.to_thread(entered.wait, 30)
        closing = create_referenced_task(backend.close())
        await asyncio.sleep(0.05)
        assert not closing.done()
        release.set()
        await closing
    finally:
        release.set()
        if closing is not None and not closing.done():
            await closing
        elif backend._thread.is_alive():
            await backend.close()


@pytest.mark.anyio
async def test_packed_slice_includes_its_last_key(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "inclusive-end.rocksdb", sync="OFF")
    _pack_limits(backend, chunk_bytes=1)
    try:
        await _flush_main(backend, range(6))
        packed = await backend.pack_file_chunk(long_sync=False, save_waiting=False, last_save_slow=False)
        assert packed is not None
        listed = backend._list_family_files(CF_MAIN_CHAIN)
        assert listed is not None
        spans = listed[1]

        def covering(height: int) -> set[int]:
            key = u32_be(height)
            return {span.number for span in spans if span.start <= key <= span.end}

        assert covering(0) & covering(1)
        assert not covering(0) & covering(2)
        for height in range(6):
            assert await backend.get(CF_MAIN_CHAIN, u32_be(height)) == b"v" + bytes([height])
    finally:
        await backend.close()


@pytest.mark.anyio
async def test_a_pack_that_does_not_shrink_is_not_retried_until_a_new_file(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "stuck-pack.rocksdb", sync="OFF")
    _pack_limits(backend, chunk_bytes=10_000_000)
    calls = 0
    try:
        await _flush_main(backend, range(3))

        def noop(_choice: object) -> None:
            nonlocal calls
            calls += 1

        backend._chip_compact = noop
        first = await backend.pack_file_chunk(long_sync=False, save_waiting=False, last_save_slow=False)
        assert first is not None
        assert calls == 1
        assert await backend.pack_file_chunk(long_sync=False, save_waiting=False, last_save_slow=False) is None
        assert calls == 1

        await backend.apply([Put(CF_MAIN_CHAIN, u32_be(3), b"v\x03")])
        await backend.flush(CF_MAIN_CHAIN)
        again = await backend.pack_file_chunk(long_sync=False, save_waiting=False, last_save_slow=False)
        assert again is not None
        assert calls == 2
        for height in range(4):
            assert await backend.get(CF_MAIN_CHAIN, u32_be(height)) == b"v" + bytes([height])
    finally:
        await backend.close()


@pytest.mark.anyio
async def test_migration_pack_does_not_start_the_chip_thread(tmp_path: Path) -> None:
    backend = RocksBackend(tmp_path / "migrate-pack.rocksdb", sync="OFF", bulk=True)
    try:
        assert backend._chip_thread is None
        await backend.apply([Put(CF_MAIN_CHAIN, u32_be(0), b"block")])
        await backend.flush(CF_MAIN_CHAIN)
        await backend.merge_families([CF_MAIN_CHAIN])
        assert backend._chip_thread is None
        assert await backend.get(CF_MAIN_CHAIN, u32_be(0)) == b"block"
    finally:
        await backend.close()
