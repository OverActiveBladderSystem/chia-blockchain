from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

from chia.full_node.db.ops import Delete, DeleteRange, KvBackend, Op, Put, key_in_range


class _View:
    """Read view. A write session subclasses this and adds mutations."""

    def __init__(self, backend: KvBackend, ops: list[Op] | None = None) -> None:
        self._backend = backend
        self._ops: list[Op] = [] if ops is None else ops

    async def get(self, cf: str, key: bytes) -> bytes | None:
        for op in reversed(self._ops):
            if op.cf != cf:
                continue
            if isinstance(op, Put) and op.key == key:
                return op.value
            if isinstance(op, Delete) and op.key == key:
                return None
            if isinstance(op, DeleteRange) and key_in_range(key, op.start, op.end):
                return None
        return await self._backend.get(cf, key)

    async def get_many(self, cf: str, keys: list[bytes]) -> dict[bytes, bytes]:
        unresolved = set(keys)
        resolved: dict[bytes, bytes | None] = {}
        for op in reversed(self._ops):
            if not unresolved or op.cf != cf:
                continue
            if isinstance(op, Put) and op.key in unresolved:
                resolved[op.key] = op.value
                unresolved.remove(op.key)
            elif isinstance(op, Delete) and op.key in unresolved:
                resolved[op.key] = None
                unresolved.remove(op.key)
            elif isinstance(op, DeleteRange):
                covered = [key for key in unresolved if key_in_range(key, op.start, op.end)]
                for key in covered:
                    resolved[key] = None
                    unresolved.remove(key)
        fetched = await self._backend.get_many(cf, list(unresolved)) if unresolved else {}
        present: dict[bytes, bytes] = {}
        for key in keys:
            if key in resolved:
                value = resolved[key]
                if value is not None:
                    present[key] = value
            elif key in fetched:
                present[key] = fetched[key]
        return present

    async def scan(self, cf: str, start: bytes, end: bytes | None) -> list[tuple[bytes, bytes]]:
        rows = dict(await self._backend.scan(cf, start, end))
        for op in self._ops:
            if op.cf != cf:
                continue
            if isinstance(op, Put):
                if key_in_range(op.key, start, end):
                    rows[op.key] = op.value
            elif isinstance(op, Delete):
                if key_in_range(op.key, start, end):
                    rows.pop(op.key, None)
            elif isinstance(op, DeleteRange):
                lo = start if start > op.start else op.start
                if end is None:
                    hi = op.end
                elif op.end is None:  # pragma: no cover
                    hi = end
                else:
                    hi = end if end < op.end else op.end
                if lo < hi:
                    for key in [key for key in rows if key_in_range(key, lo, hi)]:
                        del rows[key]
        return sorted(rows.items(), key=lambda item: item[0])


class WriteSession(_View):
    def __init__(self, backend: KvBackend) -> None:
        super().__init__(backend)
        self._savepoints: list[int] = []

    def put(self, cf: str, key: bytes, value: bytes) -> None:
        self._ops.append(Put(cf, key, value))

    def delete(self, cf: str, key: bytes) -> None:
        self._ops.append(Delete(cf, key))

    def delete_range(self, cf: str, start: bytes, end: bytes) -> None:
        if start >= end:
            return
        self._ops.append(DeleteRange(cf, start, end))

    def push_savepoint(self) -> None:
        self._savepoints.append(len(self._ops))

    def pop_savepoint(self) -> None:
        self._savepoints.pop()

    def rollback_savepoint(self) -> None:
        mark = self._savepoints.pop()
        del self._ops[mark:]

    async def commit(self) -> None:
        if self._ops:
            await self._backend.apply(self._ops)
            self._ops.clear()


class ChainDB:
    """
    One writer task at a time, with nested savepoints and read-your-writes.

    Another task's read sees the last commit. The task that holds the writer
    sees its own uncommitted puts and deletes. An exception or cancellation
    inside a nested writer rolls back to that savepoint. The outer batch
    commits only when the outermost writer exits without an error.
    """

    def __init__(self, backend: KvBackend) -> None:
        self._backend = backend
        self._lock = asyncio.Lock()
        self._sessions: dict[asyncio.Task[object], WriteSession] = {}

    async def write_column_batches(
        self,
        groups: list[tuple[str, list[tuple[bytes, bytes]]]],
        *,
        disable_wal: bool = False,
    ) -> None:
        """Write sorted keys straight to the database, outside a read-your-writes session."""
        write = getattr(self._backend, "write_column_batches", None)
        if write is None:
            raise RuntimeError("this database cannot bulk-write columns")
        async with self._lock:
            await write(groups, disable_wal=disable_wal)

    async def merge_families(self, names: list[str]) -> None:
        merge = getattr(self._backend, "merge_families", None)
        if merge is None:
            raise RuntimeError("this database cannot merge its stored files")
        async with self._lock:
            await merge(names)

    @asynccontextmanager
    async def writer(self) -> AsyncIterator[WriteSession]:
        task = asyncio.current_task()
        assert task is not None
        existing = self._sessions.get(task)
        if existing is not None:
            existing.push_savepoint()
            try:
                yield existing
            except BaseException:
                existing.rollback_savepoint()
                raise
            else:
                existing.pop_savepoint()
            return

        async with self._lock:
            session = WriteSession(self._backend)
            self._sessions[task] = session
            try:
                yield session
            except BaseException:
                raise
            else:
                await session.commit()
            finally:
                self._sessions.pop(task, None)

    @asynccontextmanager
    async def writer_maybe_transaction(self) -> AsyncIterator[WriteSession]:
        task = asyncio.current_task()
        assert task is not None
        existing = self._sessions.get(task)
        if existing is not None:
            yield existing
            return
        async with self.writer() as session:
            yield session

    @asynccontextmanager
    async def reader_no_transaction(self) -> AsyncIterator[_View]:
        task = asyncio.current_task()
        assert task is not None
        existing = self._sessions.get(task)
        if existing is not None:
            yield existing
            return
        yield _View(self._backend)

    async def checkpoint(self, destination: Path) -> None:
        checkpoint = getattr(self._backend, "checkpoint", None)
        if checkpoint is None:
            raise RuntimeError("this database does not support checkpoints")
        await checkpoint(destination)

    async def close(self) -> None:
        await self._backend.close()
