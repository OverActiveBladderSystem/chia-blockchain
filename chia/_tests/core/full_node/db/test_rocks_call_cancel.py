from __future__ import annotations

import asyncio
import threading

import pytest

from chia.full_node.db.rocks import _deliver_call_result


class _LoopErrors:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.errors: list[BaseException] = []
        self._loop = loop
        self._previous = loop.get_exception_handler()
        loop.set_exception_handler(self._handle)

    def _handle(self, _loop: asyncio.AbstractEventLoop, context: dict[str, object]) -> None:
        exc = context.get("exception")
        if isinstance(exc, BaseException):
            self.errors.append(exc)

    def close(self) -> None:
        self._loop.set_exception_handler(self._previous)


@pytest.mark.anyio
async def test_a_cancelled_database_call_drops_the_late_result() -> None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[object] = loop.create_future()
    future.cancel()
    caught = _LoopErrors(loop)
    try:
        _deliver_call_result(future, (b"late",))
        _deliver_call_result(future, RuntimeError("late"))
    finally:
        caught.close()
    assert future.cancelled()
    assert caught.errors == []


@pytest.mark.anyio
async def test_a_database_call_still_receives_its_result() -> None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[object] = loop.create_future()
    _deliver_call_result(future, (b"ok",))
    assert await future == b"ok"


@pytest.mark.anyio
async def test_a_database_call_still_receives_its_error() -> None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[object] = loop.create_future()
    _deliver_call_result(future, RuntimeError("boom"))
    with pytest.raises(RuntimeError, match="boom"):
        await future


@pytest.mark.anyio
async def test_the_database_thread_can_finish_after_cancel() -> None:
    loop = asyncio.get_running_loop()
    future: asyncio.Future[object] = loop.create_future()
    caught = _LoopErrors(loop)
    ready = threading.Event()

    def worker() -> None:
        ready.wait()
        loop.call_soon_threadsafe(_deliver_call_result, future, (b"late",))

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    future.cancel()
    ready.set()
    try:
        thread.join(timeout=2)
        await asyncio.sleep(0.05)
        assert future.cancelled()
        assert caught.errors == []
    finally:
        ready.set()
        thread.join(timeout=2)
        caught.close()
