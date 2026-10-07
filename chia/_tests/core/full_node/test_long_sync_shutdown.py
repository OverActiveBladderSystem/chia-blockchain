from __future__ import annotations

import asyncio
import logging
import time

import pytest

from chia.full_node.full_node import _await_cancelled_long_sync
from chia.util.task_referencer import create_referenced_task


@pytest.mark.anyio
async def test_a_cancelled_long_sync_that_leaves_is_finished() -> None:
    async def leave() -> None:
        try:
            await asyncio.sleep(30)
        except asyncio.CancelledError:
            return

    task = create_referenced_task(leave(), name="Task-294")
    await asyncio.sleep(0)
    task.cancel()
    await _await_cancelled_long_sync(logging.getLogger(__name__), task, timeout=2)
    assert task.done()


@pytest.mark.anyio
async def test_shutdown_closes_when_a_cancelled_long_sync_stays_running(caplog: pytest.LogCaptureFixture) -> None:
    stop = asyncio.Event()

    async def ignore_cancel() -> None:
        while not stop.is_set():
            try:
                await asyncio.sleep(3600)
            except asyncio.CancelledError:
                continue

    task = create_referenced_task(ignore_cancel(), name="Task-294")
    await asyncio.sleep(0)
    task.cancel()
    caplog.set_level(logging.WARNING)
    started = time.monotonic()
    try:
        await asyncio.wait_for(
            _await_cancelled_long_sync(logging.getLogger(__name__), task, timeout=0.2),
            timeout=2,
        )
        assert time.monotonic() - started < 1.5
        assert not task.done()
        assert any("did not stop" in message for message in caplog.messages)
    finally:
        stop.set()
        task.cancel()
        await asyncio.wait({task}, timeout=2)


@pytest.mark.anyio
async def test_an_already_finished_long_sync_is_not_waited_on() -> None:
    async def immediate() -> None:
        return

    task = create_referenced_task(immediate(), name="Task-294")
    await task
    started = time.monotonic()
    await _await_cancelled_long_sync(logging.getLogger(__name__), task, timeout=5)
    assert time.monotonic() - started < 1
