from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

import pytest

from chia.daemon.server import kill_processes, kill_service_process


class _Process:
    def __init__(self, pid: int, *, running: bool) -> None:
        self.pid = pid
        self.killed = False
        self._running = running

    def poll(self) -> int | None:
        if self._running:
            return None
        return 0

    def kill(self) -> None:
        self.killed = True
        self._running = False

    def wait(self) -> int:
        return 1


def _windows_pid_is_running(pid: int) -> bool:
    import ctypes

    synchronize = 0x00100000
    kernel = ctypes.windll.kernel32
    handle = kernel.OpenProcess(synchronize, False, pid)
    if not handle:
        return False
    try:
        # WAIT_TIMEOUT means the process is still alive.
        return kernel.WaitForSingleObject(handle, 0) == 0x102
    finally:
        kernel.CloseHandle(handle)


def test_windows_stop_kills_the_whole_tree(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chia.daemon.server.sys.platform", "win32")
    commands: list[list[str]] = []

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        commands.append([str(part) for part in args])
        return subprocess.CompletedProcess(args, 0, b"", b"")

    monkeypatch.setattr("chia.daemon.server.subprocess.run", fake_run)
    process = _Process(pid=2300, running=True)
    kill_service_process(process)  # type: ignore[arg-type]
    assert commands == [["taskkill", "/F", "/T", "/PID", "2300"]]
    assert process.killed is False


def test_windows_stop_falls_back_when_taskkill_cannot_run(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chia.daemon.server.sys.platform", "win32")

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise OSError("taskkill missing")

    monkeypatch.setattr("chia.daemon.server.subprocess.run", fake_run)
    process = _Process(pid=2300, running=True)
    kill_service_process(process)  # type: ignore[arg-type]
    assert process.killed is True


def test_windows_stop_leaves_an_exited_process_alone(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chia.daemon.server.sys.platform", "win32")

    def fake_run(args: list[str], **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        return subprocess.CompletedProcess(args, 128, b"", b"")

    monkeypatch.setattr("chia.daemon.server.subprocess.run", fake_run)
    process = _Process(pid=2300, running=False)
    kill_service_process(process)  # type: ignore[arg-type]
    assert process.killed is False


def test_other_platforms_stop_the_recorded_process(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chia.daemon.server.sys.platform", "linux")

    def fake_run(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[bytes]:
        raise AssertionError("taskkill is only for Windows")

    monkeypatch.setattr("chia.daemon.server.subprocess.run", fake_run)
    process = _Process(pid=2300, running=True)
    kill_service_process(process)  # type: ignore[arg-type]
    assert process.killed is True


@pytest.mark.anyio
async def test_stop_timeout_uses_the_tree_kill(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr("chia.daemon.server.sys.platform", "win32")
    monkeypatch.setattr("chia.daemon.server.kill", lambda *_args, **_kwargs: None)
    seen: list[int] = []

    def record(process: _Process) -> None:
        seen.append(process.pid)

    monkeypatch.setattr("chia.daemon.server.kill_service_process", record)
    process = _Process(pid=2300, running=True)
    await kill_processes([process], tmp_path, "chia_full_node", "", delay_before_kill=0)  # type: ignore[list-item]
    assert seen == [2300]


@pytest.mark.skipif(sys.platform != "win32", reason="Windows .cmd launchers")
def test_stop_ends_the_cmd_launcher_and_its_python(tmp_path: Path) -> None:
    pid_path = tmp_path / "pid.txt"
    script = tmp_path / "sleep_node.py"
    script.write_text(
        "import os\n"
        "import time\n"
        "from pathlib import Path\n"
        f"Path({str(pid_path)!r}).write_text(str(os.getpid()))\n"
        "time.sleep(120)\n",
        encoding="utf-8",
    )
    launcher = tmp_path / "chia_full_node.cmd"
    launcher.write_text(f'@echo off\r\n"{sys.executable}" "{script}"\r\n', encoding="utf-8")
    process = subprocess.Popen(
        [str(launcher)],
        shell=False,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
    )
    child_pid = 0
    try:
        deadline = time.monotonic() + 10
        while not pid_path.exists():
            if process.poll() is not None:
                raise AssertionError(f"launcher exited early with {process.returncode}")
            if time.monotonic() > deadline:
                raise AssertionError("the interpreter did not start")
            time.sleep(0.05)
        child_pid = int(pid_path.read_text(encoding="utf-8").strip())
        assert _windows_pid_is_running(child_pid)
        kill_service_process(process)
        process.wait(timeout=10)
        assert not _windows_pid_is_running(child_pid)
    finally:
        if process.poll() is None:
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                check=False,
                capture_output=True,
            )
        if child_pid and _windows_pid_is_running(child_pid):
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(child_pid)],
                check=False,
                capture_output=True,
            )
