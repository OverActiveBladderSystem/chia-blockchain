from __future__ import annotations

import asyncio
import shutil
import sys
import time
from pathlib import Path

import click

from chia.consensus.constants import replace_str_to_bytes
from chia.consensus.default_constants import DEFAULT_CONSTANTS, update_testnet_overrides
from chia.full_node.db.keys import ROCKS_SUFFIX, SQLITE_SUFFIX
from chia.full_node.db.migrate import (
    MigrateProgress,
    MigrationPaused,
    ProgressClock,
    abort_migration,
    fit_terminal_line,
    migrate_database,
    offered_database_path,
    replay_progress,
    require_destination_space,
    rocks_directory_for,
    source_headroom_warning,
)
from chia.full_node.db.path import substitute_network
from chia.full_node.db.startup import persist_full_node_database_path
from chia.util.config import load_config
from chia.util.path import path_from_root
from chia.util.task_referencer import create_referenced_task


def db_migrate_func(
    root_path: Path,
    output_parent: Path,
    *,
    yes: bool,
    no_update_config: bool,
    abort: bool = False,
) -> None:
    config = load_config(root_path, "config.yaml")
    service_config = config["full_node"]
    selected_network = service_config["selected_network"]
    if abort:
        print(asyncio.run(abort_migration(output_parent, selected_network)))
        return
    database_pattern = service_config["database_path"]
    sqlite_path = path_from_root(root_path, substitute_network(database_pattern, selected_network))
    if not sqlite_path.name.endswith(SQLITE_SUFFIX):
        print(f"database_path is already {database_pattern}. Nothing to migrate.")
        return
    if not sqlite_path.is_file():
        raise RuntimeError(f"SQLite database does not exist: {sqlite_path}")

    size, needed = require_destination_space(sqlite_path, output_parent)
    free_note = source_headroom_warning(sqlite_path)
    rocks_dir = rocks_directory_for(output_parent, selected_network)
    print(f"Source SQLite: {sqlite_path} ({size:,} bytes)")
    print(f"Destination:   {rocks_dir}")
    print(f"Free space required there: {needed:,} bytes (1.5x)")
    if free_note is not None:
        print(free_note)
    print("The SQLite file stays where it is. This command never deletes it.")
    print("The full node cannot open this RocksDB directory until the last line says Migration finished.")
    print("The steps are blocks, chain, coins, hints, summaries, pack, follow, height-to-hash,")
    print("then sub-epoch-summaries. Pack rewrites the database files into fewer pieces and can take a long time.")
    print("Leave the window open. The full node cannot use the new database until Migration finished.")
    print("Closing this window or pressing Ctrl+C stops the copy.")
    print("Run the same command again to continue the unfinished step. config.yaml is not changed until the end.")

    overrides = service_config["network_overrides"]["constants"][selected_network]
    update_testnet_overrides(selected_network, overrides)
    constants = replace_str_to_bytes(DEFAULT_CONSTANTS, **overrides)
    tty = sys.stdout.isatty()
    last_print = 0.0
    now = time.monotonic()
    clock = ProgressClock(started=now, phase_started=now)
    latest: MigrateProgress | None = None

    def paint(update: MigrateProgress, now: float) -> None:
        nonlocal last_print
        line = clock.line(update, now)
        if tty and update.phase != "complete":
            width = shutil.get_terminal_size(fallback=(120, 20)).columns
            print("\r" + fit_terminal_line(line, width), end="", flush=True)
            last_print = now
        elif now - last_print >= 15 or update.phase == "complete":
            if tty and update.phase == "complete":
                print()
            print(line, flush=True)
            last_print = now

    def show(update: MigrateProgress) -> None:
        nonlocal latest
        latest = update
        paint(update, time.monotonic())

    async def copy_and_tick() -> None:
        stop = asyncio.Event()
        ticker = create_referenced_task(
            replay_progress(stop, lambda: latest, paint),
            name="db-migrate-progress",
        )
        try:
            await migrate_database(
                sqlite_path,
                output_parent,
                selected_network=selected_network,
                constants=constants,
                progress=show,
                assume_source_headroom=yes,
            )
        finally:
            stop.set()
            ticker.cancel()
            try:
                await ticker
            except asyncio.CancelledError:
                pass

    try:
        asyncio.run(copy_and_tick())
    except MigrationPaused as paused:
        print()
        print(paused)
        return
    except KeyboardInterrupt:
        print()
        print("Copy stopped. Finished batches are saved.")
        print("Run the same command again to continue from the saved position.")
        print("config.yaml was not changed.")
        return
    _offer_config_update(
        root_path,
        database_pattern,
        output_parent,
        rocks_dir,
        sqlite_path,
        no_update_config=no_update_config,
    )


def _offer_config_update(
    root_path: Path,
    database_pattern: str,
    output_parent: Path,
    rocks_dir: Path,
    sqlite_path: Path,
    *,
    no_update_config: bool,
) -> None:
    print()
    print(f"Migration finished: {rocks_dir}")
    print("Every step is done, including the file merge and the two startup files.")
    print(f"SQLite is still at {sqlite_path}. This process does not switch the running full node.")
    print("Restart the full node after database_path changes, to open RocksDB.")
    print("To go back later, set database_path to the SQLite path and restart.")
    offered = offered_database_path(database_pattern, output_parent, rocks_dir)
    print(f"database_path: {database_pattern}")
    print(f"database_path: {offered}")
    if no_update_config:
        return
    if not click.confirm("Update config.yaml so the next start uses this RocksDB directory?", default=False):
        print("config.yaml was not changed.")
        return
    persist_full_node_database_path(root_path, offered)
    print("config.yaml updated. Restart the full node to use RocksDB.")
    if not offered.endswith(ROCKS_SUFFIX):
        print(f"Expected the new path to end in {ROCKS_SUFFIX}.")
