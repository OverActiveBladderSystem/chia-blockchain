from __future__ import annotations

from pathlib import Path

from chia.full_node.db.chain_db import ChainDB
from chia.full_node.db.keys import (
    CF_META,
    META_COMPLETE,
    META_MIGRATE_PHASE,
    META_MIGRATE_SOURCE,
    ROCKS_SUFFIX,
    SQLITE_SUFFIX,
)
from chia.full_node.db.rocks import RocksBackend
from chia.util.config import lock_and_load_config, save_config


async def unfinished_migration_reason(path: Path) -> str | None:
    """Return an error string when `path` is a RocksDB copy that has not finished."""
    if not (path / "CURRENT").exists():
        return None
    database = ChainDB(RocksBackend(path, sync="OFF"))
    try:
        async with database.reader_no_transaction() as view:
            phase = await view.get(CF_META, META_MIGRATE_PHASE)
            complete = await view.get(CF_META, META_COMPLETE)
            source = await view.get(CF_META, META_MIGRATE_SOURCE)
    finally:
        await database.close()
    if complete == b"1" or phase in {None, b"complete"}:
        return None
    phase_name = phase.decode() if phase is not None else "unknown"
    sqlite_path = _sqlite_path_to_restore(path, source)
    return (
        f"{path} is an unfinished RocksDB migration (phase {phase_name}). "
        f"Set database_path back to {sqlite_path}, rerun `chia db migrate`, "
        "and restart only after it finishes."
    )


def _sqlite_path_to_restore(rocks_path: Path, source: bytes | None) -> Path:
    """The SQLite file to put back in database_path. Prefer the path saved at copy start."""
    if source:
        try:
            text = source.decode()
        except UnicodeDecodeError:
            text = ""
        if text:
            return Path(text)
    if rocks_path.name.endswith(ROCKS_SUFFIX):
        return rocks_path.with_name(rocks_path.name[: -len(ROCKS_SUFFIX)] + SQLITE_SUFFIX)
    return rocks_path.with_suffix(SQLITE_SUFFIX)


def persist_full_node_database_path(root_path: Path, database_path: str) -> None:
    with lock_and_load_config(root_path, "config.yaml") as config:
        config["full_node"]["database_path"] = database_path
        save_config(root_path, "config.yaml", config)
