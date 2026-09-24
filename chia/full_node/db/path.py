from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from chia.full_node.db.keys import ROCKS_SUFFIX, SQLITE_SUFFIX
from chia.util.path import path_from_root


@dataclass(frozen=True)
class DatabaseChoice:
    """Which full-node database to open, and an optional config rewrite."""

    engine: str
    path: Path
    # Replacement for the config `database_path` value, still containing CHALLENGE
    # when the previous value did. None means leave config.yaml alone.
    config_database_path: str | None = None
    # True when the node should log that an existing SQLite file can be copied.
    suggest_migrate: bool = False


def substitute_network(database_path: str, selected_network: str) -> str:
    return database_path.replace("CHALLENGE", selected_network)


def engine_for_path(path: Path) -> str:
    name = path.name
    if name.endswith(SQLITE_SUFFIX):
        return "sqlite"
    if name.endswith(ROCKS_SUFFIX):
        return "rocksdb"
    raise ValueError(
        f"database_path {path} must end in {SQLITE_SUFFIX} or {ROCKS_SUFFIX}"
    )


def rocks_pattern(database_path: str) -> str:
    if database_path.endswith(SQLITE_SUFFIX):
        return database_path[: -len(SQLITE_SUFFIX)] + ROCKS_SUFFIX
    return database_path


def sqlite_sibling(resolved_sqlite: Path) -> Path:
    name = resolved_sqlite.name
    if not name.endswith(SQLITE_SUFFIX):
        raise ValueError(f"{resolved_sqlite} is not a sqlite database path")
    return resolved_sqlite.with_name(name[: -len(SQLITE_SUFFIX)] + ROCKS_SUFFIX)


def any_sqlite_sibling(resolved_sqlite: Path) -> bool:
    """True when this folder already holds a full-node sqlite file for some network."""
    folder = resolved_sqlite.parent
    if not folder.is_dir():
        return False
    return any(
        candidate.is_file() and candidate.name.endswith(SQLITE_SUFFIX) for candidate in folder.glob("blockchain_v2_*")
    )


def choose_full_node_database(
    root_path: Path,
    database_path: str | None,
    selected_network: str,
) -> DatabaseChoice:
    """
    Pick the live full-node database.

    An existing `.sqlite` file is opened so an upgraded farmer keeps farming.
    A missing sqlite file, a `.rocksdb` path, and `chia init` all use RocksDB.
    This function does not create files.
    """
    pattern = database_path if database_path else f"db/blockchain_v2_CHALLENGE{SQLITE_SUFFIX}"
    resolved = path_from_root(root_path, substitute_network(pattern, selected_network))
    try:
        engine = engine_for_path(resolved)
    except ValueError:
        if database_path is None and not resolved.exists():
            pattern = f"db/blockchain_v2_CHALLENGE{ROCKS_SUFFIX}"
            resolved = path_from_root(root_path, substitute_network(pattern, selected_network))
            return DatabaseChoice("rocksdb", resolved, pattern)
        raise

    if engine == "rocksdb":
        return DatabaseChoice("rocksdb", resolved, None)

    if resolved.is_file():
        return DatabaseChoice("sqlite", resolved, None, suggest_migrate=True)

    rocks = sqlite_sibling(resolved)
    rewrite = rocks_pattern(pattern) if not any_sqlite_sibling(resolved) else None
    return DatabaseChoice("rocksdb", rocks, rewrite)
