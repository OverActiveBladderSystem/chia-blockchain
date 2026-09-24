from __future__ import annotations

# Column families for the full-node chain database. "default" is required by RocksDB.
CF_DEFAULT = "default"
CF_META = "meta"
CF_COINS = "coins"
CF_COINS_BY_CONFIRMED = "coins_by_confirmed"
CF_COINS_BY_SPENT = "coins_by_spent"
CF_COINS_BY_PUZZLE_CONFIRMED = "coins_by_puzzle_confirmed"
CF_COINS_BY_PUZZLE_SPENT = "coins_by_puzzle_spent"
CF_COINS_BY_PARENT = "coins_by_parent"
CF_FF_UNSPENT = "ff_unspent"
CF_BLOCK_BLOBS = "block_blobs"
CF_BLOCK_META = "block_meta"
CF_BLOCKS_AT_HEIGHT = "blocks_at_height"
CF_MAIN_CHAIN = "main_chain"
CF_UNCOMPACTIFIED = "uncompactified"
CF_SES = "ses"
CF_HINTS_BY_COIN = "hints_by_coin"
CF_HINTS_BY_HINT = "hints_by_hint"

COLUMN_FAMILIES: tuple[str, ...] = (
    CF_DEFAULT,
    CF_META,
    CF_COINS,
    CF_COINS_BY_CONFIRMED,
    CF_COINS_BY_SPENT,
    CF_COINS_BY_PUZZLE_CONFIRMED,
    CF_COINS_BY_PUZZLE_SPENT,
    CF_COINS_BY_PARENT,
    CF_FF_UNSPENT,
    CF_BLOCK_BLOBS,
    CF_BLOCK_META,
    CF_BLOCKS_AT_HEIGHT,
    CF_MAIN_CHAIN,
    CF_UNCOMPACTIFIED,
    CF_SES,
    CF_HINTS_BY_COIN,
    CF_HINTS_BY_HINT,
)

# Families whose values are already compressed, or are tiny flags.
UNCOMPRESSED_FAMILIES = frozenset({CF_DEFAULT, CF_BLOCK_BLOBS, CF_META, CF_MAIN_CHAIN, CF_UNCOMPACTIFIED})

META_FORMAT = b"db_format"
META_FORMAT_VALUE = b"rocksdb-v1"
META_SCHEMA_VERSION = b"schema_version"
META_UNSPENT = b"unspent_count"
META_COMPLETE = b"complete"
META_PEAK = b"peak"
META_COMPACT_COUNT = b"compact_count"
META_UNCOMPACT_COUNT = b"uncompact_count"
META_HINT_COUNT = b"hint_count"
META_MIGRATE_PHASE = b"migrate_phase"
META_MIGRATE_ROWID = b"migrate_rowid"
META_MIGRATE_HEIGHT = b"migrate_height"
META_MIGRATE_SOURCE = b"migrate_source"

SQLITE_SUFFIX = ".sqlite"
ROCKS_SUFFIX = ".rocksdb"


def u32_be(value: int) -> bytes:
    return int(value).to_bytes(4, "big", signed=False)


def u32_from_be(raw: bytes) -> int:
    return int.from_bytes(raw, "big", signed=False)


def prefix_end(prefix: bytes) -> bytes | None:
    """Exclusive end key for every key that starts with `prefix`. None means unbounded."""
    if not prefix:
        return None
    stripped = prefix.rstrip(b"\xff")
    if not stripped:
        return None
    return stripped[:-1] + bytes([stripped[-1] + 1])
