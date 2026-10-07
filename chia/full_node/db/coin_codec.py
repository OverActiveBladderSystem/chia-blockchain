from __future__ import annotations

from dataclasses import dataclass

from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint64

from chia.full_node.db.keys import (
    CF_COINS,
    CF_COINS_BY_CONFIRMED,
    CF_COINS_BY_PUZZLE_CONFIRMED,
    CF_COINS_BY_PUZZLE_SPENT,
    CF_COINS_BY_SPENT,
    CF_FF_UNSPENT,
    u32_be,
)
from chia.types.blockchain_format.coin import Coin

_RECORD_LEN = 4 + 8 + 1 + 32 + 32 + 8 + 8


@dataclass(frozen=True)
class StoredCoin:
    coin_name: bytes32
    confirmed_index: int
    spent_index: int
    coinbase: bool
    puzzle_hash: bytes32
    parent: bytes32
    amount: uint64
    timestamp: int

    def coin(self) -> Coin:
        return Coin(self.parent, self.puzzle_hash, self.amount)


def encode_coin(record: StoredCoin) -> bytes:
    return (
        int(record.confirmed_index).to_bytes(4, "little", signed=False)
        + int(record.spent_index).to_bytes(8, "little", signed=True)
        + bytes([1 if record.coinbase else 0])
        + bytes(record.puzzle_hash)
        + bytes(record.parent)
        + int(record.amount).to_bytes(8, "big", signed=False)
        + int(record.timestamp).to_bytes(8, "little", signed=False)
    )


def decode_coin(coin_name: bytes32, raw: bytes) -> StoredCoin:
    if len(raw) != _RECORD_LEN:
        raise ValueError(f"coin record for {coin_name.hex()} is {len(raw)} bytes")
    confirmed = int.from_bytes(raw[0:4], "little", signed=False)
    spent = int.from_bytes(raw[4:12], "little", signed=True)
    coinbase = raw[12] == 1
    puzzle_hash = bytes32(raw[13:45])
    parent = bytes32(raw[45:77])
    amount = uint64(int.from_bytes(raw[77:85], "big", signed=False))
    timestamp = int.from_bytes(raw[85:93], "little", signed=False)
    return StoredCoin(coin_name, confirmed, spent, coinbase, puzzle_hash, parent, amount, timestamp)


def index_entries(record: StoredCoin) -> list[tuple[str, bytes]]:
    name = bytes(record.coin_name)
    confirmed = u32_be(record.confirmed_index)
    entries = [
        (CF_COINS, name),
        (CF_COINS_BY_CONFIRMED, confirmed + name),
        (CF_COINS_BY_PUZZLE_CONFIRMED, bytes(record.puzzle_hash) + confirmed + name),
    ]
    if record.spent_index > 0:
        spent = u32_be(record.spent_index)
        entries.append((CF_COINS_BY_SPENT, spent + name))
        entries.append((CF_COINS_BY_PUZZLE_SPENT, bytes(record.puzzle_hash) + spent + name))
    if record.spent_index == -1:
        entries.append((CF_FF_UNSPENT, bytes(record.puzzle_hash) + name))
    return entries


def lookup_entries(record: StoredCoin) -> list[tuple[str, bytes]]:
    """Wallet lookup keys only. Height scans use the per-block journal instead."""
    name = bytes(record.coin_name)
    confirmed = u32_be(record.confirmed_index)
    entries = [
        (CF_COINS_BY_PUZZLE_CONFIRMED, bytes(record.puzzle_hash) + confirmed + name),
    ]
    if record.spent_index > 0:
        spent = u32_be(record.spent_index)
        entries.append((CF_COINS_BY_PUZZLE_SPENT, bytes(record.puzzle_hash) + spent + name))
    if record.spent_index == -1:
        entries.append((CF_FF_UNSPENT, bytes(record.puzzle_hash) + name))
    return entries


_DELTA_MAGIC = b"CD1"


def encode_delta(added: list[StoredCoin], removed: list[tuple[bytes32, int]]) -> bytes:
    parts = [_DELTA_MAGIC, len(added).to_bytes(4, "big", signed=False)]
    for record in added:
        parts.append(bytes(record.coin_name))
        parts.append(encode_coin(record))
    parts.append(len(removed).to_bytes(4, "big", signed=False))
    for name, previous_spent in removed:
        parts.append(bytes(name))
        parts.append(int(previous_spent).to_bytes(8, "big", signed=True))
    return b"".join(parts)


_NAME_LEN = 32
_REMOVAL_LEN = _NAME_LEN + 8
_ADDED_STRIDE = _NAME_LEN + _RECORD_LEN
# Offsets inside encode_coin's record. Height is little-endian there.
_SPENT_OFF = 4
_SPENT_END = 12
_PUZZLE_OFF = 13
_PUZZLE_END = 45


def _delta_layout(raw: bytes) -> tuple[int, int]:
    """Return (added_end, removed_count) for a CD1 journal."""
    if len(raw) < 7 or raw[:3] != _DELTA_MAGIC:
        raise ValueError("coin journal is not a CD1 record")
    added_count = int.from_bytes(raw[3:7], "big", signed=False)
    added_end = 7 + added_count * _ADDED_STRIDE
    if added_end + 4 > len(raw):
        raise ValueError("coin journal additions are truncated")
    removed_count = int.from_bytes(raw[added_end : added_end + 4], "big", signed=False)
    if added_end + 4 + removed_count * _REMOVAL_LEN != len(raw):
        raise ValueError("coin journal removals are truncated")
    return added_end, removed_count


def added_lookup_keys(raw: bytes) -> tuple[int, list[tuple[str, bytes]]]:
    """Wallet lookup keys for coins added in a journal.

    The journal already stores each coin record. Reading those bytes avoids building a
    StoredCoin per coin. The record keeps height little-endian; lookup keys use big-endian.
    """
    added_end, _removed_count = _delta_layout(raw)
    keys: list[tuple[str, bytes]] = []
    cursor = 7
    while cursor < added_end:
        name = raw[cursor : cursor + _NAME_LEN]
        record = cursor + _NAME_LEN
        confirmed = raw[record : record + 4][::-1]
        puzzle = raw[record + _PUZZLE_OFF : record + _PUZZLE_END]
        keys.append((CF_COINS_BY_PUZZLE_CONFIRMED, puzzle + confirmed + name))
        spent = int.from_bytes(raw[record + _SPENT_OFF : record + _SPENT_END], "little", signed=True)
        if spent > 0:
            keys.append((CF_COINS_BY_PUZZLE_SPENT, puzzle + u32_be(spent) + name))
        elif spent == -1:
            keys.append((CF_FF_UNSPENT, puzzle + name))
        cursor += _ADDED_STRIDE
    return (added_end - 7) // _ADDED_STRIDE, keys


def delta_removals(raw: bytes) -> list[tuple[bytes32, int]]:
    """Names and previous spent heights for coins spent in a journal."""
    added_end, removed_count = _delta_layout(raw)
    cursor = added_end + 4
    end = cursor + removed_count * _REMOVAL_LEN
    removed: list[tuple[bytes32, int]] = []
    while cursor < end:
        name = bytes32(raw[cursor : cursor + _NAME_LEN])
        previous = int.from_bytes(raw[cursor + _NAME_LEN : cursor + _REMOVAL_LEN], "big", signed=True)
        removed.append((name, previous))
        cursor += _REMOVAL_LEN
    return removed


def decode_delta(raw: bytes) -> tuple[list[StoredCoin], list[tuple[bytes32, int]]]:
    added_end, _removed_count = _delta_layout(raw)
    cursor = 7
    added: list[StoredCoin] = []
    while cursor < added_end:
        name = bytes32(raw[cursor : cursor + _NAME_LEN])
        cursor += _NAME_LEN
        added.append(decode_coin(name, raw[cursor : cursor + _RECORD_LEN]))
        cursor += _RECORD_LEN
    return added, delta_removals(raw)
