from __future__ import annotations

from dataclasses import dataclass

from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint64

from chia.full_node.db.keys import (
    CF_COINS,
    CF_COINS_BY_CONFIRMED,
    CF_COINS_BY_PARENT,
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
        (CF_COINS_BY_PARENT, bytes(record.parent) + confirmed + name),
    ]
    if record.spent_index > 0:
        spent = u32_be(record.spent_index)
        entries.append((CF_COINS_BY_SPENT, spent + name))
        entries.append((CF_COINS_BY_PUZZLE_SPENT, bytes(record.puzzle_hash) + spent + name))
    if record.spent_index == -1:
        entries.append((CF_FF_UNSPENT, bytes(record.puzzle_hash) + name))
    return entries
