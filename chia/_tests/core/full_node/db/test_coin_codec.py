from __future__ import annotations

import pytest
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint64

from chia.full_node.db.coin_codec import (
    StoredCoin,
    added_lookup_keys,
    decode_delta,
    delta_removals,
    encode_delta,
    lookup_entries,
)

_NAME = bytes32(b"\x11" * 32)
_PUZZLE = bytes32(b"\x22" * 32)
_PARENT = bytes32(b"\x33" * 32)


def _record(name: bytes, spent: int, *, coinbase: bool = False) -> StoredCoin:
    return StoredCoin(bytes32(name), 9_141_759, spent, coinbase, _PUZZLE, _PARENT, uint64(100), 1_700_000_000)


def test_journal_lookup_keys_match_the_decoded_coins() -> None:
    added = [
        _record(b"\x01" * 32, 0),
        _record(b"\x02" * 32, -1),
        _record(b"\x03" * 32, 9_352_167),
        _record(b"\x04" * 32, -2, coinbase=True),
    ]
    removed = [(_NAME, 0), (bytes32(b"\x44" * 32), -1)]
    raw = encode_delta(added, removed)
    count, keys = added_lookup_keys(raw)
    expected: list[tuple[str, bytes]] = []
    for record in added:
        expected.extend(lookup_entries(record))
    assert count == len(added)
    assert keys == expected
    assert delta_removals(raw) == removed
    assert decode_delta(raw) == (added, removed)


def test_empty_journal_has_no_lookup_keys() -> None:
    raw = encode_delta([], [])
    assert added_lookup_keys(raw) == (0, [])
    assert delta_removals(raw) == []
    assert decode_delta(raw) == ([], [])


def test_truncated_journal_is_rejected() -> None:
    raw = encode_delta([_record(b"\x05" * 32, 0)], [])
    with pytest.raises(ValueError, match="truncated"):
        added_lookup_keys(raw[:-1])
    with pytest.raises(ValueError, match="not a CD1"):
        delta_removals(b"not-a-journal")
