from __future__ import annotations

from dataclasses import dataclass

from chia_rs.sized_bytes import bytes32


@dataclass(frozen=True)
class BlockMeta:
    prev_hash: bytes32
    height: int
    compact: bool
    in_main_chain: bool
    ses: bytes | None
    record: bytes


def encode_block_meta(meta: BlockMeta) -> bytes:
    ses = b"" if meta.ses is None else meta.ses
    return (
        bytes(meta.prev_hash)
        + int(meta.height).to_bytes(4, "little", signed=False)
        + bytes([1 if meta.compact else 0, 1 if meta.in_main_chain else 0])
        + len(ses).to_bytes(4, "little", signed=False)
        + ses
        + meta.record
    )


def decode_block_meta(raw: bytes) -> BlockMeta:
    if len(raw) < 32 + 4 + 2 + 4:
        raise ValueError(f"block meta is {len(raw)} bytes")
    prev_hash = bytes32(raw[0:32])
    height = int.from_bytes(raw[32:36], "little", signed=False)
    compact = raw[36] == 1
    in_main_chain = raw[37] == 1
    ses_len = int.from_bytes(raw[38:42], "little", signed=False)
    ses_end = 42 + ses_len
    if ses_end > len(raw):
        raise ValueError("block meta sub-epoch summary overruns the value")
    ses = raw[42:ses_end] if ses_len else None
    return BlockMeta(prev_hash, height, compact, in_main_chain, ses, raw[ses_end:])
