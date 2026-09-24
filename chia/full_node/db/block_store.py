from __future__ import annotations

import logging
import random
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import zstd
from chia_rs import BlockRecord, FullBlock, SubEpochChallengeSegment, SubEpochSegments
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32

from chia.consensus.block_generator_info import get_transactions_generator_bytes
from chia.full_node.block_store import compress, decompress, decompress_blob
from chia.full_node.db.block_codec import BlockMeta, decode_block_meta, encode_block_meta
from chia.full_node.db.chain_db import ChainDB, WriteSession, _View
from chia.full_node.db.keys import (
    CF_BLOCK_BLOBS,
    CF_BLOCK_META,
    CF_BLOCKS_AT_HEIGHT,
    CF_MAIN_CHAIN,
    CF_META,
    CF_SES,
    CF_UNCOMPACTIFIED,
    META_COMPACT_COUNT,
    META_COMPLETE,
    META_FORMAT,
    META_FORMAT_VALUE,
    META_MIGRATE_PHASE,
    META_PEAK,
    META_SCHEMA_VERSION,
    META_UNCOMPACT_COUNT,
    prefix_end,
    u32_be,
    u32_from_be,
)
from chia.full_node.full_block_utils import GeneratorBlockInfo, block_info_from_block, generator_from_block
from chia.util.errors import Err
from chia.util.lru_cache import LRUCache

log = logging.getLogger(__name__)


class RocksBlockStore:
    """Full blocks and block records in the chain RocksDB."""

    def __init__(
        self,
        db: ChainDB,
        block_cache: LRUCache[bytes32, FullBlock],
        ses_challenge_cache: LRUCache[bytes32, list[SubEpochChallengeSegment]],
    ) -> None:
        self.db = db
        self.block_cache = block_cache
        self.ses_challenge_cache = ses_challenge_cache

    @classmethod
    async def create(cls, db: ChainDB, *, use_cache: bool = True) -> RocksBlockStore:
        if use_cache:
            self = cls(db, LRUCache(1000), LRUCache(50))
        else:
            self = cls(db, LRUCache(0), LRUCache(0))
        async with db.writer_maybe_transaction() as session:
            if await session.get(CF_META, META_FORMAT) is None:
                session.put(CF_META, META_FORMAT, META_FORMAT_VALUE)
                session.put(CF_META, META_SCHEMA_VERSION, (2).to_bytes(4, "little", signed=False))
            phase = await session.get(CF_META, META_MIGRATE_PHASE)
            if phase is None and await session.get(CF_META, META_COMPLETE) is None:
                session.put(CF_META, META_COMPLETE, b"1")
        return self

    @asynccontextmanager
    async def transaction(self) -> AsyncIterator[None]:
        async with self.db.writer():
            yield

    def get_block_from_cache(self, header_hash: bytes32) -> FullBlock | None:
        return self.block_cache.get(header_hash)

    def rollback_cache_block(self, header_hash: bytes32) -> None:
        try:
            self.block_cache.remove(header_hash)
        except KeyError:
            pass

    async def add_full_block(self, header_hash: bytes32, block: FullBlock, block_record: BlockRecord) -> None:
        self.block_cache.put(header_hash, block)
        ses = (
            None
            if block_record.sub_epoch_summary_included is None
            else bytes(block_record.sub_epoch_summary_included)
        )
        async with self.db.writer_maybe_transaction() as session:
            if await session.get(CF_BLOCK_META, bytes(header_hash)) is not None:
                return
            meta = BlockMeta(
                block.prev_header_hash,
                int(block.height),
                bool(block.is_fully_compactified()),
                False,
                ses,
                bytes(block_record),
            )
            session.put(CF_BLOCK_BLOBS, bytes(header_hash), compress(block))
            session.put(CF_BLOCK_META, bytes(header_hash), encode_block_meta(meta))
            session.put(CF_BLOCKS_AT_HEIGHT, u32_be(block.height) + bytes(header_hash), b"")

    async def import_block_row(
        self,
        header_hash: bytes32,
        prev_hash: bytes32,
        height: int,
        ses: bytes | None,
        compact: bool,
        in_main_chain: bool,
        block_blob: bytes,
        block_record: bytes,
    ) -> None:
        """Insert a block copied from SQLite. `block_blob` is already compressed."""
        await self.import_block_rows(
            [(header_hash, prev_hash, height, ses, compact, in_main_chain, block_blob, block_record)]
        )

    async def import_block_rows(
        self,
        rows: list[tuple[bytes32, bytes32, int, bytes | None, bool, bool, bytes, bytes]],
    ) -> None:
        """Insert many copied blocks in one write. A block that is already stored is left as it is."""
        if len(rows) == 0:
            return
        async with self.db.writer_maybe_transaction() as session:
            existing = await session.get_many(CF_BLOCK_META, [bytes(row[0]) for row in rows])
            seen = set(existing)
            compact_delta = 0
            uncompact_delta = 0
            for header_hash, prev_hash, height, ses, compact, in_main_chain, block_blob, block_record in rows:
                raw_hash = bytes(header_hash)
                if raw_hash in seen:
                    continue
                seen.add(raw_hash)
                meta = BlockMeta(prev_hash, height, compact, in_main_chain, ses, block_record)
                session.put(CF_BLOCK_BLOBS, raw_hash, block_blob)
                session.put(CF_BLOCK_META, raw_hash, encode_block_meta(meta))
                session.put(CF_BLOCKS_AT_HEIGHT, u32_be(height) + raw_hash, b"")
                if not in_main_chain:
                    continue
                session.put(CF_MAIN_CHAIN, u32_be(height), raw_hash)
                if compact:
                    compact_delta += 1
                else:
                    session.put(CF_UNCOMPACTIFIED, u32_be(height), raw_hash)
                    uncompact_delta += 1
            if compact_delta:
                await self._add_count(session, META_COMPACT_COUNT, compact_delta)
            if uncompact_delta:
                await self._add_count(session, META_UNCOMPACT_COUNT, uncompact_delta)

    async def clear_main_chain_index(self) -> None:
        """Drop the main-chain index so a snapshot can write it again."""
        async with self.db.writer_maybe_transaction() as session:
            session.delete_range(CF_MAIN_CHAIN, b"", b"\xff" * 8)
            session.delete_range(CF_UNCOMPACTIFIED, b"", b"\xff" * 8)
            session.put(CF_META, META_COMPACT_COUNT, (0).to_bytes(8, "little", signed=False))
            session.put(CF_META, META_UNCOMPACT_COUNT, (0).to_bytes(8, "little", signed=False))

    async def apply_membership(self, membership: list[tuple[bytes32, bool, bool]]) -> None:
        """Record one batch of (hash, in_chain, compact) from a chain snapshot."""
        if len(membership) == 0:
            return
        async with self.db.writer_maybe_transaction() as session:
            found = await session.get_many(CF_BLOCK_META, [bytes(header_hash) for header_hash, _, _ in membership])
            compact_delta = 0
            uncompact_delta = 0
            for header_hash, in_main_chain, compact in membership:
                raw = found.get(bytes(header_hash))
                if raw is None:
                    continue
                meta = decode_block_meta(raw)
                updated = BlockMeta(meta.prev_hash, meta.height, compact, in_main_chain, meta.ses, meta.record)
                session.put(CF_BLOCK_META, bytes(header_hash), encode_block_meta(updated))
                if not in_main_chain:
                    continue
                session.put(CF_MAIN_CHAIN, u32_be(meta.height), bytes(header_hash))
                if compact:
                    compact_delta += 1
                else:
                    session.put(CF_UNCOMPACTIFIED, u32_be(meta.height), bytes(header_hash))
                    uncompact_delta += 1
            if compact_delta:
                await self._add_count(session, META_COMPACT_COUNT, compact_delta)
            if uncompact_delta:
                await self._add_count(session, META_UNCOMPACT_COUNT, uncompact_delta)

    async def rebuild_main_chain(self, membership: list[tuple[bytes32, bool, bool]]) -> None:
        """Replace main-chain indexes from a consistent snapshot. Membership is (hash, in_chain, compact)."""
        await self.clear_main_chain_index()
        step = 16000
        for start in range(0, len(membership), step):
            await self.apply_membership(membership[start : start + step])

    async def main_chain_hash_at(self, height: int) -> bytes32 | None:
        async with self.db.reader_no_transaction() as view:
            raw = await view.get(CF_MAIN_CHAIN, u32_be(height))
        if raw is None:
            return None
        return bytes32(raw)

    async def peak_height_map_row(self) -> tuple[bytes32, bytes32, uint32, bytes | None] | None:
        async with self.db.reader_no_transaction() as view:
            raw = await view.get(CF_META, META_PEAK)
            if raw is None:
                return None
            header_hash = bytes32(raw)
            meta = await self._load_meta(view, header_hash)
        if meta is None:
            return None
        return header_hash, meta.prev_hash, uint32(meta.height), meta.ses

    async def main_chain_height_map_window(
        self, start: int, end_exclusive: int
    ) -> dict[bytes32, tuple[uint32, bytes32, bytes | None]]:
        if end_exclusive <= start:
            return {}
        async with self.db.reader_no_transaction() as view:
            chain = await self._main_chain_between(view, start, end_exclusive - 1)
            ordered: dict[bytes32, tuple[uint32, bytes32, bytes | None]] = {}
            for height, header_hash in chain:
                meta = await self._load_meta(view, header_hash)
                if meta is None:
                    continue
                ordered[header_hash] = (uint32(height), meta.prev_hash, meta.ses)
        return ordered

    async def _mark_imported_in_chain(self, session: WriteSession, header_hash: bytes32, meta: BlockMeta) -> None:
        height_key = u32_be(meta.height)
        session.put(CF_MAIN_CHAIN, height_key, bytes(header_hash))
        if meta.compact:
            await self._add_count(session, META_COMPACT_COUNT, 1)
        else:
            session.put(CF_UNCOMPACTIFIED, height_key, bytes(header_hash))
            await self._add_count(session, META_UNCOMPACT_COUNT, 1)

    async def rollback(self, height: int) -> None:
        async with self.db.writer_maybe_transaction() as session:
            start = u32_be(height + 1) if height >= 0 else b""
            rows = await session.scan(CF_MAIN_CHAIN, start, None)
            for key, value in rows:
                await self._clear_main_chain(session, u32_from_be(key), bytes32(value))

    async def set_in_chain(self, header_hashes: list[tuple[bytes32]]) -> None:
        async with self.db.writer_maybe_transaction() as session:
            for (header_hash,) in header_hashes:
                meta = await self._load_meta(session, header_hash)
                if meta is None:
                    raise RuntimeError(f"The blockchain database is corrupt. All of {header_hashes} should exist")
                if meta.in_main_chain:
                    continue
                height_key = u32_be(meta.height)
                previous = await session.get(CF_MAIN_CHAIN, height_key)
                if previous is not None and bytes32(previous) != header_hash:
                    await self._clear_main_chain(session, meta.height, bytes32(previous))
                updated = BlockMeta(
                    meta.prev_hash, meta.height, meta.compact, True, meta.ses, meta.record
                )
                session.put(CF_BLOCK_META, bytes(header_hash), encode_block_meta(updated))
                session.put(CF_MAIN_CHAIN, height_key, bytes(header_hash))
                if meta.compact:
                    await self._add_count(session, META_COMPACT_COUNT, 1)
                else:
                    session.put(CF_UNCOMPACTIFIED, height_key, bytes(header_hash))
                    await self._add_count(session, META_UNCOMPACT_COUNT, 1)

    async def set_peak(self, header_hash: bytes32) -> None:
        async with self.db.writer_maybe_transaction() as session:
            session.put(CF_META, META_PEAK, bytes(header_hash))

    async def replace_proof(self, header_hash: bytes32, block: FullBlock) -> None:
        assert header_hash == block.header_hash
        self.block_cache.put(header_hash, block)
        async with self.db.writer() as session:
            meta = await self._load_meta(session, header_hash)
            if meta is None:
                return
            compact = bool(block.is_fully_compactified())
            if meta.in_main_chain and compact != meta.compact:
                height_key = u32_be(meta.height)
                if compact:
                    session.delete(CF_UNCOMPACTIFIED, height_key)
                    await self._add_count(session, META_COMPACT_COUNT, 1)
                    await self._add_count(session, META_UNCOMPACT_COUNT, -1)
                else:
                    session.put(CF_UNCOMPACTIFIED, height_key, bytes(header_hash))
                    await self._add_count(session, META_COMPACT_COUNT, -1)
                    await self._add_count(session, META_UNCOMPACT_COUNT, 1)
            updated = BlockMeta(meta.prev_hash, meta.height, compact, meta.in_main_chain, meta.ses, meta.record)
            session.put(CF_BLOCK_BLOBS, bytes(header_hash), compress(block))
            session.put(CF_BLOCK_META, bytes(header_hash), encode_block_meta(updated))

    async def get_full_block(self, header_hash: bytes32) -> FullBlock | None:
        cached = self.block_cache.get(header_hash)
        if cached is not None:
            return cached
        raw = await self._blob(header_hash)
        if raw is None:
            return None
        block = decompress(raw)
        self.block_cache.put(header_hash, block)
        return block

    async def get_full_block_bytes(self, header_hash: bytes32) -> bytes | None:
        cached = self.block_cache.get(header_hash)
        if cached is not None:
            return bytes(cached)
        raw = await self._blob(header_hash)
        if raw is None:
            return None
        return zstd.decompress(raw)

    async def get_full_blocks_at(self, heights: list[uint32]) -> list[FullBlock]:
        if len(heights) == 0:
            return []
        blocks: list[FullBlock] = []
        async with self.db.reader_no_transaction() as view:
            for height in heights:
                prefix = u32_be(int(height))
                rows = await view.scan(CF_BLOCKS_AT_HEIGHT, prefix, prefix_end(prefix))
                for key, _value in rows:
                    raw = await view.get(CF_BLOCK_BLOBS, key[-32:])
                    if raw is not None:
                        blocks.append(decompress(raw))
        return blocks

    async def get_block_info(self, header_hash: bytes32) -> GeneratorBlockInfo | None:
        cached = self.block_cache.get(header_hash)
        if cached is not None:
            return GeneratorBlockInfo(
                cached.foliage.prev_block_hash,
                cached.transactions_generator,
                cached.transactions_generator_ref_list,
                cached.transactions_generator_buffer,
                cached.version,
            )
        loaded = await self._decompressed_with_height(header_hash)
        if loaded is None:
            return None
        block_bytes, height = loaded
        try:
            return block_info_from_block(memoryview(block_bytes))
        except Exception as e:
            log.exception(f"cheap parser failed for block at height {height}: {e}")
            block = FullBlock.from_bytes(block_bytes)
            return GeneratorBlockInfo(
                block.foliage.prev_block_hash,
                block.transactions_generator,
                block.transactions_generator_ref_list,
                block.transactions_generator_buffer,
                block.version,
            )

    async def get_generator(self, header_hash: bytes32) -> bytes | None:
        cached = self.block_cache.get(header_hash)
        if cached is not None:
            return get_transactions_generator_bytes(cached)
        loaded = await self._decompressed_with_height(header_hash)
        if loaded is None:
            return None
        block_bytes, height = loaded
        try:
            return generator_from_block(memoryview(block_bytes))
        except Exception as e:  # pragma: no cover
            log.error(f"cheap parser failed for block at height {height}: {e}")
            return get_transactions_generator_bytes(FullBlock.from_bytes(block_bytes))

    async def get_generators_at(self, heights: set[uint32]) -> dict[uint32, bytes]:
        if len(heights) == 0:
            return {}
        generators: dict[uint32, bytes] = {}
        async with self.db.reader_no_transaction() as view:
            for height in heights:
                raw_hash = await view.get(CF_MAIN_CHAIN, u32_be(int(height)))
                if raw_hash is None:
                    continue
                raw = await view.get(CF_BLOCK_BLOBS, raw_hash)
                if raw is None:
                    continue
                block_bytes = zstd.decompress(raw)
                try:
                    gen = generator_from_block(memoryview(block_bytes))
                except Exception as e:  # pragma: no cover
                    log.error(f"cheap parser failed for block at height {height}: {e}")
                    gen = get_transactions_generator_bytes(FullBlock.from_bytes(block_bytes))
                if gen is None:
                    raise ValueError(Err.GENERATOR_REF_HAS_NO_GENERATOR)
                generators[uint32(height)] = gen
        if len(generators) != len(heights):
            raise KeyError(Err.GENERATOR_REF_HAS_NO_GENERATOR)
        return generators

    async def get_block_records_by_hash(self, header_hashes: list[bytes32]) -> list[BlockRecord]:
        if len(header_hashes) == 0:
            return []
        found: dict[bytes32, BlockRecord] = {}
        async with self.db.reader_no_transaction() as view:
            for header_hash in header_hashes:
                meta = await self._load_meta(view, header_hash)
                if meta is not None:
                    found[header_hash] = BlockRecord.from_bytes(meta.record)
        records: list[BlockRecord] = []
        for header_hash in header_hashes:
            if header_hash not in found:
                raise ValueError(f"Header hash {header_hash} not in the blockchain")
            records.append(found[header_hash])
        return records

    async def get_prev_hash(self, header_hash: bytes32) -> bytes32:
        cached = self.block_cache.get(header_hash)
        if cached is not None:
            return cached.prev_header_hash
        async with self.db.reader_no_transaction() as view:
            meta = await self._load_meta(view, header_hash)
        if meta is None:
            raise KeyError("missing block in chain")
        return meta.prev_hash

    async def get_block_bytes_by_hash(self, header_hashes: list[bytes32]) -> list[bytes]:
        return [bytes(block) for block in await self._blocks_by_hash(header_hashes, cache=False)]

    async def get_blocks_by_hash(self, header_hashes: list[bytes32]) -> list[FullBlock]:
        return await self._blocks_by_hash(header_hashes, cache=True)

    async def stored_block_metas(self) -> list[tuple[bytes32, BlockMeta]]:
        async with self.db.reader_no_transaction() as view:
            rows = await view.scan(CF_BLOCK_META, b"", None)
        return [(bytes32(key), decode_block_meta(value)) for key, value in rows]

    async def get_block_record(self, header_hash: bytes32) -> BlockRecord | None:
        async with self.db.reader_no_transaction() as view:
            meta = await self._load_meta(view, header_hash)
        if meta is None:
            return None
        return BlockRecord.from_bytes(meta.record)

    async def get_block_records_in_range(self, start: int, stop: int) -> dict[bytes32, BlockRecord]:
        records: dict[bytes32, BlockRecord] = {}
        async with self.db.reader_no_transaction() as view:
            for _height, header_hash in await self._main_chain_between(view, start, stop):
                meta = await self._load_meta(view, header_hash)
                if meta is not None:
                    records[header_hash] = BlockRecord.from_bytes(meta.record)
        return records

    async def get_block_bytes_in_range(self, start: int, stop: int) -> list[bytes]:
        async with self.db.reader_no_transaction() as view:
            chain = await self._main_chain_between(view, start, stop)
            if len(chain) != (stop - start) + 1:
                raise ValueError(f"Some blocks in range {start}-{stop} were not found.")
            blobs: list[bytes] = []
            for _height, header_hash in chain:
                raw = await view.get(CF_BLOCK_BLOBS, bytes(header_hash))
                if raw is None:
                    raise ValueError(f"Some blocks in range {start}-{stop} were not found.")
                blobs.append(decompress_blob(raw))
        return blobs

    async def get_peak(self) -> tuple[bytes32, uint32] | None:
        async with self.db.reader_no_transaction() as view:
            raw = await view.get(CF_META, META_PEAK)
            if raw is None:
                return None
            header_hash = bytes32(raw)
            meta = await self._load_meta(view, header_hash)
        if meta is None:
            return None
        return header_hash, uint32(meta.height)

    async def get_block_records_close_to_peak(self, blocks_n: int) -> tuple[dict[bytes32, BlockRecord], bytes32 | None]:
        peak = await self.get_peak()
        if peak is None:
            return {}, None
        start = max(0, int(peak[1]) - blocks_n)
        return await self.get_block_records_in_range(start, int(peak[1])), peak[0]

    async def is_fully_compactified(self, header_hash: bytes32) -> bool | None:
        async with self.db.reader_no_transaction() as view:
            meta = await self._load_meta(view, header_hash)
        if meta is None:
            return None
        return meta.compact

    async def get_random_not_compactified(self, number: int) -> list[int]:
        if number <= 0:
            return []
        async with self.db.reader_no_transaction() as view:
            rows = await view.scan(CF_UNCOMPACTIFIED, b"", None)
        heights = [u32_from_be(key) for key, _value in rows]
        if number >= len(heights):
            return heights
        return random.sample(heights, number)

    async def count_compactified_blocks(self) -> int:
        return await self._count(META_COMPACT_COUNT)

    async def count_uncompactified_blocks(self) -> int:
        return await self._count(META_UNCOMPACT_COUNT)

    async def persist_sub_epoch_challenge_segments(
        self, ses_block_hash: bytes32, segments: list[SubEpochChallengeSegment]
    ) -> None:
        async with self.db.writer_maybe_transaction() as session:
            session.put(CF_SES, bytes(ses_block_hash), bytes(SubEpochSegments(segments)))
        self.ses_challenge_cache.put(ses_block_hash, segments)

    async def get_sub_epoch_challenge_segments(
        self, ses_block_hash: bytes32
    ) -> list[SubEpochChallengeSegment] | None:
        cached = self.ses_challenge_cache.get(ses_block_hash)
        if cached is not None:
            return cached
        async with self.db.reader_no_transaction() as view:
            raw = await view.get(CF_SES, bytes(ses_block_hash))
        if raw is None:
            return None
        segments = SubEpochSegments.from_bytes(raw).challenge_segments
        self.ses_challenge_cache.put(ses_block_hash, segments)
        return segments

    async def _blob(self, header_hash: bytes32) -> bytes | None:
        async with self.db.reader_no_transaction() as view:
            return await view.get(CF_BLOCK_BLOBS, bytes(header_hash))

    async def _decompressed_with_height(self, header_hash: bytes32) -> tuple[bytes, int] | None:
        async with self.db.reader_no_transaction() as view:
            raw = await view.get(CF_BLOCK_BLOBS, bytes(header_hash))
            meta = await self._load_meta(view, header_hash)
        if raw is None or meta is None:
            return None
        return zstd.decompress(raw), meta.height

    async def _blocks_by_hash(self, header_hashes: list[bytes32], *, cache: bool) -> list[FullBlock]:
        if len(header_hashes) == 0:
            return []
        found: dict[bytes32, FullBlock] = {}
        async with self.db.reader_no_transaction() as view:
            for header_hash in header_hashes:
                raw = await view.get(CF_BLOCK_BLOBS, bytes(header_hash))
                if raw is None:
                    continue
                block = decompress(raw)
                found[header_hash] = block
                if cache:
                    self.block_cache.put(header_hash, block)
        ordered: list[FullBlock] = []
        for header_hash in header_hashes:
            block = found.get(header_hash)
            if block is None:
                raise ValueError(f"Header hash {header_hash} not in the blockchain")
            ordered.append(block)
        return ordered

    async def _load_meta(self, view: _View, header_hash: bytes32) -> BlockMeta | None:
        raw = await view.get(CF_BLOCK_META, bytes(header_hash))
        if raw is None:
            return None
        return decode_block_meta(raw)

    async def _main_chain_between(self, view: _View, start: int, stop: int) -> list[tuple[int, bytes32]]:
        if stop < start:
            return []
        end = None if stop >= 2**32 - 1 else u32_be(stop + 1)
        rows = await view.scan(CF_MAIN_CHAIN, u32_be(max(start, 0)), end)
        return [(u32_from_be(key), bytes32(value)) for key, value in rows]

    async def _clear_main_chain(self, session: WriteSession, height: int, header_hash: bytes32) -> None:
        meta = await self._load_meta(session, header_hash)
        session.delete(CF_MAIN_CHAIN, u32_be(height))
        if meta is None or not meta.in_main_chain:
            return
        if meta.compact:
            await self._add_count(session, META_COMPACT_COUNT, -1)
        else:
            session.delete(CF_UNCOMPACTIFIED, u32_be(height))
            await self._add_count(session, META_UNCOMPACT_COUNT, -1)
        cleared = BlockMeta(meta.prev_hash, meta.height, meta.compact, False, meta.ses, meta.record)
        session.put(CF_BLOCK_META, bytes(header_hash), encode_block_meta(cleared))

    async def _add_count(self, session: WriteSession, key: bytes, delta: int) -> None:
        current = await self._count_from(session, key)
        updated = current + delta
        if updated < 0:
            raise ValueError(f"{key!r} counter went negative ({updated})")
        session.put(CF_META, key, updated.to_bytes(8, "little", signed=False))

    async def _count(self, key: bytes) -> int:
        async with self.db.reader_no_transaction() as view:
            return await self._count_from(view, key)

    async def _count_from(self, view: _View, key: bytes) -> int:
        raw = await view.get(CF_META, key)
        if raw is None:
            return 0
        return int.from_bytes(raw, "little", signed=False)
