"""Choose one small slice of a height-ordered column family to pack.

Those families are stored in block-height order. Each flush is its own file,
the files do not overlap, and ordinary compaction moves them down without
merging them. The file count grows until a pack merges a slice on purpose.

A reorg writes those same heights again. The slice is chosen from the file
list read at the moment of the pack. Nothing about the next height is stored.
Heights still close to the tip are left unpacked, because that is the part a
reorg usually rewrites. A deeper reorg shows up as new files on the next list.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass

from chia.full_node.db.keys import CF_BLOCKS_AT_HEIGHT, CF_COIN_DELTA, CF_MAIN_CHAIN, CF_UNCOMPACTIFIED

# Height-ordered families. Coin records and block bodies already land in large
# files and are left to ordinary compaction.
CHIP_FAMILIES: tuple[str, ...] = (CF_MAIN_CHAIN, CF_BLOCKS_AT_HEIGHT, CF_UNCOMPACTIFIED, CF_COIN_DELTA)

# Stay near the open time of a few thousand files, not tens of thousands.
FILE_CAP = 400
# RocksDB's target file size. One pack rewrites about this much.
CHUNK_BYTES = 64 * 1024 * 1024
# A file already near the target size is a finished pack. Repacking it does
# not cut the count.
SMALL_BYTES = 32 * 1024 * 1024
# Blocks newer than this stay unpacked. 2048 blocks is about 11 hours, which
# covers an ordinary fork. The pack reads the live files again after a deeper one.
REORG_HEADROOM_BLOCKS = 2048
# A save at least this slow is a heavy stretch. Skip the pack so the disk can
# finish the block instead of rewriting old files.
SLOW_SAVE_SECONDS = 0.5

_FILE_LINE = re.compile(r"^\s*(\d+):(\d+)\[[^\]]*\]\['([0-9a-fA-F]*)' seq:\d+, type:\d+ \.\. '([0-9a-fA-F]*)'")


@dataclass(frozen=True)
class SstSpan:
    """One on-disk file: its RocksDB number, inclusive key bounds, and size."""

    number: int
    start: bytes
    end: bytes
    size: int


@dataclass(frozen=True)
class ChipChoice:
    """Inclusive key range to pack, and the small files that suggested it."""

    start: bytes
    end: bytes
    file_numbers: tuple[int, ...]


def parse_sstables(text: str) -> list[SstSpan] | None:
    """Read file bounds from a `rocksdb.sstables` property.

    Returns None when a file line is present but not understood. An empty
    property is an empty list, which means this family has no files yet.
    """
    spans: list[SstSpan] = []
    for line in text.splitlines():
        if "seq:" not in line:
            continue
        match = _FILE_LINE.match(line)
        if match is None:
            return None
        number_text, size_text, start_hex, end_hex = match.groups()
        try:
            start = bytes.fromhex(start_hex)
            end = bytes.fromhex(end_hex)
        except ValueError:
            return None
        spans.append(SstSpan(int(number_text), start, end, int(size_text)))
    return spans


def families_over_cap(counts: Mapping[str, int], *, file_cap: int) -> list[str]:
    """Families past the cap, the most over first. Ties keep the family order."""
    over = [name for name in CHIP_FAMILIES if counts.get(name, 0) > file_cap]
    over.sort(key=lambda name: -counts[name])
    return over


def _height_prefix(key: bytes) -> int | None:
    if len(key) < 4:
        return None
    return int.from_bytes(key[:4], "big")


def _overlaps(start: bytes, end: bytes, other: SstSpan) -> bool:
    return start <= other.end and other.start <= end


def select_chip_range(
    files: list[SstSpan],
    *,
    file_count: int,
    file_cap: int = FILE_CAP,
    chunk_bytes: int = CHUNK_BYTES,
    small_bytes: int = SMALL_BYTES,
    reorg_headroom: int = REORG_HEADROOM_BLOCKS,
    save_waiting: bool = False,
    last_save_slow: bool = False,
    chip_running: bool = False,
    long_sync: bool = False,
) -> ChipChoice | None:
    """The oldest packed slice, or None when this is not a moment to pack.

    The end key is inclusive. Files at the tip are left out. A slice stops
    before a large file so the pack does not pull that file in. When a reorg
    has written new small files on top of one already-packed file, that one
    packed file may be included so the new values replace the old ones.
    """
    if long_sync or save_waiting or last_save_slow or chip_running or file_count <= file_cap:
        return None
    if not files or chunk_bytes < 0 or small_bytes <= 0:
        return None

    tip_height: int | None = None
    for item in files:
        height = _height_prefix(item.end)
        if height is not None and (tip_height is None or height > tip_height):
            tip_height = height
    newest_settled = None if tip_height is None else tip_height - max(reorg_headroom, 0)

    def settled(item: SstSpan) -> bool:
        if newest_settled is None:
            return True
        height = _height_prefix(item.end)
        return height is not None and height <= newest_settled

    def small(item: SstSpan) -> bool:
        return item.size < small_bytes and settled(item) and len(item.start) > 0 and len(item.end) > 0

    ordered = sorted(files, key=lambda item: (item.start, item.end, item.number))
    smalls = [item for item in ordered if small(item)]

    def overlap_bytes(start: bytes, end: bytes, chosen: list[SstSpan]) -> int:
        chosen_ids = {id(item) for item in chosen}
        total = 0
        for item in files:
            if id(item) in chosen_ids or not _overlaps(start, end, item):
                continue
            total += item.size
        return total

    def close(chosen: list[SstSpan]) -> ChipChoice | None:
        # Drop files from the newer end until the extra packed data is at most one chunk.
        for end_index in range(len(chosen), 1, -1):
            part = chosen[:end_index]
            start = min(item.start for item in part)
            end = max(item.end for item in part)
            if overlap_bytes(start, end, part) > chunk_bytes:
                continue
            return ChipChoice(start, end, tuple(item.number for item in part))
        return None

    run: list[SstSpan] = []
    run_bytes = 0
    for item in smalls:
        if run:
            previous_end = max(existing.end for existing in run)
            split = False
            for other in ordered:
                if small(other):
                    continue
                # A file we are not packing sits strictly between these two small files.
                if previous_end < other.start < item.start:
                    split = True
                    break
            start = min(existing.start for existing in run)
            end = max(previous_end, item.end)
            # Adding this file would pull in more than one chunk of already-packed data.
            blown = overlap_bytes(start, end, [*run, item]) > chunk_bytes
            full = len(run) >= 2 and (run_bytes >= chunk_bytes or run_bytes + item.size > chunk_bytes)
            if split or blown or full:
                choice = close(run)
                if choice is not None:
                    return choice
                run = []
                run_bytes = 0
        run.append(item)
        run_bytes += item.size
    return close(run)
