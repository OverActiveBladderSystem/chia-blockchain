from __future__ import annotations

import asyncio
import logging
import re
import time
from dataclasses import replace
from typing import cast

import pytest
from chia_rs import (
    BlockRecord,
    FullBlock,
    G2Element,
    MerkleSet,
    SpendBundle,
    TransactionsInfo,
    is_canonical_serialization,
)
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint8, uint32, uint64

from chia._tests.blockchain.blockchain_test_utils import (
    _validate_and_add_block,
    _validate_and_add_block_multi_error,
    _validate_and_add_block_multi_result,
    _validate_and_add_block_no_error,
    check_block_store_invariant,
)
from chia._tests.conftest import ConsensusMode
from chia._tests.core.full_node.test_full_node import find_reward_coin
from chia._tests.util.get_name_puzzle_conditions import get_name_puzzle_conditions
from chia.consensus.augmented_chain import AugmentedBlockchain
from chia.consensus.block_body_validation import ForkInfo
from chia.consensus.block_generator_info import (
    block_has_transactions_generator,
    get_transactions_generator_bytes,
    get_transactions_generator_program,
)
from chia.consensus.block_header_validation import validate_finished_header_block
from chia.consensus.block_rewards import calculate_base_farmer_reward
from chia.consensus.blockchain import AddBlockResult, Blockchain
from chia.consensus.coinbase import create_farmer_coin
from chia.consensus.find_fork_point import lookup_fork_chain
from chia.consensus.full_block_to_block_record import block_to_block_record
from chia.consensus.generator_tools import get_block_header
from chia.consensus.multiprocess_validation import PreValidationResult, pre_validate_block
from chia.simulator.block_tools import BlockTools, make_unfinished_block
from chia.simulator.wallet_tools import WalletTool
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.serialized_program import SerializedProgram
from chia.types.condition_opcodes import ConditionOpcode
from chia.types.condition_with_args import ConditionWithArgs
from chia.types.generator_types import BlockGenerator
from chia.types.validation_state import ValidationState
from chia.util.casts import int_to_bytes
from chia.util.errors import Err
from chia.util.hash import std_hash
from chia.util.recursive_replace import recursive_replace
from chia.wallet.conditions import AssertCoinAnnouncement, AssertPuzzleAnnouncement
from chia.wallet.puzzles.p2_delegated_puzzle_or_hidden_puzzle import (
    DEFAULT_HIDDEN_PUZZLE_HASH,
    calculate_synthetic_secret_key,
)

log = logging.getLogger(__name__)

MY_COIN_ASSERTION_OPCODES = [
    ConditionOpcode.ASSERT_MY_AMOUNT,
    ConditionOpcode.ASSERT_MY_PUZZLEHASH,
    ConditionOpcode.ASSERT_MY_COIN_ID,
    ConditionOpcode.ASSERT_MY_PARENT_ID,
]

AGG_SIG_OPCODES = [
    ConditionOpcode.AGG_SIG_ME,
    ConditionOpcode.AGG_SIG_UNSAFE,
    ConditionOpcode.AGG_SIG_PARENT,
    ConditionOpcode.AGG_SIG_PUZZLE,
    ConditionOpcode.AGG_SIG_AMOUNT,
    ConditionOpcode.AGG_SIG_PUZZLE_AMOUNT,
    ConditionOpcode.AGG_SIG_PARENT_AMOUNT,
    ConditionOpcode.AGG_SIG_PARENT_PUZZLE,
]

TIMELOCK_CASES = [
    (ConditionOpcode.ASSERT_MY_BIRTH_HEIGHT, -1, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_MY_BIRTH_HEIGHT, 0x100000000, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_MY_BIRTH_HEIGHT, 2, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_MY_BIRTH_HEIGHT, 3, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, -1, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, 0x10000000000000000, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, 10019, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, 10020, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, 10021, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_SECONDS_RELATIVE, -2, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_SECONDS_RELATIVE, -1, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_SECONDS_RELATIVE, 0, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_SECONDS_RELATIVE, 1, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_SECONDS_RELATIVE, 9, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_SECONDS_RELATIVE, 10, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_SECONDS_RELATIVE, 11, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, -2, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, -1, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, 0, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, 1, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, 9, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, 10, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, 11, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_HEIGHT_RELATIVE, -2, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_HEIGHT_RELATIVE, -1, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_HEIGHT_RELATIVE, 0, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_HEIGHT_RELATIVE, 1, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_HEIGHT_RELATIVE, -2, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_HEIGHT_RELATIVE, -1, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_HEIGHT_RELATIVE, 0, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_HEIGHT_RELATIVE, 1, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_HEIGHT_ABSOLUTE, 1, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_HEIGHT_ABSOLUTE, 2, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_HEIGHT_ABSOLUTE, 3, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_HEIGHT_ABSOLUTE, 4, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_HEIGHT_ABSOLUTE, 1, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_HEIGHT_ABSOLUTE, 2, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_HEIGHT_ABSOLUTE, 3, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_BEFORE_HEIGHT_ABSOLUTE, 4, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10019, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10020, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10021, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10029, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10030, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10031, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10032, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10019, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10020, AddBlockResult.INVALID_BLOCK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10021, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10029, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10030, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10031, AddBlockResult.NEW_PEAK),
    (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10032, AddBlockResult.NEW_PEAK),
]

EPHEMERAL_TIMELOCK_CASES = [
        # we don't allow any birth assertions, not
        # relative time locks on ephemeral coins. This test is only for
        # ephemeral coins, so these cases should always fail
        # MY BIRHT HEIGHT
        (ConditionOpcode.ASSERT_MY_BIRTH_HEIGHT, -1, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_MY_BIRTH_HEIGHT, 0x100000000, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_MY_BIRTH_HEIGHT, 2, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_MY_BIRTH_HEIGHT, 3, AddBlockResult.INVALID_BLOCK),
        # MY BIRHT SECONDS
        (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, -1, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, 0x10000000000000000, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, 10029, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, 10030, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_MY_BIRTH_SECONDS, 10031, AddBlockResult.INVALID_BLOCK),
        # SECONDS RELATIVE
        # genesis timestamp is 10000 and each block is 10 seconds
        (ConditionOpcode.ASSERT_SECONDS_RELATIVE, -2, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_SECONDS_RELATIVE, -1, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_SECONDS_RELATIVE, 0, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_SECONDS_RELATIVE, 1, AddBlockResult.INVALID_BLOCK),
        # BEFORE SECONDS RELATIVE
        # relative conditions are not allowed on ephemeral spends
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, -2, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, -1, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, 0, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, 10, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_RELATIVE, 0x10000000000000000, AddBlockResult.INVALID_BLOCK),
        # HEIGHT RELATIVE
        (ConditionOpcode.ASSERT_HEIGHT_RELATIVE, -2, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_HEIGHT_RELATIVE, -1, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_HEIGHT_RELATIVE, 0, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_HEIGHT_RELATIVE, 1, AddBlockResult.INVALID_BLOCK),
        # BEFORE HEIGHT RELATIVE
        # relative conditions are not allowed on ephemeral spends
        (ConditionOpcode.ASSERT_BEFORE_HEIGHT_RELATIVE, -2, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_HEIGHT_RELATIVE, -1, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_HEIGHT_RELATIVE, 0, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_HEIGHT_RELATIVE, 1, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_HEIGHT_RELATIVE, 0x100000000, AddBlockResult.INVALID_BLOCK),
        # HEIGHT ABSOLUTE
        (ConditionOpcode.ASSERT_HEIGHT_ABSOLUTE, 2, AddBlockResult.NEW_PEAK),
        (ConditionOpcode.ASSERT_HEIGHT_ABSOLUTE, 3, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_HEIGHT_ABSOLUTE, 4, AddBlockResult.INVALID_BLOCK),
        # BEFORE HEIGHT ABSOLUTE
        (ConditionOpcode.ASSERT_BEFORE_HEIGHT_ABSOLUTE, 2, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_HEIGHT_ABSOLUTE, 3, AddBlockResult.NEW_PEAK),
        (ConditionOpcode.ASSERT_BEFORE_HEIGHT_ABSOLUTE, 4, AddBlockResult.NEW_PEAK),
        # SECONDS ABSOLUTE
        # genesis timestamp is 10000 and each block is 10 seconds
        (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10020, AddBlockResult.NEW_PEAK),  # <- previous tx-block
        (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10021, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10029, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10030, AddBlockResult.INVALID_BLOCK),  # <- current tx-block
        (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10031, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_SECONDS_ABSOLUTE, 10032, AddBlockResult.INVALID_BLOCK),
        # BEFORE SECONDS ABSOLUTE
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10020, AddBlockResult.INVALID_BLOCK),
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10021, AddBlockResult.NEW_PEAK),
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10030, AddBlockResult.NEW_PEAK),
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10031, AddBlockResult.NEW_PEAK),
        (ConditionOpcode.ASSERT_BEFORE_SECONDS_ABSOLUTE, 10032, AddBlockResult.NEW_PEAK),
    ]


async def run_basic_reorg(b: Blockchain, bt: BlockTools) -> None:
    blocks = bt.get_consecutive_blocks(15)

    for block in blocks:
        await _validate_and_add_block(b, block)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 14

    blocks_reorg_chain = bt.get_consecutive_blocks(7, blocks[:10], seed=b"2")
    fork_info = ForkInfo(-1, -1, bt.constants.GENESIS_CHALLENGE)
    aug_chain = AugmentedBlockchain(b)
    for reorg_block in blocks_reorg_chain:
        if reorg_block.height < 10:
            await _validate_and_add_block(
                b,
                reorg_block,
                expected_result=AddBlockResult.ALREADY_HAVE_BLOCK,
                fork_info=fork_info,
                augmented_blockchain=aug_chain,
            )
        elif reorg_block.height < 15:
            await _validate_and_add_block(
                b,
                reorg_block,
                expected_result=AddBlockResult.ADDED_AS_ORPHAN,
                fork_info=fork_info,
                augmented_blockchain=aug_chain,
            )
        elif reorg_block.height >= 15:
            await _validate_and_add_block(b, reorg_block, fork_info=fork_info, augmented_blockchain=aug_chain)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 16


def _header_hash(block: BlockRecord | None) -> bytes32 | None:
    if block is None:
        return None
    return block.header_hash


async def run_get_tx_peak_reorg(b: Blockchain, bt: BlockTools, reorg_point: int) -> None:
    """The transaction-block peak follows the winning chain through a fork."""
    blocks = bt.get_consecutive_blocks(reorg_point)
    last_tx_block: bytes32 | None = None
    for block in blocks:
        assert _header_hash(b.get_tx_peak()) == last_tx_block
        await _validate_and_add_block(b, block)
        if block.is_transaction_block():
            last_tx_block = block.header_hash
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == reorg_point - 1
    assert _header_hash(b.get_tx_peak()) == last_tx_block

    reorg_last_tx_block: bytes32 | None = None
    fork_block = blocks[9]
    fork_info = ForkInfo(fork_block.height, fork_block.height, fork_block.header_hash)
    blocks_reorg_chain = bt.get_consecutive_blocks(7, blocks[:10], seed=b"2")
    assert blocks_reorg_chain[reorg_point].is_transaction_block() is False
    aug_chain = AugmentedBlockchain(b)
    for reorg_block in blocks_reorg_chain:
        if reorg_block.height < 10:
            await _validate_and_add_block(
                b, reorg_block, expected_result=AddBlockResult.ALREADY_HAVE_BLOCK, augmented_blockchain=aug_chain
            )
        elif reorg_block.height < reorg_point:
            await _validate_and_add_block(
                b,
                reorg_block,
                expected_result=AddBlockResult.ADDED_AS_ORPHAN,
                fork_info=fork_info,
                augmented_blockchain=aug_chain,
            )
        else:
            await _validate_and_add_block(b, reorg_block, fork_info=fork_info, augmented_blockchain=aug_chain)
        if reorg_block.is_transaction_block():
            reorg_last_tx_block = reorg_block.header_hash
        if reorg_block.height >= reorg_point:
            last_tx_block = reorg_last_tx_block
        assert _header_hash(b.get_tx_peak()) == last_tx_block
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 16


async def run_reorg_from_genesis(b: Blockchain, bt: BlockTools) -> None:
    """Replace the chain with a longer one from genesis, then switch back to a longer original."""
    blocks = bt.get_consecutive_blocks(15)
    for block in blocks:
        await _validate_and_add_block(b, block)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 14

    blocks_reorg_chain = bt.get_consecutive_blocks(16, [], seed=b"2")
    fork_info = ForkInfo(-1, -1, bt.constants.GENESIS_CHALLENGE)
    aug_chain = AugmentedBlockchain(b)
    for reorg_block in blocks_reorg_chain:
        if reorg_block.height < 15:
            await _validate_and_add_block_multi_result(
                b,
                reorg_block,
                expected_result=[AddBlockResult.ADDED_AS_ORPHAN, AddBlockResult.ALREADY_HAVE_BLOCK],
                fork_info=fork_info,
                augmented_blockchain=aug_chain,
            )
        elif reorg_block.height >= 15:
            await _validate_and_add_block(b, reorg_block, fork_info=fork_info, augmented_blockchain=aug_chain)

    blocks_reorg_chain_2 = bt.get_consecutive_blocks(3, blocks, seed=b"3")
    fork_info = ForkInfo(-1, -1, bt.constants.GENESIS_CHALLENGE)
    aug_chain2 = AugmentedBlockchain(b)
    for reorg_block in blocks_reorg_chain_2:
        if reorg_block.height < 15:
            await _validate_and_add_block(
                b,
                reorg_block,
                expected_result=AddBlockResult.ALREADY_HAVE_BLOCK,
                fork_info=fork_info,
                augmented_blockchain=aug_chain2,
            )
        elif reorg_block.height < 16:
            await _validate_and_add_block(
                b,
                reorg_block,
                expected_result=AddBlockResult.ADDED_AS_ORPHAN,
                fork_info=fork_info,
                augmented_blockchain=aug_chain2,
            )
        else:
            await _validate_and_add_block(b, reorg_block, fork_info=fork_info, augmented_blockchain=aug_chain2)

    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 17
    assert await b.coin_store.num_unspent() > 0


async def run_reorg_new_ref(b: Blockchain, bt: BlockTools, *, expect_same_height_becomes_peak: bool) -> None:
    """A heavier fork whose block refers to generators that live on that fork."""
    wallet_a = WalletTool(b.constants)
    puzzle_hashes = [wallet_a.get_new_puzzlehash() for _ in range(5)]
    coinbase_puzzlehash = puzzle_hashes[0]
    receiver_puzzlehash = puzzle_hashes[1]

    blocks = bt.get_consecutive_blocks(
        5,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        guarantee_transaction_block=True,
    )
    all_coins = []
    for spend_block in blocks[:5]:
        for coin in spend_block.get_included_reward_coins():
            if coin.puzzle_hash == coinbase_puzzlehash:
                all_coins.append(coin)
    spend_bundle_0 = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    blocks = bt.get_consecutive_blocks(
        15,
        block_list_input=blocks,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        transaction_data=spend_bundle_0,
        guarantee_transaction_block=True,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 19

    blocks_reorg_chain = bt.get_consecutive_blocks(
        1,
        blocks[:10],
        seed=b"2",
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
    )
    spend_bundle = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    blocks_reorg_chain = bt.get_consecutive_blocks(
        2,
        blocks_reorg_chain,
        seed=b"2",
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        transaction_data=spend_bundle,
        guarantee_transaction_block=True,
    )
    spend_bundle2 = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    blocks_reorg_chain = bt.get_consecutive_blocks(
        4, blocks_reorg_chain, seed=b"2", block_refs=[uint32(5), uint32(11)], transaction_data=spend_bundle2
    )
    blocks_reorg_chain = bt.get_consecutive_blocks(4, blocks_reorg_chain, seed=b"2")

    fork_info = ForkInfo(-1, -1, b.constants.GENESIS_CHALLENGE)
    for i, block in enumerate(blocks_reorg_chain):
        if i < 10:
            expected = AddBlockResult.ALREADY_HAVE_BLOCK
        elif i < 19:
            expected = AddBlockResult.ADDED_AS_ORPHAN
        elif i == 19:
            peak = b.get_peak()
            assert peak is not None
            if block.total_iters < peak.total_iters:
                expected = AddBlockResult.NEW_PEAK
            else:
                expected = AddBlockResult.ADDED_AS_ORPHAN
            if expect_same_height_becomes_peak:
                assert expected == AddBlockResult.NEW_PEAK
        else:
            expected = AddBlockResult.NEW_PEAK
        await _validate_and_add_block(b, block, expected_result=expected, fork_info=fork_info)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 20


async def run_chain_failed_rollback(b: Blockchain, bt: BlockTools) -> None:
    """Roll the coin store back while the peak stays put, then reject the next spend."""
    wallet_a = WalletTool(b.constants)
    puzzle_hashes = [wallet_a.get_new_puzzlehash() for _ in range(5)]
    coinbase_puzzlehash = puzzle_hashes[0]
    receiver_puzzlehash = puzzle_hashes[1]

    blocks = bt.get_consecutive_blocks(
        20,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        guarantee_transaction_block=True,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 19

    all_coins = []
    for spend_block in blocks[:10]:
        for coin in spend_block.get_included_reward_coins():
            if coin.puzzle_hash == coinbase_puzzlehash:
                all_coins.append(coin)
    spend_bundle = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    blocks_reorg_chain = bt.get_consecutive_blocks(
        11,
        blocks[:10],
        seed=b"2",
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        transaction_data=spend_bundle,
        guarantee_transaction_block=True,
    )
    fork_block = blocks_reorg_chain[9]
    fork_info = ForkInfo(fork_block.height, fork_block.height, fork_block.header_hash)
    for block in blocks_reorg_chain[10:-1]:
        await _validate_and_add_block(b, block, expected_result=AddBlockResult.ADDED_AS_ORPHAN, fork_info=fork_info)

    await b.coin_store.rollback_to_block(2)
    with pytest.raises(ValueError, match="Invalid operation to set spent"):
        await _validate_and_add_block(b, blocks_reorg_chain[-1], fork_info=fork_info)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 19


async def get_fork_info(blockchain: Blockchain, block: FullBlock, peak: BlockRecord) -> ForkInfo:
    fork_chain, fork_hash = await lookup_fork_chain(
        blockchain,
        (peak.height, peak.header_hash),
        (block.height - 1, block.prev_header_hash),
        blockchain.constants,
    )
    fork_height = block.height - len(fork_chain) - 1
    fork_info = ForkInfo(fork_height, fork_height, fork_hash)
    counter = 0
    start = time.monotonic()
    for height in range(fork_info.fork_height + 1, block.height):
        fork_block = await blockchain.block_store.get_full_block(fork_chain[uint32(height)])
        assert fork_block is not None
        assert fork_block.height - 1 == fork_info.peak_height
        assert fork_block.height == 0 or fork_block.prev_header_hash == fork_info.peak_hash
        await blockchain.run_single_block(fork_block, fork_info)
        counter += 1
    log.info(
        "executed %s block generators in %.2fs. %s additions, %s removals",
        counter,
        time.monotonic() - start,
        len(fork_info.additions_since_fork),
        len(fork_info.removals_since_fork),
    )
    return fork_info


async def run_reorg_flip_flop(b: Blockchain, bt: BlockTools) -> None:
    """Two equal-length chains take turns being heavier, then one chain adds ten more blocks."""
    wallet_a = WalletTool(b.constants)
    puzzle_hashes = [wallet_a.get_new_puzzlehash() for _ in range(5)]
    coinbase_puzzlehash = puzzle_hashes[0]
    receiver_puzzlehash = puzzle_hashes[1]

    chain_a = bt.get_consecutive_blocks(
        10,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        guarantee_transaction_block=True,
    )
    all_coins = []
    for spend_block in chain_a:
        for coin in spend_block.get_included_reward_coins():
            if coin.puzzle_hash == coinbase_puzzlehash:
                all_coins.append(coin)

    spend_bundle = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    chain_a = bt.get_consecutive_blocks(
        5,
        chain_a,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        transaction_data=spend_bundle,
        guarantee_transaction_block=True,
    )
    spend_bundle = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    chain_a = bt.get_consecutive_blocks(
        5,
        chain_a,
        block_refs=[uint32(10)],
        transaction_data=spend_bundle,
        guarantee_transaction_block=True,
    )
    spend_bundle = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    chain_a = bt.get_consecutive_blocks(
        20,
        chain_a,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        transaction_data=spend_bundle,
        guarantee_transaction_block=True,
    )

    chain_b = bt.get_consecutive_blocks(
        5,
        chain_a[:20],
        seed=b"2",
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
    )
    spend_bundle = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    chain_b = bt.get_consecutive_blocks(
        5,
        chain_b,
        seed=b"2",
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        transaction_data=spend_bundle,
        guarantee_transaction_block=True,
    )
    spend_bundle = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    chain_b = bt.get_consecutive_blocks(10, chain_b, seed=b"2", block_refs=[uint32(15)], transaction_data=spend_bundle)
    assert len(chain_a) == len(chain_b)

    counter = 0
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    difficulty = b.constants.DIFFICULTY_STARTING
    for block_a, block_b in zip(chain_a, chain_b):
        if counter % 2 == 0:
            block1, block2 = block_b, block_a
        else:
            block1, block2 = block_a, block_b
        counter += 1
        for block in (block1, block2):
            prevalidation = await (
                await pre_validate_block(
                    b.constants,
                    AugmentedBlockchain(b),
                    block,
                    b.pool,
                    None,
                    ValidationState(sub_slot_iters, difficulty, None),
                )
            )
            peak = b.get_peak()
            if peak is None:
                fork_info = ForkInfo(-1, -1, bt.constants.GENESIS_CHALLENGE)
            else:
                fork_info = await get_fork_info(b, block, peak)
            _, err, _ = await b.add_block(block, prevalidation, sub_slot_iters=sub_slot_iters, fork_info=fork_info)
            assert err is None

    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 39
    chain_b = bt.get_consecutive_blocks(
        10,
        chain_b,
        seed=b"2",
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
    )
    for block in chain_b[40:]:
        await _validate_and_add_block(b, block)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 49
    assert await b.coin_store.num_unspent() > 0


async def run_reorg_transaction(b: Blockchain, bt: BlockTools) -> None:
    """A chain that spends a reward coin, then a fork that spends the same coin."""
    wallet_a = WalletTool(b.constants)
    puzzle_hashes = [wallet_a.get_new_puzzlehash() for _ in range(5)]
    coinbase_puzzlehash = puzzle_hashes[0]
    receiver_puzzlehash = puzzle_hashes[1]

    blocks = bt.get_consecutive_blocks(10, farmer_reward_puzzle_hash=coinbase_puzzlehash)
    blocks = bt.get_consecutive_blocks(
        2, blocks, farmer_reward_puzzle_hash=coinbase_puzzlehash, guarantee_transaction_block=True
    )

    spend_coin = None
    for bl in blocks[-2:]:
        for coin in bl.get_included_reward_coins():
            if coin.puzzle_hash == coinbase_puzzlehash:
                spend_coin = coin
                break

    assert spend_coin is not None
    spend_bundle = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, spend_coin)

    blocks = bt.get_consecutive_blocks(
        2,
        blocks,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        transaction_data=spend_bundle,
        guarantee_transaction_block=True,
    )

    blocks_fork = bt.get_consecutive_blocks(
        1, blocks[:12], farmer_reward_puzzle_hash=coinbase_puzzlehash, seed=b"123", guarantee_transaction_block=True
    )
    blocks_fork = bt.get_consecutive_blocks(
        2,
        blocks_fork,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        transaction_data=spend_bundle,
        guarantee_transaction_block=True,
        seed=b"1245",
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    fork_block = blocks[11]
    fork_info = ForkInfo(fork_block.height, fork_block.height, fork_block.header_hash)
    aug_chain = AugmentedBlockchain(b)
    for block in blocks_fork[12:]:
        await _validate_and_add_block_no_error(b, block, fork_info=fork_info, augmented_blockchain=aug_chain)

    created = Coin(spend_coin.name(), receiver_puzzlehash, uint64(1_000))
    created_record = await b.coin_store.get_coin_record(created.name())
    spent_record = await b.coin_store.get_coin_record(spend_coin.name())
    assert created_record is not None
    assert created_record.spent_block_index == 0
    assert spent_record is not None
    assert spent_record.spent_block_index > 0


async def run_reorg_stale_fork_height(b: Blockchain, bt: BlockTools) -> None:
    """Pass a stale fork height while adding the same chain, including a generator reference."""
    wallet_a = WalletTool(b.constants)
    puzzle_hashes = [wallet_a.get_new_puzzlehash() for _ in range(5)]
    coinbase_puzzlehash = puzzle_hashes[0]
    receiver_puzzlehash = puzzle_hashes[1]
    blocks = bt.get_consecutive_blocks(
        5,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        guarantee_transaction_block=True,
    )
    all_coins = []
    for spend_block in blocks:
        for coin in spend_block.get_included_reward_coins():
            if coin.puzzle_hash == coinbase_puzzlehash:
                all_coins.append(coin)
    spend_bundle = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    blocks = bt.get_consecutive_blocks(
        5,
        blocks,
        farmer_reward_puzzle_hash=coinbase_puzzlehash,
        pool_reward_puzzle_hash=receiver_puzzlehash,
        transaction_data=spend_bundle,
        guarantee_transaction_block=True,
    )
    spend_bundle2 = wallet_a.generate_signed_transaction(uint64(1_000), receiver_puzzlehash, all_coins.pop())
    blocks = bt.get_consecutive_blocks(4, blocks, block_refs=[uint32(5)], transaction_data=spend_bundle2)
    for block in blocks[:5]:
        await _validate_and_add_block(b, block, expected_result=AddBlockResult.NEW_PEAK)
    fork_info = ForkInfo(blocks[4].height, blocks[4].height, blocks[4].header_hash)
    for block in blocks[5:]:
        await _validate_and_add_block(b, block, expected_result=AddBlockResult.NEW_PEAK, fork_info=fork_info)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 13
    assert await b.coin_store.num_unspent() > 0


async def run_double_spent_in_reorg(b: Blockchain, bt: BlockTools) -> None:
    """A fork that spends a coin the canonical chain already spent, then an ephemeral double spend."""
    blocks = bt.get_consecutive_blocks(
        3,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)

    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    await _validate_and_add_block(b, blocks[-1])

    new_coin = tx.additions()[0]
    tx_2 = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), new_coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx_2
    )
    await _validate_and_add_block(b, blocks[-1])
    blocks = bt.get_consecutive_blocks(5, block_list_input=blocks, guarantee_transaction_block=True)
    for block in blocks[-5:]:
        await _validate_and_add_block(b, block)

    blocks_reorg = bt.get_consecutive_blocks(2, block_list_input=blocks[:-7], guarantee_transaction_block=True)
    fork_info = ForkInfo(blocks[-8].height, blocks[-8].height, blocks[-8].header_hash)
    await _validate_and_add_block(
        b, blocks_reorg[-2], expected_result=AddBlockResult.ADDED_AS_ORPHAN, fork_info=fork_info
    )
    await _validate_and_add_block(
        b, blocks_reorg[-1], expected_result=AddBlockResult.ADDED_AS_ORPHAN, fork_info=fork_info
    )

    blocks_reorg = bt.get_consecutive_blocks(
        1, block_list_input=blocks_reorg, guarantee_transaction_block=True, transaction_data=tx_2
    )
    await _validate_and_add_block(b, blocks_reorg[-1], expected_error=Err.UNKNOWN_UNSPENT, fork_info=fork_info)

    agg = SpendBundle.aggregate([tx, tx_2])
    blocks_reorg = bt.get_consecutive_blocks(
        1, block_list_input=blocks_reorg[:-1], guarantee_transaction_block=True, transaction_data=agg
    )
    await _validate_and_add_block(
        b, blocks_reorg[-1], expected_result=AddBlockResult.ADDED_AS_ORPHAN, fork_info=fork_info
    )

    blocks_reorg = bt.get_consecutive_blocks(
        1, block_list_input=blocks_reorg, guarantee_transaction_block=True, transaction_data=tx_2
    )
    await _validate_and_add_block(b, blocks_reorg[-1], expected_error=Err.DOUBLE_SPEND_IN_FORK, fork_info=fork_info)

    rewards_ph = wt.get_new_puzzlehash()
    blocks_reorg = bt.get_consecutive_blocks(
        10,
        block_list_input=blocks_reorg[:-1],
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=rewards_ph,
    )
    for block in blocks_reorg[-10:]:
        await _validate_and_add_block_multi_result(
            b, block, expected_result=[AddBlockResult.ADDED_AS_ORPHAN, AddBlockResult.NEW_PEAK], fork_info=fork_info
        )

    first_coin = await b.coin_store.get_coin_record(new_coin.name())
    assert first_coin is not None and first_coin.spent
    second_coin = await b.coin_store.get_coin_record(tx_2.additions()[0].name())
    assert second_coin is not None and not second_coin.spent

    farmer_coin = create_farmer_coin(
        blocks_reorg[-1].height,
        rewards_ph,
        calculate_base_farmer_reward(blocks_reorg[-1].height),
        bt.constants.GENESIS_CHALLENGE,
    )
    tx_3 = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), farmer_coin)
    blocks_reorg = bt.get_consecutive_blocks(
        1, block_list_input=blocks_reorg, guarantee_transaction_block=True, transaction_data=tx_3
    )
    await _validate_and_add_block(b, blocks_reorg[-1])
    farmer_coin_record = await b.coin_store.get_coin_record(farmer_coin.name())
    assert farmer_coin_record is not None and farmer_coin_record.spent


async def _three_reward_blocks(b: Blockchain, bt: BlockTools) -> list[FullBlock]:
    blocks = bt.get_consecutive_blocks(
        3,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    return blocks


async def run_duplicate_outputs(b: Blockchain, bt: BlockTools) -> None:
    blocks = await _three_reward_blocks(b, bt)
    wt = bt.get_pool_wallet_tool()
    condition_dict: dict[ConditionOpcode, list[ConditionWithArgs]] = {ConditionOpcode.CREATE_COIN: []}
    for _ in range(2):
        output = ConditionWithArgs(ConditionOpcode.CREATE_COIN, [bt.pool_ph, int_to_bytes(1)])
        condition_dict[ConditionOpcode.CREATE_COIN].append(output)
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin, condition_dic=condition_dict)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    await _validate_and_add_block(b, blocks[-1], expected_error=Err.DUPLICATE_OUTPUT)


async def run_duplicate_removals(b: Blockchain, bt: BlockTools) -> None:
    blocks = await _three_reward_blocks(b, bt)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    tx_2 = wt.generate_signed_transaction(uint64(11), wt.get_new_puzzlehash(), coin)
    agg = SpendBundle.aggregate([tx, tx_2])
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=agg
    )
    await _validate_and_add_block(b, blocks[-1], expected_error=Err.DOUBLE_SPEND)


async def run_double_spent_in_coin_store(b: Blockchain, bt: BlockTools) -> None:
    blocks = await _three_reward_blocks(b, bt)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    await _validate_and_add_block(b, blocks[-1])
    tx_2 = wt.generate_signed_transaction(
        uint64(10), wt.get_new_puzzlehash(), blocks[-2].get_included_reward_coins()[0]
    )
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx_2
    )
    await _validate_and_add_block(b, blocks[-1], expected_error=Err.DOUBLE_SPEND)


async def run_minting_coin(b: Blockchain, bt: BlockTools) -> None:
    blocks = await _three_reward_blocks(b, bt)
    wt = bt.get_pool_wallet_tool()
    spend = find_reward_coin(blocks[-1], bt.pool_ph)
    output = ConditionWithArgs(ConditionOpcode.CREATE_COIN, [bt.pool_ph, int_to_bytes(spend.amount)])
    tx = wt.generate_signed_transaction(
        uint64(10),
        wt.get_new_puzzlehash(),
        spend,
        condition_dic={ConditionOpcode.CREATE_COIN: [output]},
    )
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    await _validate_and_add_block(b, blocks[-1], expected_error=Err.MINTING_COIN)


async def run_invalid_fees_in_block(b: Blockchain, bt: BlockTools) -> None:
    blocks = await _three_reward_blocks(b, bt)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    block = blocks[-1]
    block_2 = recursive_replace(block, "transactions_info.fees", uint64(1239))
    assert block_2.transactions_info is not None
    block_2 = recursive_replace(
        block_2, "foliage_transaction_block.transactions_info_hash", block_2.transactions_info.get_hash()
    )
    assert block_2.foliage_transaction_block is not None
    block_2 = recursive_replace(
        block_2, "foliage.foliage_transaction_block_hash", block_2.foliage_transaction_block.get_hash()
    )
    new_m = block_2.foliage.foliage_transaction_block_hash
    assert new_m is not None
    new_fsb_sig = bt.get_plot_signature(new_m, block.reward_chain_block.proof_of_space.plot_public_key)
    block_2 = recursive_replace(block_2, "foliage.foliage_transaction_block_signature", new_fsb_sig)
    await _validate_and_add_block(b, block_2, expected_error=Err.INVALID_BLOCK_FEE_AMOUNT)


async def run_invalid_agg_sig(b: Blockchain, bt: BlockTools) -> None:
    blocks = await _three_reward_blocks(b, bt)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    last_block = recursive_replace(blocks[-1], "transactions_info.aggregated_signature", G2Element.generator())
    assert last_block.transactions_info is not None
    last_block = recursive_replace(
        last_block, "foliage_transaction_block.transactions_info_hash", last_block.transactions_info.get_hash()
    )
    assert last_block.foliage_transaction_block is not None
    last_block = recursive_replace(
        last_block, "foliage.foliage_transaction_block_hash", last_block.foliage_transaction_block.get_hash()
    )
    new_m = last_block.foliage.foliage_transaction_block_hash
    assert new_m is not None
    new_fsb_sig = bt.get_plot_signature(new_m, last_block.reward_chain_block.proof_of_space.plot_public_key)
    last_block = recursive_replace(last_block, "foliage.foliage_transaction_block_signature", new_fsb_sig)
    await _validate_and_add_block(b, last_block, expected_error=Err.BAD_AGGREGATE_SIGNATURE)
    future = await pre_validate_block(
        b.constants,
        AugmentedBlockchain(b),
        last_block,
        b.pool,
        None,
        ValidationState(b.constants.SUB_SLOT_ITERS_STARTING, b.constants.DIFFICULTY_STARTING, None),
    )
    preval_result = await future
    assert preval_result.error == Err.BAD_AGGREGATE_SIGNATURE.value


async def run_header_blocks_tx_filter(b: Blockchain, bt: BlockTools) -> None:
    """A transaction block keeps its header hash when the transactions filter is omitted."""
    blocks = bt.get_consecutive_blocks(
        3,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[2], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1,
        block_list_input=blocks,
        guarantee_transaction_block=True,
        transaction_data=tx,
    )
    await _validate_and_add_block(b, blocks[-1])
    blocks_with_filter = await b.get_header_blocks_in_range(0, 10, tx_filter=True)
    blocks_without_filter = await b.get_header_blocks_in_range(0, 10, tx_filter=False)
    header_hash = blocks[-1].header_hash
    assert blocks_with_filter[header_hash].transactions_filter != blocks_without_filter[header_hash].transactions_filter
    assert blocks_with_filter[header_hash].header_hash == blocks_without_filter[header_hash].header_hash


async def run_non_tx_header_filter(b: Blockchain, bt: BlockTools) -> None:
    """A non-transaction block has an empty transactions filter."""
    blocks = bt.get_consecutive_blocks(10)
    for block in blocks:
        await _validate_and_add_block(b, block)
    non_tx_block = next(block for block in blocks if not block.is_transaction_block())
    blocks_with_filter = await b.get_header_blocks_in_range(0, 42, tx_filter=True)
    assert blocks_with_filter[non_tx_block.header_hash].transactions_filter == b"\x00"


async def run_overlong_generator_encoding(b: Blockchain, bt: BlockTools, *, expect_invalid: bool) -> None:
    """A non-canonical generator encoding is stored before soft fork 2.7 and rejected after."""
    blocks = bt.get_consecutive_blocks(10)
    for block in blocks[:-1]:
        await _validate_and_add_block(b, block)
    while not blocks[-1].is_transaction_block():
        await _validate_and_add_block(b, blocks[-1])
        blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
    original_block = blocks[-1]
    generator = SerializedProgram.fromhex("c00101")
    assert not is_canonical_serialization(bytes(generator))
    if original_block.version == 0:
        block = recursive_replace(original_block, "transactions_generator", generator)
    else:
        block = recursive_replace(original_block, "transactions_generator_buffer", bytes(generator))
    block = recursive_replace(block, "transactions_info.generator_root", std_hash(bytes(generator)))
    block = recursive_replace(
        block, "foliage_transaction_block.transactions_info_hash", std_hash(bytes(block.transactions_info))
    )
    block = recursive_replace(
        block, "foliage.foliage_transaction_block_hash", std_hash(bytes(block.foliage_transaction_block))
    )
    expected_error = Err.INVALID_TRANSACTIONS_GENERATOR_ENCODING if expect_invalid else None
    await _validate_and_add_block(b, block, expected_error=expected_error, skip_prevalidation=True)
    stored = await b.block_store.get_full_block(block.header_hash)
    if expect_invalid:
        assert stored is None
    else:
        assert stored is not None
        assert stored.header_hash == block.header_hash


def _resign_transaction_block(bt: BlockTools, block: FullBlock, updated: FullBlock) -> FullBlock:
    assert updated.foliage_transaction_block is not None
    signed = recursive_replace(
        updated, "foliage.foliage_transaction_block_hash", updated.foliage_transaction_block.get_hash()
    )
    message = signed.foliage.foliage_transaction_block_hash
    assert message is not None
    signature = bt.get_plot_signature(message, block.reward_chain_block.proof_of_space.plot_public_key)
    return recursive_replace(signed, "foliage.foliage_transaction_block_signature", signature)


async def run_invalid_merkle_roots(b: Blockchain, bt: BlockTools) -> None:
    """A block with the wrong additions or removals root is rejected and not stored."""
    blocks = await _three_reward_blocks(b, bt)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    block = blocks[-1]
    bad_additions = _resign_transaction_block(
        bt, block, recursive_replace(block, "foliage_transaction_block.additions_root", MerkleSet([]).get_root())
    )
    await _validate_and_add_block(b, bad_additions, expected_error=Err.BAD_ADDITION_ROOT)
    assert await b.block_store.get_full_block(bad_additions.header_hash) is None

    bad_removals = _resign_transaction_block(
        bt,
        block,
        recursive_replace(block, "foliage_transaction_block.removals_root", MerkleSet([std_hash(b"1")]).get_root()),
    )
    await _validate_and_add_block(b, bad_removals, expected_error=Err.BAD_REMOVAL_ROOT)
    assert await b.block_store.get_full_block(bad_removals.header_hash) is None


async def run_invalid_filter(b: Blockchain, bt: BlockTools) -> None:
    """A block with the wrong transactions filter hash is rejected and not stored."""
    blocks = await _three_reward_blocks(b, bt)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    block = blocks[-1]
    bad_filter = _resign_transaction_block(
        bt, block, recursive_replace(block, "foliage_transaction_block.filter_hash", std_hash(b"3"))
    )
    await _validate_and_add_block(b, bad_filter, expected_error=Err.INVALID_TRANSACTIONS_FILTER_HASH)
    assert await b.block_store.get_full_block(bad_filter.header_hash) is None


async def run_invalid_reward_claims(b: Blockchain, bt: BlockTools) -> None:
    """Too few, too many, or duplicate reward coins are rejected and not stored."""
    blocks = bt.get_consecutive_blocks(2, guarantee_transaction_block=True)
    await _validate_and_add_block(b, blocks[0])
    block = blocks[-1]
    assert block.transactions_info is not None
    claims = block.transactions_info.reward_claims_incorporated
    name = std_hash(b"")
    variants = (
        claims[:-1],
        [*claims, Coin(name, name, claims[0].amount)],
        [*claims, claims[-1]],
    )
    for rewards in variants:
        updated = recursive_replace(block, "transactions_info.reward_claims_incorporated", rewards)
        assert updated.transactions_info is not None
        updated = recursive_replace(
            updated, "foliage_transaction_block.transactions_info_hash", updated.transactions_info.get_hash()
        )
        bad = _resign_transaction_block(bt, block, updated)
        await _validate_and_add_block(b, bad, expected_error=Err.INVALID_REWARD_COINS, skip_prevalidation=True)
        assert await b.block_store.get_full_block(bad.header_hash) is None


def _with_generator_root(bt: BlockTools, block: FullBlock, root: bytes) -> FullBlock:
    updated = recursive_replace(block, "transactions_info.generator_root", root)
    assert updated.transactions_info is not None
    updated = recursive_replace(
        updated, "foliage_transaction_block.transactions_info_hash", updated.transactions_info.get_hash()
    )
    return _resign_transaction_block(bt, block, updated)


async def run_invalid_transactions_generator_hash(b: Blockchain, bt: BlockTools) -> None:
    """A generator root that does not match the generator is rejected and not stored."""
    blocks = bt.get_consecutive_blocks(2, guarantee_transaction_block=True)
    await _validate_and_add_block(b, blocks[0])
    empty_generator = _with_generator_root(bt, blocks[-1], bytes([1] * 32))
    await _validate_and_add_block(
        b, empty_generator, expected_error=Err.INVALID_TRANSACTIONS_GENERATOR_HASH, skip_prevalidation=True
    )
    assert await b.block_store.get_full_block(empty_generator.header_hash) is None

    await _validate_and_add_block(b, blocks[1])
    blocks = bt.get_consecutive_blocks(
        2,
        block_list_input=blocks,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
    )
    await _validate_and_add_block(b, blocks[2])
    await _validate_and_add_block(b, blocks[3])
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    wrong_root = _with_generator_root(bt, blocks[-1], bytes(32))
    await _validate_and_add_block(b, wrong_root, expected_error=Err.INVALID_TRANSACTIONS_GENERATOR_HASH)
    assert await b.block_store.get_full_block(wrong_root.header_hash) is None


async def run_invalid_transactions_ref_list(b: Blockchain, bt: BlockTools, *, refs_allowed: bool) -> None:
    """A generator reference list that does not match its signed root is rejected."""
    blocks = bt.get_consecutive_blocks(
        3,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
    )
    await _validate_and_add_block(b, blocks[0])
    await _validate_and_add_block(b, blocks[1])
    block = blocks[-1]
    updated = recursive_replace(block, "transactions_info.generator_refs_root", bytes(32))
    assert updated.transactions_info is not None
    updated = recursive_replace(
        updated, "foliage_transaction_block.transactions_info_hash", updated.transactions_info.get_hash()
    )
    bad_root = _resign_transaction_block(bt, block, updated)
    await _validate_and_add_block(
        b, bad_root, expected_error=Err.INVALID_TRANSACTIONS_GENERATOR_REFS_ROOT, skip_prevalidation=True
    )
    assert await b.block_store.get_full_block(bad_root.header_hash) is None

    bad_list = recursive_replace(block, "transactions_generator_ref_list", [uint32(0)])
    expected_error = (
        Err.INVALID_TRANSACTIONS_GENERATOR_REFS_ROOT if refs_allowed else Err.TOO_MANY_GENERATOR_REFS
    )
    await _validate_and_add_block(b, bad_list, expected_error=expected_error)

    await _validate_and_add_block(b, blocks[-1])
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(5, block_list_input=blocks, guarantee_transaction_block=False)
    for added in blocks[-5:]:
        await _validate_and_add_block(b, added)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    await _validate_and_add_block(b, blocks[-1])
    assert block_has_transactions_generator(blocks[-1])
    if refs_allowed:
        blocks = bt.get_consecutive_blocks(
            1,
            block_list_input=blocks,
            guarantee_transaction_block=True,
            transaction_data=tx,
            block_refs=[blocks[-1].height],
        )
        assert len(blocks[-1].transactions_generator_ref_list) == 0


async def run_cost_exceeds_max(b: Blockchain, bt: BlockTools, *, softfork_height: uint32, extra_coins: bool) -> None:
    """A block whose transaction cost is above the maximum is rejected and not stored."""
    blocks = bt.get_consecutive_blocks(
        3,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    wt = bt.get_pool_wallet_tool()
    condition_dict: dict[ConditionOpcode, list[ConditionWithArgs]] = {ConditionOpcode.CREATE_COIN: []}
    num_coins = 7_000 + (7_000 // 3 if extra_coins else 0)
    for i in range(num_coins):
        condition_dict[ConditionOpcode.CREATE_COIN].append(
            ConditionWithArgs(ConditionOpcode.CREATE_COIN, [bt.pool_ph, int_to_bytes(i)])
        )
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin, condition_dic=condition_dict)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    generator = get_transactions_generator_program(blocks[-1])
    assert generator is not None
    assert blocks[-1].transactions_info is not None
    npc_result = get_name_puzzle_conditions(
        BlockGenerator(generator, []),
        b.constants.MAX_BLOCK_COST_CLVM * 1000,
        mempool_mode=False,
        height=softfork_height,
        constants=bt.constants,
    )
    assert npc_result.conds is not None
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    block = blocks[-1]
    fork_info = ForkInfo(block.height - 1, block.height - 1, block.prev_header_hash)
    err = (
        await b.add_block(
            block,
            PreValidationResult(None, None, uint64(1), npc_result.conds.replace(validated_signature=True), uint32(0)),
            sub_slot_iters=sub_slot_iters,
            fork_info=fork_info,
        )
    )[1]
    assert err == Err.BLOCK_COST_EXCEEDS_MAX
    assert await b.block_store.get_full_block(block.header_hash) is None
    future = await pre_validate_block(
        b.constants,
        AugmentedBlockchain(b),
        block,
        b.pool,
        None,
        ValidationState(sub_slot_iters, b.constants.DIFFICULTY_STARTING, None),
    )
    result = await future
    assert Err(result.error) == Err.BLOCK_COST_EXCEEDS_MAX


async def run_invalid_cost_in_block(b: Blockchain, bt: BlockTools, *, softfork_height: uint32) -> None:
    """A block that reports the wrong transaction cost is rejected and not stored."""
    blocks = await _three_reward_blocks(b, bt)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    block = blocks[-1]
    assert block.transactions_info is not None
    real_cost = block.transactions_info.cost
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    for claimed in (uint64(0), uint64(1), uint64(1_000_000)):
        updated = recursive_replace(block, "transactions_info.cost", claimed)
        assert updated.transactions_info is not None
        updated = recursive_replace(
            updated, "foliage_transaction_block.transactions_info_hash", updated.transactions_info.get_hash()
        )
        bad = _resign_transaction_block(bt, block, updated)
        generator = get_transactions_generator_program(bad)
        assert generator is not None
        npc_result = get_name_puzzle_conditions(
            BlockGenerator(generator, []),
            min(b.constants.MAX_BLOCK_COST_CLVM * 1000, real_cost),
            mempool_mode=False,
            height=softfork_height,
            constants=bt.constants,
        )
        assert npc_result.conds is not None
        fork_info = ForkInfo(bad.height - 1, bad.height - 1, bad.prev_header_hash)
        _, err, _ = await b.add_block(
            bad,
            PreValidationResult(None, None, uint64(1), npc_result.conds.replace(validated_signature=True), uint32(0)),
            sub_slot_iters=sub_slot_iters,
            fork_info=fork_info,
        )
        assert err == Err.INVALID_BLOCK_COST
        assert await b.block_store.get_full_block(bad.header_hash) is None


async def _reject_and_not_stored(
    b: Blockchain, block: FullBlock, error: Err, *, skip_prevalidation: bool = False
) -> None:
    await _validate_and_add_block(b, block, expected_error=error, skip_prevalidation=skip_prevalidation)
    assert await b.block_store.get_full_block(block.header_hash) is None


async def run_not_tx_block_but_has_data(b: Blockchain, bt: BlockTools) -> None:
    """A non-transaction block that carries transaction data is rejected."""
    blocks = bt.get_consecutive_blocks(1)
    while blocks[-1].foliage_transaction_block is not None:
        await _validate_and_add_block(b, blocks[-1])
        blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
    original = blocks[-1]
    if original.version == 0:
        block = recursive_replace(original, "transactions_generator", SerializedProgram.to(None))
    else:
        block = recursive_replace(original, "transactions_generator_buffer", b"\xff")
    await _reject_and_not_stored(b, block, Err.NOT_BLOCK_BUT_HAS_DATA, skip_prevalidation=True)
    name = std_hash(b"")
    block = recursive_replace(
        original,
        "transactions_info",
        TransactionsInfo(name, name, G2Element(), uint64(1), uint64(1), []),
    )
    await _reject_and_not_stored(b, block, Err.NOT_BLOCK_BUT_HAS_DATA, skip_prevalidation=True)
    block = recursive_replace(original, "transactions_generator_ref_list", [uint64(1)])
    await _reject_and_not_stored(b, block, Err.NOT_BLOCK_BUT_HAS_DATA, skip_prevalidation=True)


async def run_invalid_block_version(
    b: Blockchain, bt: BlockTools, wrong_version: uint8, *, transaction_block: bool
) -> None:
    """An unknown or wrong block version is rejected for full and unfinished blocks."""
    blocks = bt.get_consecutive_blocks(1)
    while transaction_block != (blocks[-1].foliage_transaction_block is not None):
        await _validate_and_add_block(b, blocks[-1])
        blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
    block = blocks[-1]
    assert block.is_transaction_block() == transaction_block
    if wrong_version == block.version:
        pytest.skip(f"version {wrong_version} is valid for this consensus mode")
    bad_block = recursive_replace(block, "version", wrong_version)
    await _reject_and_not_stored(b, bad_block, Err.INVALID_BLOCK_VERSION, skip_prevalidation=True)
    unfinished = make_unfinished_block(block, bt.constants)
    bad_unfinished = recursive_replace(unfinished, "version", wrong_version)
    _, err = await b.validate_unfinished_block_header(bad_unfinished)
    assert err == Err.INVALID_BLOCK_VERSION


async def run_ephemeral_timelock(
    b: Blockchain,
    bt: BlockTools,
    opcode: ConditionOpcode,
    lock_value: int,
    expected: AddBlockResult,
    *,
    with_garbage: bool,
) -> None:
    """A timelock on a coin created and spent in the same block."""
    blocks = bt.get_consecutive_blocks(
        3,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
        genesis_timestamp=uint64(10_000),
        time_per_block=10,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    wt = bt.get_pool_wallet_tool()
    extra = [b"garbage"] if with_garbage else []
    conditions = {opcode: [ConditionWithArgs(opcode, [int_to_bytes(lock_value), *extra])]}
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx1 = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    coin1 = tx1.additions()[0]
    tx2 = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin1, condition_dic=conditions)
    assert coin1 in tx2.removals()
    coin2 = tx2.additions()[0]
    bundle = SpendBundle.aggregate([tx1, tx2])
    blocks = bt.get_consecutive_blocks(
        1,
        block_list_input=blocks,
        guarantee_transaction_block=True,
        transaction_data=bundle,
        time_per_block=10,
    )
    block = blocks[-1]
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    prevalidation = await (
        await pre_validate_block(
            b.constants,
            AugmentedBlockchain(b),
            block,
            b.pool,
            None,
            ValidationState(sub_slot_iters, b.constants.DIFFICULTY_STARTING, None),
        )
    )
    fork_info = ForkInfo(block.height - 1, block.height - 1, block.prev_header_hash)
    result, _, _ = await b.add_block(block, prevalidation, sub_slot_iters=sub_slot_iters, fork_info=fork_info)
    assert result == expected
    if expected == AddBlockResult.NEW_PEAK:
        spent = await b.coin_store.get_coin_record(coin1.name())
        created = await b.coin_store.get_coin_record(coin2.name())
        assert spent is not None and spent.spent
        assert created is not None and not created.spent


async def run_timelock_conditions(
    b: Blockchain,
    bt: BlockTools,
    opcode: ConditionOpcode,
    lock_value: int,
    expected: AddBlockResult,
) -> None:
    """A timelock on a reward coin created in an earlier block."""
    blocks = bt.get_consecutive_blocks(
        3,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
        genesis_timestamp=uint64(10_000),
        time_per_block=10,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    wt = bt.get_pool_wallet_tool()
    conditions = {opcode: [ConditionWithArgs(opcode, [int_to_bytes(lock_value)])]}
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin, condition_dic=conditions)
    blocks = bt.get_consecutive_blocks(
        1,
        block_list_input=blocks,
        guarantee_transaction_block=True,
        transaction_data=tx,
        time_per_block=10,
    )
    block = blocks[-1]
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    prevalidation = await (
        await pre_validate_block(
            b.constants,
            AugmentedBlockchain(b),
            block,
            b.pool,
            None,
            ValidationState(sub_slot_iters, b.constants.DIFFICULTY_STARTING, None),
        )
    )
    fork_info = ForkInfo(block.height - 1, block.height - 1, block.prev_header_hash)
    result, _, _ = await b.add_block(block, prevalidation, sub_slot_iters=sub_slot_iters, fork_info=fork_info)
    assert result == expected
    record = await b.coin_store.get_coin_record(coin.name())
    assert record is not None
    if expected == AddBlockResult.NEW_PEAK:
        assert record.spent
    else:
        assert not record.spent


async def run_aggsig_garbage(
    b: Blockchain, bt: BlockTools, opcode: ConditionOpcode, *, with_garbage: bool
) -> None:
    """An aggregate-signature condition is accepted with or without extra arguments."""
    blocks = bt.get_consecutive_blocks(
        3,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
        genesis_timestamp=uint64(10_000),
        time_per_block=10,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx1 = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    coin1 = tx1.additions()[0]
    secret_key = wt.get_private_key_for_puzzle_hash(coin1.puzzle_hash)
    synthetic_secret_key = calculate_synthetic_secret_key(secret_key, DEFAULT_HIDDEN_PUZZLE_HASH)
    public_key = synthetic_secret_key.get_g1()
    args = [bytes(public_key), b"msg"] + ([b"garbage"] if with_garbage else [])
    tx2 = wt.generate_signed_transaction(
        uint64(10),
        wt.get_new_puzzlehash(),
        coin1,
        condition_dic={opcode: [ConditionWithArgs(opcode, args)]},
    )
    assert coin1 in tx2.removals()
    bundle = SpendBundle.aggregate([tx1, tx2])
    blocks = bt.get_consecutive_blocks(
        1,
        block_list_input=blocks,
        guarantee_transaction_block=True,
        transaction_data=bundle,
        time_per_block=10,
    )
    block = blocks[-1]
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    prevalidation = await (
        await pre_validate_block(
            b.constants,
            AugmentedBlockchain(b),
            block,
            b.pool,
            None,
            ValidationState(sub_slot_iters, b.constants.DIFFICULTY_STARTING, None),
        )
    )
    # Body validation is what this checks. Clear a prevalidation error so add_block reaches it.
    prevalidation = replace(prevalidation, error=None, required_iters=uint64(1))
    fork_info = ForkInfo(block.height - 1, block.height - 1, block.prev_header_hash)
    result, error, state_change = await b.add_block(
        block, prevalidation, sub_slot_iters=sub_slot_iters, fork_info=fork_info
    )
    assert result == AddBlockResult.NEW_PEAK
    assert error is None
    assert state_change is not None and state_change.fork_height == uint32(2)
    spent = await b.coin_store.get_coin_record(coin.name())
    created = await b.coin_store.get_coin_record(coin1.name())
    assert spent is not None and spent.spent
    assert created is not None and created.spent


async def run_coin_assertions(
    b: Blockchain, bt: BlockTools, opcode: ConditionOpcode, *, with_garbage: bool
) -> None:
    """A coin asserting its amount, puzzle hash, id, or parent is accepted."""
    blocks = bt.get_consecutive_blocks(
        3,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
        genesis_timestamp=uint64(10_000),
        time_per_block=10,
    )
    for block in blocks:
        await _validate_and_add_block(b, block)
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx1 = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    coin1 = tx1.additions()[0]
    if opcode == ConditionOpcode.ASSERT_MY_AMOUNT:
        args: list[bytes] = [int_to_bytes(coin1.amount)]
    elif opcode == ConditionOpcode.ASSERT_MY_PUZZLEHASH:
        args = [coin1.puzzle_hash]
    elif opcode == ConditionOpcode.ASSERT_MY_COIN_ID:
        args = [coin1.name()]
    elif opcode == ConditionOpcode.ASSERT_MY_PARENT_ID:
        args = [coin1.parent_coin_info]
    else:
        raise ValueError(opcode)
    if with_garbage:
        args.append(b"garbage")
    tx2 = wt.generate_signed_transaction(
        uint64(10),
        wt.get_new_puzzlehash(),
        coin1,
        condition_dic={opcode: [ConditionWithArgs(opcode, args)]},
    )
    assert coin1 in tx2.removals()
    bundle = SpendBundle.aggregate([tx1, tx2])
    blocks = bt.get_consecutive_blocks(
        1,
        block_list_input=blocks,
        guarantee_transaction_block=True,
        transaction_data=bundle,
        time_per_block=10,
    )
    block = blocks[-1]
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    prevalidation = await (
        await pre_validate_block(
            b.constants,
            AugmentedBlockchain(b),
            block,
            b.pool,
            None,
            ValidationState(sub_slot_iters, b.constants.DIFFICULTY_STARTING, None),
        )
    )
    prevalidation = replace(prevalidation, error=None, required_iters=uint64(1))
    fork_info = ForkInfo(block.height - 1, block.height - 1, block.prev_header_hash)
    result, error, state_change = await b.add_block(
        block, prevalidation, sub_slot_iters=sub_slot_iters, fork_info=fork_info
    )
    assert result == AddBlockResult.NEW_PEAK
    assert error is None
    assert state_change is not None and state_change.fork_height == uint32(2)
    spent = await b.coin_store.get_coin_record(coin.name())
    created = await b.coin_store.get_coin_record(coin1.name())
    assert spent is not None and spent.spent
    assert created is not None and created.spent


def _reward_blocks(blocks: list[FullBlock], puzzle_hash: bytes32) -> list[FullBlock]:
    found = []
    for block in blocks:
        if any(coin.puzzle_hash == puzzle_hash for coin in block.get_included_reward_coins()):
            found.append(block)
    return found


async def run_announcements(b: Blockchain, bt: BlockTools, *, puzzle: bool) -> None:
    """An announcement assert fails alone and succeeds when the same block creates it."""
    blocks = bt.get_consecutive_blocks(
        1,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
        genesis_timestamp=uint64(10_000),
        time_per_block=10,
    )
    while len(_reward_blocks(blocks, bt.pool_ph)) < 2:
        blocks = bt.get_consecutive_blocks(
            1,
            block_list_input=blocks,
            guarantee_transaction_block=True,
            farmer_reward_puzzle_hash=bt.pool_ph,
            time_per_block=10,
        )
    for block in blocks:
        await _validate_and_add_block(b, block)
    rewarding = _reward_blocks(blocks, bt.pool_ph)
    coin_assert = find_reward_coin(rewarding[0], bt.pool_ph)
    coin_create = find_reward_coin(rewarding[1], bt.pool_ph)
    message = b"test"
    wt = bt.get_pool_wallet_tool()
    if puzzle:
        assert_opcode = ConditionOpcode.ASSERT_PUZZLE_ANNOUNCEMENT
        create_opcode = ConditionOpcode.CREATE_PUZZLE_ANNOUNCEMENT
        asserted = AssertPuzzleAnnouncement(asserted_ph=coin_create.puzzle_hash, asserted_msg=message).msg_calc
    else:
        assert_opcode = ConditionOpcode.ASSERT_COIN_ANNOUNCEMENT
        create_opcode = ConditionOpcode.CREATE_COIN_ANNOUNCEMENT
        asserted = AssertCoinAnnouncement(asserted_id=coin_create.name(), asserted_msg=message).msg_calc
    assert_tx = wt.generate_signed_transaction(
        uint64(1000),
        wt.get_new_puzzlehash(),
        coin_assert,
        condition_dic={assert_opcode: [ConditionWithArgs(assert_opcode, [asserted])]},
    )
    missing = bt.get_consecutive_blocks(
        1,
        block_list_input=blocks,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
        transaction_data=assert_tx,
        time_per_block=10,
    )
    await _reject_and_not_stored(b, missing[-1], Err.ASSERT_ANNOUNCE_CONSUMED_FAILED)
    create_tx = wt.generate_signed_transaction(
        uint64(1000),
        wt.get_new_puzzlehash(),
        coin_create,
        condition_dic={create_opcode: [ConditionWithArgs(create_opcode, [message])]},
    )
    bundle = SpendBundle.aggregate([assert_tx, create_tx])
    accepted = bt.get_consecutive_blocks(
        1,
        block_list_input=blocks,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
        transaction_data=bundle,
        time_per_block=10,
    )
    await _validate_and_add_block(b, accepted[-1])
    for coin in (coin_assert, coin_create):
        record = await b.coin_store.get_coin_record(coin.name())
        assert record is not None and record.spent


async def run_pre_validation_fails_bad_blocks(b: Blockchain, bt: BlockTools) -> None:
    """A valid block pre-checks cleanly, and a block with the wrong iteration count does not."""
    blocks = bt.get_consecutive_blocks(2)
    await _validate_and_add_block(b, blocks[0])
    bad = recursive_replace(
        blocks[-1], "reward_chain_block.total_iters", blocks[-1].reward_chain_block.total_iters + 1
    )
    state = ValidationState(b.constants.SUB_SLOT_ITERS_STARTING, b.constants.DIFFICULTY_STARTING, None)
    chain = AugmentedBlockchain(b)
    checked = []
    for block in (blocks[0], bad):
        checked.append(
            await (
                await pre_validate_block(
                    b.constants,
                    chain,
                    block,
                    b.pool,
                    None,
                    state,
                )
            )
        )
    assert checked[0].error is None
    assert checked[1].error is not None


async def run_pre_validation_batch(b: Blockchain, blocks: list[FullBlock]) -> None:
    """One hundred pre-checked blocks are added as the new peak."""
    batch = blocks[:100]
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    state = ValidationState(sub_slot_iters, b.constants.DIFFICULTY_STARTING, None)
    chain = AugmentedBlockchain(b)
    checked = [
        await (
            await pre_validate_block(
                b.constants,
                chain,
                block,
                b.pool,
                None,
                state,
            )
        )
        for block in batch
    ]
    fork_info = ForkInfo(-1, -1, b.constants.GENESIS_CHALLENGE)
    for block, result in zip(batch, checked):
        assert result.error is None
        added, err, _ = await b.add_block(block, result, sub_slot_iters, fork_info=fork_info)
        assert err is None
        assert added == AddBlockResult.NEW_PEAK
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == 99
    assert await b.coin_store.num_unspent() > 0


async def run_long_chain(b: Blockchain, blocks: list[FullBlock]) -> None:
    """Add 1000 blocks, and reject bad sub-slot difficulty and sub-epoch summaries along the way."""
    fork_info = ForkInfo(blocks[0].height - 1, blocks[0].height - 1, blocks[0].prev_header_hash)
    for block in blocks:
        if (
            len(block.finished_sub_slots) == 0
            or block.finished_sub_slots[0].challenge_chain.subepoch_summary_hash is None
        ):
            await _validate_and_add_block(b, block, fork_info=fork_info)
            if block.height % 100 == 0:
                print(f"long chain: {block.height}", flush=True)
            continue
        new_finished_ss = recursive_replace(
            block.finished_sub_slots[0],
            "challenge_chain.new_sub_slot_iters",
            uint64(10_000_000),
        )
        block_bad = recursive_replace(block, "finished_sub_slots", [new_finished_ss, *block.finished_sub_slots[1:]])
        header_block_bad = get_block_header(block_bad)
        expected_difficulty = block.finished_sub_slots[0].challenge_chain.new_difficulty or uint64(0)
        expected_sub_slot_iters = block.finished_sub_slots[0].challenge_chain.new_sub_slot_iters or uint64(0)
        expected_vs = ValidationState(expected_sub_slot_iters, expected_difficulty, None)
        _, error = validate_finished_header_block(b.constants, b, header_block_bad, False, expected_vs)
        assert error is not None
        assert error.code == Err.INVALID_NEW_SUB_SLOT_ITERS
        await _validate_and_add_block(b, block_bad, expected_result=AddBlockResult.INVALID_BLOCK, fork_info=fork_info)

        new_finished_ss = recursive_replace(
            block.finished_sub_slots[0], "challenge_chain.new_difficulty", uint64(10_000_000)
        )
        block_bad = recursive_replace(block, "finished_sub_slots", [new_finished_ss, *block.finished_sub_slots[1:]])
        header_block_bad = get_block_header(block_bad)
        _, error = validate_finished_header_block(b.constants, b, header_block_bad, False, expected_vs)
        assert error is not None
        assert error.code == Err.INVALID_NEW_DIFFICULTY
        await _validate_and_add_block(b, block_bad, expected_result=AddBlockResult.INVALID_BLOCK, fork_info=fork_info)

        new_finished_ss = recursive_replace(
            block.finished_sub_slots[0], "challenge_chain.subepoch_summary_hash", bytes(32)
        )
        new_finished_ss = recursive_replace(
            new_finished_ss, "reward_chain.challenge_chain_sub_slot_hash", new_finished_ss.challenge_chain.get_hash()
        )
        block_bad = recursive_replace(block, "finished_sub_slots", [new_finished_ss])
        header_block_bad = get_block_header(block_bad)
        _, error = validate_finished_header_block(b.constants, b, header_block_bad, False, expected_vs)
        assert error is not None
        assert error.code == Err.INVALID_SUB_EPOCH_SUMMARY
        await _validate_and_add_block(b, block_bad, expected_result=AddBlockResult.INVALID_BLOCK, fork_info=fork_info)

        new_finished_ss = recursive_replace(
            block.finished_sub_slots[0], "challenge_chain.subepoch_summary_hash", std_hash(b"123")
        )
        new_finished_ss = recursive_replace(
            new_finished_ss, "reward_chain.challenge_chain_sub_slot_hash", new_finished_ss.challenge_chain.get_hash()
        )
        block_bad = recursive_replace(block, "finished_sub_slots", [new_finished_ss])
        header_block_bad = get_block_header(block_bad)
        _, error = validate_finished_header_block(b.constants, b, header_block_bad, False, expected_vs)
        assert error is not None
        assert error.code == Err.INVALID_SUB_EPOCH_SUMMARY
        await _validate_and_add_block(b, block_bad, expected_result=AddBlockResult.INVALID_BLOCK, fork_info=fork_info)

        await _validate_and_add_block(b, block, fork_info=fork_info)
        if block.height % 100 == 0:
            print(f"long chain: {block.height}", flush=True)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == len(blocks) - 1


async def run_long_compact_chain(b: Blockchain, blocks: list[FullBlock]) -> None:
    for block in blocks:
        await _validate_and_add_block(b, block, skip_prevalidation=True)
        if block.height % 100 == 0:
            print(f"compact chain: {block.height}", flush=True)
    peak = b.get_peak()
    assert peak is not None
    assert peak.height == len(blocks) - 1
    assert await b.coin_store.num_unspent() > 0


async def run_long_reorg(
    b: Blockchain,
    default_10000_blocks: list[FullBlock],
    reorg_blocks: list[FullBlock],
    *,
    light_blocks: bool,
    consensus_mode: ConsensusMode,
) -> None:
    """A heavier chain takes over, then the original chain takes over again."""
    num_blocks_chain_1 = 1600
    num_blocks_chain_2_start = 500
    blocks = default_10000_blocks[:num_blocks_chain_1]
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    chain = AugmentedBlockchain(b)
    state = ValidationState(sub_slot_iters, b.constants.DIFFICULTY_STARTING, None)
    print(f"pre-validating {len(blocks)} blocks", flush=True)
    futures = [
        await pre_validate_block(
            b.constants,
            chain,
            block,
            b.pool,
            None,
            state,
        )
        for block in blocks
    ]
    pre_validation_results: list[PreValidationResult] = list(await asyncio.gather(*futures))
    for i, block in enumerate(blocks):
        if block.height != 0 and len(block.finished_sub_slots) > 0:
            new_iters = block.finished_sub_slots[0].challenge_chain.new_sub_slot_iters
            if new_iters is not None:
                sub_slot_iters = new_iters
        assert pre_validation_results[i].error is None
        if block.height % 100 == 0:
            print(f"main chain: {block.height:4} weight: {block.weight}", flush=True)
        fork_info = ForkInfo(block.height - 1, block.height - 1, block.prev_header_hash)
        result, err, _ = await b.add_block(
            block, pre_validation_results[i], sub_slot_iters=sub_slot_iters, fork_info=fork_info
        )
        await check_block_store_invariant(b)
        assert err is None
        assert result == AddBlockResult.NEW_PEAK
    peak = b.get_peak()
    assert peak is not None
    chain_1_height = peak.height
    chain_1_weight = peak.weight
    assert chain_1_height == num_blocks_chain_1 - 1
    assert reorg_blocks[num_blocks_chain_2_start - 1] == default_10000_blocks[num_blocks_chain_2_start - 1]
    assert reorg_blocks[num_blocks_chain_2_start] != default_10000_blocks[num_blocks_chain_2_start]
    b.clean_block_records()
    first_peak = b.get_peak()
    fork_info2 = None
    aug_chain: AugmentedBlockchain | None = AugmentedBlockchain(b)
    for reorg_block in reorg_blocks:
        if reorg_block.height % 100 == 0:
            peak = b.get_peak()
            assert peak is not None
            print(
                f"reorg chain: {reorg_block.height:4} weight: {reorg_block.weight} peak: {str(peak.header_hash)[:6]}",
                flush=True,
            )
        if reorg_block.height < num_blocks_chain_2_start:
            await _validate_and_add_block(
                b, reorg_block, expected_result=AddBlockResult.ALREADY_HAVE_BLOCK, augmented_blockchain=aug_chain
            )
            continue
        if fork_info2 is None:
            fork_info2 = ForkInfo(reorg_block.height - 1, reorg_block.height - 1, reorg_block.prev_header_hash)
        if consensus_mode < ConsensusMode.HARD_FORK_3_0:
            expected_result = (
                AddBlockResult.ADDED_AS_ORPHAN if reorg_block.weight <= chain_1_weight else AddBlockResult.NEW_PEAK
            )
        else:
            peak = b.get_peak()
            assert peak is not None
            is_new_peak = reorg_block.weight > peak.weight or (
                reorg_block.weight == peak.weight and reorg_block.total_iters < peak.total_iters
            )
            expected_result = AddBlockResult.NEW_PEAK if is_new_peak else AddBlockResult.ADDED_AS_ORPHAN
        if expected_result == AddBlockResult.NEW_PEAK:
            aug_chain = None
        await _validate_and_add_block(
            b,
            reorg_block,
            expected_result=expected_result,
            fork_info=fork_info2,
            augmented_blockchain=aug_chain,
        )
    peak = b.get_peak()
    assert peak is not None
    assert first_peak != peak
    assert peak.weight > chain_1_weight
    if light_blocks:
        assert peak.height > chain_1_height
    else:
        assert peak.height < chain_1_height
    second_peak = peak
    chain_2_weight = peak.weight
    b.clean_block_records()
    if light_blocks:
        blocks = default_10000_blocks[num_blocks_chain_2_start - 100 : 1800]
    else:
        blocks = default_10000_blocks[num_blocks_chain_2_start - 100 : 2600]
    record = await b.get_block_record_from_db(blocks[0].prev_header_hash)
    for _ in range(200):
        assert record is not None
        b.add_block_record(record)
        record = await b.get_block_record_from_db(record.prev_hash)
    assert record is not None
    b.add_block_record(record)
    fork_block = default_10000_blocks[num_blocks_chain_2_start - 101]
    fork_info = ForkInfo(fork_block.height, fork_block.height, fork_block.header_hash)
    await b.warmup(fork_block.height)
    aug_chain = AugmentedBlockchain(b)
    for block in blocks:
        if block.height % 128 == 0:
            peak = b.get_peak()
            assert peak is not None
            print(
                f"original chain: {block.height:4} weight: {block.weight} peak: {str(peak.header_hash)[:6]}",
                flush=True,
            )
        if block.height <= chain_1_height:
            expect = AddBlockResult.ALREADY_HAVE_BLOCK
        elif consensus_mode < ConsensusMode.HARD_FORK_3_0:
            expect = AddBlockResult.ADDED_AS_ORPHAN if block.weight < chain_2_weight else AddBlockResult.NEW_PEAK
        else:
            peak = b.get_peak()
            assert peak is not None
            is_new_peak = block.weight > peak.weight or (
                block.weight == peak.weight and block.total_iters < peak.total_iters
            )
            expect = AddBlockResult.NEW_PEAK if is_new_peak else AddBlockResult.ADDED_AS_ORPHAN
        await _validate_and_add_block(
            b, block, fork_info=fork_info, expected_result=expect, augmented_blockchain=aug_chain
        )
    peak = b.get_peak()
    assert peak is not None
    assert peak.header_hash != second_peak.header_hash
    assert peak.weight > chain_2_weight


def _generator_bytes(block: FullBlock) -> bytes:
    generator = get_transactions_generator_bytes(block)
    assert generator is not None
    return generator


async def run_lookup_block_generators(
    b: Blockchain,
    blocks_1: list[FullBlock],
    blocks_2: list[FullBlock],
    *,
    clear_cache: bool,
) -> None:
    """Generator lookup follows each fork and does not cross from one fork to the other."""
    print("lookup: adding main chain", flush=True)
    fork_info = ForkInfo(-1, -1, b.constants.GENESIS_CHALLENGE)
    for block in blocks_2[:550]:
        await _validate_and_add_block(b, block, expected_result=AddBlockResult.NEW_PEAK, fork_info=fork_info)
        if block.height % 100 == 0:
            print(f"lookup main: {block.height}", flush=True)
    fork_info = ForkInfo(blocks_1[500].height - 1, blocks_1[500].height - 1, blocks_1[500].prev_header_hash)
    for block in blocks_1[500:550]:
        await _validate_and_add_block(b, block, expected_result=AddBlockResult.ADDED_AS_ORPHAN, fork_info=fork_info)
    peak_1 = blocks_1[550]
    peak_2 = blocks_2[550]
    for peak in (peak_1, peak_2):
        if clear_cache:
            b.clean_block_records()
        generators = await b.lookup_block_generators(peak.prev_header_hash, {uint32(2)})
        assert generators == {uint32(2): _generator_bytes(blocks_1[2])}
    for peak in (peak_1, peak_2):
        if clear_cache:
            b.clean_block_records()
        generators = await b.lookup_block_generators(peak.prev_header_hash, {uint32(2), uint32(10), uint32(26)})
        assert generators == {
            uint32(2): _generator_bytes(blocks_1[2]),
            uint32(10): _generator_bytes(blocks_1[10]),
            uint32(26): _generator_bytes(blocks_1[26]),
        }
    if clear_cache:
        b.clean_block_records()
    generators = await b.lookup_block_generators(peak_1.prev_header_hash, {uint32(503)})
    assert generators == {uint32(503): _generator_bytes(blocks_1[503])}
    if clear_cache:
        b.clean_block_records()
    generators = await b.lookup_block_generators(peak_2.prev_header_hash, {uint32(516)})
    assert generators == {uint32(516): _generator_bytes(blocks_2[516])}
    if clear_cache:
        b.clean_block_records()
    with pytest.raises(ValueError, match=re.escape(Err.GENERATOR_REF_HAS_NO_GENERATOR.name)):
        await b.lookup_block_generators(peak_1.prev_header_hash, {uint32(516)})
    if clear_cache:
        b.clean_block_records()
    with pytest.raises(ValueError, match=re.escape(Err.GENERATOR_REF_HAS_NO_GENERATOR.name)):
        await b.lookup_block_generators(peak_2.prev_header_hash, {uint32(503)})
    if clear_cache:
        b.clean_block_records()
    with pytest.raises(ValueError, match=re.escape(Err.GENERATOR_REF_HAS_NO_GENERATOR.name)):
        await b.lookup_block_generators(peak_1.prev_header_hash, {uint32(8)})
    if clear_cache:
        b.clean_block_records()
    with pytest.raises(ValueError, match=re.escape(Err.GENERATOR_REF_HAS_NO_GENERATOR.name)):
        await b.lookup_block_generators(peak_2.prev_header_hash, {uint32(8)})
    if clear_cache:
        b.clean_block_records()
    with pytest.raises(AssertionError):
        await b.lookup_block_generators(blocks_2[600].prev_header_hash, {uint32(3)})
    if clear_cache:
        b.clean_block_records()
    with pytest.raises(AssertionError):
        await b.lookup_block_generators(blocks_1[600].prev_header_hash, {uint32(3)})


async def run_reward_block_hash(b: Blockchain, bt: BlockTools) -> None:
    """A block whose foliage names the wrong reward block is rejected."""
    blocks = bt.get_consecutive_blocks(2)
    await _validate_and_add_block(b, blocks[0])
    bad = recursive_replace(blocks[-1], "foliage.reward_block_hash", std_hash(b""))
    await _reject_and_not_stored(b, bad, Err.INVALID_REWARD_BLOCK_HASH)


async def run_reward_block_presence(b: Blockchain, bt: BlockTools) -> None:
    """A block that lies about whether it is a transaction block is rejected."""
    blocks = bt.get_consecutive_blocks(1)
    bad = recursive_replace(blocks[0], "reward_chain_block.is_transaction_block", False)
    bad = recursive_replace(bad, "foliage.reward_block_hash", bad.reward_chain_block.get_hash())
    await _reject_and_not_stored(b, bad, Err.INVALID_FOLIAGE_BLOCK_PRESENCE)
    await _validate_and_add_block(b, blocks[0])
    while True:
        blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
        if not blocks[-1].is_transaction_block():
            bad = recursive_replace(blocks[-1], "reward_chain_block.is_transaction_block", True)
            bad = recursive_replace(bad, "foliage.reward_block_hash", bad.reward_chain_block.get_hash())
            await _reject_and_not_stored(b, bad, Err.INVALID_FOLIAGE_BLOCK_PRESENCE)
            return
        await _validate_and_add_block(b, blocks[-1])


async def run_tx_block_missing_data(b: Blockchain, bt: BlockTools) -> None:
    """A transaction block without its transaction body is rejected."""
    blocks = bt.get_consecutive_blocks(2, guarantee_transaction_block=True)
    await _validate_and_add_block(b, blocks[0])
    missing_body = recursive_replace(blocks[-1], "foliage_transaction_block", None)
    await _validate_and_add_block_multi_error(
        b, missing_body, [Err.IS_TRANSACTION_BLOCK_BUT_NO_DATA, Err.INVALID_FOLIAGE_BLOCK_PRESENCE]
    )
    assert await b.block_store.get_full_block(missing_body.header_hash) is None
    missing_info = recursive_replace(blocks[-1], "transactions_info", None)
    with pytest.raises(AssertionError):
        await _validate_and_add_block_multi_error(
            b, missing_info, [Err.IS_TRANSACTION_BLOCK_BUT_NO_DATA, Err.INVALID_FOLIAGE_BLOCK_PRESENCE]
        )


async def run_invalid_transactions_info_hash(b: Blockchain, bt: BlockTools) -> None:
    """A transaction block whose signed info hash does not match is rejected."""
    blocks = bt.get_consecutive_blocks(2, guarantee_transaction_block=True)
    await _validate_and_add_block(b, blocks[0])
    block = recursive_replace(blocks[-1], "foliage_transaction_block.transactions_info_hash", std_hash(b""))
    bad = _resign_transaction_block(bt, blocks[-1], block)
    await _reject_and_not_stored(b, bad, Err.INVALID_TRANSACTIONS_INFO_HASH)


async def run_invalid_transactions_block_hash(b: Blockchain, bt: BlockTools) -> None:
    """A block whose foliage points at the wrong transaction block is rejected."""
    blocks = bt.get_consecutive_blocks(2, guarantee_transaction_block=True)
    await _validate_and_add_block(b, blocks[0])
    block = recursive_replace(blocks[-1], "foliage.foliage_transaction_block_hash", std_hash(b""))
    message = block.foliage.foliage_transaction_block_hash
    assert message is not None
    signature = bt.get_plot_signature(message, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
    bad = recursive_replace(block, "foliage.foliage_transaction_block_signature", signature)
    await _reject_and_not_stored(b, bad, Err.INVALID_FOLIAGE_BLOCK_HASH)


async def run_prevalidation_fast_fail(
    b: Blockchain, bt: BlockTools, monkeypatch: pytest.MonkeyPatch, *, expect_version_1: bool
) -> None:
    """A swapped generator is rejected before CLVM runs, and the candidate is not kept."""
    blocks = bt.get_consecutive_blocks(2, guarantee_transaction_block=True)
    await _validate_and_add_block(b, blocks[0])
    await _validate_and_add_block(b, blocks[1])
    blocks = bt.get_consecutive_blocks(
        2,
        block_list_input=blocks,
        guarantee_transaction_block=True,
        farmer_reward_puzzle_hash=bt.pool_ph,
    )
    await _validate_and_add_block(b, blocks[2])
    await _validate_and_add_block(b, blocks[3])
    wt = bt.get_pool_wallet_tool()
    coin = find_reward_coin(blocks[-1], bt.pool_ph)
    tx = wt.generate_signed_transaction(uint64(10), wt.get_new_puzzlehash(), coin)
    blocks = bt.get_consecutive_blocks(
        1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
    )
    block = blocks[-1]
    assert block_has_transactions_generator(block)
    assert block.transactions_info is not None
    assert block.version == (1 if expect_version_1 else 0)

    def replace_generator(blk: FullBlock, generator: SerializedProgram) -> FullBlock:
        if blk.version == 0:
            return cast(FullBlock, recursive_replace(blk, "transactions_generator", generator))
        return cast(FullBlock, recursive_replace(blk, "transactions_generator_buffer", bytes(generator)))

    malicious_generator = SerializedProgram.fromhex("80")
    mutated = replace_generator(block, malicious_generator)
    mutated_generator_bytes = get_transactions_generator_bytes(mutated)
    assert mutated_generator_bytes is not None
    assert mutated.transactions_info is not None
    assert std_hash(mutated_generator_bytes) != mutated.transactions_info.generator_root

    def _trap_run_block(*args: object, **kwargs: object) -> None:
        raise AssertionError("CLVM generator execution must not run for a mutated generator")

    monkeypatch.setattr("chia.consensus.multiprocess_validation._run_block", _trap_run_block)
    await _validate_and_add_block(b, mutated, expected_error=Err.INVALID_TRANSACTIONS_GENERATOR_HASH)
    assert await b.block_store.get_full_block(mutated.header_hash) is None

    mutated = recursive_replace(block, "transactions_info", None)
    await _validate_and_add_block(b, mutated, expected_error=Err.IS_TRANSACTION_BLOCK_BUT_NO_DATA)

    mutated = replace_generator(block, malicious_generator)
    mutated = recursive_replace(mutated, "transactions_info.generator_root", std_hash(bytes(malicious_generator)))
    mutated = recursive_replace(mutated, "foliage_transaction_block", None)
    await _validate_and_add_block(b, mutated, expected_error=Err.INVALID_TRANSACTIONS_INFO_HASH)

    async def _trap_generator_lookup(*args: object, **kwargs: object) -> None:
        raise AssertionError("generator lookup must not run for too many references")

    monkeypatch.setattr(AugmentedBlockchain, "lookup_block_generators", _trap_generator_lookup)
    mutated = recursive_replace(
        block,
        "transactions_generator_ref_list",
        [uint32(0)] * (b.constants.MAX_GENERATOR_REF_LIST_SIZE + 1),
    )
    await _validate_and_add_block(b, mutated, expected_error=Err.TOO_MANY_GENERATOR_REFS)

    augmented_blockchain = AugmentedBlockchain(b)

    async def _fail_generator_resolution(*args: object, **kwargs: object) -> None:
        assert augmented_blockchain.try_block_record(block.header_hash) is None
        raise ValueError("generator reference is unavailable")

    monkeypatch.setattr("chia.consensus.multiprocess_validation.get_block_generator", _fail_generator_resolution)
    await _validate_and_add_block(
        b,
        block,
        expected_error=Err.FAILED_GETTING_GENERATOR_MULTIPROCESSING,
        augmented_blockchain=augmented_blockchain,
    )
    assert augmented_blockchain.try_block_record(block.header_hash) is None


async def run_get_tx_peak(b: Blockchain, blocks: list[FullBlock]) -> None:
    """The transaction-block peak advances only on transaction blocks."""
    test_blocks = blocks[:100]
    sub_slot_iters = b.constants.SUB_SLOT_ITERS_STARTING
    difficulty = b.constants.DIFFICULTY_STARTING
    chain = AugmentedBlockchain(b)
    state = ValidationState(sub_slot_iters, difficulty, None)
    prevalidation = [
        await (
            await pre_validate_block(
                b.constants,
                chain,
                block,
                b.pool,
                None,
                state,
            )
        )
        for block in test_blocks
    ]
    last_tx_block_record = None
    for block, prevalidation_res in zip(test_blocks, prevalidation):
        assert b.get_tx_peak() == last_tx_block_record
        fork_info = ForkInfo(block.height - 1, block.height - 1, block.prev_header_hash)
        _, err, _ = await b.add_block(block, prevalidation_res, sub_slot_iters=sub_slot_iters, fork_info=fork_info)
        assert err is None
        if block.is_transaction_block():
            assert prevalidation_res.required_iters is not None
            last_tx_block_record = block_to_block_record(
                b.constants,
                b,
                prevalidation_res.required_iters,
                block,
                sub_slot_iters,
            )
    assert b.get_tx_peak() == last_tx_block_record


async def run_get_blocks_at(b: Blockchain, blocks: list[FullBlock]) -> None:
    heights = [block.height for block in blocks[:200]]
    for block in blocks[:200]:
        await _validate_and_add_block(b, block)
    records = await b.get_block_records_at(heights)
    assert len(records) == 200
    assert records[-1].height == 199
