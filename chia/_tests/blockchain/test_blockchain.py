from __future__ import annotations

import copy
import logging
import platform
import random
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass

import pytest
from chia_rs import (
    AugSchemeMPL,
    BlockRecord,
    CoinRecord,
    ConsensusConstants,
    EndOfSubSlotBundle,
    FullBlock,
    G2Element,
    InfusedChallengeChainSubSlot,
    SpendBundleConditions,
    SpendConditions,
    UnfinishedBlock,
)
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint8, uint32, uint64

from chia._tests.blockchain.blockchain_test_utils import (
    _validate_and_add_block,
    _validate_and_add_block_multi_error,
)
from chia._tests.blockchain.reorg_cases import (
    AGG_SIG_OPCODES,
    EPHEMERAL_TIMELOCK_CASES,
    MY_COIN_ASSERTION_OPCODES,
    TIMELOCK_CASES,
    run_aggsig_garbage,
    run_announcements,
    run_basic_reorg,
    run_chain_failed_rollback,
    run_coin_assertions,
    run_cost_exceeds_max,
    run_double_spent_in_coin_store,
    run_double_spent_in_reorg,
    run_duplicate_outputs,
    run_duplicate_removals,
    run_ephemeral_timelock,
    run_get_blocks_at,
    run_get_tx_peak,
    run_get_tx_peak_reorg,
    run_header_blocks_tx_filter,
    run_invalid_agg_sig,
    run_invalid_block_version,
    run_invalid_cost_in_block,
    run_invalid_fees_in_block,
    run_invalid_filter,
    run_invalid_merkle_roots,
    run_invalid_reward_claims,
    run_invalid_transactions_block_hash,
    run_invalid_transactions_generator_hash,
    run_invalid_transactions_info_hash,
    run_invalid_transactions_ref_list,
    run_long_chain,
    run_long_compact_chain,
    run_long_reorg,
    run_lookup_block_generators,
    run_minting_coin,
    run_non_tx_header_filter,
    run_not_tx_block_but_has_data,
    run_overlong_generator_encoding,
    run_pre_validation_batch,
    run_pre_validation_fails_bad_blocks,
    run_prevalidation_fast_fail,
    run_reorg_flip_flop,
    run_reorg_from_genesis,
    run_reorg_new_ref,
    run_reorg_stale_fork_height,
    run_reorg_transaction,
    run_reward_block_hash,
    run_reward_block_presence,
    run_timelock_conditions,
    run_tx_block_missing_data,
)
from chia._tests.conftest import ConsensusMode
from chia._tests.util.blockchain import create_blockchain
from chia._tests.util.get_name_puzzle_conditions import get_name_puzzle_conditions
from chia.consensus.block_body_validation import ForkAdd, ForkInfo
from chia.consensus.block_generator_info import block_has_transactions_generator
from chia.consensus.block_header_validation import validate_finished_header_block
from chia.consensus.blockchain import AddBlockResult, Blockchain
from chia.consensus.generator_tools import get_block_header
from chia.consensus.get_block_generator import get_block_generator
from chia.consensus.pot_iterations import is_overflow_block
from chia.full_node.coin_store import CoinStore
from chia.full_node.db.coin_store import RocksCoinStore
from chia.simulator.block_tools import BlockTools, create_block_tools_async
from chia.simulator.keyring import TempKeyring
from chia.simulator.vdf_prover import get_vdf_info_and_proof
from chia.types.blockchain_format.classgroup import ClassgroupElement
from chia.types.blockchain_format.coin import Coin
from chia.types.blockchain_format.vdf import VDFInfo, VDFProof, validate_vdf
from chia.types.condition_opcodes import ConditionOpcode
from chia.types.validation_state import ValidationState
from chia.util.errors import Err
from chia.util.hash import std_hash
from chia.util.keychain import Keychain
from chia.util.recursive_replace import recursive_replace


def _is_macos_intel() -> bool:
    """True when running on macOS with an Intel CPU (x86_64). Used to skip slow test params."""
    return platform.system() == "Darwin" and platform.machine() in {"x86_64", "i386"}


log = logging.getLogger(__name__)
bad_element = ClassgroupElement.create(b"\x00")


async def get_coin_record(b: Blockchain, coin_id: bytes32) -> CoinRecord | None:
    # single-record lookup is not part of the consensus coin store protocol,
    # but tests know the concrete store
    coin_store = b.coin_store
    assert isinstance(coin_store, CoinStore | RocksCoinStore)
    return await coin_store.get_coin_record(coin_id)


@asynccontextmanager
async def make_empty_blockchain(constants: ConsensusConstants) -> AsyncIterator[Blockchain]:
    """
    Provides a list of 10 valid blocks, as well as a blockchain with 9 blocks added to it.
    """

    async with create_blockchain(constants, 2) as (bc, _):
        yield bc


class TestGenesisBlock:
    @pytest.mark.anyio
    async def test_block_tools_proofs_400(
        self, default_400_blocks: list[FullBlock], blockchain_constants: ConsensusConstants
    ) -> None:
        vdf, proof = get_vdf_info_and_proof(
            blockchain_constants,
            ClassgroupElement.get_default_element(),
            blockchain_constants.GENESIS_CHALLENGE,
            uint64(231),
        )
        if validate_vdf(proof, blockchain_constants, ClassgroupElement.get_default_element(), vdf) is False:
            raise Exception("invalid proof")

    @pytest.mark.anyio
    async def test_block_tools_proofs_1000(
        self, default_1000_blocks: list[FullBlock], blockchain_constants: ConsensusConstants
    ) -> None:
        vdf, proof = get_vdf_info_and_proof(
            blockchain_constants,
            ClassgroupElement.get_default_element(),
            blockchain_constants.GENESIS_CHALLENGE,
            uint64(231),
        )
        if validate_vdf(proof, blockchain_constants, ClassgroupElement.get_default_element(), vdf) is False:
            raise Exception("invalid proof")

    @pytest.mark.anyio
    async def test_block_tools_proofs(self, blockchain_constants: ConsensusConstants) -> None:
        vdf, proof = get_vdf_info_and_proof(
            blockchain_constants,
            ClassgroupElement.get_default_element(),
            blockchain_constants.GENESIS_CHALLENGE,
            uint64(231),
        )
        if validate_vdf(proof, blockchain_constants, ClassgroupElement.get_default_element(), vdf) is False:
            raise Exception("invalid proof")

    @pytest.mark.anyio
    async def test_non_overflow_genesis(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        assert empty_blockchain.get_peak() is None
        genesis = bt.get_consecutive_blocks(1, force_overflow=False)[0]
        await _validate_and_add_block(empty_blockchain, genesis)
        peak = empty_blockchain.get_peak()
        assert peak is not None
        assert peak.height == 0

    @pytest.mark.anyio
    async def test_overflow_genesis(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        genesis = bt.get_consecutive_blocks(1, force_overflow=True)[0]
        await _validate_and_add_block(empty_blockchain, genesis)

    @pytest.mark.anyio
    @pytest.mark.skipif(_is_macos_intel(), reason="Slow on macOS Intel")
    async def test_genesis_empty_slots(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        genesis = bt.get_consecutive_blocks(1, force_overflow=False, skip_slots=30)[0]
        await _validate_and_add_block(empty_blockchain, genesis)

    @pytest.mark.anyio
    async def test_overflow_genesis_empty_slots(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        genesis = bt.get_consecutive_blocks(1, force_overflow=True, skip_slots=3)[0]
        await _validate_and_add_block(empty_blockchain, genesis)

    @pytest.mark.anyio
    async def test_genesis_validate_1(
        self, empty_blockchain: Blockchain, bt: BlockTools, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Monkey patch pre_sp_tx_block so we dont throw there
        monkeypatch.setattr(
            "chia.consensus.multiprocess_validation.pre_sp_tx_block_height",
            lambda *args, **kwargs: uint32(0),
        )
        genesis = bt.get_consecutive_blocks(1, force_overflow=False)[0]
        bad_prev = bytes([1] * 32)
        genesis = recursive_replace(genesis, "foliage.prev_block_hash", bad_prev)
        await _validate_and_add_block(empty_blockchain, genesis, expected_error=Err.INVALID_PREV_BLOCK_HASH)


class TestBlockHeaderValidation:
    @pytest.mark.anyio
    async def test_long_chain(self, empty_blockchain: Blockchain, default_1000_blocks: list[FullBlock]) -> None:
        await run_long_chain(empty_blockchain, default_1000_blocks)

    @pytest.mark.anyio
    async def test_unfinished_blocks(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blockchain = empty_blockchain
        blocks = bt.get_consecutive_blocks(3)
        for block in blocks[:-1]:
            await _validate_and_add_block(empty_blockchain, block)
        block = blocks[-1]
        unf = UnfinishedBlock(
            block.finished_sub_slots,
            block.reward_chain_block.get_unfinished(),
            block.challenge_chain_sp_proof,
            block.reward_chain_sp_proof,
            block.foliage,
            block.foliage_transaction_block,
            block.transactions_info,
            block.transactions_generator,
            [],
            block.transactions_generator_buffer,
            block.version,
        )
        conds = None
        # if this assert fires, remove it along with the block below
        assert not block_has_transactions_generator(unf)
        if block_has_transactions_generator(unf):  # pragma: no cover
            block_generator = await get_block_generator(blockchain.lookup_block_generators, unf)
            assert block_generator is not None
            assert unf.transactions_info is not None
            npc_result = get_name_puzzle_conditions(
                block_generator,
                unf.transactions_info.cost,
                mempool_mode=False,
                height=block.height,
                constants=bt.constants,
            )
            conds = npc_result.conds

        validate_res = await blockchain.validate_unfinished_block(unf, conds, False)
        err = validate_res.error
        assert err is None

        await _validate_and_add_block(empty_blockchain, block)
        blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, force_overflow=True)
        block = blocks[-1]
        unf = UnfinishedBlock(
            block.finished_sub_slots,
            block.reward_chain_block.get_unfinished(),
            block.challenge_chain_sp_proof,
            block.reward_chain_sp_proof,
            block.foliage,
            block.foliage_transaction_block,
            block.transactions_info,
            block.transactions_generator,
            [],
            block.transactions_generator_buffer,
            block.version,
        )
        conds = None
        # if this assert fires, remove it along with the block below
        assert not block_has_transactions_generator(unf)
        if block_has_transactions_generator(unf):
            block_generator = await get_block_generator(blockchain.lookup_block_generators, unf)
            assert block_generator is not None
            assert unf.transactions_info is not None
            npc_result = get_name_puzzle_conditions(
                block_generator,
                unf.transactions_info.cost,
                mempool_mode=False,
                height=block.height,
                constants=bt.constants,
            )
            conds = npc_result.conds
        validate_res = await blockchain.validate_unfinished_block(unf, conds, False)
        assert validate_res.error is None

    @pytest.mark.anyio
    async def test_empty_genesis(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        for block in bt.get_consecutive_blocks(2, skip_slots=3):
            await _validate_and_add_block(empty_blockchain, block)

    @pytest.mark.anyio
    async def test_empty_slots_non_genesis(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blockchain = empty_blockchain
        blocks = bt.get_consecutive_blocks(10)
        for block in blocks:
            await _validate_and_add_block(empty_blockchain, block)

        blocks = bt.get_consecutive_blocks(10, skip_slots=2, block_list_input=blocks)
        for block in blocks[10:]:
            await _validate_and_add_block(empty_blockchain, block)
        peak = blockchain.get_peak()
        assert peak is not None
        assert peak.height == 19

    @pytest.mark.anyio
    async def test_one_sb_per_slot(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blockchain = empty_blockchain
        num_blocks = 20
        blocks: list[FullBlock] = []
        for _ in range(num_blocks):
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, skip_slots=1)
            await _validate_and_add_block(empty_blockchain, blocks[-1])
        peak = blockchain.get_peak()
        assert peak is not None
        assert peak.height == num_blocks - 1

    @pytest.mark.anyio
    async def test_all_overflow(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blockchain = empty_blockchain
        num_rounds = 5
        blocks: list[FullBlock] = []
        num_blocks = 0
        for i in range(1, num_rounds):
            num_blocks += i
            blocks = bt.get_consecutive_blocks(i, block_list_input=blocks, skip_slots=1, force_overflow=True)
            for block in blocks[-i:]:
                await _validate_and_add_block(empty_blockchain, block)
        peak = blockchain.get_peak()
        assert peak is not None
        assert peak.height == num_blocks - 1

    @pytest.mark.anyio
    async def test_unf_block_overflow(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blockchain = empty_blockchain

        blocks: list[FullBlock] = []
        while True:
            # This creates an overflow block, then a normal block, and then an overflow in the next sub-slot
            # blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, force_overflow=True)
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, force_overflow=True)

            await _validate_and_add_block(blockchain, blocks[-2])

            sb_1 = blockchain.block_record(blocks[-2].header_hash)

            sb_2_next_ss = blocks[-1].total_iters - blocks[-2].total_iters < sb_1.sub_slot_iters
            # We might not get a normal block for sb_2, and we might not get them in the right slots
            # So this while loop keeps trying
            if sb_1.overflow and sb_2_next_ss:
                block = blocks[-1]
                unf = UnfinishedBlock(
                    [],
                    block.reward_chain_block.get_unfinished(),
                    block.challenge_chain_sp_proof,
                    block.reward_chain_sp_proof,
                    block.foliage,
                    block.foliage_transaction_block,
                    block.transactions_info,
                    block.transactions_generator,
                    [],
                    block.transactions_generator_buffer,
                    block.version,
                )
                conds = None
                # if this assert fires, remove it along with the block below
                assert not block_has_transactions_generator(block)
                if block_has_transactions_generator(block):
                    block_generator = await get_block_generator(blockchain.lookup_block_generators, unf)
                    assert block_generator is not None
                    assert unf.transactions_info is not None
                    npc_result = get_name_puzzle_conditions(
                        block_generator,
                        unf.transactions_info.cost,
                        mempool_mode=False,
                        height=block.height,
                        constants=bt.constants,
                    )
                    conds = npc_result.conds
                validate_res = await blockchain.validate_unfinished_block(unf, conds, skip_overflow_ss_validation=True)
                assert validate_res.error is None
                return None

            await _validate_and_add_block(blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_one_sb_per_two_slots(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blockchain = empty_blockchain
        num_blocks = 20
        blocks: list[FullBlock] = []
        for _ in range(num_blocks):  # Same thing, but 2 sub-slots per block
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, skip_slots=2)
            await _validate_and_add_block(blockchain, blocks[-1])
        peak = blockchain.get_peak()
        assert peak is not None
        assert peak.height == num_blocks - 1

    @pytest.mark.anyio
    async def test_one_sb_per_five_slots(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blockchain = empty_blockchain
        num_blocks = 10
        blocks: list[FullBlock] = []
        for _ in range(num_blocks):  # Same thing, but 5 sub-slots per block
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, skip_slots=5)
            await _validate_and_add_block(blockchain, blocks[-1])
        peak = blockchain.get_peak()
        assert peak is not None
        assert peak.height == num_blocks - 1

    @pytest.mark.anyio
    async def test_basic_chain_overflow(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blocks = bt.get_consecutive_blocks(5, force_overflow=True)
        for block in blocks:
            await _validate_and_add_block(empty_blockchain, block)
        peak = empty_blockchain.get_peak()
        assert peak is not None
        assert peak.height == len(blocks) - 1

    @pytest.mark.anyio
    async def test_one_sb_per_two_slots_force_overflow(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blockchain = empty_blockchain
        num_blocks = 10
        blocks: list[FullBlock] = []
        for _ in range(num_blocks):
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, skip_slots=2, force_overflow=True)
            await _validate_and_add_block(blockchain, blocks[-1])
        peak = blockchain.get_peak()
        assert peak is not None
        assert peak.height == num_blocks - 1

    @pytest.mark.anyio
    async def test_invalid_prev(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 1
        blocks = bt.get_consecutive_blocks(2, force_overflow=False)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        block_1_bad = recursive_replace(blocks[-1], "foliage.prev_block_hash", bytes([0] * 32))

        await _validate_and_add_block(empty_blockchain, block_1_bad, expected_error=Err.INVALID_PREV_BLOCK_HASH)

    @pytest.mark.anyio
    async def test_invalid_pospace(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2
        blocks = bt.get_consecutive_blocks(2, force_overflow=False)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        block_1_bad = recursive_replace(blocks[-1], "reward_chain_block.proof_of_space.proof", bytes([0] * 32))

        await _validate_and_add_block(empty_blockchain, block_1_bad, expected_error=Err.INVALID_POSPACE)

    @pytest.mark.anyio
    async def test_invalid_sub_slot_challenge_hash_genesis(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2a
        blocks = bt.get_consecutive_blocks(1, force_overflow=False, skip_slots=1)
        new_finished_ss = recursive_replace(
            blocks[0].finished_sub_slots[0],
            "challenge_chain.challenge_chain_end_of_slot_vdf.challenge",
            bytes([2] * 32),
        )
        block_0_bad = recursive_replace(
            blocks[0], "finished_sub_slots", [new_finished_ss, *blocks[0].finished_sub_slots[1:]]
        )

        header_block_bad = get_block_header(block_0_bad)
        expected_vs = ValidationState(
            empty_blockchain.constants.SUB_SLOT_ITERS_STARTING, empty_blockchain.constants.DIFFICULTY_STARTING, None
        )
        _, error = validate_finished_header_block(
            empty_blockchain.constants, empty_blockchain, header_block_bad, False, expected_vs
        )

        assert error is not None
        assert error.code == Err.INVALID_PREV_CHALLENGE_SLOT_HASH
        await _validate_and_add_block(empty_blockchain, block_0_bad, expected_result=AddBlockResult.INVALID_BLOCK)

    @pytest.mark.anyio
    async def test_invalid_sub_slot_challenge_hash_non_genesis(
        self, empty_blockchain: Blockchain, bt: BlockTools
    ) -> None:
        # 2b
        blocks = bt.get_consecutive_blocks(1, force_overflow=False, skip_slots=0)
        blocks = bt.get_consecutive_blocks(1, force_overflow=False, skip_slots=1, block_list_input=blocks)
        new_finished_ss = recursive_replace(
            blocks[1].finished_sub_slots[0],
            "challenge_chain.challenge_chain_end_of_slot_vdf.challenge",
            bytes([2] * 32),
        )
        block_1_bad = recursive_replace(
            blocks[1], "finished_sub_slots", [new_finished_ss, *blocks[1].finished_sub_slots[1:]]
        )

        await _validate_and_add_block(empty_blockchain, blocks[0])
        header_block_bad = get_block_header(block_1_bad)
        # TODO: Inspect these block values as they are currently None
        expected_difficulty = blocks[1].finished_sub_slots[0].challenge_chain.new_difficulty or uint64(0)
        expected_sub_slot_iters = blocks[1].finished_sub_slots[0].challenge_chain.new_sub_slot_iters or uint64(0)
        expected_vs = ValidationState(expected_sub_slot_iters, expected_difficulty, None)
        _, error = validate_finished_header_block(
            empty_blockchain.constants, empty_blockchain, header_block_bad, False, expected_vs
        )
        assert error is not None
        assert error.code == Err.INVALID_PREV_CHALLENGE_SLOT_HASH
        await _validate_and_add_block(empty_blockchain, block_1_bad, expected_result=AddBlockResult.INVALID_BLOCK)

    @pytest.mark.anyio
    async def test_invalid_sub_slot_challenge_hash_empty_ss(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2c
        blocks = bt.get_consecutive_blocks(1, force_overflow=False, skip_slots=0)
        blocks = bt.get_consecutive_blocks(1, force_overflow=False, skip_slots=2, block_list_input=blocks)
        new_finished_ss = recursive_replace(
            blocks[1].finished_sub_slots[-1],
            "challenge_chain.challenge_chain_end_of_slot_vdf.challenge",
            bytes([2] * 32),
        )
        block_1_bad = recursive_replace(
            blocks[1], "finished_sub_slots", [*blocks[1].finished_sub_slots[:-1], new_finished_ss]
        )
        await _validate_and_add_block(empty_blockchain, blocks[0])

        header_block_bad = get_block_header(block_1_bad)
        # TODO: Inspect these block values as they are currently None
        expected_difficulty = blocks[1].finished_sub_slots[0].challenge_chain.new_difficulty or uint64(0)
        expected_sub_slot_iters = blocks[1].finished_sub_slots[0].challenge_chain.new_sub_slot_iters or uint64(0)
        expected_vs = ValidationState(expected_sub_slot_iters, expected_difficulty, None)
        _, error = validate_finished_header_block(
            empty_blockchain.constants, empty_blockchain, header_block_bad, False, expected_vs
        )
        assert error is not None
        assert error.code == Err.INVALID_PREV_CHALLENGE_SLOT_HASH
        await _validate_and_add_block(empty_blockchain, block_1_bad, expected_result=AddBlockResult.INVALID_BLOCK)

    @pytest.mark.anyio
    async def test_genesis_no_icc(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2d
        blocks = bt.get_consecutive_blocks(1, force_overflow=False, skip_slots=1)
        new_finished_ss = recursive_replace(
            blocks[0].finished_sub_slots[0],
            "infused_challenge_chain",
            InfusedChallengeChainSubSlot(
                VDFInfo(
                    bytes32.zeros,
                    uint64(1200),
                    ClassgroupElement.get_default_element(),
                )
            ),
        )
        block_0_bad = recursive_replace(
            blocks[0], "finished_sub_slots", [new_finished_ss, *blocks[0].finished_sub_slots[1:]]
        )
        await _validate_and_add_block(empty_blockchain, block_0_bad, expected_error=Err.SHOULD_NOT_HAVE_ICC)

    async def do_test_invalid_icc_sub_slot_vdf(
        self, keychain: Keychain, db_version: int, constants: ConsensusConstants
    ) -> None:
        async with (
            create_block_tools_async(
                constants=constants.replace(
                    SUB_SLOT_ITERS_STARTING=uint64(2**12),
                    DIFFICULTY_STARTING=uint64(constants.DIFFICULTY_STARTING * 2),
                ),
                keychain=keychain,
            ) as bt_high_iters,
            create_blockchain(bt_high_iters.constants, db_version) as (bc1, _),
        ):
            blocks = bt_high_iters.get_consecutive_blocks(10)
            for block in blocks:
                if (
                    len(block.finished_sub_slots) > 0
                    and block.finished_sub_slots[-1].infused_challenge_chain is not None
                ):
                    # Bad iters
                    new_finished_ss = recursive_replace(
                        block.finished_sub_slots[-1],
                        "infused_challenge_chain",
                        InfusedChallengeChainSubSlot(
                            block.finished_sub_slots[
                                -1
                            ].infused_challenge_chain.infused_challenge_chain_end_of_slot_vdf.replace(
                                number_of_iterations=uint64(10000000),
                            )
                        ),
                    )
                    block_bad = recursive_replace(
                        block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss]
                    )
                    await _validate_and_add_block(bc1, block_bad, expected_error=Err.INVALID_ICC_EOS_VDF)

                    # Bad output
                    new_finished_ss_2 = recursive_replace(
                        block.finished_sub_slots[-1],
                        "infused_challenge_chain",
                        InfusedChallengeChainSubSlot(
                            block.finished_sub_slots[
                                -1
                            ].infused_challenge_chain.infused_challenge_chain_end_of_slot_vdf.replace(
                                output=ClassgroupElement.get_default_element(),
                            )
                        ),
                    )
                    log.warning(f"Proof: {block.finished_sub_slots[-1].proofs}")
                    block_bad_2 = recursive_replace(
                        block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_2]
                    )
                    await _validate_and_add_block(bc1, block_bad_2, expected_error=Err.INVALID_ICC_EOS_VDF)

                    # Bad challenge hash
                    new_finished_ss_3 = recursive_replace(
                        block.finished_sub_slots[-1],
                        "infused_challenge_chain",
                        InfusedChallengeChainSubSlot(
                            block.finished_sub_slots[
                                -1
                            ].infused_challenge_chain.infused_challenge_chain_end_of_slot_vdf.replace(
                                challenge=bytes32.zeros
                            )
                        ),
                    )
                    block_bad_3 = recursive_replace(
                        block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_3]
                    )
                    await _validate_and_add_block(bc1, block_bad_3, expected_error=Err.INVALID_ICC_EOS_VDF)

                    # Bad proof
                    new_finished_ss_5 = recursive_replace(
                        block.finished_sub_slots[-1],
                        "proofs.infused_challenge_chain_slot_proof",
                        VDFProof(uint8(0), b"1239819023890", False),
                    )
                    block_bad_5 = recursive_replace(
                        block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_5]
                    )
                    await _validate_and_add_block(bc1, block_bad_5, expected_error=Err.INVALID_ICC_EOS_VDF)

                await _validate_and_add_block(bc1, block)

    @pytest.mark.anyio
    async def test_invalid_icc_sub_slot_vdf(self, db_version: int, blockchain_constants: ConsensusConstants) -> None:
        with TempKeyring() as keychain:
            await self.do_test_invalid_icc_sub_slot_vdf(keychain, db_version, blockchain_constants)

    @pytest.mark.anyio
    async def test_invalid_icc_into_cc(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        blockchain = empty_blockchain
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(blockchain, blocks[0])
        case_1, case_2 = False, False
        while not case_1 or not case_2:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, skip_slots=1)
            block = blocks[-1]
            if len(block.finished_sub_slots) > 0 and block.finished_sub_slots[-1].infused_challenge_chain is not None:
                if block.finished_sub_slots[-1].reward_chain.deficit == bt.constants.MIN_BLOCKS_PER_CHALLENGE_BLOCK:
                    # 2g
                    case_1 = True
                    new_finished_ss = recursive_replace(
                        block.finished_sub_slots[-1],
                        "challenge_chain",
                        block.finished_sub_slots[-1].challenge_chain.replace(
                            infused_challenge_chain_sub_slot_hash=bytes32([1] * 32)
                        ),
                    )
                else:
                    # 2h
                    case_2 = True
                    new_finished_ss = recursive_replace(
                        block.finished_sub_slots[-1],
                        "challenge_chain",
                        block.finished_sub_slots[-1].challenge_chain.replace(
                            infused_challenge_chain_sub_slot_hash=block.finished_sub_slots[
                                -1
                            ].infused_challenge_chain.get_hash(),
                        ),
                    )
                block_bad = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss]
                )

                header_block_bad = get_block_header(block_bad)
                # TODO: Inspect these block values as they are currently None
                expected_difficulty = block.finished_sub_slots[0].challenge_chain.new_difficulty or uint64(0)
                expected_sub_slot_iters = block.finished_sub_slots[0].challenge_chain.new_sub_slot_iters or uint64(0)
                expected_vs = ValidationState(expected_sub_slot_iters, expected_difficulty, None)
                _, error = validate_finished_header_block(
                    empty_blockchain.constants, empty_blockchain, header_block_bad, False, expected_vs
                )
                assert error is not None
                assert error.code == Err.INVALID_ICC_HASH_CC
                await _validate_and_add_block(blockchain, block_bad, expected_result=AddBlockResult.INVALID_BLOCK)

                # 2i
                new_finished_ss_bad_rc = recursive_replace(
                    block.finished_sub_slots[-1],
                    "reward_chain",
                    block.finished_sub_slots[-1].reward_chain.replace(infused_challenge_chain_sub_slot_hash=None),
                )
                block_bad = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_bad_rc]
                )
                await _validate_and_add_block(blockchain, block_bad, expected_error=Err.INVALID_ICC_HASH_RC)
            elif len(block.finished_sub_slots) > 0 and block.finished_sub_slots[-1].infused_challenge_chain is None:
                # 2j
                # TODO: This code path is currently not exercised
                new_finished_ss_bad_cc = recursive_replace(
                    block.finished_sub_slots[-1],
                    "challenge_chain",
                    block.finished_sub_slots[-1].challenge_chain.replace(
                        infused_challenge_chain_sub_slot_hash=bytes32([1] * 32)
                    ),
                )
                block_bad = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_bad_cc]
                )
                await _validate_and_add_block(blockchain, block_bad, expected_error=Err.INVALID_ICC_HASH_CC)

                # 2k
                # TODO: This code path is currently not exercised
                new_finished_ss_bad_rc = recursive_replace(
                    block.finished_sub_slots[-1],
                    "reward_chain",
                    block.finished_sub_slots[-1].reward_chain.replace(
                        infused_challenge_chain_sub_slot_hash=bytes32([1] * 32)
                    ),
                )
                block_bad = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_bad_rc]
                )
                await _validate_and_add_block(blockchain, block_bad, expected_error=Err.INVALID_ICC_HASH_RC)

            # Finally, add the block properly
            await _validate_and_add_block(blockchain, block)

    @pytest.mark.anyio
    # todo_v2_plots fix this test and remove limit_consensus_modes
    # This can probably be fixed by:
    # index d0f7daa91b..b6c5a27d13 100644
    # --- a/chia/consensus/multiprocess_validation.py
    # +++ b/chia/consensus/multiprocess_validation.py
    # @@ -246,7 +246,7 @@ async def pre_validate_block(
    #              sub_slot_iters=vs.ssi,
    #              prev_ses_block=vs.prev_ses_block,
    #          )
    # -    except ValueError:
    # +    except Exception as e:
    #          log.exception("block_to_block_record()")
    #          return return_error(Err.INVALID_SUB_EPOCH_SUMMARY)
    @pytest.mark.limit_consensus_modes(
        allowed=[ConsensusMode.PLAIN, ConsensusMode.HARD_FORK_2_0],
        reason="In the 3.0 hard fork scenario, the last check fails with an exception "
        "(KeyError) instead of an error code. All passing tests fail because the "
        "proof-of-space is invalid (mismatching challenge). The test suggests that "
        "the specific error isn't important, but it still doesn't like exceptions",
    )
    async def test_empty_slot_no_ses(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2l
        blockchain = empty_blockchain
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(blockchain, blocks[0])
        blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, skip_slots=4)

        new_finished_ss = recursive_replace(
            blocks[-1].finished_sub_slots[-1],
            "challenge_chain",
            blocks[-1].finished_sub_slots[-1].challenge_chain.replace(subepoch_summary_hash=std_hash(b"0")),
        )
        block_bad = recursive_replace(
            blocks[-1], "finished_sub_slots", [*blocks[-1].finished_sub_slots[:-1], new_finished_ss]
        )

        header_block_bad = get_block_header(block_bad)
        expected_vs = ValidationState(
            empty_blockchain.constants.SUB_SLOT_ITERS_STARTING, empty_blockchain.constants.DIFFICULTY_STARTING, None
        )
        _, error = validate_finished_header_block(
            empty_blockchain.constants, empty_blockchain, header_block_bad, False, expected_vs
        )
        assert error is not None
        assert error.code == Err.INVALID_SUB_EPOCH_SUMMARY_HASH
        await _validate_and_add_block(blockchain, block_bad, expected_result=AddBlockResult.INVALID_BLOCK)

    @pytest.mark.anyio
    @pytest.mark.limit_consensus_modes(
        allowed=[ConsensusMode.PLAIN, ConsensusMode.HARD_FORK_2_0],
        reason="After the phase out, when we have v2-only plots. It seems like "
        "we never get an overflow block. get_consecutive_blocks(..., force_overflow=True) "
        "loops until we time out",
    )
    async def test_empty_sub_slots_epoch(
        self, empty_blockchain: Blockchain, default_400_blocks: list[FullBlock], bt: BlockTools
    ) -> None:
        # 2m
        # Tests adding an empty sub slot after the sub-epoch / epoch.
        # Also tests overflow block in epoch
        blocks_base = default_400_blocks[: bt.constants.EPOCH_BLOCKS]
        assert len(blocks_base) == bt.constants.EPOCH_BLOCKS
        blocks_1 = bt.get_consecutive_blocks(1, block_list_input=blocks_base, force_overflow=True)
        blocks_2 = bt.get_consecutive_blocks(1, skip_slots=5, block_list_input=blocks_base, force_overflow=True)
        for block in blocks_base:
            await _validate_and_add_block(empty_blockchain, block, skip_prevalidation=True)
        await _validate_and_add_block(
            empty_blockchain, blocks_1[-1], expected_result=AddBlockResult.NEW_PEAK, skip_prevalidation=True
        )
        assert blocks_1[-1].header_hash != blocks_2[-1].header_hash
        await _validate_and_add_block(
            empty_blockchain, blocks_2[-1], expected_result=AddBlockResult.ADDED_AS_ORPHAN, skip_prevalidation=True
        )

    @pytest.mark.anyio
    async def test_wrong_cc_hash_rc(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2o
        blockchain = empty_blockchain
        blocks = bt.get_consecutive_blocks(1, skip_slots=1)
        blocks = bt.get_consecutive_blocks(1, skip_slots=1, block_list_input=blocks)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        new_finished_ss = recursive_replace(
            blocks[-1].finished_sub_slots[-1],
            "reward_chain",
            blocks[-1].finished_sub_slots[-1].reward_chain.replace(challenge_chain_sub_slot_hash=bytes32([3] * 32)),
        )
        block_1_bad = recursive_replace(
            blocks[-1], "finished_sub_slots", [*blocks[-1].finished_sub_slots[:-1], new_finished_ss]
        )

        await _validate_and_add_block(blockchain, block_1_bad, expected_error=Err.INVALID_CHALLENGE_SLOT_HASH_RC)

    @pytest.mark.anyio
    async def test_invalid_cc_sub_slot_vdf(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2q
        blocks: list[FullBlock] = []
        found_overflow_slot: bool = False

        while not found_overflow_slot:
            blocks = bt.get_consecutive_blocks(1, blocks)
            block = blocks[-1]
            if (
                len(block.finished_sub_slots)
                and is_overflow_block(bt.constants, block.reward_chain_block.signage_point_index)
                and block.finished_sub_slots[-1].challenge_chain.challenge_chain_end_of_slot_vdf.output
                != ClassgroupElement.get_default_element()
            ):
                found_overflow_slot = True
                # Bad iters
                new_finished_ss = recursive_replace(
                    block.finished_sub_slots[-1],
                    "challenge_chain",
                    recursive_replace(
                        block.finished_sub_slots[-1].challenge_chain,
                        "challenge_chain_end_of_slot_vdf.number_of_iterations",
                        uint64(10000000),
                    ),
                )
                new_finished_ss = recursive_replace(
                    new_finished_ss,
                    "reward_chain.challenge_chain_sub_slot_hash",
                    new_finished_ss.challenge_chain.get_hash(),
                )
                log.warning(f"Num slots: {len(block.finished_sub_slots)}")
                block_bad = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss]
                )
                log.warning(f"Signage point index: {block_bad.reward_chain_block.signage_point_index}")
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_CC_EOS_VDF)

                # Bad output
                new_finished_ss_2 = recursive_replace(
                    block.finished_sub_slots[-1],
                    "challenge_chain",
                    recursive_replace(
                        block.finished_sub_slots[-1].challenge_chain,
                        "challenge_chain_end_of_slot_vdf.output",
                        ClassgroupElement.get_default_element(),
                    ),
                )

                new_finished_ss_2 = recursive_replace(
                    new_finished_ss_2,
                    "reward_chain.challenge_chain_sub_slot_hash",
                    new_finished_ss_2.challenge_chain.get_hash(),
                )
                block_bad_2 = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_2]
                )
                await _validate_and_add_block(empty_blockchain, block_bad_2, expected_error=Err.INVALID_CC_EOS_VDF)

                # Bad challenge hash
                new_finished_ss_3 = recursive_replace(
                    block.finished_sub_slots[-1],
                    "challenge_chain",
                    recursive_replace(
                        block.finished_sub_slots[-1].challenge_chain,
                        "challenge_chain_end_of_slot_vdf.challenge",
                        bytes([1] * 32),
                    ),
                )

                new_finished_ss_3 = recursive_replace(
                    new_finished_ss_3,
                    "reward_chain.challenge_chain_sub_slot_hash",
                    new_finished_ss_3.challenge_chain.get_hash(),
                )
                block_bad_3 = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_3]
                )

                await _validate_and_add_block_multi_error(
                    empty_blockchain,
                    block_bad_3,
                    [Err.INVALID_CC_EOS_VDF, Err.INVALID_PREV_CHALLENGE_SLOT_HASH, Err.INVALID_POSPACE],
                )

                # Bad proof
                new_finished_ss_5 = recursive_replace(
                    block.finished_sub_slots[-1],
                    "proofs.challenge_chain_slot_proof",
                    VDFProof(uint8(0), b"1239819023890", False),
                )
                block_bad_5 = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_5]
                )
                await _validate_and_add_block(empty_blockchain, block_bad_5, expected_error=Err.INVALID_CC_EOS_VDF)

            await _validate_and_add_block(empty_blockchain, block)

    @pytest.mark.anyio
    async def test_invalid_rc_sub_slot_vdf(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2p
        blocks: list[FullBlock] = []
        found_block: bool = False

        while not found_block:
            blocks = bt.get_consecutive_blocks(1, blocks)
            block = blocks[-1]
            if (
                len(block.finished_sub_slots)
                and block.finished_sub_slots[-1].reward_chain.end_of_slot_vdf.output
                != ClassgroupElement.get_default_element()
            ):
                found_block = True
                # Bad iters
                new_finished_ss = recursive_replace(
                    block.finished_sub_slots[-1],
                    "reward_chain",
                    recursive_replace(
                        block.finished_sub_slots[-1].reward_chain,
                        "end_of_slot_vdf.number_of_iterations",
                        uint64(10000000),
                    ),
                )
                block_bad = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss]
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_EOS_VDF)

                # Bad output
                new_finished_ss_2 = recursive_replace(
                    block.finished_sub_slots[-1],
                    "reward_chain",
                    recursive_replace(
                        block.finished_sub_slots[-1].reward_chain,
                        "end_of_slot_vdf.output",
                        # Don't use the default element here. With BlockTools'
                        # tiny 16-bit test because of the small discriminant in
                        # tests, byte-distinct ClassgroupElement encodings can
                        # reduce to the same classgroup element in the native
                        # VDF verifier. Some RC EOS outputs therefore still
                        # verify after replacement with the default element
                        # and the block is rejected later because the
                        # serialized reward sub-slot hash changed.
                        bad_element,
                    ),
                )
                block_bad_2 = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_2]
                )
                await _validate_and_add_block(empty_blockchain, block_bad_2, expected_error=Err.INVALID_RC_EOS_VDF)

                # Bad challenge hash
                new_finished_ss_3 = recursive_replace(
                    block.finished_sub_slots[-1],
                    "reward_chain",
                    recursive_replace(
                        block.finished_sub_slots[-1].reward_chain,
                        "end_of_slot_vdf.challenge",
                        bytes32([1] * 32),
                    ),
                )
                block_bad_3 = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_3]
                )
                await _validate_and_add_block(empty_blockchain, block_bad_3, expected_error=Err.INVALID_RC_EOS_VDF)

                # Bad proof
                new_finished_ss_5 = recursive_replace(
                    block.finished_sub_slots[-1],
                    "proofs.reward_chain_slot_proof",
                    VDFProof(uint8(0), b"1239819023890", False),
                )
                block_bad_5 = recursive_replace(
                    block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss_5]
                )
                await _validate_and_add_block(empty_blockchain, block_bad_5, expected_error=Err.INVALID_RC_EOS_VDF)

            await _validate_and_add_block(empty_blockchain, block)

    @pytest.mark.anyio
    async def test_genesis_bad_deficit(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2r
        block = bt.get_consecutive_blocks(1, skip_slots=2)[0]
        new_finished_ss = recursive_replace(
            block.finished_sub_slots[-1],
            "reward_chain",
            recursive_replace(
                block.finished_sub_slots[-1].reward_chain,
                "deficit",
                bt.constants.MIN_BLOCKS_PER_CHALLENGE_BLOCK - 1,
            ),
        )
        block_bad = recursive_replace(block, "finished_sub_slots", [*block.finished_sub_slots[:-1], new_finished_ss])
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_DEFICIT)

    @pytest.mark.anyio
    async def test_reset_deficit(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 2s, 2t
        blockchain = empty_blockchain
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        await _validate_and_add_block(empty_blockchain, blocks[1])
        case_1, case_2 = False, False
        while not case_1 or not case_2:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks, skip_slots=1)
            if len(blocks[-1].finished_sub_slots) > 0:
                new_finished_ss = recursive_replace(
                    blocks[-1].finished_sub_slots[-1],
                    "reward_chain",
                    recursive_replace(
                        blocks[-1].finished_sub_slots[-1].reward_chain,
                        "deficit",
                        uint8(0),
                    ),
                )
                if blockchain.block_record(blocks[-2].header_hash).deficit == 0:
                    case_1 = True
                else:
                    case_2 = True

                block_bad = recursive_replace(
                    blocks[-1], "finished_sub_slots", [*blocks[-1].finished_sub_slots[:-1], new_finished_ss]
                )
                await _validate_and_add_block_multi_error(
                    empty_blockchain, block_bad, [Err.INVALID_DEFICIT, Err.INVALID_ICC_HASH_CC]
                )

            await _validate_and_add_block(empty_blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_genesis_has_ses(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 3a
        block = bt.get_consecutive_blocks(1, skip_slots=1)[0]
        new_finished_ss = recursive_replace(
            block.finished_sub_slots[0],
            "challenge_chain",
            recursive_replace(
                block.finished_sub_slots[0].challenge_chain,
                "subepoch_summary_hash",
                bytes32.zeros,
            ),
        )

        new_finished_ss = recursive_replace(
            new_finished_ss,
            "reward_chain",
            new_finished_ss.reward_chain.replace(
                challenge_chain_sub_slot_hash=new_finished_ss.challenge_chain.get_hash()
            ),
        )
        block_bad = recursive_replace(block, "finished_sub_slots", [new_finished_ss, *block.finished_sub_slots[1:]])
        with pytest.raises(AssertionError):
            # Fails pre validation
            await _validate_and_add_block(
                empty_blockchain, block_bad, expected_error=Err.INVALID_SUB_EPOCH_SUMMARY_HASH
            )

    @pytest.mark.anyio
    async def test_no_ses_if_no_se(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 3b
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        while True:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            if len(blocks[-1].finished_sub_slots) > 0 and is_overflow_block(
                bt.constants, blocks[-1].reward_chain_block.signage_point_index
            ):
                new_finished_ss: EndOfSubSlotBundle = recursive_replace(
                    blocks[-1].finished_sub_slots[0],
                    "challenge_chain",
                    recursive_replace(
                        blocks[-1].finished_sub_slots[0].challenge_chain,
                        "subepoch_summary_hash",
                        bytes32.zeros,
                    ),
                )

                new_finished_ss = recursive_replace(
                    new_finished_ss,
                    "reward_chain",
                    new_finished_ss.reward_chain.replace(
                        challenge_chain_sub_slot_hash=new_finished_ss.challenge_chain.get_hash(),
                    ),
                )
                block_bad = recursive_replace(
                    blocks[-1], "finished_sub_slots", [new_finished_ss, *blocks[-1].finished_sub_slots[1:]]
                )
                await _validate_and_add_block_multi_error(
                    empty_blockchain,
                    block_bad,
                    expected_errors=[
                        Err.INVALID_SUB_EPOCH_SUMMARY_HASH,
                        Err.INVALID_SUB_EPOCH_SUMMARY,
                    ],
                )
                return None
            await _validate_and_add_block(empty_blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_too_many_blocks(self, empty_blockchain: Blockchain) -> None:
        # 4: TODO
        pass

    @pytest.mark.anyio
    async def test_bad_pos(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 5
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        block_bad = recursive_replace(blocks[-1], "reward_chain_block.proof_of_space.challenge", std_hash(b""))
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_POSPACE)

        block_bad = recursive_replace(
            blocks[-1], "reward_chain_block.proof_of_space.pool_contract_puzzle_hash", std_hash(b"")
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_POSPACE)

        if blocks[-1].reward_chain_block.proof_of_space.version == 0:
            block_bad = recursive_replace(blocks[-1], "reward_chain_block.proof_of_space.size", 62)
        else:
            block_bad = recursive_replace(blocks[-1], "reward_chain_block.proof_of_space.strength", 1)
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_POSPACE)

        block_bad = recursive_replace(
            blocks[-1],
            "reward_chain_block.proof_of_space.plot_public_key",
            AugSchemeMPL.key_gen(std_hash(b"1231n")).get_g1(),
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_POSPACE)

        if blocks[-1].reward_chain_block.proof_of_space.version == 0:
            block_bad = recursive_replace(
                blocks[-1],
                "reward_chain_block.proof_of_space.size",
                32,
            )
        else:
            block_bad = recursive_replace(
                blocks[-1],
                "reward_chain_block.proof_of_space.strength",
                67,
            )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_POSPACE)
        block_bad = recursive_replace(
            blocks[-1],
            "reward_chain_block.proof_of_space.proof",
            bytes([1] * len(blocks[-1].reward_chain_block.proof_of_space.proof)),
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_POSPACE)

        # TODO: test not passing the plot filter

    @pytest.mark.anyio
    async def test_bad_signage_point_index(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 6
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        with pytest.raises(ValueError):
            block_bad = recursive_replace(
                blocks[-1], "reward_chain_block.signage_point_index", bt.constants.NUM_SPS_SUB_SLOT
            )
            await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_SP_INDEX)
        with pytest.raises(ValueError):
            block_bad = recursive_replace(
                blocks[-1], "reward_chain_block.signage_point_index", bt.constants.NUM_SPS_SUB_SLOT + 1
            )
            await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_SP_INDEX)

    @pytest.mark.anyio
    async def test_sp_0_no_sp(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 7
        blocks: list[FullBlock] = []
        case_1, case_2 = False, False
        while not case_1 or not case_2:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            if blocks[-1].reward_chain_block.signage_point_index == 0:
                case_1 = True
                block_bad = recursive_replace(blocks[-1], "reward_chain_block.signage_point_index", uint8(1))
                if blocks[-1].reward_chain_block.proof_of_space.param().strength_v2 is not None:
                    # V2 plot filtering depends on the signage point index, so this mutation may fail
                    # PoSpace validation before reaching the SP-index consistency check.
                    await _validate_and_add_block_multi_error(
                        empty_blockchain, block_bad, [Err.INVALID_SP_INDEX, Err.INVALID_POSPACE]
                    )
                else:
                    await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_SP_INDEX)

            elif not is_overflow_block(bt.constants, blocks[-1].reward_chain_block.signage_point_index):
                case_2 = True
                block_bad = recursive_replace(blocks[-1], "reward_chain_block.signage_point_index", uint8(0))
                await _validate_and_add_block_multi_error(
                    empty_blockchain, block_bad, [Err.INVALID_SP_INDEX, Err.INVALID_POSPACE]
                )
            await _validate_and_add_block(empty_blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_epoch_overflows(self, empty_blockchain: Blockchain) -> None:
        # 9. TODO. This is hard to test because it requires modifying the block tools to make these special blocks
        pass

    @pytest.mark.anyio
    async def test_bad_total_iters(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 10
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        block_bad = recursive_replace(
            blocks[-1], "reward_chain_block.total_iters", blocks[-1].reward_chain_block.total_iters + 1
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_TOTAL_ITERS)

    @pytest.mark.anyio
    async def test_bad_rc_sp_vdf(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 11
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        while True:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            if blocks[-1].reward_chain_block.signage_point_index != 0:
                block_bad = recursive_replace(
                    blocks[-1], "reward_chain_block.reward_chain_sp_vdf.challenge", std_hash(b"1")
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_SP_VDF)
                block_bad = recursive_replace(
                    blocks[-1],
                    "reward_chain_block.reward_chain_sp_vdf.output",
                    bad_element,
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_SP_VDF)
                block_bad = recursive_replace(
                    blocks[-1],
                    "reward_chain_block.reward_chain_sp_vdf.number_of_iterations",
                    uint64(1111111111111),
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_SP_VDF)
                block_bad = recursive_replace(
                    blocks[-1],
                    "reward_chain_sp_proof",
                    VDFProof(uint8(0), std_hash(b""), False),
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_SP_VDF)
                return None
            await _validate_and_add_block(empty_blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_bad_rc_sp_sig(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 12
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        block_bad = recursive_replace(blocks[-1], "reward_chain_block.reward_chain_sp_signature", G2Element.generator())
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_SIGNATURE)

    @pytest.mark.anyio
    async def test_bad_cc_sp_vdf(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 13. Note: does not validate fully due to proof of space being validated first

        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        while True:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            if blocks[-1].reward_chain_block.signage_point_index != 0:
                block_bad = recursive_replace(
                    blocks[-1], "reward_chain_block.challenge_chain_sp_vdf.challenge", std_hash(b"1")
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_result=AddBlockResult.INVALID_BLOCK)
                block_bad = recursive_replace(
                    blocks[-1],
                    "reward_chain_block.challenge_chain_sp_vdf.output",
                    bad_element,
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_result=AddBlockResult.INVALID_BLOCK)
                block_bad = recursive_replace(
                    blocks[-1],
                    "reward_chain_block.challenge_chain_sp_vdf.number_of_iterations",
                    uint64(1111111111111),
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_result=AddBlockResult.INVALID_BLOCK)
                block_bad = recursive_replace(
                    blocks[-1],
                    "challenge_chain_sp_proof",
                    VDFProof(uint8(0), std_hash(b""), False),
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_CC_SP_VDF)
                return None
            await _validate_and_add_block(empty_blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_bad_cc_sp_sig(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 14
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        block_bad = recursive_replace(
            blocks[-1], "reward_chain_block.challenge_chain_sp_signature", G2Element.generator()
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_CC_SIGNATURE)

    @pytest.mark.anyio
    async def test_is_transaction_block(self, empty_blockchain: Blockchain) -> None:
        # 15: TODO
        pass

    @pytest.mark.anyio
    async def test_bad_foliage_sb_sig(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 16
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        block_bad = recursive_replace(blocks[-1], "foliage.foliage_block_data_signature", G2Element.generator())
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_PLOT_SIGNATURE)

    @pytest.mark.anyio
    async def test_bad_foliage_transaction_block_sig(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 17
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        while True:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            if blocks[-1].foliage_transaction_block is not None:
                block_bad = recursive_replace(
                    blocks[-1], "foliage.foliage_transaction_block_signature", G2Element.generator()
                )
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_PLOT_SIGNATURE)
                return None
            await _validate_and_add_block(empty_blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_unfinished_reward_chain_sb_hash(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 18
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        block_bad: FullBlock = recursive_replace(
            blocks[-1], "foliage.foliage_block_data.unfinished_reward_block_hash", std_hash(b"2")
        )
        new_m = block_bad.foliage.foliage_block_data.get_hash()
        assert new_m is not None
        new_fsb_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
        block_bad = recursive_replace(block_bad, "foliage.foliage_block_data_signature", new_fsb_sig)
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_URSB_HASH)

    @pytest.mark.anyio
    async def test_pool_target_height(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 19
        blocks = bt.get_consecutive_blocks(3)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        await _validate_and_add_block(empty_blockchain, blocks[1])
        block_bad: FullBlock = recursive_replace(blocks[-1], "foliage.foliage_block_data.pool_target.max_height", 1)
        new_m = block_bad.foliage.foliage_block_data.get_hash()
        assert new_m is not None
        new_fsb_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
        block_bad = recursive_replace(block_bad, "foliage.foliage_block_data_signature", new_fsb_sig)
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.OLD_POOL_TARGET)

    @pytest.mark.anyio
    async def test_pool_target_pre_farm(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 20a
        blocks = bt.get_consecutive_blocks(1)
        block_bad: FullBlock = recursive_replace(
            blocks[-1], "foliage.foliage_block_data.pool_target.puzzle_hash", std_hash(b"12")
        )
        new_m = block_bad.foliage.foliage_block_data.get_hash()
        assert new_m is not None
        new_fsb_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
        block_bad = recursive_replace(block_bad, "foliage.foliage_block_data_signature", new_fsb_sig)
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_PREFARM)

    @pytest.mark.anyio
    async def test_pool_target_signature(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 20b
        blocks_initial = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks_initial[0])
        await _validate_and_add_block(empty_blockchain, blocks_initial[1])

        attempts = 0
        while True:
            # Go until we get a block that has a pool pk, as opposed to a pool contract
            blocks = bt.get_consecutive_blocks(
                1, blocks_initial, seed=std_hash(attempts.to_bytes(4, byteorder="big", signed=False))
            )
            if blocks[-1].foliage.foliage_block_data.pool_signature is not None:
                block_bad: FullBlock = recursive_replace(
                    blocks[-1], "foliage.foliage_block_data.pool_signature", G2Element.generator()
                )
                new_m = block_bad.foliage.foliage_block_data.get_hash()
                assert new_m is not None
                new_fsb_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
                block_bad = recursive_replace(block_bad, "foliage.foliage_block_data_signature", new_fsb_sig)
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_POOL_SIGNATURE)
                return None
            attempts += 1
            assert attempts < 300

    @pytest.mark.anyio
    @pytest.mark.limit_consensus_modes(
        allowed=[
            ConsensusMode.PLAIN,
            ConsensusMode.HARD_FORK_2_0,
            ConsensusMode.SOFT_FORK_2_7,
        ],
        reason=(
            "This test asserts INVALID_POOL_TARGET; HF3 V2 plots can fail filter/PoSpace validation before reaching "
            "that pool-target check."
        ),
    )
    async def test_pool_target_contract(
        self, empty_blockchain: Blockchain, bt: BlockTools, seeded_random: random.Random
    ) -> None:
        # 20c invalid pool target with contract
        blocks_initial = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks_initial[0])
        await _validate_and_add_block(empty_blockchain, blocks_initial[1])

        attempts = 0
        while True:
            # Go until we get a block that has a pool contract opposed to a pool pk
            blocks = bt.get_consecutive_blocks(
                1, blocks_initial, seed=std_hash(attempts.to_bytes(4, byteorder="big", signed=False))
            )
            if blocks[-1].foliage.foliage_block_data.pool_signature is None:
                block_bad: FullBlock = recursive_replace(
                    blocks[-1], "foliage.foliage_block_data.pool_target.puzzle_hash", bytes32.random(seeded_random)
                )
                new_m = block_bad.foliage.foliage_block_data.get_hash()
                assert new_m is not None
                new_fsb_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
                block_bad = recursive_replace(block_bad, "foliage.foliage_block_data_signature", new_fsb_sig)
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_POOL_TARGET)
                return
            attempts += 1
            assert attempts < 400

    @pytest.mark.anyio
    async def test_foliage_data_presence(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 22
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        case_1, case_2 = False, False
        while not case_1 or not case_2:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            if blocks[-1].foliage_transaction_block is not None:
                case_1 = True
                block_bad: FullBlock = recursive_replace(blocks[-1], "foliage.foliage_transaction_block_hash", None)
            else:
                case_2 = True
                block_bad = recursive_replace(blocks[-1], "foliage.foliage_transaction_block_hash", std_hash(b""))
            await _validate_and_add_block_multi_error(
                empty_blockchain,
                block_bad,
                [
                    Err.INVALID_FOLIAGE_BLOCK_PRESENCE,
                    Err.INVALID_IS_TRANSACTION_BLOCK,
                    Err.INVALID_PREV_BLOCK_HASH,
                    Err.INVALID_PREV_BLOCK_HASH,
                ],
            )

    @pytest.mark.anyio
    async def test_foliage_transaction_block_hash(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 23
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        case_1, case_2 = False, False
        while not case_1 or not case_2:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            if blocks[-1].foliage_transaction_block is not None:
                block_bad: FullBlock = recursive_replace(
                    blocks[-1], "foliage.foliage_transaction_block_hash", std_hash(b"2")
                )

                new_m = block_bad.foliage.foliage_transaction_block_hash
                assert new_m is not None
                new_fbh_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
                block_bad = recursive_replace(block_bad, "foliage.foliage_transaction_block_signature", new_fbh_sig)
                await _validate_and_add_block(
                    empty_blockchain, block_bad, expected_error=Err.INVALID_FOLIAGE_BLOCK_HASH
                )
                return None
            await _validate_and_add_block(empty_blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_genesis_bad_prev_block(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 24a
        blocks = bt.get_consecutive_blocks(1)
        block_bad: FullBlock = recursive_replace(
            blocks[-1], "foliage_transaction_block.prev_transaction_block_hash", std_hash(b"2")
        )
        assert block_bad.foliage_transaction_block is not None
        block_bad = recursive_replace(
            block_bad, "foliage.foliage_transaction_block_hash", block_bad.foliage_transaction_block.get_hash()
        )
        new_m = block_bad.foliage.foliage_transaction_block_hash
        assert new_m is not None
        new_fbh_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
        block_bad = recursive_replace(block_bad, "foliage.foliage_transaction_block_signature", new_fbh_sig)
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_PREV_BLOCK_HASH)

    @pytest.mark.anyio
    async def test_bad_prev_block_non_genesis(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 24b
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        while True:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            if blocks[-1].foliage_transaction_block is not None:
                block_bad: FullBlock = recursive_replace(
                    blocks[-1], "foliage_transaction_block.prev_transaction_block_hash", std_hash(b"2")
                )
                assert block_bad.foliage_transaction_block is not None
                block_bad = recursive_replace(
                    block_bad, "foliage.foliage_transaction_block_hash", block_bad.foliage_transaction_block.get_hash()
                )
                new_m = block_bad.foliage.foliage_transaction_block_hash
                assert new_m is not None
                new_fbh_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
                block_bad = recursive_replace(block_bad, "foliage.foliage_transaction_block_signature", new_fbh_sig)
                await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_PREV_BLOCK_HASH)
                return None
            await _validate_and_add_block(empty_blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_bad_filter_hash(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 25
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        while True:
            blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
            if blocks[-1].foliage_transaction_block is not None:
                block_bad: FullBlock = recursive_replace(
                    blocks[-1], "foliage_transaction_block.filter_hash", std_hash(b"2")
                )
                assert block_bad.foliage_transaction_block is not None
                block_bad = recursive_replace(
                    block_bad, "foliage.foliage_transaction_block_hash", block_bad.foliage_transaction_block.get_hash()
                )
                new_m = block_bad.foliage.foliage_transaction_block_hash
                assert new_m is not None
                new_fbh_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
                block_bad = recursive_replace(block_bad, "foliage.foliage_transaction_block_signature", new_fbh_sig)
                await _validate_and_add_block(
                    empty_blockchain, block_bad, expected_error=Err.INVALID_TRANSACTIONS_FILTER_HASH
                )
                return None
            await _validate_and_add_block(empty_blockchain, blocks[-1])

    @pytest.mark.anyio
    async def test_bad_timestamp(self, bt: BlockTools) -> None:
        # 26
        # the test constants set MAX_FUTURE_TIME to 10 days, restore it to
        # default for this test
        constants = bt.constants.replace(MAX_FUTURE_TIME2=uint32(2 * 60))
        time_delta = 2 * 60 + 1

        blocks = bt.get_consecutive_blocks(1)

        async with make_empty_blockchain(constants) as b:
            await _validate_and_add_block(b, blocks[0])
            while True:
                blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
                if blocks[-1].foliage_transaction_block is None:
                    await _validate_and_add_block(b, blocks[-1])
                    continue

                assert blocks[0].foliage_transaction_block is not None
                block_bad: FullBlock = recursive_replace(
                    blocks[-1],
                    "foliage_transaction_block.timestamp",
                    blocks[0].foliage_transaction_block.timestamp - 10,
                )
                assert block_bad.foliage_transaction_block is not None
                block_bad = recursive_replace(
                    block_bad,
                    "foliage.foliage_transaction_block_hash",
                    block_bad.foliage_transaction_block.get_hash(),
                )
                new_m = block_bad.foliage.foliage_transaction_block_hash
                assert new_m is not None
                new_fbh_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
                block_bad = recursive_replace(block_bad, "foliage.foliage_transaction_block_signature", new_fbh_sig)
                await _validate_and_add_block(b, block_bad, expected_error=Err.TIMESTAMP_TOO_FAR_IN_PAST)

                assert blocks[0].foliage_transaction_block is not None
                block_bad = recursive_replace(
                    blocks[-1],
                    "foliage_transaction_block.timestamp",
                    blocks[0].foliage_transaction_block.timestamp,
                )
                assert block_bad.foliage_transaction_block is not None
                block_bad = recursive_replace(
                    block_bad,
                    "foliage.foliage_transaction_block_hash",
                    block_bad.foliage_transaction_block.get_hash(),
                )
                new_m = block_bad.foliage.foliage_transaction_block_hash
                assert new_m is not None
                new_fbh_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
                block_bad = recursive_replace(block_bad, "foliage.foliage_transaction_block_signature", new_fbh_sig)
                await _validate_and_add_block(b, block_bad, expected_error=Err.TIMESTAMP_TOO_FAR_IN_PAST)

                # Set the timestamp on the block to be too far out in the
                # future from now
                slack = 5
                block_bad = recursive_replace(
                    blocks[-1],
                    "foliage_transaction_block.timestamp",
                    uint64(time.time()) + time_delta + slack,
                )
                assert block_bad.foliage_transaction_block is not None
                block_bad = recursive_replace(
                    block_bad,
                    "foliage.foliage_transaction_block_hash",
                    block_bad.foliage_transaction_block.get_hash(),
                )
                new_m = block_bad.foliage.foliage_transaction_block_hash
                assert new_m is not None
                new_fbh_sig = bt.get_plot_signature(new_m, blocks[-1].reward_chain_block.proof_of_space.plot_public_key)
                block_bad = recursive_replace(block_bad, "foliage.foliage_transaction_block_signature", new_fbh_sig)
                await _validate_and_add_block(b, block_bad, expected_error=Err.TIMESTAMP_TOO_FAR_IN_FUTURE)
                return None

    @pytest.mark.anyio
    async def test_height(self, empty_blockchain: Blockchain, bt: BlockTools, monkeypatch: pytest.MonkeyPatch) -> None:
        # Monkey patch add_block_to_mmr otherwise we will throw an error in the mmr invariant
        # assetion and not in the block header validation code
        monkeypatch.setattr(
            "chia.consensus.blockchain_mmr.BlockchainMMRManager.add_block_to_mmr",
            lambda *args, **kwargs: None,
        )

        # 27
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        block_bad: FullBlock = recursive_replace(blocks[-1], "reward_chain_block.height", 2)
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_HEIGHT)

    @pytest.mark.anyio
    async def test_height_genesis(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 27
        blocks = bt.get_consecutive_blocks(1)
        block_bad: FullBlock = recursive_replace(blocks[-1], "reward_chain_block.height", 1)
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_PREV_BLOCK_HASH)

    @pytest.mark.anyio
    async def test_weight(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 28
        blocks = bt.get_consecutive_blocks(2)
        await _validate_and_add_block(empty_blockchain, blocks[0])
        block_bad: FullBlock = recursive_replace(blocks[-1], "reward_chain_block.weight", 22131)
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_WEIGHT)

    @pytest.mark.anyio
    async def test_weight_genesis(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 28
        blocks = bt.get_consecutive_blocks(1)
        block_bad: FullBlock = recursive_replace(blocks[-1], "reward_chain_block.weight", 0)
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_WEIGHT)

    @pytest.mark.anyio
    async def test_bad_cc_ip_vdf(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 29
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
        block_bad = recursive_replace(blocks[-1], "reward_chain_block.challenge_chain_ip_vdf.challenge", std_hash(b"1"))
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_CC_IP_VDF)
        block_bad = recursive_replace(
            blocks[-1],
            "reward_chain_block.challenge_chain_ip_vdf.output",
            bad_element,
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_CC_IP_VDF)
        block_bad = recursive_replace(
            blocks[-1],
            "reward_chain_block.challenge_chain_ip_vdf.number_of_iterations",
            uint64(1111111111111),
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_CC_IP_VDF)
        block_bad = recursive_replace(
            blocks[-1],
            "challenge_chain_ip_proof",
            VDFProof(uint8(0), std_hash(b""), False),
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_CC_IP_VDF)

    @pytest.mark.anyio
    async def test_bad_rc_ip_vdf(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 30
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
        block_bad = recursive_replace(blocks[-1], "reward_chain_block.reward_chain_ip_vdf.challenge", std_hash(b"1"))
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_IP_VDF)
        block_bad = recursive_replace(
            blocks[-1],
            "reward_chain_block.reward_chain_ip_vdf.output",
            bad_element,
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_IP_VDF)
        block_bad = recursive_replace(
            blocks[-1],
            "reward_chain_block.reward_chain_ip_vdf.number_of_iterations",
            uint64(1111111111111),
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_IP_VDF)
        block_bad = recursive_replace(
            blocks[-1],
            "reward_chain_ip_proof",
            VDFProof(uint8(0), std_hash(b""), False),
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_RC_IP_VDF)

    @pytest.mark.anyio
    async def test_bad_icc_ip_vdf(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 31
        blocks = bt.get_consecutive_blocks(1)
        await _validate_and_add_block(empty_blockchain, blocks[0])

        blocks = bt.get_consecutive_blocks(1, block_list_input=blocks)
        block_bad = recursive_replace(
            blocks[-1], "reward_chain_block.infused_challenge_chain_ip_vdf.challenge", std_hash(b"1")
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_ICC_VDF)
        block_bad = recursive_replace(
            blocks[-1],
            "reward_chain_block.infused_challenge_chain_ip_vdf.output",
            bad_element,
        )

        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_ICC_VDF)
        block_bad = recursive_replace(
            blocks[-1],
            "reward_chain_block.infused_challenge_chain_ip_vdf.number_of_iterations",
            uint64(1111111111111),
        )
        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_ICC_VDF)
        block_bad = recursive_replace(
            blocks[-1],
            "infused_challenge_chain_ip_proof",
            VDFProof(uint8(0), std_hash(b""), False),
        )

        await _validate_and_add_block(empty_blockchain, block_bad, expected_error=Err.INVALID_ICC_VDF)

    @pytest.mark.anyio
    async def test_reward_block_hash(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_reward_block_hash(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_reward_block_hash_2(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_reward_block_presence(empty_blockchain, bt)


co = ConditionOpcode
rbr = AddBlockResult


class TestPreValidation:
    @pytest.mark.anyio
    async def test_pre_validation_fails_bad_blocks(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_pre_validation_fails_bad_blocks(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_pre_validation(
        self, empty_blockchain: Blockchain, default_1000_blocks: list[FullBlock], bt: BlockTools
    ) -> None:
        await run_pre_validation_batch(empty_blockchain, default_1000_blocks)


class TestBodyValidation:
    @pytest.mark.anyio
    @pytest.mark.parametrize("puzzle", [False, True])
    async def test_announcements(self, puzzle: bool, bt: BlockTools) -> None:
        async with make_empty_blockchain(bt.constants) as blockchain:
            await run_announcements(blockchain, bt, puzzle=puzzle)

    @pytest.mark.anyio
    @pytest.mark.parametrize("opcode", MY_COIN_ASSERTION_OPCODES)
    @pytest.mark.parametrize("with_garbage", [True, False])
    async def test_conditions(
        self, empty_blockchain: Blockchain, opcode: ConditionOpcode, with_garbage: bool, bt: BlockTools
    ) -> None:
        await run_coin_assertions(empty_blockchain, bt, opcode, with_garbage=with_garbage)

    @pytest.mark.anyio
    @pytest.mark.parametrize(
        "opcode,lock_value,expected",
        TIMELOCK_CASES,
    )
    async def test_timelock_conditions(
        self, opcode: ConditionOpcode, lock_value: int, expected: AddBlockResult, bt: BlockTools
    ) -> None:
        async with make_empty_blockchain(bt.constants) as blockchain:
            await run_timelock_conditions(blockchain, bt, opcode, lock_value, expected)

    @pytest.mark.anyio
    @pytest.mark.parametrize("opcode", AGG_SIG_OPCODES)
    @pytest.mark.parametrize("with_garbage", [True, False])
    async def test_aggsig_garbage(
        self,
        empty_blockchain: Blockchain,
        opcode: ConditionOpcode,
        with_garbage: bool,
        bt: BlockTools,
        consensus_mode: ConsensusMode,
    ) -> None:
        await run_aggsig_garbage(empty_blockchain, bt, opcode, with_garbage=with_garbage)

    @pytest.mark.anyio
    @pytest.mark.parametrize("with_garbage", [True, False])
    @pytest.mark.parametrize(
        "opcode,lock_value,expected",
        EPHEMERAL_TIMELOCK_CASES,
    )
    async def test_ephemeral_timelock(
        self, opcode: ConditionOpcode, lock_value: int, expected: AddBlockResult, with_garbage: bool, bt: BlockTools
    ) -> None:
        async with make_empty_blockchain(bt.constants) as blockchain:
            await run_ephemeral_timelock(
                blockchain, bt, opcode, lock_value, expected, with_garbage=with_garbage
            )

    @pytest.mark.anyio
    async def test_not_tx_block_but_has_data(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_not_tx_block_but_has_data(empty_blockchain, bt)

    @pytest.mark.anyio
    @pytest.mark.parametrize("wrong_version", [uint8(0), uint8(1), uint8(2)])
    @pytest.mark.parametrize("transaction_block", [True, False])
    async def test_invalid_block_version(
        self, empty_blockchain: Blockchain, bt: BlockTools, wrong_version: uint8, transaction_block: bool
    ) -> None:
        await run_invalid_block_version(empty_blockchain, bt, wrong_version, transaction_block=transaction_block)

    @pytest.mark.anyio
    async def test_tx_block_missing_data(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_tx_block_missing_data(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_invalid_transactions_info_hash(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_invalid_transactions_info_hash(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_invalid_transactions_block_hash(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_invalid_transactions_block_hash(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_invalid_reward_claims(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_invalid_reward_claims(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_invalid_transactions_generator_hash(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_invalid_transactions_generator_hash(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_prevalidation_fast_fail(
        self,
        empty_blockchain: Blockchain,
        bt: BlockTools,
        monkeypatch: pytest.MonkeyPatch,
        consensus_mode: ConsensusMode,
    ) -> None:
        await run_prevalidation_fast_fail(
            empty_blockchain,
            bt,
            monkeypatch,
            expect_version_1=consensus_mode >= ConsensusMode.HARD_FORK_3_0,
        )

    @pytest.mark.anyio
    async def test_invalid_transactions_ref_list(
        self, empty_blockchain: Blockchain, bt: BlockTools, consensus_mode: ConsensusMode
    ) -> None:
        await run_invalid_transactions_ref_list(
            empty_blockchain,
            bt,
            refs_allowed=consensus_mode < ConsensusMode.SOFT_FORK_2_7,
        )

    @pytest.mark.anyio
    @pytest.mark.skipif(_is_macos_intel(), reason="Slow on macOS Intel")
    async def test_cost_exceeds_max(
        self, empty_blockchain: Blockchain, softfork_height: uint32, bt: BlockTools, consensus_mode: ConsensusMode
    ) -> None:
        await run_cost_exceeds_max(
            empty_blockchain,
            bt,
            softfork_height=softfork_height,
            extra_coins=consensus_mode >= ConsensusMode.HARD_FORK_3_0,
        )

    @pytest.mark.anyio
    async def test_clvm_must_not_fail(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        # 8
        pass

    @pytest.mark.anyio
    async def test_invalid_cost_in_block(
        self, empty_blockchain: Blockchain, softfork_height: uint32, bt: BlockTools
    ) -> None:
        await run_invalid_cost_in_block(empty_blockchain, bt, softfork_height=softfork_height)

    @pytest.mark.anyio
    async def test_max_coin_amount(self, db_version: int, bt: BlockTools) -> None:
        # 10
        # TODO: fix, this is not reaching validation. Because we can't create a block with such amounts due to uint64
        # limit in Coin
        pass
        #
        # with TempKeyring() as keychain:
        #     new_test_constants = bt.constants.replace(
        #         GENESIS_PRE_FARM_POOL_PUZZLE_HASH=bt.pool_ph,
        #         GENESIS_PRE_FARM_FARMER_PUZZLE_HASH=bt.pool_ph,
        #     )
        #     b, db_wrapper = await create_blockchain(new_test_constants, db_version)
        #     bt_2 = await create_block_tools_async(constants=new_test_constants, keychain=keychain)
        #     bt_2.constants = bt_2.constants.replace(
        #         GENESIS_PRE_FARM_POOL_PUZZLE_HASH=bt.pool_ph,
        #         GENESIS_PRE_FARM_FARMER_PUZZLE_HASH=bt.pool_ph,
        #     )
        #     blocks = bt_2.get_consecutive_blocks(
        #         3,
        #         guarantee_transaction_block=True,
        #         farmer_reward_puzzle_hash=bt.pool_ph,
        #     )
        #     assert (await b.add_block(blocks[0]))[0] == AddBlockResult.NEW_PEAK
        #     assert (await b.add_block(blocks[1]))[0] == AddBlockResult.NEW_PEAK
        #     assert (await b.add_block(blocks[2]))[0] == AddBlockResult.NEW_PEAK

        #     wt: WalletTool = bt_2.get_pool_wallet_tool()

        #     condition_dict: dict[ConditionOpcode, list[ConditionWithArgs]] = {ConditionOpcode.CREATE_COIN: []}
        #     output = ConditionWithArgs(ConditionOpcode.CREATE_COIN, [bt_2.pool_ph, int_to_bytes(2 ** 64)])
        #     condition_dict[ConditionOpcode.CREATE_COIN].append(output)

        #     coin = find_reward_coin(blocks[1], bt.pool_ph)
        #     tx = wt.generate_signed_transaction_multiple_coins(
        #         uint64(10),
        #         wt.get_new_puzzlehash(),
        #         coin,
        #         condition_dic=condition_dict,
        #     )
        #     with pytest.raises(Exception):
        #         blocks = bt_2.get_consecutive_blocks(
        #             1, block_list_input=blocks, guarantee_transaction_block=True, transaction_data=tx
        #         )
        #     await db_wrapper.close()
        #     b.shut_down()

    @pytest.mark.anyio
    async def test_invalid_merkle_roots(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_invalid_merkle_roots(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_invalid_filter(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_invalid_filter(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_duplicate_outputs(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_duplicate_outputs(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_duplicate_removals(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_duplicate_removals(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_double_spent_in_coin_store(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_double_spent_in_coin_store(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_double_spent_in_reorg(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_double_spent_in_reorg(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_minting_coin(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_minting_coin(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_max_coin_amount_fee(self) -> None:
        # 18 TODO: we can't create a block with such amounts due to uint64
        pass

    @pytest.mark.anyio
    async def test_invalid_fees_in_block(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_invalid_fees_in_block(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_invalid_agg_sig(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_invalid_agg_sig(empty_blockchain, bt)


def maybe_header_hash(block: BlockRecord | None) -> bytes32 | None:
    if block is None:
        return None
    return block.header_hash


class TestReorgs:
    @pytest.mark.anyio
    async def test_basic_reorg(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_basic_reorg(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_get_tx_peak_reorg(
        self, empty_blockchain: Blockchain, bt: BlockTools, consensus_mode: ConsensusMode
    ) -> None:
        if consensus_mode >= ConsensusMode.HARD_FORK_3_0_AFTER_PHASE_OUT:
            reorg_point = 14
        elif consensus_mode not in {
            ConsensusMode.HARD_FORK_2_0,
            ConsensusMode.SOFT_FORK_2_7,
        }:
            reorg_point = 13
        else:
            reorg_point = 12
        await run_get_tx_peak_reorg(empty_blockchain, bt, reorg_point)

    @pytest.mark.anyio
    @pytest.mark.parametrize("light_blocks", [True, False])
    @pytest.mark.skipif(_is_macos_intel(), reason="Slow on macOS Intel")
    @pytest.mark.limit_consensus_modes(
        allowed=[
            ConsensusMode.PLAIN,
            ConsensusMode.HARD_FORK_2_0,
            ConsensusMode.HARD_FORK_3_0,
            ConsensusMode.HARD_FORK_3_0_AFTER_PHASE_OUT,
        ],
        reason="save time",
    )
    async def test_long_reorg(
        self,
        light_blocks: bool,
        empty_blockchain: Blockchain,
        default_10000_blocks: list[FullBlock],
        test_long_reorg_blocks: list[FullBlock],
        test_long_reorg_blocks_light: list[FullBlock],
        consensus_mode: ConsensusMode,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        monkeypatch.setattr("chia.consensus.block_header_validation.validate_vdf", lambda *a, **kw: True)
        monkeypatch.setattr("chia.consensus.block_header_validation.AugSchemeMPL.verify", lambda *a, **kw: True)
        if light_blocks:
            reorg_blocks = test_long_reorg_blocks_light[:1650]
        elif consensus_mode >= ConsensusMode.HARD_FORK_3_0:
            reorg_blocks = test_long_reorg_blocks[:1350]
        else:
            reorg_blocks = test_long_reorg_blocks[:1200]
        await run_long_reorg(
            empty_blockchain,
            default_10000_blocks,
            reorg_blocks,
            light_blocks=light_blocks,
            consensus_mode=consensus_mode,
        )

    @pytest.mark.anyio
    @pytest.mark.skipif(_is_macos_intel(), reason="Slow on macOS Intel")
    @pytest.mark.limit_consensus_modes(
        allowed=[
            ConsensusMode.PLAIN,
            ConsensusMode.HARD_FORK_2_0,
            ConsensusMode.HARD_FORK_3_0,
            ConsensusMode.HARD_FORK_3_0_AFTER_PHASE_OUT,
        ],
        reason="save time",
    )
    async def test_long_compact_blockchain(
        self, empty_blockchain: Blockchain, default_2000_blocks_compact: list[FullBlock]
    ) -> None:
        await run_long_compact_chain(empty_blockchain, default_2000_blocks_compact)

    @pytest.mark.anyio
    async def test_reorg_from_genesis(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_reorg_from_genesis(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_reorg_transaction(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_reorg_transaction(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_get_header_blocks_in_range_tx_filter(self, empty_blockchain: Blockchain, bt: BlockTools) -> None:
        await run_header_blocks_tx_filter(empty_blockchain, bt)

    @pytest.mark.anyio
    async def test_get_blocks_at(self, empty_blockchain: Blockchain, default_1000_blocks: list[FullBlock]) -> None:
        await run_get_blocks_at(empty_blockchain, default_1000_blocks)

    @pytest.mark.anyio
    async def test_overlong_generator_encoding(
        self, empty_blockchain: Blockchain, bt: BlockTools, consensus_mode: ConsensusMode
    ) -> None:
        await run_overlong_generator_encoding(
            empty_blockchain,
            bt,
            expect_invalid=consensus_mode >= ConsensusMode.SOFT_FORK_2_7,
        )


@pytest.mark.anyio
@pytest.mark.skipif(_is_macos_intel(), reason="Slow on macOS Intel")
async def test_reorg_new_ref(empty_blockchain: Blockchain, bt: BlockTools, consensus_mode: ConsensusMode) -> None:
    await run_reorg_new_ref(
        empty_blockchain,
        bt,
        expect_same_height_becomes_peak=consensus_mode < ConsensusMode.HARD_FORK_3_0,
    )


# this test doesn't reorg, but _reconsider_peak() is passed a stale
# "fork_height" to make it look like it's in a reorg, but all the same blocks
# are just added back.
@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(
    allowed=[ConsensusMode.HARD_FORK_2_0], reason="after hard fork 2 we no longer allow block references"
)
async def test_reorg_stale_fork_height(empty_blockchain: Blockchain, bt: BlockTools) -> None:
    await run_reorg_stale_fork_height(empty_blockchain, bt)


@pytest.mark.anyio
@pytest.mark.skipif(_is_macos_intel(), reason="Slow on macOS Intel")
async def test_chain_failed_rollback(empty_blockchain: Blockchain, bt: BlockTools) -> None:
    await run_chain_failed_rollback(empty_blockchain, bt)


@pytest.mark.anyio
@pytest.mark.skipif(_is_macos_intel(), reason="Slow on macOS Intel")
async def test_reorg_flip_flop(empty_blockchain: Blockchain, bt: BlockTools, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("chia.consensus.block_header_validation.validate_vdf", lambda *a, **kw: True)
    monkeypatch.setattr("chia.consensus.block_header_validation.AugSchemeMPL.verify", lambda *a, **kw: True)
    await run_reorg_flip_flop(empty_blockchain, bt)


@pytest.mark.anyio
async def test_get_tx_peak(default_400_blocks: list[FullBlock], empty_blockchain: Blockchain) -> None:
    await run_get_tx_peak(empty_blockchain, default_400_blocks)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="block heights for generators differ between test chains in different modes")
@pytest.mark.parametrize("clear_cache", [True, False])
async def test_lookup_block_generators(
    default_10000_blocks: list[FullBlock],
    test_long_reorg_blocks_light: list[FullBlock],
    empty_blockchain: Blockchain,
    clear_cache: bool,
) -> None:
    await run_lookup_block_generators(
        empty_blockchain, default_10000_blocks, test_long_reorg_blocks_light, clear_cache=clear_cache
    )


@pytest.mark.anyio
async def test_get_header_blocks_in_range_tx_filter_non_tx_block(empty_blockchain: Blockchain, bt: BlockTools) -> None:
    await run_non_tx_header_filter(empty_blockchain, bt)


@dataclass(frozen=True)
class ForkInfoTestSetup:
    fork_info: ForkInfo
    initial_additions_since_fork: dict[bytes32, ForkAdd]
    test_block: FullBlock
    coin: Coin
    child_coin: Coin

    @classmethod
    def create(cls, same_ph_as_parent: bool, same_amount_as_parent: bool) -> ForkInfoTestSetup:
        from chia._tests.util.network_protocol_data import full_block as test_block

        unrelated_coin = Coin(bytes32([0] * 32), bytes32([1] * 32), uint64(42))
        # We add this initial state with an unrelated addition, to create a
        # difference between the `rollback` state and the completely empty
        # `reset` state.
        initial_additions_since_fork = {
            unrelated_coin.name(): ForkAdd(
                coin=unrelated_coin,
                confirmed_height=uint32(1),
                timestamp=uint64(0),
                hint=None,
                is_coinbase=False,
                same_as_parent=False,
            )
        }
        fork_info = ForkInfo(
            test_block.height - 1,
            test_block.height - 1,
            test_block.prev_header_hash,
            additions_since_fork=copy.copy(initial_additions_since_fork),
        )
        puzzle_hash = bytes32([2] * 32)
        amount = uint64(1337)
        coin = Coin(bytes32([3] * 32), puzzle_hash, amount)
        child_coin_ph = puzzle_hash if same_ph_as_parent else bytes32([4] * 32)
        child_coin_amount = amount if same_amount_as_parent else uint64(0)
        child_coin = Coin(coin.name(), child_coin_ph, child_coin_amount)
        return cls(
            fork_info=fork_info,
            initial_additions_since_fork=initial_additions_since_fork,
            test_block=test_block,
            coin=coin,
            child_coin=child_coin,
        )

    def check_additions(self, expected_same_parent_additions: set[bytes32]) -> None:
        assert all(
            a in self.fork_info.additions_since_fork and self.fork_info.additions_since_fork[a].same_as_parent
            for a in expected_same_parent_additions
        )
        remaining_additions = set(self.fork_info.additions_since_fork) - expected_same_parent_additions
        assert not any(self.fork_info.additions_since_fork[a].same_as_parent for a in remaining_additions)


@pytest.mark.parametrize("same_ph_as_parent", [True, False])
@pytest.mark.parametrize("same_amount_as_parent", [True, False])
@pytest.mark.parametrize("rollback", [True, False])
@pytest.mark.parametrize("reset", [True, False])
@pytest.mark.anyio
async def test_include_spends_same_as_parent(
    same_ph_as_parent: bool, same_amount_as_parent: bool, rollback: bool, reset: bool
) -> None:
    """
    Tests that `ForkInfo` properly tracks same-as-parent created coins.
    A created coin is tracked as such if its puzzle hash and amount match the
    parent. We're covering here `include_spends`, `rollback` and `reset` in the
    context of same-as-parent coins.
    """
    test_setup = ForkInfoTestSetup.create(same_ph_as_parent, same_amount_as_parent)
    # Now let's prepare the test spend bundle conditions
    create_coin = [(test_setup.child_coin.puzzle_hash, test_setup.child_coin.amount, None)]
    conds = SpendBundleConditions(
        [
            SpendConditions(
                test_setup.coin.name(),
                test_setup.coin.parent_coin_info,
                test_setup.coin.puzzle_hash,
                test_setup.coin.amount,
                None,
                None,
                None,
                None,
                None,
                None,
                create_coin,
                [],
                [],
                [],
                [],
                [],
                [],
                [],
                0,
                execution_cost=0,
                condition_cost=0,
                atom_count=0,
                pair_count=0,
                fingerprint=b"",
            )
        ],
        0,
        0,
        0,
        None,
        None,
        [],
        0,
        0,
        0,
        True,
        0,
        0,
        0,
        0,
        0,
    )
    # Now let's run the test
    test_setup.fork_info.include_spends(conds, test_setup.test_block, test_setup.test_block.header_hash)
    # Let's make sure the results are as expected
    expected_same_parent_additions = (
        {test_setup.child_coin.name()} if same_ph_as_parent and same_amount_as_parent else set()
    )
    test_setup.check_additions(expected_same_parent_additions)
    if rollback:
        # Now we rollback before the spend that belongs to the test conditions
        test_setup.fork_info.rollback(test_setup.test_block.prev_header_hash, test_setup.test_block.height - 1)
        # That should leave only the initial additions we started with, which
        # are unrelated to the test conditions. We added this initial state to
        # create a difference between `rollback` state and the completely empty
        # `reset` state.
        assert test_setup.fork_info.additions_since_fork == test_setup.initial_additions_since_fork
    if reset:
        # Now we reset to a test height and header hash
        test_setup.fork_info.reset(1, bytes32([0] * 32))
        # That should leave this empty
        assert test_setup.fork_info.additions_since_fork == {}


@pytest.mark.parametrize("same_ph_as_parent", [True, False])
@pytest.mark.parametrize("same_amount_as_parent", [True, False])
@pytest.mark.parametrize("rollback", [True, False])
@pytest.mark.parametrize("reset", [True, False])
@pytest.mark.anyio
async def test_include_block_same_as_parent_coins(
    same_ph_as_parent: bool, same_amount_as_parent: bool, rollback: bool, reset: bool
) -> None:
    """
    Tests that `ForkInfo` properly tracks same-as-parent created coins.
    A created coin is tracked as such if its puzzle hash and amount match the
    parent. We're covering here `include_block`, `rollback` and `reset` in the
    context of such coins.
    """
    test_setup = ForkInfoTestSetup.create(same_ph_as_parent, same_amount_as_parent)
    # Now let's run the test
    test_setup.fork_info.include_block(
        [(test_setup.child_coin, None)],
        [(test_setup.coin.name(), test_setup.coin)],
        test_setup.test_block,
        test_setup.test_block.header_hash,
    )
    # Let's make sure the results are as expected
    expected_same_as_parent_additions = (
        {test_setup.child_coin.name()} if same_ph_as_parent and same_amount_as_parent else set()
    )
    test_setup.check_additions(expected_same_as_parent_additions)
    if rollback:
        # Now we rollback before the spend that belongs to the test conditions
        test_setup.fork_info.rollback(test_setup.test_block.prev_header_hash, test_setup.test_block.height - 1)
        # That should leave only the initial additions we started with
        assert test_setup.fork_info.additions_since_fork == test_setup.initial_additions_since_fork
    if reset:
        # Now we reset to a test height and header hash
        test_setup.fork_info.reset(1, bytes32([0] * 32))
        # That should leave this empty
        assert test_setup.fork_info.additions_since_fork == {}
