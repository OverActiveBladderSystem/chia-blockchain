from __future__ import annotations

from pathlib import Path

import pytest
from chia_rs import FullBlock
from chia_rs.sized_ints import uint8, uint32

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
from chia.consensus.blockchain import AddBlockResult
from chia.simulator.block_tools import BlockTools
from chia.types.condition_opcodes import ConditionOpcode


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_basic_reorg_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_basic_reorg(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_reorg_transaction_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_reorg_transaction(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_reorg_from_genesis_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_reorg_from_genesis(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="block references are allowed before hard fork 3")
async def test_reorg_new_ref_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_reorg_new_ref(
            blockchain,
            bt,
            expect_same_height_becomes_peak=consensus_mode < ConsensusMode.HARD_FORK_3_0,
        )


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_chain_failed_rollback_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_chain_failed_rollback(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="block references in this leapfrog reorg are for the plain constants")
async def test_reorg_flip_flop_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    del consensus_mode
    monkeypatch.setattr("chia.consensus.block_header_validation.validate_vdf", lambda *a, **kw: True)
    monkeypatch.setattr("chia.consensus.block_header_validation.AugSchemeMPL.verify", lambda *a, **kw: True)
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_reorg_flip_flop(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(
    allowed=[ConsensusMode.HARD_FORK_2_0],
    reason="after hard fork 2 we no longer allow block references",
)
async def test_reorg_stale_fork_height_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_reorg_stale_fork_height(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_double_spent_in_reorg_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_double_spent_in_reorg(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_spend_rejections_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    for runner in (run_duplicate_outputs, run_duplicate_removals, run_double_spent_in_coin_store):
        async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path / runner.__name__) as (
            blockchain,
            wrapper,
        ):
            assert wrapper is None
            await runner(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_rejected_blocks_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    for runner in (run_minting_coin, run_invalid_fees_in_block, run_invalid_agg_sig):
        async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path / runner.__name__) as (
            blockchain,
            wrapper,
        ):
            assert wrapper is None
            await runner(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="plain constants put the transaction-block fork at height 13")
async def test_get_tx_peak_reorg_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_get_tx_peak_reorg(blockchain, bt, 13)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_header_blocks_tx_filter_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_header_blocks_tx_filter(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_get_blocks_at_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    default_1000_blocks: list[FullBlock],
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_get_blocks_at(blockchain, default_1000_blocks)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_non_tx_header_filter_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_non_tx_header_filter(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_get_tx_peak_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    default_400_blocks: list[FullBlock],
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_get_tx_peak(blockchain, default_400_blocks)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(
    allowed=[ConsensusMode.PLAIN, ConsensusMode.SOFT_FORK_2_7],
    reason="one mode stores the old encoding and one mode rejects it",
)
async def test_overlong_generator_encoding_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_overlong_generator_encoding(
            blockchain,
            bt,
            expect_invalid=consensus_mode >= ConsensusMode.SOFT_FORK_2_7,
        )


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_invalid_roots_and_filter_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    del consensus_mode
    for runner in (run_invalid_merkle_roots, run_invalid_filter):
        async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path / runner.__name__) as (
            blockchain,
            wrapper,
        ):
            assert wrapper is None
            await runner(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_invalid_reward_claims_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_invalid_reward_claims(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_invalid_transactions_generator_hash_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_invalid_transactions_generator_hash(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(
    allowed=[ConsensusMode.PLAIN, ConsensusMode.SOFT_FORK_2_7],
    reason="one mode still allows generator references and one mode rejects them",
)
async def test_invalid_transactions_ref_list_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_invalid_transactions_ref_list(
            blockchain,
            bt,
            refs_allowed=consensus_mode < ConsensusMode.SOFT_FORK_2_7,
        )


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="one plain run is enough to reject an over-cost block")
async def test_cost_exceeds_max_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_cost_exceeds_max(blockchain, bt, softfork_height=uint32(1_000_000), extra_coins=False)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="one plain run is enough to reject a wrong reported cost")
async def test_invalid_cost_in_block_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_invalid_cost_in_block(blockchain, bt, softfork_height=uint32(1_000_000))


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_transaction_block_shape_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    del consensus_mode
    for runner in (
        run_not_tx_block_but_has_data,
        run_tx_block_missing_data,
        run_invalid_transactions_info_hash,
        run_invalid_transactions_block_hash,
    ):
        async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path / runner.__name__) as (
            blockchain,
            wrapper,
        ):
            assert wrapper is None
            await runner(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.parametrize("wrong_version", [uint8(0), uint8(1), uint8(2)])
@pytest.mark.parametrize("transaction_block", [True, False])
@pytest.mark.limit_consensus_modes(reason="plain constants accept version 0 and reject the others")
async def test_invalid_block_version_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    wrong_version: uint8,
    transaction_block: bool,
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_invalid_block_version(blockchain, bt, wrong_version, transaction_block=transaction_block)


@pytest.mark.anyio
@pytest.mark.parametrize("with_garbage", [True, False])
@pytest.mark.parametrize("opcode,lock_value,expected", EPHEMERAL_TIMELOCK_CASES)
@pytest.mark.limit_consensus_modes(reason="plain constants are enough for ephemeral timelocks")
async def test_ephemeral_timelock_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    opcode: ConditionOpcode,
    lock_value: int,
    expected: AddBlockResult,
    with_garbage: bool,
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_ephemeral_timelock(
            blockchain, bt, opcode, lock_value, expected, with_garbage=with_garbage
        )


@pytest.mark.anyio
@pytest.mark.parametrize("opcode,lock_value,expected", TIMELOCK_CASES)
@pytest.mark.limit_consensus_modes(reason="plain constants are enough for timelocks on an earlier coin")
async def test_timelock_conditions_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    opcode: ConditionOpcode,
    lock_value: int,
    expected: AddBlockResult,
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_timelock_conditions(blockchain, bt, opcode, lock_value, expected)


@pytest.mark.anyio
@pytest.mark.parametrize("opcode", AGG_SIG_OPCODES)
@pytest.mark.parametrize("with_garbage", [True, False])
@pytest.mark.limit_consensus_modes(reason="plain constants are enough for aggregate-signature arguments")
async def test_aggsig_garbage_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    opcode: ConditionOpcode,
    with_garbage: bool,
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_aggsig_garbage(blockchain, bt, opcode, with_garbage=with_garbage)


@pytest.mark.anyio
@pytest.mark.parametrize("puzzle", [False, True])
@pytest.mark.limit_consensus_modes(reason="plain constants are enough for announcement conditions")
async def test_announcements_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    puzzle: bool,
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_announcements(blockchain, bt, puzzle=puzzle)


@pytest.mark.anyio
@pytest.mark.parametrize("opcode", MY_COIN_ASSERTION_OPCODES)
@pytest.mark.parametrize("with_garbage", [True, False])
@pytest.mark.limit_consensus_modes(reason="plain constants are enough for coin identity assertions")
async def test_coin_assertions_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    opcode: ConditionOpcode,
    with_garbage: bool,
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_coin_assertions(blockchain, bt, opcode, with_garbage=with_garbage)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="plain constants are enough for pre-checking blocks")
async def test_pre_validation_fails_bad_blocks_on_rocksdb(
    bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_pre_validation_fails_bad_blocks(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="plain constants are enough for pre-checking blocks")
async def test_pre_validation_batch_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    default_1000_blocks: list[FullBlock],
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_pre_validation_batch(blockchain, default_1000_blocks)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="plain constants are enough for reward-block hash checks")
async def test_reward_block_hash_on_rocksdb(bt: BlockTools, consensus_mode: ConsensusMode, tmp_path: Path) -> None:
    del consensus_mode
    for runner in (run_reward_block_hash, run_reward_block_presence):
        async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path / runner.__name__) as (
            blockchain,
            wrapper,
        ):
            assert wrapper is None
            await runner(blockchain, bt)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(
    allowed=[ConsensusMode.PLAIN, ConsensusMode.HARD_FORK_3_0],
    reason="one mode uses the old generator field and one uses the buffer field",
)
async def test_prevalidation_fast_fail_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_prevalidation_fast_fail(
            blockchain,
            bt,
            monkeypatch,
            expect_version_1=consensus_mode >= ConsensusMode.HARD_FORK_3_0,
        )


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_long_chain_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    default_1000_blocks: list[FullBlock],
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_long_chain(blockchain, default_1000_blocks)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="the sqlite suite already covers every consensus mode")
async def test_long_compact_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    default_2000_blocks_compact: list[FullBlock],
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_long_compact_chain(blockchain, default_2000_blocks_compact)


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="block heights for generators differ between test chains in different modes")
@pytest.mark.parametrize("clear_cache", [True, False])
async def test_lookup_block_generators_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    default_10000_blocks: list[FullBlock],
    test_long_reorg_blocks_light: list[FullBlock],
    clear_cache: bool,
) -> None:
    del consensus_mode
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_lookup_block_generators(
            blockchain,
            default_10000_blocks,
            test_long_reorg_blocks_light,
            clear_cache=clear_cache,
        )


@pytest.mark.anyio
@pytest.mark.limit_consensus_modes(reason="plain constants; the heavier shorter fork is the rocksdb case")
async def test_long_reorg_on_rocksdb(
    bt: BlockTools,
    consensus_mode: ConsensusMode,
    tmp_path: Path,
    default_10000_blocks: list[FullBlock],
    test_long_reorg_blocks: list[FullBlock],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("chia.consensus.block_header_validation.validate_vdf", lambda *a, **kw: True)
    monkeypatch.setattr("chia.consensus.block_header_validation.AugSchemeMPL.verify", lambda *a, **kw: True)
    async with create_blockchain(bt.constants, 2, engine="rocksdb", root_path=tmp_path) as (blockchain, wrapper):
        assert wrapper is None
        await run_long_reorg(
            blockchain,
            default_10000_blocks,
            test_long_reorg_blocks[:1200],
            light_blocks=False,
            consensus_mode=consensus_mode,
        )
