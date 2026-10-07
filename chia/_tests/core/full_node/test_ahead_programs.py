"""Background work for blocks downloaded ahead of the one being checked.

Every block can run the proof math stored inside it. A transaction program
runs when its older programs are already known. A saved result is reused only
when the rules and those older programs match.
"""

from __future__ import annotations

import asyncio
import logging
import threading
from concurrent.futures import Future
from types import SimpleNamespace

import pytest
from chia_rs.sized_bytes import bytes32
from chia_rs.sized_ints import uint32

from chia.consensus import multiprocess_validation as validation
from chia.consensus.multiprocess_validation import (
    _await_block_math,
    earlier_programs_for_block,
    precompute_block_proofs,
    program_can_be_precomputed,
    program_refs_digest,
    remember_precomputed_program,
    take_precomputed_program,
)
from chia.full_node.full_node import (
    _AHEAD_GENERATOR_KEEP,
    FullNode,
    _ahead_block_work,
    _next_ahead_request,
    _remember_block_generators,
    _saved_generator_heights,
    _submit_block_math,
)
from chia.util.errors import Err
from chia.util.priority_thread_pool_executor import PriorityThreadPoolExecutor


@pytest.fixture(autouse=True)
def _clear_program_cache() -> None:
    validation._program_cache.clear()


def _block(
    *,
    generator: object | None = None,
    refs: list[int] | None = None,
    transactions_info: object | None = None,
    foliage_transaction_block: object | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        transactions_generator=generator,
        transactions_generator_buffer=None,
        transactions_generator_ref_list=[uint32(height) for height in refs or []],
        transactions_info=transactions_info,
        foliage_transaction_block=foliage_transaction_block,
    )


def test_non_transaction_block_has_no_program_to_run() -> None:
    block = _block()
    assert program_can_be_precomputed(block) is False  # type: ignore[arg-type]
    assert _ahead_block_work(block, {}) == ("non-transaction", None)  # type: ignore[arg-type]


def test_reward_only_transaction_block_is_held_without_a_program() -> None:
    block = _block(transactions_info=object(), foliage_transaction_block=object())
    assert program_can_be_precomputed(block) is False  # type: ignore[arg-type]
    assert _ahead_block_work(block, {}) == ("reward-only", None)  # type: ignore[arg-type]


def test_program_that_names_an_older_block_runs_once_that_block_is_in_hand() -> None:
    block = _block(generator=object(), refs=[5], transactions_info=object(), foliage_transaction_block=object())
    assert program_can_be_precomputed(block) is False  # type: ignore[arg-type]
    assert _ahead_block_work(block, {})[0] == "needs-older"  # type: ignore[arg-type]
    assert earlier_programs_for_block(block, {5: b"older"}) == [b"older"]  # type: ignore[arg-type]
    assert _ahead_block_work(block, {5: b"older"}) == ("program", [b"older"])  # type: ignore[arg-type]


def test_saved_program_is_used_only_for_the_same_rules_and_older_programs() -> None:
    header = bytes32(b"\x11" * 32)
    digest = program_refs_digest([b"older"])
    remember_precomputed_program(header, 7, digest, None, Err.INVALID_TRANSACTIONS_GENERATOR_HASH, None)
    # Different rules drop the saved result.
    assert take_precomputed_program(header, 8, [b"older"]) is None
    assert take_precomputed_program(header, 7, [b"older"]) is None

    remember_precomputed_program(header, 7, digest, None, Err.INVALID_TRANSACTIONS_GENERATOR_HASH, "bad")
    # Different older programs drop the saved result.
    assert take_precomputed_program(header, 7, [b"other"]) is None

    remember_precomputed_program(header, 7, program_refs_digest([]), None, Err.BLOCK_COST_EXCEEDS_MAX, "cost")
    assert take_precomputed_program(header, 7, []) == (Err.BLOCK_COST_EXCEEDS_MAX, "cost", None)


def test_generator_memory_keeps_blocks_just_downloaded() -> None:
    def fake(height: int, program: bytes) -> SimpleNamespace:
        return SimpleNamespace(height=height, transactions_generator=None, transactions_generator_buffer=program)

    generators: dict[int, bytes] = {}
    first = [fake(height, b"p") for height in range(1, _AHEAD_GENERATOR_KEEP + 1)]
    _remember_block_generators(generators, first)  # type: ignore[arg-type]
    assert len(generators) == _AHEAD_GENERATOR_KEEP
    assert 1 in generators

    # A height just downloaded is kept for this call even when it is the oldest.
    _remember_block_generators(generators, [fake(0, b"fresh")])  # type: ignore[arg-type]
    assert generators[0] == b"fresh"

    newest = _AHEAD_GENERATOR_KEEP + 5
    _remember_block_generators(generators, [fake(newest, b"new")])  # type: ignore[arg-type]
    assert 0 not in generators
    assert generators[newest] == b"new"
    assert len(generators) == _AHEAD_GENERATOR_KEEP


def test_database_lookup_is_only_for_programs_already_at_or_below_the_peak() -> None:
    block = _block(
        generator=object(),
        refs=[3, 9, 20],
        transactions_info=object(),
        foliage_transaction_block=object(),
    )
    # 3 is already in memory. 20 is ahead of the saved peak. 9 is saved.
    assert _saved_generator_heights([block], {3: b"have"}, 10) == {9}  # type: ignore[list-item]
    non_tx = _block(refs=[1])
    assert _saved_generator_heights([non_tx], {}, 100) == set()  # type: ignore[list-item]


def test_non_transaction_block_queues_its_math_once() -> None:
    calls: list[tuple[str, object, object]] = []

    def run_in_loop(fn: object, *_args: object, **kwargs: object) -> None:
        calls.append((getattr(fn, "__name__", ""), kwargs.get("nice"), kwargs.get("dedicated")))

    node = SimpleNamespace(
        constants=object(),
        pool=SimpleNamespace(run_in_loop=run_in_loop),
        log=logging.getLogger("t"),
        _block_math={},
    )
    block = _block()
    block.header_hash = bytes32(b"\x22" * 32)
    block.height = 5
    FullNode._queue_ahead_programs(node, [block], "5-5", {}, set())  # type: ignore[arg-type]
    assert calls == [("_run_block_math", (0, 5), False)]
    FullNode._queue_ahead_programs(node, [block], "5-5", {}, {"5-5"})  # type: ignore[arg-type]
    assert calls == [("_run_block_math", (0, 5), False)]


def test_a_finished_proof_job_is_followed_by_one_program_job(monkeypatch: pytest.MonkeyPatch) -> None:
    """The program waits until the proof job is done, then runs once."""
    monkeypatch.setattr("chia.full_node.full_node.precompute_block_proofs", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("chia.full_node.full_node.precompute_transaction_program", lambda *_args, **_kwargs: None)
    release = threading.Event()
    block = _block(generator=object(), refs=[5], transactions_info=object(), foliage_transaction_block=object())
    block.header_hash = bytes32(b"\x33" * 32)
    block.height = 9
    generators: dict[int, bytes] = {}
    node = SimpleNamespace(constants=object(), _block_math={})

    with PriorityThreadPoolExecutor(max_workers=1) as pool:
        node.pool = pool
        pool.submit(release.wait, 5)
        assert _submit_block_math(node, block, generators, program_only=False) is True  # type: ignore[arg-type]
        # The proof job still holds the only worker, so a second job is not added.
        assert _submit_block_math(node, block, generators, program_only=True) is False  # type: ignore[arg-type]
        assert len(node._block_math[block.header_hash].futures) == 1
        release.set()
        node._block_math[block.header_hash].futures[0].result(timeout=5)
        assert node._block_math[block.header_hash].program_started is False

        generators[5] = b"older"
        assert _submit_block_math(node, block, generators, program_only=True) is True  # type: ignore[arg-type]
        assert len(node._block_math[block.header_hash].futures) == 2
        node._block_math[block.header_hash].futures[1].result(timeout=5)
        assert _submit_block_math(node, block, generators, program_only=True) is False  # type: ignore[arg-type]
        assert _submit_block_math(node, block, generators, program_only=False) is False  # type: ignore[arg-type]


@pytest.mark.anyio
async def test_check_waits_for_math_on_the_event_loop() -> None:
    """A finished job is recorded immediately. A running job is awaited, then the check starts."""
    done: Future[None] = Future()
    done.set_result(None)
    finished: dict[str, float] = {}
    await _await_block_math([done], finished, 5)
    assert finished["math_ready"] == pytest.approx(1)
    assert finished["math_queued"] == pytest.approx(1)
    assert finished["math_wait"] < 0.1

    pending: Future[None] = Future()
    running: dict[str, float] = {}
    order: list[str] = []

    async def check_after_math() -> None:
        await _await_block_math([pending], running, 6)
        order.append("check")

    task = asyncio.ensure_future(check_after_math())
    await asyncio.sleep(0.05)
    assert order == []
    assert "math_ready" not in running
    pending.set_result(None)
    await task
    assert order == ["check"]
    assert running["math_queued"] == pytest.approx(1)
    assert "math_ready" not in running


@pytest.mark.anyio
async def test_a_failed_math_job_still_reaches_the_check() -> None:
    failed: Future[None] = Future()
    failed.set_exception(RuntimeError("proof math failed"))
    phase: dict[str, float] = {}
    await _await_block_math([failed], phase, 7)
    assert phase["math_queued"] == pytest.approx(1)
    assert phase["math_ready"] == pytest.approx(1)


def test_block_proofs_use_the_numbers_stored_in_the_block(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[object] = []

    def fake_validate(*_args: object, **_kwargs: object) -> bool:
        seen.append(_args[3])
        return True

    proof = SimpleNamespace(normalized_to_identity=True)
    skipped = SimpleNamespace(normalized_to_identity=False)
    reward = SimpleNamespace(
        reward_chain_ip_vdf="rc-ip",
        reward_chain_sp_vdf="rc-sp",
        challenge_chain_ip_vdf="cc-ip",
        challenge_chain_sp_vdf="cc-sp",
    )
    sub_slot = SimpleNamespace(
        proofs=SimpleNamespace(
            reward_chain_slot_proof=proof,
            challenge_chain_slot_proof=proof,
            infused_challenge_chain_slot_proof=skipped,
        ),
        reward_chain=SimpleNamespace(end_of_slot_vdf="rc-eos"),
        challenge_chain=SimpleNamespace(challenge_chain_end_of_slot_vdf="cc-eos"),
        infused_challenge_chain=SimpleNamespace(infused_challenge_chain_end_of_slot_vdf="icc-eos"),
    )
    block = SimpleNamespace(
        reward_chain_block=reward,
        reward_chain_ip_proof=proof,
        reward_chain_sp_proof=proof,
        challenge_chain_ip_proof=proof,
        challenge_chain_sp_proof=skipped,
        finished_sub_slots=[sub_slot],
    )
    monkeypatch.setattr(validation, "validate_vdf", fake_validate)
    precompute_block_proofs(object(), block)  # type: ignore[arg-type]
    assert seen == ["rc-ip", "rc-sp", "cc-ip", "rc-eos", "cc-eos"]


def test_ahead_request_stops_at_the_target_and_the_caps() -> None:
    # One range covers the whole chain. Nothing past the target is asked for.
    assert _next_ahead_request(0, 4, 32, 4, 2, [], []) is None
    # The range being checked is in flight. Ask for the next one.
    assert _next_ahead_request(0, 1000, 32, 4, 2, [0], []) == (32, 63)
    # Two later requests are already in flight.
    assert _next_ahead_request(0, 1000, 32, 4, 2, [0, 32, 64], []) is None
    # Four later ranges are already held.
    assert _next_ahead_request(0, 1000, 32, 4, 2, [0], [32, 64, 96, 128]) is None
    # The first missing range is asked for before a later one that is already held.
    assert _next_ahead_request(0, 1000, 32, 4, 2, [0], [32, 96]) == (64, 95)
    # The last partial range is the one being checked, so there is no later range.
    assert _next_ahead_request(96, 100, 32, 4, 2, [96], []) is None
    assert _next_ahead_request(64, 100, 32, 4, 2, [64], []) == (96, 100)
