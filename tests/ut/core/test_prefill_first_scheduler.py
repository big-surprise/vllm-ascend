# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Exercise real Request/KV scheduling on CPU, including output consumption.

The contract is a single compute phase per SchedulerOutput, prefill preference
when admission succeeds, and decode progress when it does not. These tests
target scheduling/accounting errors without loading weights or running kernels.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.core.test_dyntra_lb_scheduler import create_dyntra_lb_scheduler, make_dyntra_test_config
from tests.ut.kv_offload.utils import create_request
from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.core.prefill_first_scheduler import PrefillFirstAsyncScheduler, PrefillFirstScheduler

BLOCK_SIZE = 4


@pytest.fixture(autouse=True)
def reset_ascend_config():
    clear_ascend_config()
    yield
    clear_ascend_config()


@pytest.fixture(params=[PrefillFirstScheduler, PrefillFirstAsyncScheduler])
def scheduler_cls(request):
    return request.param


def make_scheduler(scheduler_cls, budget=8, max_seqs=8, num_blocks=256, use_v2=True):
    config = make_dyntra_test_config(max_num_seqs=max_seqs, max_num_batched_tokens=64, block_size=BLOCK_SIZE)
    config.kv_transfer_config = None
    config.scheduler_config.max_num_batched_tokens = budget
    config.scheduler_config.async_scheduling = issubclass(scheduler_cls, AsyncScheduler)
    scheduler = create_dyntra_lb_scheduler(config, scheduler_cls=scheduler_cls, num_blocks=num_blocks)
    scheduler.use_v2_model_runner = use_v2
    return scheduler


def req(index, length=4, max_tokens=20):
    return create_request(index, num_tokens=length, max_tokens=max_tokens, block_size=BLOCK_SIZE)


def runner_output(scheduler, output):
    ids = list(output.num_scheduled_tokens)
    return ModelRunnerOutput(
        req_ids=ids,
        req_id_to_index={rid: i for i, rid in enumerate(ids)},
        sampled_token_ids=[[] if scheduler.requests[rid].is_prefill_chunk else [100] for rid in ids],
        logprobs=None,
        prompt_logprobs_dict={},
        pooler_output=[],
    )


def consume(scheduler, output):
    scheduler.update_from_output(output, runner_output(scheduler, output))


def start_decode(scheduler, request):
    scheduler.add_request(request)
    output = scheduler.schedule()
    assert set(output.num_scheduled_tokens) == {request.request_id}
    consume(scheduler, output)
    return request


@pytest.mark.parametrize("use_v2", [False, True])
def test_new_prefill_wins_without_advancing_or_freeing_decode(scheduler_cls, use_v2):
    scheduler = make_scheduler(scheduler_cls, use_v2=use_v2)
    decode = start_decode(scheduler, req(1))
    old_computed = decode.num_computed_tokens
    old_blocks = scheduler.kv_cache_manager.get_block_ids(decode.request_id)
    prefill = req(2, length=6)
    scheduler.add_request(prefill)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {prefill.request_id: 6}
    assert decode in scheduler.running
    assert decode.status == RequestStatus.RUNNING
    assert decode.num_computed_tokens == old_computed
    assert scheduler.kv_cache_manager.get_block_ids(decode.request_id) == old_blocks
    consume(scheduler, output)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1, prefill.request_id: 1}
    consume(scheduler, output)
    assert scheduler.phase_batch_counts == {"prefill": 2, "decode": 1, "idle": 0}


def test_chunked_prefill_remains_pure_and_yields_to_waiting_decode(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    prefill = req(2, length=20)
    scheduler.add_request(prefill)
    for index, chunk_size in enumerate((8, 8, 4)):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {prefill.request_id: chunk_size}
        assert decode.num_computed_tokens == 4 + index
        consume(scheduler, output)
        if index < 2:
            decode_output = scheduler.schedule()
            assert decode_output.num_scheduled_tokens == {decode.request_id: 1}
            consume(scheduler, decode_output)
    assert scheduler.schedule().num_scheduled_tokens == {decode.request_id: 1, prefill.request_id: 1}


def test_continuous_arrivals_cannot_indefinitely_defer_running_decode(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    scheduler.add_request(req(2))
    for i in range(2, 5):
        prefill_id = f"id-{i}"
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {prefill_id: 4}
        consume(scheduler, output)
        waiting = req(i + 1)
        scheduler.add_request(waiting)
        output = scheduler.schedule()
        assert decode.request_id in output.num_scheduled_tokens
        assert waiting.request_id not in output.num_scheduled_tokens
        assert set(output.num_scheduled_tokens.values()) == {1}
        assert waiting.status == RequestStatus.WAITING
        consume(scheduler, output)
    assert decode.num_computed_tokens == 7
    assert scheduler.phase_batch_counts == {"prefill": 4, "decode": 3, "idle": 0}


def test_full_running_slots_fall_back_to_decode(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls, max_seqs=1)
    decode = start_decode(scheduler, req(1))
    waiting = req(2)
    scheduler.add_request(waiting)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1}
    assert waiting.status == RequestStatus.WAITING


def test_insufficient_prefill_kv_falls_back_to_decode(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls, num_blocks=4)
    decode = start_decode(scheduler, req(1))
    waiting = req(2, length=16)
    scheduler.add_request(waiting)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1}
    assert waiting.status == RequestStatus.WAITING
    assert not output.preempted_req_ids


def test_blocked_waiting_request_does_not_starve_decode(scheduler_cls, monkeypatch):
    scheduler = make_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    blocked = req(2)
    blocked.status = RequestStatus.WAITING_FOR_REMOTE_KVS
    monkeypatch.setattr(scheduler, "_try_promote_blocked_waiting_request", lambda request: False)
    scheduler.add_request(blocked)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1}
    assert blocked not in scheduler.running


def test_single_token_first_prefill_does_not_mix_with_decode(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    fresh = req(2, length=1)
    scheduler.add_request(fresh)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {fresh.request_id: 1}
    assert decode.num_computed_tokens == 4
    assert scheduler.phase_batch_counts["prefill"] == 2


def test_stateful_single_token_prompt_tail_joins_decode(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    tail = req(2, length=9)
    scheduler.add_request(tail)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {tail.request_id: 8}
    consume(scheduler, output)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1, tail.request_id: 1}
    assert scheduler.phase_batch_counts["decode"] == 1


def test_stateful_one_token_budget_remainder_is_not_mixed(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    scheduler.scheduler_config.long_prefill_token_threshold = 7
    first, second = req(1, length=20), req(2, length=20)
    scheduler.add_request(first)
    scheduler.add_request(second)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {first.request_id: 7, second.request_id: 1}
    consume(scheduler, output)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {first.request_id: 7}
    assert second.num_computed_tokens == 1


def test_prefix_hit_with_one_token_left_is_deferred_to_decode_batch(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls, budget=16)
    priming = req(1, length=9, max_tokens=1)
    scheduler.add_request(priming)
    consume(scheduler, scheduler.schedule())
    assert priming.is_finished()
    hit = req(1, length=9)
    hit.request_id = "cached-hit"
    fresh = req(3, length=6)
    scheduler.add_request(hit)
    scheduler.add_request(fresh)

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {fresh.request_id: 6}
    consume(scheduler, output)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {fresh.request_id: 1, hit.request_id: 1}
    assert hit.num_computed_tokens == 9


def test_decode_preemption_then_recompute_keeps_phase_and_accounting(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls, num_blocks=3)
    first, second = req(1, max_tokens=2), req(2, max_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)
    consume(scheduler, scheduler.schedule())
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {first.request_id: 1}
    assert second.request_id in output.preempted_req_ids
    assert second.status == RequestStatus.PREEMPTED
    assert second.num_computed_tokens == 0
    consume(scheduler, output)
    assert first.is_finished()

    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {second.request_id: 5}
    consume(scheduler, output)
    assert second.is_finished()
    assert not scheduler.requests
    assert not scheduler.running


def test_async_inflight_outputs_survive_a_prefill_interruption():
    scheduler = make_scheduler(PrefillFirstAsyncScheduler)
    old = req(1)
    scheduler.add_request(old)
    first = scheduler.schedule()
    first_result = runner_output(scheduler, first)
    assert old.num_output_placeholders == 1
    new = req(2)
    scheduler.add_request(new)
    second = scheduler.schedule()
    second_result = runner_output(scheduler, second)
    assert second.num_scheduled_tokens == {new.request_id: 4}
    assert old.num_output_placeholders == 1
    third = scheduler.schedule()
    third_result = runner_output(scheduler, third)
    assert third.num_scheduled_tokens == {old.request_id: 1, new.request_id: 1}
    assert old.num_output_placeholders == new.num_output_placeholders == 2
    for output, result in ((first, first_result), (second, second_result), (third, third_result)):
        scheduler.update_from_output(output, result)
    assert old.num_output_placeholders == new.num_output_placeholders == 0
    assert old.num_in_flight_tokens == new.num_in_flight_tokens == 0
    assert old.num_output_tokens == new.num_output_tokens == 2
    assert scheduler.current_step == 3


def test_paused_step_does_not_advance_requests_or_run_two_updates(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    request = start_decode(scheduler, req(1))
    scheduler._pause_state = PauseState.PAUSED_ALL
    before = scheduler.current_step
    output = scheduler.schedule()
    assert not output.num_scheduled_tokens
    assert scheduler.current_step == before + 1
    assert request.num_computed_tokens == 4
    assert scheduler.phase_batch_counts["idle"] == 1


def test_async_preempted_request_waits_for_inflight_output_before_recompute():
    scheduler = make_scheduler(PrefillFirstAsyncScheduler, num_blocks=3)
    first, second = req(1, max_tokens=2), req(2, max_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)
    prefill = scheduler.schedule()
    prefill_result = runner_output(scheduler, prefill)
    decode = scheduler.schedule()
    decode_result = runner_output(scheduler, decode)
    assert decode.num_scheduled_tokens == {first.request_id: 1}
    assert second.status == RequestStatus.PREEMPTED
    assert second.num_stale_output_tokens > 0
    pending = scheduler.schedule()
    pending_result = runner_output(scheduler, pending)
    assert second.request_id not in pending.num_scheduled_tokens
    scheduler.update_from_output(prefill, prefill_result)
    scheduler.update_from_output(decode, decode_result)
    scheduler.update_from_output(pending, pending_result)
    assert second.num_stale_output_tokens == 0
    resumed = scheduler.schedule()
    assert resumed.num_scheduled_tokens == {second.request_id: 5}
    consume(scheduler, resumed)
    assert not scheduler.requests


def test_hybrid_mamba_alignment_preserves_pure_batches_and_output_metadata(scheduler_cls):
    config = make_dyntra_test_config(max_num_batched_tokens=64, block_size=BLOCK_SIZE)
    config.kv_transfer_config = None
    config.scheduler_config.max_num_batched_tokens = 6
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.num_gpu_blocks = 128
    config.scheduler_config.async_scheduling = scheduler_cls is PrefillFirstAsyncScheduler
    kv_config = KVCacheConfig(
        num_blocks=128,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(block_size=BLOCK_SIZE, num_kv_heads=1, head_size=1, dtype=torch.float16),
            ),
            KVCacheGroupSpec(
                ["gdn"],
                MambaSpec(
                    block_size=BLOCK_SIZE,
                    shapes=((1, 1, 1),),
                    dtypes=(torch.float32,),
                    page_size_padded=16,
                    mamba_cache_mode="align",
                ),
            ),
        ],
    )
    scheduler = scheduler_cls(
        vllm_config=config,
        kv_cache_config=kv_config,
        structured_output_manager=StructuredOutputManager(config),
        block_size=BLOCK_SIZE,
    )
    scheduler.use_v2_model_runner = True
    decode = start_decode(scheduler, req(1))
    prefill = req(2, length=12)
    scheduler.add_request(prefill)
    assert scheduler.need_mamba_block_aligned_split
    for index in range(3):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {prefill.request_id: 4}
        assert decode.num_computed_tokens == 4 + index
        assert output.kv_connector_block_state is None
        consume(scheduler, output)
        if index < 2:
            decode_output = scheduler.schedule()
            assert decode_output.num_scheduled_tokens == {decode.request_id: 1}
            consume(scheduler, decode_output)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1, prefill.request_id: 1}
    consume(scheduler, output)


def test_scheduler_class_option_resolves_the_async_implementation():
    config = make_dyntra_test_config()
    config.scheduler_config.scheduler_cls = "vllm_ascend.core.prefill_first_scheduler.PrefillFirstAsyncScheduler"
    assert config.scheduler_config.get_scheduler_cls() is PrefillFirstAsyncScheduler


def test_prefill_only_chunks_continue_without_idle_decode_steps(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    prefill = req(1, length=20)
    scheduler.add_request(prefill)
    for size in (8, 8, 4):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {prefill.request_id: size}
        consume(scheduler, output)
    assert scheduler.phase_batch_counts == {"prefill": 3, "decode": 0, "idle": 0}


def test_decode_is_protected_when_earlier_prefill_exhausts_entire_budget(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    scheduler.scheduler_config.long_prefill_token_threshold = 4
    prefill, decode = req(1, length=20), req(2, length=4)
    scheduler.add_request(prefill)
    scheduler.add_request(decode)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {prefill.request_id: 4, decode.request_id: 4}
    consume(scheduler, output)
    scheduler.scheduler_config.long_prefill_token_threshold = 0
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {prefill.request_id: 8}
    consume(scheduler, output)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1}
    consume(scheduler, output)


def test_blocked_protected_decode_falls_back_to_prefill_without_idling(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    start_decode(scheduler, req(1))
    prefill = req(2)
    scheduler.add_request(prefill)
    consume(scheduler, scheduler.schedule())
    for running in scheduler.running:
        running.next_decode_eligible_step = scheduler.current_step + 100
    fresh = req(3, length=6)
    scheduler.add_request(fresh)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {fresh.request_id: 6}
    consume(scheduler, output)
    assert scheduler.phase_batch_counts["idle"] == 0


def test_pause_does_not_cancel_decode_protection(scheduler_cls):
    scheduler = make_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    scheduler.add_request(req(2))
    consume(scheduler, scheduler.schedule())
    waiting = req(3)
    scheduler.add_request(waiting)
    scheduler._pause_state = PauseState.PAUSED_ALL
    assert not scheduler.schedule().num_scheduled_tokens
    scheduler._pause_state = PauseState.UNPAUSED
    output = scheduler.schedule()
    assert decode.request_id in output.num_scheduled_tokens
    assert waiting.request_id not in output.num_scheduled_tokens
    assert set(output.num_scheduled_tokens.values()) == {1}


@pytest.mark.parametrize("setting", ["speculation", "dp", "pp"])
def test_unsupported_modes_fail_before_scheduler_initialization(scheduler_cls, setting):
    config = make_dyntra_test_config()
    if setting == "speculation":
        config.speculative_config = SimpleNamespace()
    elif setting == "dp":
        config.parallel_config.data_parallel_size = 2
    else:
        config.parallel_config.pipeline_parallel_size = 2
    with pytest.raises(ValueError, match="does not support|requires DP"):
        scheduler_cls(vllm_config=config)
