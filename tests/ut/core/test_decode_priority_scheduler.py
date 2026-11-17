# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Bound deliberate prefill deferral while preserving running decode progress.

Use the real scheduler and KV manager with a controlled monotonic policy clock;
assert scheduled batches and consumed output state, without loading a model.
"""

from types import SimpleNamespace

import pytest
import torch
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.interface import PauseState
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheGroupSpec, MambaSpec
from vllm.v1.request import RequestStatus
from vllm.v1.structured_output import StructuredOutputManager

from tests.ut.core.test_dyntra_lb_scheduler import make_dyntra_test_config
from tests.ut.core.test_prefill_first_scheduler import (
    BLOCK_SIZE,
    consume,
    make_scheduler,
    req,
    runner_output,
    start_decode,
)
from vllm_ascend.ascend_config import clear_ascend_config
from vllm_ascend.core.decode_priority_scheduler import DecodePriorityAsyncScheduler, DecodePriorityScheduler


@pytest.fixture(autouse=True)
def clean_policy_config(monkeypatch):
    clear_ascend_config()
    monkeypatch.setenv("VLLM_ASCEND_DECODE_BURST_STEPS", "4")
    monkeypatch.setenv("VLLM_ASCEND_PREFILL_MAX_DELAY_MS", "20")
    yield
    clear_ascend_config()


@pytest.fixture(params=[DecodePriorityScheduler, DecodePriorityAsyncScheduler])
def scheduler_cls(request):
    return request.param


def setup_scheduler(scheduler_cls, **kwargs):
    scheduler = make_scheduler(scheduler_cls, **kwargs)
    clock = SimpleNamespace(now=100.0)
    scheduler._policy_clock = lambda: clock.now
    return scheduler, clock


@pytest.mark.parametrize("use_v2", [False, True])
def test_four_decode_steps_then_prefill_joins_decode_in_one_batch(scheduler_cls, use_v2):
    scheduler, clock = setup_scheduler(scheduler_cls, use_v2=use_v2)
    decode = start_decode(scheduler, req(1))
    prefill = req(2, length=6)
    scheduler.add_request(prefill)
    old_block_ids = scheduler.kv_cache_manager.get_block_ids(decode.request_id)
    for _ in range(4):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {decode.request_id: 1}
        assert prefill.num_computed_tokens == 0
        consume(scheduler, output)
        clock.now += 0.001
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1, prefill.request_id: 6}
    assert scheduler._policy_reason == "decode_step_limit"
    assert scheduler.kv_cache_manager.get_block_ids(decode.request_id)[0][:len(old_block_ids[0])] == old_block_ids[0]
    consume(scheduler, output)
    assert scheduler.phase_batch_counts == {"prefill": 1, "decode": 4, "idle": 0, "mixed": 1}
    assert decode.num_output_tokens == 6
    assert prefill.num_output_tokens == 1


def test_time_limit_releases_before_step_limit(scheduler_cls):
    scheduler, clock = setup_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    prefill = req(2)
    scheduler.add_request(prefill)
    consume(scheduler, scheduler.schedule())
    clock.now += 0.021
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1, prefill.request_id: 4}
    assert scheduler._policy_reason == "prefill_wait_limit"


@pytest.mark.parametrize("setting", ["VLLM_ASCEND_DECODE_BURST_STEPS", "VLLM_ASCEND_PREFILL_MAX_DELAY_MS"])
def test_zero_setting_disables_deliberate_deferral(scheduler_cls, monkeypatch, setting):
    monkeypatch.setenv(setting, "0")
    scheduler, _ = setup_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    prefill = req(2)
    scheduler.add_request(prefill)
    assert scheduler.schedule().num_scheduled_tokens == {decode.request_id: 1, prefill.request_id: 4}


def test_configurable_two_step_burst(scheduler_cls, monkeypatch):
    monkeypatch.setenv("VLLM_ASCEND_DECODE_BURST_STEPS", "2")
    monkeypatch.setenv("VLLM_ASCEND_PREFILL_MAX_DELAY_MS", "10")
    scheduler, clock = setup_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    prefill = req(2)
    scheduler.add_request(prefill)
    for _ in range(2):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {decode.request_id: 1}
        consume(scheduler, output)
        clock.now += 0.001
    assert prefill.request_id in scheduler.schedule().num_scheduled_tokens


def test_prefill_only_runs_immediately_and_chunks_keep_progressing(scheduler_cls):
    scheduler, _ = setup_scheduler(scheduler_cls)
    prefill = req(1, length=20)
    scheduler.add_request(prefill)
    for size in (8, 8, 4):
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {prefill.request_id: size}
        consume(scheduler, output)
    assert scheduler.phase_batch_counts == {"prefill": 3, "decode": 0, "idle": 0, "mixed": 0}


def test_no_runnable_decode_falls_back_in_same_step(scheduler_cls):
    scheduler, _ = setup_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    decode.next_decode_eligible_step = scheduler.current_step + 100
    prefill = req(2)
    scheduler.add_request(prefill)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {prefill.request_id: 4}
    assert scheduler.phase_batch_counts["idle"] == 0


def test_wait_limit_does_not_override_full_request_slots(scheduler_cls):
    scheduler, clock = setup_scheduler(scheduler_cls, max_seqs=1)
    decode = start_decode(scheduler, req(1, max_tokens=8))
    waiting = req(2)
    scheduler.add_request(waiting)
    consume(scheduler, scheduler.schedule())
    clock.now += 0.021
    output = scheduler.schedule()
    assert scheduler._policy_reason == "prefill_wait_limit"
    assert output.num_scheduled_tokens == {decode.request_id: 1}
    assert waiting.status == RequestStatus.WAITING
    consume(scheduler, output)
    while not decode.is_finished():
        consume(scheduler, scheduler.schedule())
    assert waiting.request_id in scheduler.schedule().num_scheduled_tokens


def test_wait_limit_does_not_stall_decode_when_prefill_cannot_fit_kv(scheduler_cls):
    scheduler, clock = setup_scheduler(scheduler_cls, num_blocks=4)
    decode = start_decode(scheduler, req(1))
    waiting = req(2, length=16)
    scheduler.add_request(waiting)
    consume(scheduler, scheduler.schedule())
    clock.now += 0.021
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1}
    assert waiting.status == RequestStatus.WAITING


def test_serving_one_prefill_does_not_reset_other_waiting_deadlines(scheduler_cls):
    scheduler, clock = setup_scheduler(scheduler_cls, budget=5)
    start_decode(scheduler, req(1))
    first, second = req(2), req(3)
    scheduler.add_request(first)
    scheduler.add_request(second)
    consume(scheduler, scheduler.schedule())
    clock.now += 0.021
    output = scheduler.schedule()
    assert first.request_id in output.num_scheduled_tokens
    assert second.request_id not in output.num_scheduled_tokens
    consume(scheduler, output)
    assert scheduler._prefill_pending_since[second] == 100.0
    output = scheduler.schedule()
    assert second.request_id in output.num_scheduled_tokens
    assert scheduler._policy_reason == "prefill_wait_limit"


def test_finished_or_aborted_prefills_do_not_keep_stale_timers(scheduler_cls):
    scheduler, clock = setup_scheduler(scheduler_cls)
    start_decode(scheduler, req(1))
    old = req(2)
    scheduler.add_request(old)
    consume(scheduler, scheduler.schedule())
    scheduler.finish_requests([old.request_id], RequestStatus.FINISHED_ABORTED)
    clock.now += 1
    consume(scheduler, scheduler.schedule())
    assert not scheduler._prefill_pending_since
    new = req(3)
    scheduler.add_request(new)
    output = scheduler.schedule()
    assert new.request_id not in output.num_scheduled_tokens
    assert scheduler._oldest_prefill_wait_ms == 0


def test_pause_keeps_deadline_without_consuming_steps(scheduler_cls):
    scheduler, clock = setup_scheduler(scheduler_cls)
    decode = start_decode(scheduler, req(1))
    prefill = req(2)
    scheduler.add_request(prefill)
    consume(scheduler, scheduler.schedule())
    scheduler._pause_state = PauseState.PAUSED_ALL
    for _ in range(3):
        assert not scheduler.schedule().num_scheduled_tokens
    assert scheduler._decode_steps_with_pending_prefill == 1
    clock.now += 0.021
    scheduler._pause_state = PauseState.UNPAUSED
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1, prefill.request_id: 4}


def test_prefix_hit_one_token_can_join_decode_without_waiting_for_release(scheduler_cls):
    scheduler, _ = setup_scheduler(scheduler_cls, budget=16)
    priming = req(1, length=9, max_tokens=1)
    scheduler.add_request(priming)
    consume(scheduler, scheduler.schedule())
    decode = start_decode(scheduler, req(2))
    hit = req(1, length=9)
    hit.request_id = "prefix-hit"
    scheduler.add_request(hit)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {decode.request_id: 1, hit.request_id: 1}
    assert hit.num_computed_tokens == 9


def test_async_inflight_outputs_survive_decode_burst_and_mixed_release():
    scheduler, clock = setup_scheduler(DecodePriorityAsyncScheduler)
    decode = req(1)
    scheduler.add_request(decode)
    output = scheduler.schedule()
    pending = [(output, runner_output(scheduler, output))]
    prefill = req(2)
    scheduler.add_request(prefill)
    for i in range(5):
        output = scheduler.schedule()
        pending.append((output, runner_output(scheduler, output)))
        assert decode.request_id in output.num_scheduled_tokens
        assert (prefill.request_id in output.num_scheduled_tokens) == (i == 4)
        if len(pending) >= 2:
            previous, result = pending.pop(0)
            scheduler.update_from_output(previous, result)
        clock.now += 0.001
    for previous, result in pending:
        scheduler.update_from_output(previous, result)
    assert decode.num_output_placeholders == prefill.num_output_placeholders == 0
    assert decode.num_in_flight_tokens == prefill.num_in_flight_tokens == 0
    assert decode.num_output_tokens == 6
    assert prefill.num_output_tokens == 1


def test_decode_retraction_and_recompute_progress_with_limited_kv(scheduler_cls):
    scheduler, _ = setup_scheduler(scheduler_cls, num_blocks=3)
    first, second = req(1, max_tokens=2), req(2, max_tokens=2)
    scheduler.add_request(first)
    scheduler.add_request(second)
    consume(scheduler, scheduler.schedule())
    output = scheduler.schedule()
    assert second.request_id in output.preempted_req_ids
    consume(scheduler, output)
    output = scheduler.schedule()
    assert output.num_scheduled_tokens == {second.request_id: 5}
    consume(scheduler, output)
    assert not scheduler.requests


def test_hybrid_mamba_chunks_are_deferred_then_released_with_decode(scheduler_cls):
    config = make_dyntra_test_config(max_num_batched_tokens=64, block_size=BLOCK_SIZE)
    config.kv_transfer_config = None
    config.scheduler_config.max_num_batched_tokens = 6
    config.cache_config.mamba_cache_mode = "align"
    config.cache_config.num_gpu_blocks = 128
    config.scheduler_config.async_scheduling = issubclass(scheduler_cls, AsyncScheduler)
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
    scheduler._policy_clock = lambda: 100.0
    decode = start_decode(scheduler, req(1, max_tokens=30))
    prefill = req(2, length=12)
    scheduler.add_request(prefill)
    for _ in range(3):
        for _ in range(4):
            output = scheduler.schedule()
            assert output.num_scheduled_tokens == {decode.request_id: 1}
            consume(scheduler, output)
        output = scheduler.schedule()
        assert output.num_scheduled_tokens == {decode.request_id: 1, prefill.request_id: 4}
        assert output.kv_connector_block_state is None
        consume(scheduler, output)
    assert prefill.num_computed_tokens == 12
    assert scheduler.phase_batch_counts["mixed"] == 3


@pytest.mark.parametrize("setting,value", [
    ("VLLM_ASCEND_DECODE_BURST_STEPS", "-1"),
    ("VLLM_ASCEND_DECODE_BURST_STEPS", "1.5"),
    ("VLLM_ASCEND_PREFILL_MAX_DELAY_MS", "-1"),
    ("VLLM_ASCEND_PREFILL_MAX_DELAY_MS", "nan"),
    ("VLLM_ASCEND_PREFILL_MAX_DELAY_MS", "inf"),
])
def test_invalid_policy_limits_fail_early(monkeypatch, setting, value):
    monkeypatch.setenv(setting, value)
    with pytest.raises(ValueError):
        DecodePriorityScheduler(vllm_config=SimpleNamespace())


def test_async_class_is_resolved_by_scheduler_option():
    config = make_dyntra_test_config()
    config.scheduler_config.scheduler_cls = "vllm_ascend.core.decode_priority_scheduler.DecodePriorityAsyncScheduler"
    assert config.scheduler_config.get_scheduler_cls() is DecodePriorityAsyncScheduler
