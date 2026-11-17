# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project
"""Run bounded decode bursts, then admit prefill into ordinary mixed batches."""

import math
import time

from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.request import Request, RequestStatus

from vllm_ascend import envs
from vllm_ascend.core.prefill_first_scheduler import PrefillFirstScheduler

logger = init_logger(__name__)


class DecodePriorityScheduler(PrefillFirstScheduler):
    """Delay prefill while decode progresses, with step and elapsed-time caps.

    A release uses the ordinary running-then-waiting traversal and may mix
    prefill with decode. No-decode and blocked-decode cases release immediately.
    The time cap bounds policy deferral, not resource waits or execution time.
    """

    def __init__(self, vllm_config: VllmConfig, *args, **kwargs):
        self.decode_burst_steps = envs.VLLM_ASCEND_DECODE_BURST_STEPS
        self.prefill_max_delay_ms = envs.VLLM_ASCEND_PREFILL_MAX_DELAY_MS
        if self.decode_burst_steps < 0:
            raise ValueError("VLLM_ASCEND_DECODE_BURST_STEPS must be >= 0")
        if not math.isfinite(self.prefill_max_delay_ms) or self.prefill_max_delay_ms < 0:
            raise ValueError("VLLM_ASCEND_PREFILL_MAX_DELAY_MS must be finite and >= 0")
        self._prefill_pending_since: dict[Request, float] = {}
        self._decode_steps_with_pending_prefill = 0
        self._policy_now = 0.0
        self._oldest_prefill_wait_ms = 0.0
        self._policy_reason = "initial"
        super().__init__(vllm_config, *args, **kwargs)
        self.phase_batch_counts["mixed"] = 0

    def _log_policy(self) -> None:
        logger.info(
            "DecodePriorityScheduler enabled: decode_burst_steps=%d prefill_max_delay_ms=%.3f; "
            "prefill release permits mixed batches",
            self.decode_burst_steps,
            self.prefill_max_delay_ms,
        )

    @staticmethod
    def _policy_clock() -> float:
        return time.monotonic()

    def _pending_prefills(self) -> set[Request]:
        pending = set()
        for request in self.running:
            tokens = request.num_tokens_with_spec + request.num_output_placeholders - request.num_computed_tokens
            if tokens > 0 and self._is_prefill_work(request.num_computed_tokens, tokens):
                pending.add(request)
        # Readiness promotion remains in the scheduling pass. Blocked statuses
        # start their policy timer once they become WAITING/PREEMPTED.
        for queue in (self.waiting, self.skipped_waiting):
            for request in queue:
                if request.status in (RequestStatus.WAITING, RequestStatus.PREEMPTED):
                    pending.add(request)
        return pending

    def _get_phase_order(self) -> tuple[bool | None, ...]:
        self._policy_now = self._policy_clock()
        pending = self._pending_prefills()
        self._prefill_pending_since = {
            request: self._prefill_pending_since.get(request, self._policy_now) for request in pending
        }
        self._oldest_prefill_wait_ms = (
            max(0.0, (self._policy_now - min(self._prefill_pending_since.values())) * 1000) if pending else 0.0
        )
        if not pending:
            self._decode_steps_with_pending_prefill = 0
            self._policy_reason = "no_pending_prefill"
        elif not any(self._has_pending_decode(request) for request in self.running):
            self._policy_reason = "no_running_decode"
        elif self._oldest_prefill_wait_ms >= self.prefill_max_delay_ms:
            self._policy_reason = "prefill_wait_limit"
        elif self._decode_steps_with_pending_prefill >= self.decode_burst_steps:
            self._policy_reason = "decode_step_limit"
        else:
            self._policy_reason = "decode_burst"
            # None is the ordinary mixed-admission pass, used only if the
            # decode-only pass cannot schedule any compute tokens.
            return (False, None)
        return (None,)

    def _record_phase(self, batch_kind: str, decode_waiting: bool, num_scheduled_tokens: dict[str, int]) -> None:
        made_prefill_progress = False
        for req_id in num_scheduled_tokens:
            request = self.requests[req_id]
            if request in self._prefill_pending_since:
                made_prefill_progress = True
                # Reset only this request: admitting a younger request must
                # not erase the oldest unserved request's deadline.
                self._prefill_pending_since[request] = self._policy_now
        if made_prefill_progress:
            self._decode_steps_with_pending_prefill = 0
        elif batch_kind == "decode" and self._prefill_pending_since:
            self._decode_steps_with_pending_prefill = min(
                self._decode_steps_with_pending_prefill + 1, self.decode_burst_steps
            )
        # Idle or paused steps do not consume/reset the step allowance. Time
        # continues to advance so an overdue prefill is released on resume.
        num_prefills = sum(
            self._is_prefill_work(self.requests[req_id].num_computed_tokens, tokens)
            for req_id, tokens in num_scheduled_tokens.items()
        )
        logger.debug(
            "DecodePriority batch=%s reason=%s prefill_reqs=%d decode_reqs=%d "
            "tokens=%d deferred_decode_steps=%d oldest_prefill_wait_ms=%.3f",
            batch_kind,
            self._policy_reason,
            num_prefills,
            len(num_scheduled_tokens) - num_prefills,
            sum(num_scheduled_tokens.values()),
            self._decode_steps_with_pending_prefill,
            self._oldest_prefill_wait_ms,
        )


class DecodePriorityAsyncScheduler(AsyncScheduler, DecodePriorityScheduler):
    """Keep upstream async placeholder and output-consumption semantics."""
