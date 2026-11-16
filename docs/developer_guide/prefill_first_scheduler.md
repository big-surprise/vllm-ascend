# Prefill-first scheduling

This opt-in scheduler follows SGLang's ordinary prefill-first selection: try a
prefill batch first, then run decode if no prefill compute can be admitted. It
does not fill a prefill batch's unused token budget with decode requests and does
not impose a fixed prefill/decode alternation.

## Enable

With asynchronous scheduling:

```bash
vllm serve MODEL \
    --async-scheduling \
    --scheduler-cls vllm_ascend.core.prefill_first_scheduler.PrefillFirstAsyncScheduler
```

For synchronous scheduling, select
`vllm_ascend.core.prefill_first_scheduler.PrefillFirstScheduler` and disable async
scheduling. Removing `--scheduler-cls` restores the ordinary scheduler.

The initial supported scope is non-speculative autoregressive serving with
DP=1 and PP=1. These constraints are checked at initialization. The scheduler
retains tensor parallelism, chunked prefill, prefix caching, Mamba alignment and
the upstream async placeholder/output accounting. Other custom schedulers
(profiling chunk, recompute, DyntraLB, batch-job-aware scheduling) are not combined
with this policy. Their automatic scheduler-class selection can override the CLI
class; verify the initialization log when enabling a policy.

## Scheduling behavior

Each scheduling step first traverses running requests and then waiting requests
for prefill work. If no compute tokens were scheduled, it repeats that traversal
for decode work. Allocation, preemption, request-slot limits, encoder constraints
and KV dependencies still apply. A waiting queue that cannot be admitted does not
prevent existing decodes from progressing.

The phase is determined after prefix-cache matching, token caps and Mamba block
alignment. More than one input token, or a first input token with no previous
state, belongs to the prefill pass. One input token with existing state belongs
to the decode pass. This matches the Ascend GDN handling of stateful one-token
prompt tails and full-prefix-cache hits. It deliberately distinguishes the
kernel execution phase from the request's lifetime label.

Unscheduled decodes stay in `running` and retain their state; they are counted
toward the request limit and remain visible to the ordinary KV/preemption logic.
The scheduler constructs one output and advances async accounting once per step.
It does not filter an already-finalized upstream scheduling result.

## Validation and performance

Run the scheduler tests with the repository's usual unit-test environment:

```bash
pytest -q tests/ut/core/test_prefill_first_scheduler.py
```

The tests use real scheduler requests and KV managers on CPU, including hybrid
Mamba caches. They cover prefills interrupting decodes, chunk continuation,
stateful one-token tails, prefix hits, full request slots, insufficient KV,
blocked requests, preemption/recomputation and asynchronous output delivery.
Hardware model accuracy, graph replay and throughput must also be verified on
the intended model before drawing performance conclusions.

The instance's `phase_batch_counts` counts prefill, decode and idle batches. Debug
logs contain `PrefillFirst batch=... requests=... tokens=...`. Backend metadata
should also be sampled when measuring actual GDN mixed-batch frequency. Keep
logging settings identical between compared runs and avoid per-layer logging.

Strict prefill preference can delay decode under sustained arrivals. A prefill
chunk can be long even when only one prefill batch is executed. Compare TTFT,
TPOT, end-to-end latency, throughput and cache hit rate at identical request-rate
and concurrency settings. Pure decode batches larger than 32 still fall back
from the current MegaGDN implementation; this scheduler does not change that
operator's batch limit.

## Upstream compatibility

The `schedule()` body is based on vLLM commit
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`. Its request-selection passes and
preemption rollback differ from upstream. KV block copies, Mamba state handoff,
encoder metadata, finished/preempted request notifications and async state
updates remain in the same single finalization step. Rebase the copied body and
rerun the tests when upgrading vLLM; do not assume compatibility across releases.
