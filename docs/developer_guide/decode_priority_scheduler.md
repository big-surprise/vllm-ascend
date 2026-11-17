# Decode priority with bounded prefill delay

This experimental policy favors consecutive decode steps to reduce interference
with ongoing generation. It defers prefill deliberately while decode can run,
then permits ordinary mixed admission when the step or time limit is reached.
It aims to trade some TTFT for TPOT; performance improvement is not guaranteed.

## Enable

```bash
export VLLM_ASCEND_DECODE_BURST_STEPS=4
export VLLM_ASCEND_PREFILL_MAX_DELAY_MS=20
vllm serve MODEL \
    --async-scheduling \
    --scheduler-cls vllm_ascend.core.decode_priority_scheduler.DecodePriorityAsyncScheduler
```

For synchronous execution, disable async scheduling and use `DecodePriorityScheduler`.
The two environment variables are read once at scheduler initialization:

| Variable | Default | Accepted values |
| --- | --- | --- |
| `VLLM_ASCEND_DECODE_BURST_STEPS` | 4 | Integer >= 0 |
| `VLLM_ASCEND_PREFILL_MAX_DELAY_MS` | 20 | Finite number >= 0 |

Setting either to zero disables deliberate prefill deferral and permits ordinary
mixed admission each step. Removing `--scheduler-cls` restores the upstream
scheduler, which is the preferred baseline for comparisons.

## Policy

When decode and prefill are pending, run up to four successful pure-decode
batches before allowing prefill again. At each scheduling decision, also check
the oldest prefill timer. If 20 ms has elapsed, release prefill early. The release
pass schedules running requests before waiting requests and can combine decode
and prefill in one model execution. This is different from strict P/D separation.

With the default limits and a clock that has not yet reached the time cap:

```text
D -> D -> D -> D -> [D + P] -> D -> D -> ...
```

If only prefill can run, it proceeds immediately. If a decode-only attempt cannot
schedule any compute, the scheduler falls through to ordinary admission in the
same step. It does not finalize an empty result and then call the scheduler again.
Idle/paused steps do not consume or reset the decode-step allowance.

Each pending request has a monotonic timer, starting when the scheduler first
observes it in a schedulable waiting state or with unfinished running prefill.
Blocked grammar/KV/streaming statuses do not start a policy timer until ready.
Scheduled progress resets only that request's timer; other waiting requests keep
their original age. Aborted/finished/no-longer-prefilling requests are pruned.
Running prefill chunks are included, so TTFT may include waits between chunks.
Stateful one-token prompt tails are allowed on the decode kernel path.

The time cap means **stop deliberately deferring** at the next scheduling
decision. It is not a 20 ms TTFT guarantee or a device execution deadline.
It cannot interrupt submitted work or bypass request-slot, token-budget, KV,
encoder or other dependency constraints. With async scheduling, a limit on
scheduler submissions also does not translate directly to elapsed device time.

The initial supported scope is non-speculative autoregressive serving with
DP=1 and PP=1. The policy shares the inspected vLLM revision's schedule body
through `PrefillFirstScheduler`, but overrides phase selection and policy
accounting. The previous prefill-first policy remains separately selectable.
Other automatic custom scheduler selections must not override this class.

## Observability and evaluation

Startup logs print the resolved limits. DEBUG logs include the actual batch kind,
policy reason (`decode_burst`, `decode_step_limit`, `prefill_wait_limit`, etc.),
prefill/decode request counts, deferred decode steps and oldest prefill wait.
The existing `--enable-logging-iteration-details` remains available, with its
request-lifecycle classification and CPU-time caveats. This policy's phase
classification follows the GDN-compatible stateful-one-token rule.

Start with 4 steps / 20 ms, then compare 2 / 10 ms and the original scheduler.
Use identical input data, rate, concurrency cap, cache reset/warmup, logging,
MegaGDN switch and code/model builds. Compare mean and tail TTFT, TPOT/ITL, E2E,
successful throughput and waiting queue. Bursts can accumulate prefills and make
release batches longer; lower TPOT alone does not establish overall improvement.

The policy does not enable MegaGDN or change its supported batch sizes. Hardware
accuracy, actual graph/kernel execution and performance require model-level
validation. CPU tests cover scheduling behavior and cache/state accounting only.

## Tests

```bash
pytest -q tests/ut/core/test_decode_priority_scheduler.py \
    tests/ut/core/test_prefill_first_scheduler.py \
    tests/ut/core/test_scheduler_connector_block_state.py
```

Tests cover step/time release, parameter validation, no-decode and blocked-decode
fallback, independent request deadlines, cancellation, pause/resume, KV pressure,
prefix hits, async output delivery, hybrid Mamba alignment and mixed release.
The shared schedule body is pinned to vLLM
`ced6857afa0ea7b2e3f0846a62e1394e90f15607`; rebase and revalidate on upgrades.
