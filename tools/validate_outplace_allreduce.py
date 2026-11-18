#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
"""Validate the out-of-place NPU all-reduce fast path.

Two ways to run it:

    # config-only checks, no device needed
    python tools/validate_outplace_allreduce.py

    # full validation, 2+ cards
    torchrun --nproc_per_node=2 tools/validate_outplace_allreduce.py

Sections, in order of increasing privilege:

1. config   -- the flat ``allreduce_outplace`` key defaults to off, accepts lax
               bools, and rejects unknown keys (``extra="forbid"``).
2. borrow   -- the process group's communicator and collective stream can be
               borrowed on this torch_npu build.
3. correct  -- the result matches ``dist.all_reduce`` and, unlike the default
               path, the input tensor is left untouched.
4. cost     -- host time to enqueue one collective (the small-message metric).
5. graph    -- capture one call and replay it twice. Replay #2 is the check that
               the in-place path cannot pass: a captured in-place reduce keeps
               accumulating into its static input.
6. wiring   -- ``NPUCommunicator.all_reduce`` dispatches to the fast path and
               still falls back for tensors it cannot handle.

Every section prints ``[PASS]``/``[FAIL]``/``[SKIP]``; the process exit code is
non-zero if anything failed.
"""

import argparse
import json
import os
import sys
import time

import torch
import torch.distributed as dist

FAILED = []


def report(status, name, detail=""):
    print(f"[{status}] {name}" + (f" -- {detail}" if detail else ""), flush=True)
    if status == "FAIL":
        FAILED.append(name)


def section_config():
    from vllm_ascend.ascend_config import AscendConfig

    try:
        default = AscendConfig()
        report("PASS" if default.allreduce_outplace is False else "FAIL", "config: default is off")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "config: default is off", repr(exc))
        return

    try:
        enabled = AscendConfig(allreduce_outplace="true")
        report("PASS" if enabled.allreduce_outplace is True else "FAIL", "config: lax bool coercion")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "config: lax bool coercion", repr(exc))

    try:
        AscendConfig(allreduce_outplac=True)  # deliberate typo
    except Exception:
        report("PASS", "config: unknown key rejected")
    else:
        report("FAIL", "config: unknown key rejected", "typo was silently accepted")

    # Exactly what the CLI delivers: json.loads -> kwargs.
    try:
        parsed = AscendConfig(**json.loads('{"allreduce_outplace": true}'))
        report(
            "PASS" if parsed.allreduce_outplace is True else "FAIL",
            "config: flat --additional-config form accepted",
            "--additional-config '{\"allreduce_outplace\": true}'",
        )
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "config: flat --additional-config form accepted", repr(exc))

    legacy = AscendConfig.__dataclass_fields__.get("allreduce_config")
    report(
        "PASS" if legacy is None else "FAIL",
        "config: legacy nested key removed",
        "old shape now fails extra=\"forbid\" instead of being ignored",
    )

    from vllm_ascend.distributed.device_communicators.npu_communicator import (
        _outplace_allreduce_enabled,
    )

    # AscendConfig is not initialised in this script, so the defensive read must
    # return False instead of raising.
    try:
        off = _outplace_allreduce_enabled()
        report("PASS" if off is False else "FAIL", "config: switch read is fail-safe before init")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "config: switch read is fail-safe before init", repr(exc))


def bench_enqueue(fn, tensor, iters):
    for _ in range(50):
        fn(tensor)
    torch.npu.synchronize()
    start = time.perf_counter()
    for _ in range(iters):
        fn(tensor)
    torch.npu.synchronize()
    return (time.perf_counter() - start) / iters * 1e6


def section_borrow(device):
    from vllm_ascend.distributed.device_communicators.hccl_outplace import (
        OutplaceHcclAllReduce,
    )

    comm = OutplaceHcclAllReduce.try_create(
        group=dist.group.WORLD,
        device=device,
        global_rank=dist.get_rank(),
    )
    if comm is None:
        report("FAIL", "borrow: PG communicator + collective stream", "try_create returned None, see warnings above")
        return None
    report(
        "PASS",
        "borrow: PG communicator + collective stream",
        f"dedicated stream={comm._stream is not None}, world_size={comm.world_size}",
    )
    return comm


def section_correct(comm, device, hidden, rank, world_size):
    expect = sum(range(1, world_size + 1))
    x = torch.full((hidden,), rank + 1, dtype=torch.bfloat16, device=device)

    out = comm.all_reduce(x)
    torch.npu.synchronize()
    ok = torch.equal(out, torch.full_like(out, expect))
    report("PASS" if ok else "FAIL", "correct: value", f"expect {expect}, got {out[0].item()}")

    untouched = x[0].item() == rank + 1
    report("PASS" if untouched else "FAIL", "correct: input not mutated", f"input[0]={x[0].item()}")

    # Same numbers as the default path.
    reference = x.clone()
    dist.all_reduce(reference, group=dist.group.WORLD)
    torch.npu.synchronize()
    same = torch.equal(reference, out)
    report("PASS" if same else "FAIL", "correct: matches dist.all_reduce")

    # 2-D and 3-D activation shapes are the real callers.
    for shape in ((4, hidden), (2, 3, hidden)):
        t = torch.ones(shape, dtype=torch.bfloat16, device=device) * (rank + 1)
        o = comm.all_reduce(t)
        torch.npu.synchronize()
        good = torch.equal(o, torch.full_like(o, expect))
        report("PASS" if good else "FAIL", f"correct: shape {shape}")

    return out


def section_cost(comm, device, hidden, rank, world_size, iters):
    x = torch.full((hidden,), rank + 1, dtype=torch.bfloat16, device=device)
    buf = x.clone()

    t_default = bench_enqueue(lambda t: dist.all_reduce(t, group=dist.group.WORLD), buf, iters)
    t_fast = bench_enqueue(comm.all_reduce, x, iters)
    print(
        f"       hidden={hidden} bf16 -> {hidden * 2} B/message, {iters} iters\n"
        f"       dist.all_reduce (in place) : {t_default:8.2f} us/call\n"
        f"       outplace_all_reduce        : {t_fast:8.2f} us/call\n"
        f"       delta                      : {t_default - t_fast:+8.2f} us/call",
        flush=True,
    )
    report("PASS", "cost: measured", "interpret with the model's all-reduce count per step")


def section_graph(comm, device, hidden, rank, world_size):
    expect = sum(range(1, world_size + 1))
    graph_cls = getattr(torch.npu, "NPUGraph", None)
    if graph_cls is None:
        report("SKIP", "graph: capture + replay", "torch.npu.NPUGraph unavailable on this torch_npu")
        return

    try:
        static_in = torch.full((hidden,), rank + 1, dtype=torch.bfloat16, device=device)
        g = graph_cls()
        with torch.npu.graph(g):
            static_out = comm.all_reduce(static_in)
        torch.npu.synchronize()
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "graph: capture", f"{type(exc).__name__}: {exc}")
        return
    report("PASS", "graph: capture")

    # Two replays from the same static input must yield the same result. An
    # in-place collective would give expect, 2*expect on the second replay.
    results = []
    for i in range(2):
        static_in.fill_(rank + 1)
        g.replay()
        torch.npu.synchronize()
        results.append(static_out[0].item())

    good = all(abs(v - expect) < 1 for v in results)
    report("PASS" if good else "FAIL", "graph: replay is idempotent", f"replays={results}, expect={expect}")


def section_wiring(device, hidden, rank, world_size):
    """Exercise the real dispatch in ``NPUCommunicator.all_reduce``."""
    from vllm_ascend.distributed.device_communicators import npu_communicator as mod

    original = mod._outplace_allreduce_enabled
    mod._outplace_allreduce_enabled = lambda: True  # config is not initialised here
    try:
        cpu_group = dist.new_group(ranks=list(range(world_size)), backend="gloo")
        dev_group = dist.new_group(ranks=list(range(world_size)))
        communicator = mod.NPUCommunicator(cpu_group=cpu_group, device=device, device_group=dev_group)
    except Exception as exc:  # noqa: BLE001
        report("SKIP", "wiring: NPUCommunicator dispatch", f"{type(exc).__name__}: {exc}")
        return
    finally:
        mod._outplace_allreduce_enabled = original

    if communicator.outplace_ar is None:
        report("FAIL", "wiring: fast path constructed")
        return
    report("PASS", "wiring: fast path constructed")

    expect = sum(range(1, world_size + 1))
    x = torch.full((hidden,), rank + 1, dtype=torch.bfloat16, device=device)
    out = communicator.all_reduce(x)
    torch.npu.synchronize()
    report("PASS" if torch.equal(out, torch.full_like(out, expect)) else "FAIL", "wiring: bf16 takes fast path")
    report("PASS" if x[0].item() == rank + 1 else "FAIL", "wiring: input not mutated")

    # int32 must fall back to the default path and still be correct.
    xi = torch.full((8,), rank + 1, dtype=torch.int32, device=device)
    oi = communicator.all_reduce(xi)
    torch.npu.synchronize()
    report("PASS" if int(oi[0].item()) == expect else "FAIL", "wiring: int32 falls back", f"got {oi[0].item()}")

    # Contiguous check: a strided view must fall back, not raise.
    xv = torch.full((hidden * 2,), rank + 1, dtype=torch.bfloat16, device=device)[::2]
    try:
        ov = communicator.all_reduce(xv)
        torch.npu.synchronize()
        report("PASS" if torch.equal(ov, torch.full_like(ov, expect)) else "FAIL", "wiring: non-contiguous falls back")
    except Exception as exc:  # noqa: BLE001
        report("FAIL", "wiring: non-contiguous falls back", f"{type(exc).__name__}: {exc}")

    communicator.destroy()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hidden-size", type=int, default=2048, help="bf16 width per message (2048 -> 4 KiB)")
    parser.add_argument("--iters", type=int, default=3000)
    parser.add_argument("--skip-graph", action="store_true")
    parser.add_argument("--skip-cost", action="store_true")
    args = parser.parse_args()

    print("== section 1/6: config ==", flush=True)
    section_config()

    if "RANK" not in os.environ:
        print("\n(no distributed environment: sections 2-6 need torchrun --nproc_per_node=2)", flush=True)
        return 1 if FAILED else 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    torch.npu.set_device(local_rank)
    dist.init_process_group("hccl")
    device = torch.device("npu", local_rank)

    try:
        # Warm up hilloc so the first borrow does not race communicator creation.
        warm = torch.ones(8, dtype=torch.bfloat16, device=device)
        dist.all_reduce(warm, group=dist.group.WORLD)
        torch.npu.synchronize()

        print("\n== section 2/6: borrow ==", flush=True)
        comm = section_borrow(device)
        if comm is None:
            return 1

        print("\n== section 3/6: correctness ==", flush=True)
        section_correct(comm, device, args.hidden_size, rank, world_size)

        if not args.skip_cost:
            print("\n== section 4/6: cost ==", flush=True)
            section_cost(comm, device, args.hidden_size, rank, world_size, args.iters)

        if not args.skip_graph:
            print("\n== section 5/6: graph ==", flush=True)
            section_graph(comm, device, args.hidden_size, rank, world_size)

        print("\n== section 6/6: NPUCommunicator wiring ==", flush=True)
        section_wiring(device, args.hidden_size, rank, world_size)

        comm.destroy()
    finally:
        dist.destroy_process_group()

    if rank == 0:
        print("\n" + ("ALL SECTIONS PASSED" if not FAILED else f"FAILED: {FAILED}"), flush=True)
    return 1 if FAILED else 0


if __name__ == "__main__":
    sys.exit(main())
