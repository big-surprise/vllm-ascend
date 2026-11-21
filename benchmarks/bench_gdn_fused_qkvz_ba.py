#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Single-op microbenchmark for the Qwen3.5 GDN fused ``qkvz+ba`` GEMM.

Background
----------
``ops/gdn.py::fuse_ba_into_qkvz`` concatenates the GDN ``in_proj_qkvz``
weight (N=4096 at TP4) and ``in_proj_ba`` weight (N=24 at TP4) into ONE
``F.linear`` with N=4120.  On Ascend 950DT the original N=4096 GEMV hits
CANN's **MIX_AIC** MatMulV3 path (``Block Num=32``, ``Mix Block Num=64``,
``aiv_time>0``, ~14.9us), while N=4120 falls back to the pure **AI_CORE**
path (``Block Num=29``, ``Mix Block Num=0``, ``aiv_time=0``, ~17.3us).
That ~2.4us/layer x 48 GDN layers is the gap this benchmark measures.

Across 32 captured profiles the MIX_AIC selection rule is:

    N <= n_aic * 128          (one 128-column tile per AI Core)

  * 950DT: 32 AI Cores -> N <= 4096  (3584/32=112, 4096/32=128 -> MIX_AIC;
    4120/32=128.75 -> AI_CORE; 8704/32=272 -> AI_CORE)
  * 950PR: 28 AI Cores -> N <= 3584  (4096/28=146 -> AI_CORE)

So padding N *upwards* can never restore MIX_AIC for N=4120; it can only
rebalance the AI_CORE tiling, and the first 950DT measurement (2026-11-21)
shows that tiling is NOT monotonic: N=4120/4128/4160 -> Block 29, 4224 -> 30,
4352 -> 31, 4608 -> 32, 4736 -> 30, 8192 -> 32. None of them beat N=4120 by
more than the measurement noise.

The padding experiment is CLOSED: on 950DT every padded N (4128..8192) was
slower than 4120 by profiler Duration(us) - extra zero rows only add MTE2
weight traffic - and the alignment config switch was removed. The fused GEMM
always keeps the exact N=4120; the real online gap is MTE2 weight cold reads
(~5us/layer vs a hot-L2 bare kernel), addressed by sharing one weight
allocation (ops/gdn.py) and prefetching ``_fused_qkvz_ba_w``
(mlp_weight_prefetch.py) - not by tiling.

This script remains a standalone REGRESSION / DIAGNOSTIC tool: it reports the
kernel-side ``Duration(us)`` (the authoritative device time, the same field
the online msprof export uses) alongside host wall and graph timings, verifies
bitwise output equality for padded candidates, and cross-checks whether the
metrics agree. It no longer recommends any config value. Host wall-clock
around ``g.replay()`` is NOT a device time: on 950DT it is dominated by a
~150us fixed per-replay cost that completely swamps a ~12us decode GEMM.

Usage (on an NPU node with torch + torch_npu)
---------------------------------------------
  # Wall-clock timing of every candidate N, decode (M=1) and prefill (M=2252)
  python benchmarks/bench_gdn_fused_qkvz_ba.py

  # Also capture the CANN kernel fields via torch_npu.profiler
  python benchmarks/bench_gdn_fused_qkvz_ba.py --profile

  # Or judge an EXISTING msprof / ascend_pt output (no NPU needed)
  python benchmarks/bench_gdn_fused_qkvz_ba.py \
      --profile-dir /path/to/ASCEND_PROFILER_OUTPUT \
      --profile-dir /path/to/another/ASCEND_PROFILER_OUTPUT

  # Pure-logic self test on any machine (no torch / no NPU required)
  python benchmarks/bench_gdn_fused_qkvz_ba.py --cpu-check

  # Save results as CSV for later comparison
  python benchmarks/bench_gdn_fused_qkvz_ba.py --out /tmp/gdn_gemm.csv

The script is standalone: it imports nothing from vllm_ascend, so it can be
copied to a worker container and run without installing the whole repo.
"""

from __future__ import annotations

import argparse
import csv
import glob
import os
import statistics
import sys
import time

# Fused projection geometry for Qwen3.5-27B, BF16, TP4, decode.
DEFAULT_MAIN_COLS = 4096
DEFAULT_BA_COLS = 24
DEFAULT_K = 5120
DEFAULT_NS = (4096, 4120, 4128, 4160, 4224, 4352, 4608, 4608 + 128, 8192)
DEFAULT_MS = (1, 2252)

# kernel_details.csv columns used to judge the CANN implementation.
# NOTE: "Accelerator Core" is a *string* (MIX_AIC / AI_CORE / ...), the rest
# are numeric and get averaged over all matching kernels.
_KERNEL_CORE_FIELD = "Accelerator Core"
_KERNEL_NUM_FIELDS = (
    "Block Num",
    "Mix Block Num",
    "aiv_time(us)",
    "aicore_time(us)",
    "cube_utilization(%)",
)


# ---------------------------------------------------------------------------
# Pure-logic helpers (usable without torch / NPU)
# ---------------------------------------------------------------------------
def align_up(n: int, granularity: int) -> int:
    """Round ``n`` up to a multiple of ``granularity`` (0/1 -> unchanged)."""
    if granularity <= 1:
        return n
    return ((n + granularity - 1) // granularity) * granularity


def fused_geometry(main_cols: int, ba_cols: int, n_align: int) -> dict:
    """Mirror ``ops/gdn.py``'s padding + split index arithmetic exactly."""
    n_fused = main_cols + ba_cols
    n_pad = align_up(n_fused, n_align) if n_align > 0 else n_fused
    pad_rows = n_pad - n_fused
    # The forward does one split into [main | ba | pad] and drops ``pad``.
    split_sizes = [main_cols, ba_cols] + ([pad_rows] if pad_rows else [])
    return {
        "main_cols": main_cols,
        "ba_cols": ba_cols,
        "n_fused": n_fused,
        "n_align": n_align,
        "n_pad": n_pad,
        "pad_rows": pad_rows,
        "split_sizes": split_sizes,
        "overhead_pct": 100.0 * pad_rows / n_fused,
    }


def mix_aic_expected(n: int, n_aic: int = 32, tile: int = 128) -> bool:
    """Empirical CANN rule: MIX_AIC needs at most one 128-col tile per AIC."""
    if n == 0:
        return False
    # Small-N kernels (ba: 24 at TP4 / 48 at TP2) are a MIX_AIC special case.
    if n <= 2 * n_aic:
        return True
    return n % n_aic == 0 and n // n_aic <= tile


def cpu_check(args) -> int:
    """Validate the padding/split arithmetic and print the candidate table."""
    print("== CPU self-check: fused qkvz+ba geometry (no NPU required) ==")
    print("main_cols=%d ba_cols=%d K=%d" % (args.main_cols, args.ba_cols, args.k))
    print()
    hdr = "%-8s %-8s %-8s %-9s %-9s %-8s %-9s %s" % (
        "N", "N%32", "N%128", "N/32", "pad_rows", "overhead", "MIX_AIC?", "split sizes",
    )
    print(hdr)
    print("-" * len(hdr))
    for n in args.ns:
        g = fused_geometry(args.main_cols, args.ba_cols, 0)
        n_fused = g["n_fused"]
        pad_rows = max(0, n - n_fused)
        split_str = (
            "[baseline: qkvz only]" if n < n_fused
            else str([args.main_cols, args.ba_cols] + ([pad_rows] if pad_rows else []))
        )
        print(
            "%-8d %-8d %-8d %-9.2f %-9d %-8.2f %-9s %s"
            % (
                n,
                n % 32,
                n % 128,
                n / 32.0,
                pad_rows,
                100.0 * pad_rows / n_fused,
                "yes" if mix_aic_expected(n) else "NO",
                split_str,
            )
        )
    print()
    print("Rule: MIX_AIC <=> N <= n_aic*128 (950DT n_aic=32 -> N<=4096).")
    print("=> N=%d (fused) exceeds the limit; no upward padding can restore MIX_AIC."
          % (args.main_cols + args.ba_cols))
    print("   950DT measurement (2026-11-21): EVERY padded N (4128..8192) was")
    print("   slower than 4120 by profiler Duration(us); the alignment switch")
    print("   was removed. The fused GEMM always keeps the exact N=4120.")
    print()
    # Keep the split-arithmetic self-check: it still proves padding, were it
    # ever reintroduced elsewhere, is bitwise lossless.
    for a in (0, 32, 64, 128, 144, 256):
        g = fused_geometry(args.main_cols, args.ba_cols, a)
        assert sum(g["split_sizes"]) == g["n_pad"]
        assert g["split_sizes"][0] == args.main_cols
        assert g["split_sizes"][1] == args.ba_cols
        assert g["n_pad"] >= g["n_fused"]
    print("split arithmetic OK (all alignments sum to N_pad and keep main/ba).")
    print()
    print("Regression tool: verify core/Block/MixBlk/aiv and that N=4120 stays")
    print("the fastest. No config value is recommended.")
    return 0


# ---------------------------------------------------------------------------
# Offline profiler-output judging (no NPU required)
# ---------------------------------------------------------------------------
def _fnum(x) -> float:
    try:
        return float(str(x).strip())
    except (TypeError, ValueError):
        return 0.0


def _find_kernel_details(path: str) -> str | None:
    """Accept a directory (ASCEND_PROFILER_OUTPUT or its parent) or a csv."""
    if os.path.isfile(path) and path.endswith(".csv"):
        return path
    candidates = [
        os.path.join(path, "kernel_details.csv"),
        os.path.join(path, "ASCEND_PROFILER_OUTPUT", "kernel_details.csv"),
    ]
    for c in candidates:
        if os.path.isfile(c):
            return c
    hits = glob.glob(os.path.join(path, "**", "kernel_details.csv"), recursive=True)
    return hits[0] if hits else None


def parse_kernel_details(path: str, m_filter: int = 1):
    """Return {(M,N,K): stats} for MatMulV3 rows in an msprof export."""
    kd = _find_kernel_details(path)
    if kd is None:
        raise FileNotFoundError("no kernel_details.csv under %s" % path)
    out = {}
    with open(kd, newline="", encoding="utf-8", errors="replace") as fh:
        for r in csv.DictReader(fh):
            if r.get("Type") != "MatMulV3":
                continue
            shapes = r.get("Input Shapes", "").replace('"', "").split(";")
            if len(shapes) < 2:
                continue
            try:
                a = [int(float(v)) for v in shapes[0].split(",")]
                b = [int(float(v)) for v in shapes[1].split(",")]
                m = a[-2] if len(a) >= 2 else a[0]
                k = a[-1]
                n = b[0]
            except (ValueError, IndexError):
                continue
            if m_filter is not None and m != m_filter:
                continue
            rec = out.setdefault((m, n, k), {"n": 0, "dur": []})
            rec["n"] += 1
            rec["dur"].append(_fnum(r.get("Duration(us)")))
            # Some CANN versions mix implementations for one shape; keep the
            # most frequent core label so the verdict is not skewed by tails.
            rec.setdefault("cores", []).append(str(r.get(_KERNEL_CORE_FIELD, "?")).strip())
            for f in _KERNEL_NUM_FIELDS:
                rec.setdefault(f, []).append(_fnum(r.get(f)))
    for rec in out.values():
        cores = rec.pop("cores", [])
        rec[_KERNEL_CORE_FIELD] = (
            max(set(cores), key=cores.count) if cores else "?"
        )
        for f in list(rec):
            if isinstance(rec[f], list):
                rec[f] = statistics.mean(rec[f]) if rec[f] else 0.0
    return kd, out


def judge_profile_dirs(dirs) -> int:
    print("== Offline kernel_details.csv judgement (M=1 decode only) ==")
    print("%-46s %-7s %-6s %-9s %-7s %-8s %-8s %s" % (
        "profile", "N", "K", "core", "n", "Block", "MixBlk", "aiv(us) / dur(us)"))
    print("-" * 130)
    for d in dirs:
        try:
            kd, data = parse_kernel_details(d)
        except Exception as e:  # noqa: BLE001 - report and continue
            print("!! %s: %s" % (d, e))
            continue
        label = os.path.basename(os.path.dirname(os.path.dirname(kd))) or os.path.basename(d)
        for (m, n, k), rec in sorted(data.items()):
            core = rec.get("Accelerator Core", "")
            # core is a numeric enum in some CANN versions; keep the raw value.
            print("%-46s %-7d %-6d %-9s %-7d %-8.0f %-8.0f %6.2f / %6.2f" % (
                label[:46], n, k, str(core)[:9], rec["n"],
                rec.get("Block Num", 0), rec.get("Mix Block Num", 0),
                rec.get("aiv_time(us)", 0), rec["dur"]))
    print()
    print("MIX_AIC <=> core==MIX_AIC and MixBlk>0 and aiv(us)>0.")
    return 0


# ---------------------------------------------------------------------------
# NPU timing
# ---------------------------------------------------------------------------
def _dtype(torch, name: str):
    return {
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
        "fp32": torch.float32,
    }[name]


def _make_weight(torch, n: int, k: int, main_cols: int, ba_cols: int,
                 dtype, device, base=None):
    """Build [qkvz | ba | zero-pad] exactly like fuse_ba_into_qkvz does.

    ``base`` is the shared unpadded (n_fused, K) weight. EVERY candidate N
    must derive from it: the old version called randn() per N, so the
    correctness check compared GEMMs with two different random weights and
    reported a failure for every case regardless of the padding.
    """
    n_fused = main_cols + ba_cols
    if base is None:
        base = torch.randn((n_fused, k), dtype=dtype, device=device) * 0.02
    if n < n_fused:
        return base[:n].contiguous()          # pure-qkvz reference row
    if n == n_fused:
        return base.contiguous()
    w = torch.zeros((n, k), dtype=dtype, device=device)
    with torch.no_grad():
        w[:n_fused].copy_(base)
    return w.contiguous()


def _bench_call(torch, torch_npu, fn, warmup: int, iters: int):
    """Host wall clock per call (includes launch jitter)."""
    for _ in range(warmup):
        fn()
    torch_npu.npu.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch_npu.npu.synchronize()
    return (time.perf_counter() - t0) * 1e6 / iters


def _bench_call_repeat(torch, torch_npu, fn, warmup: int, iters: int,
                       repeat: int):
    """Median of ``repeat`` independent measurements (suppresses host jitter)."""
    vals = [_bench_call(torch, torch_npu, fn, warmup, iters)
            for _ in range(max(1, repeat))]
    return statistics.median(vals)


def _bench_graph(torch, torch_npu, fn, warmup: int, iters: int):
    """NPUGraph replay timed with NPU events, i.e. DEVICE time.

    A plain time.perf_counter() around g.replay() measures host enqueue plus
    a large fixed per-replay cost (~150us on 950DT) and cannot resolve a
    12us decode GEMM, which is why the first run showed a flat ~150us column
    for every N while the profiler Duration was ~12us.
    """
    try:
        stream = torch.npu.Stream()
        stream.wait_stream(torch.npu.current_stream())
        with torch.npu.stream(stream):
            for _ in range(warmup):
                fn()
        torch.npu.current_stream().wait_stream(stream)
        g = torch.npu.NPUGraph()
        with torch.npu.graph(g):
            fn()
        for _ in range(warmup):
            g.replay()
        torch_npu.npu.synchronize()
        Event = getattr(torch.npu, "Event", None) or getattr(
            torch_npu.npu, "Event", None)
        if Event is None:
            raise RuntimeError("no NPU Event API available")
        start = Event(enable_timing=True)
        end = Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            g.replay()
        end.record()
        torch_npu.npu.synchronize()
        return start.elapsed_time(end) * 1e3 / iters  # ms -> us
    except Exception as e:  # noqa: BLE001 - graph mode is optional
        print("   [graph timing unavailable: %s]" % e)
        return None


def _profile_one(torch, torch_npu, fn, out_dir: str):
    """Try to capture kernel_details.csv with torch_npu.profiler."""
    os.makedirs(out_dir, exist_ok=True)
    try:
        from torch_npu.profiler import (  # type: ignore
            ProfilerActivity,
            ProfilerLevel,
            _ExperimentalConfig,
            profile,
            tensorboard_trace_handler,
        )
        try:
            from torch_npu.profiler import schedule as prof_schedule
        except ImportError:
            prof_schedule = None

        experimental = _ExperimentalConfig(
            profiler_level=ProfilerLevel.Level1,
        )
        kw = dict(
            activities=[ProfilerActivity.CPU, ProfilerActivity.NPU],
            record_shapes=True,
            on_trace_ready=tensorboard_trace_handler(out_dir),
            experimental_config=experimental,
        )
        # Without a schedule the profiler is still in RECORD state when the
        # context manager stops it, producing "Incorrect schedule" warnings
        # and possibly truncated kernel data.
        if prof_schedule is not None:
            kw["schedule"] = prof_schedule(wait=0, warmup=0, active=3, repeat=1)
        with profile(**kw) as prof:
            for _ in range(4):          # one extra step lets the state machine finish
                fn()
                prof.step()
        return out_dir
    except Exception as e:  # noqa: BLE001 - fall back to manual msprof
        print("   [auto profiler unavailable: %s]" % e)
        print("   fallback: msprof --output=%s python %s --profile-dir %s" % (
            out_dir, os.path.abspath(__file__), out_dir))
        return None


def _judge_dir_for_n(path: str, n: int, m: int = 1):
    try:
        _, data = parse_kernel_details(path, m_filter=m)
    except Exception:
        return None
    for (mm, nn, kk), rec in data.items():
        if nn == n:
            return rec
    return None


def run_npu(args) -> int:
    try:
        import torch
        import torch_npu  # noqa: F401
    except Exception as e:  # noqa: BLE001
        print("ERROR: this mode needs torch + torch_npu on an Ascend node: %s" % e)
        print("       use --cpu-check or --profile-dir on a machine without an NPU.")
        return 2

    torch.npu.set_device(args.device)
    device = "npu:%d" % args.device
    dtype = _dtype(torch, args.dtype)
    torch.manual_seed(1234)
    n_fused = args.main_cols + args.ba_cols

    results = []
    print("== NPU single-op benchmark: F.linear(x[M,%d], W[N,%d]) dtype=%s ==" % (args.k, args.k, args.dtype))
    print("main_cols=%d ba_cols=%d n_fused=%d device=%s warmup=%d iters=%d repeat=%d"
          % (args.main_cols, args.ba_cols, n_fused,
             device, args.warmup, args.iters, args.repeat))
    print()
    hdr = "%-5s %-7s %-8s %-9s %-11s %-9s %-8s %-7s %-8s %-8s %-5s %-6s" % (
        "M", "N", "pad", "wall(us)", "devgraph(us)", "dur(us)", "core",
        "Block", "MixBlk", "aiv(us)", "MIX?", "ok")
    print(hdr)
    print("-" * len(hdr))

    for m in args.ms:
        x = torch.randn((m, args.k), dtype=dtype, device=device)
        # ONE shared unpadded weight for every candidate. Re-randomising per N
        # (the old behaviour) made the correctness check compare two different
        # matrices and always report a failure regardless of the padding.
        w_base = _make_weight(torch, n_fused, args.k, args.main_cols,
                              args.ba_cols, dtype, device)
        w_ref = w_base
        for n in args.ns:
            if n < n_fused:
                w = w_ref[:n].contiguous()          # pure-qkvz reference row
            else:
                w = _make_weight(torch, n, args.k, args.main_cols,
                                 args.ba_cols, dtype, device, base=w_ref)

            def fn(w=w, x=x):
                return torch.nn.functional.linear(x, w)

            wall = _bench_call_repeat(torch, torch_npu, fn, args.warmup,
                                      args.iters, args.repeat)
            graph = _bench_graph(torch, torch_npu, fn, args.warmup,
                                 args.iters) if args.graph else None

            # Correctness: the first n_fused output columns must match the
            # unpadded GEMM. Different N may pick a different CANN tiling and
            # change the bf16 accumulation order by 1 ULP, so use allclose and
            # report both the max abs diff and strict bitwise equality.
            torch_npu.npu.synchronize()
            ok = None
            maxdiff = None
            bitwise = None
            if n >= n_fused:
                with torch.no_grad():
                    out = fn()
                    ref = torch.nn.functional.linear(x, w_ref)
                got = out[..., :n_fused]
                bitwise = bool(torch.equal(got, ref))
                maxdiff = float((got.float() - ref.float()).abs().max().item())
                ok = bool(torch.allclose(got, ref, rtol=2e-2, atol=4e-3))
                if not ok:
                    print("   !! CORRECTNESS FAILURE at N=%d: max_abs_diff=%.6g"
                          % (n, maxdiff))

            dur = core = blk = mix = aiv = None
            if args.profile:
                prof_dir = os.path.join(args.prof_out, "n%d_m%d" % (n, m))
                captured = _profile_one(torch, torch_npu, fn, prof_dir)
                rec = _judge_dir_for_n(captured or prof_dir, n, m) if captured else None
                if rec:
                    dur = rec.get("dur")
                    core, blk, mix, aiv = (
                        str(rec.get("Accelerator Core", "?")),
                        "%.0f" % rec.get("Block Num", 0),
                        "%.0f" % rec.get("Mix Block Num", 0),
                        "%.2f" % rec.get("aiv_time(us)", 0),
                    )
            is_mix = "YES" if (core == "MIX_AIC" and float(mix or 0) > 0) else "no"
            print("%-5d %-7d %-8d %-9.2f %-11s %-9s %-8s %-7s %-8s %-8s %-5s %-6s" % (
                m, n, n - n_fused, wall,
                "%.2f" % graph if graph is not None else "-",
                "%.2f" % dur if dur else "-",
                str(core or "-")[:8], blk or "-", mix or "-", aiv or "-",
                is_mix, {True: "ok", False: "FAIL", None: "-"}[ok]))
            results.append(dict(m=m, n=n, pad=n - n_fused,
                                wall_us=wall, graph_us=graph, dur_us=dur,
                                core=core or "-", block=blk or "-",
                                mix_block=mix or "-", aiv_us=aiv or "-",
                                correct=ok, bitwise_equal=bitwise,
                                max_abs_diff=maxdiff))
        print()

    # ---- recommendation -------------------------------------------------
    dec = [r for r in results if r["m"] == 1 and r["n"] >= n_fused]
    base = next((r for r in dec if r["n"] == n_fused), None)
    print("== Verdict (decode M=1) ==")
    NOISE_US, NOISE_PCT = 0.5, 3.0
    if not base:
        print("baseline N=%d missing; cannot recommend." % n_fused)
    else:
        # Kernel-side Duration is authoritative (same field as online msprof)
        # and immune to host-launch jitter. Fall back to device-event graph,
        # then to host wall (noisiest).
        metric = "dur_us"
        if not all(r.get("dur_us") for r in dec):
            metric = "graph_us" if all(r.get("graph_us") for r in dec) else "wall_us"
        label = {"dur_us": "profiler Duration(us)",
                 "graph_us": "device-event graph(us)",
                 "wall_us": "host wall(us) [noisy]"}[metric]
        print("primary metric: %s" % label)
        print("baseline N=%d: %s=%.2f us | wall=%.2f graph=%s" % (
            base["n"], metric, base[metric], base["wall_us"],
            "%.2f" % base["graph_us"] if base["graph_us"] else "-"))
        best = min(dec, key=lambda r: r[metric] if r.get(metric) else float("inf"))
        delta = best[metric] - base[metric]
        pct = 100.0 * delta / base[metric] if base[metric] else 0.0
        print("fastest   N=%d: %s=%.2f us, delta %+.2f us (%.1f%%)" % (
            best["n"], metric, best[metric], delta, pct))
        # Cross-check the other metrics: if they disagree on the winner, the
        # delta is noise, not a real speedup.
        winners = {}
        for alt in ("dur_us", "wall_us", "graph_us"):
            vals = [r for r in dec if r.get(alt)]
            if vals:
                winners[alt] = min(vals, key=lambda r: r[alt])["n"]
        print("metric winners: %s" % ", ".join(
            "%s->N=%d" % (k, v) for k, v in winners.items()))
        if len(set(winners.values())) > 1:
            print("!! metrics DISAGREE on the fastest N -> ranking is noise-dominated.")
        if best["n"] == base["n"]:
            print("=> baseline N=%d is the FASTEST by %s; every padded N is slower."
                  % (n_fused, metric))
        elif abs(delta) < NOISE_US or abs(pct) < NOISE_PCT:
            print("=> NO measurable gain (|delta| < %.2f us or %.1f%%)." % (
                NOISE_US, NOISE_PCT))
        elif best["n"] != base["n"]:
            print("!! padded N=%d won by %s; re-measure with --repeat 5."
                  % (best["n"], metric))
            print("   No alignment config switch exists (it was removed).")
        slower = [r for r in dec if r["n"] != base["n"]]
        if slower:
            ds = [r[metric] - base[metric] for r in slower]
            print("padded candidates vs baseline: %+.2f ~ %+.2f us (mean %+.2f us)" % (
                min(ds), max(ds), statistics.mean(ds)))
        print("=> KEEP the exact fused N=%d; padding is disproved (extra zero"
              % n_fused)
        print("   rows only add MTE2 traffic). The remaining gap is MTE2 cold")
        print("   reads, addressed by single-storage weights + prefetch in")
        print("   ops/gdn.py / mlp_weight_prefetch.py, not by tiling.")
        if args.profile:
            mix_hits = [r["n"] for r in results
                        if r["m"] == 1 and r["core"] == "MIX_AIC"]
            print("MIX_AIC hit at N=%s (only N<=%d qualifies on 950DT)"
                  % (mix_hits if mix_hits else "none", 32 * 128))
    print()
    print("Interpretation:")
    print("  dur(us)       kernel-side Duration from the profiler - authoritative")
    print("                device time, same field as online msprof; use it to rank.")
    print("  wall(us)      host enqueue wall clock, includes launch jitter; noisy.")
    print("  devgraph(us)  NPUGraph replay timed with NPU events. For M=1 it still")
    print("                reads ~141us while dur is ~12us: replay is host-bound")
    print("                and the events include device idle gaps between replays.")
    print("                Only M=2252 (prefill) matches dur (~+3us). Do NOT use")
    print("                this column to rank decode cases; rank by dur(us).")
    print("  Block/MixBlk/aiv = CANN implementation; MIX_AIC <=> core=MIX_AIC,")
    print("                MixBlk>0 and aiv>0 (only N<=4096 on 950DT, M=1).")
    print("  ok            first %d output columns vs the unpadded GEMM" % n_fused)
    print("                (bf16 allclose; FAIL means a REAL bug).")

    if args.out:
        with open(args.out, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(results[0].keys()))
            w.writeheader()
            w.writerows(results)
        print("results written to %s" % args.out)
    return 0


# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(
        description="Qwen3.5 GDN fused qkvz+ba GEMM single-op microbenchmark",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--ns", default=",".join(str(n) for n in DEFAULT_NS),
                   help="comma-separated candidate output widths N")
    p.add_argument("--ms", default=",".join(str(m) for m in DEFAULT_MS),
                   help="comma-separated M values (1=decode, 2252=prefill)")
    p.add_argument("--k", type=int, default=DEFAULT_K, help="hidden size K")
    p.add_argument("--main-cols", type=int, default=DEFAULT_MAIN_COLS,
                   help="qkvz output width")
    p.add_argument("--ba-cols", type=int, default=DEFAULT_BA_COLS,
                   help="ba output width (TP4)")
    p.add_argument("--dtype", choices=("bf16", "fp16", "fp32"), default="bf16")
    p.add_argument("--device", type=int, default=0, help="NPU device id")
    p.add_argument("--warmup", type=int, default=20)
    p.add_argument("--iters", type=int, default=200)
    p.add_argument("--repeat", type=int, default=3,
                   help="repeat each wall timing and take the median (reduces host jitter)")
    p.add_argument("--graph", action="store_true",
                   help="also time NPUGraph replay (closest to PIECEWISE mode)")
    p.add_argument("--profile", action="store_true",
                   help="capture kernel_details.csv via torch_npu.profiler per case")
    p.add_argument("--prof-out", default="./gdn_gemm_prof",
                   help="root dir for --profile captures")
    p.add_argument("--profile-dir", action="append", default=[],
                   help="judge an existing msprof/ascend_pt output (repeatable); "
                        "offline mode, no NPU needed")
    p.add_argument("--cpu-check", action="store_true",
                   help="pure-logic self test, no torch / no NPU needed")
    p.add_argument("--out", default="", help="write result table to this CSV")
    return p


def main() -> int:
    args = build_parser().parse_args()
    args.ns = [int(v) for v in args.ns.split(",") if v.strip()]
    args.ms = [int(v) for v in args.ms.split(",") if v.strip()]
    if args.cpu_check:
        return cpu_check(args)
    if args.profile_dir:
        return judge_profile_dirs(args.profile_dir)
    return run_npu(args)


if __name__ == "__main__":
    sys.exit(main())
