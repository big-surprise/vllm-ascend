#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""P0 prefill-scheduling configuration sweep for Qwen3.5-27B on 950DT / TP4.

Why this experiment exists
--------------------------
The con4 / 20-request / in2252 / out10 profiling run showed that the 5 rounds
each merged THREE requests into a single M=6787 prefill (310ms) instead of
running three separate M=2262 prefills (158ms).  Those merged prefills account
for ~79% of the 3.35s trace window and stall every in-flight decode, which is
what blows up P90 TTFT and P90 E2E by 5-6x.  TPOT itself only moved +4~8%.

P0 is therefore about the *prefill scheduling policy*, not decode kernels.
This script restarts the server once per configuration, replays the same fixed
20 requests at concurrency 4, and reports P50/P90/P99 of TTFT / TPOT / E2E so
the configurations are directly comparable.

Cases (edit DEFAULT_CASES or supply --cases-file)
-------------------------------------------------
  baseline                 current G2 config (default max-num-batched-tokens)
  mbt-4096 / mbt-2252 / mbt-2048   cap the batch, forbidding 3-request merges
  mbt-2252-chunked         cap + chunked prefill
  prefix-cache             enable prefix caching (only pays off if prompts share a prefix)
  mbt-2252-prefix          both levers
  mbt-2252-no-ar-prefetch  P2 control: turn off enable_decode_ar_prefetch

Note: parameter names differ between vLLM versions.  If your build rejects
e.g. --enable-chunked-prefill, edit the case list (or --cases-file) rather than
the script.  Everything else (allreduce fast path, ba fusion, TP4, model,
dataset, concurrency) is held constant across cases.

Measurement
-----------
Streaming SSE against /v1/completions (or /v1/chat/completions with --chat):
  TTFT  = first non-empty token - request send
  TPOT  = (last token - first token) / (n_tokens - 1)
  E2E   = last token - request send
`ignore_eos=true` keeps output length fixed at --max-tokens for comparability.
2 warmup requests per case are discarded.  P90 over 20 samples is noisy; use
--repeats 3 (60 samples) before quoting a number.

Only the Python standard library is required.  No NPU / vLLM import happens at
module load time, so --dry-run works on a laptop.

Examples
--------
  # 1. Inspect exactly what would run (no server, no requests)
  python benchmarks/p0_prefill_schedule_sweep.py --dry-run

  # 2. Real sweep on 950DT
  python benchmarks/p0_prefill_schedule_sweep.py \
      --server-cmd 'vllm serve /data/Qwen3.5-27B --tensor-parallel-size 4 \
        --port 8000 --served-model-name qwen35 \
        --additional-config '"'"'{"allreduce_outplace": true, "enable_decode_ar_prefetch": true}'"'"' {extra}' \
      --model qwen35 --dataset /data/prompts_2252.jsonl \
      --concurrency 4 --num-requests 20 --max-tokens 10 --repeats 3 \
      --out /data/p0_sweep

  # 3. Benchmark an already-running server (single case only)
  python benchmarks/p0_prefill_schedule_sweep.py \
      --no-manage-server --cases baseline --url http://127.0.0.1:8000 \
      --model qwen35 --out /tmp/p0_one
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shlex
import statistics
import subprocess
import sys
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

# Each entry's `extra_args` is appended to --server-cmd at the {extra} marker.
# `note` is carried into the report so the numbers remain self-describing.
DEFAULT_CASES: list[dict] = [
    {"name": "baseline", "extra_args": [], "note": "G2 current config: default max-num-batched-tokens"},
    {"name": "mbt-4096", "extra_args": ["--max-num-batched-tokens", "4096"], "note": "cap below 2x2262"},
    {"name": "mbt-2252", "extra_args": ["--max-num-batched-tokens", "2252"], "note": "one prefill request per forward"},
    {"name": "mbt-2048", "extra_args": ["--max-num-batched-tokens", "2048"], "note": "more aggressive cap"},
    {
        "name": "mbt-2252-chunked",
        "extra_args": ["--max-num-batched-tokens", "2252", "--enable-chunked-prefill"],
        "note": "cap + chunked prefill (verify flag name for your vLLM build)",
    },
    {
        "name": "prefix-cache",
        "extra_args": ["--enable-prefix-caching"],
        "note": "only pays off when the 20 prompts share a prefix",
    },
    {
        "name": "mbt-2252-prefix",
        "extra_args": ["--max-num-batched-tokens", "2252", "--enable-prefix-caching"],
        "note": "both P0 levers",
    },
    {
        "name": "mbt-2252-no-ar-prefetch",
        "extra_args": [
            "--max-num-batched-tokens",
            "2252",
            "--additional-config",
            '{"allreduce_outplace": true, "enable_decode_ar_prefetch": false}',
        ],
        "note": "P2 control: A switch off at con4",
    },
]


@dataclass
class Case:
    name: str
    extra_args: list[str] = field(default_factory=list)
    note: str = ""


@dataclass
class Sample:
    case: str
    repeat: int
    req_id: int
    ttft_ms: float
    tpot_ms: float
    e2e_ms: float
    tokens: int


def load_cases(path: str | None, only: list[str] | None) -> list[Case]:
    raw = json.loads(Path(path).read_text(encoding="utf-8")) if path else DEFAULT_CASES
    cases = [Case(c["name"], list(c.get("extra_args", [])), c.get("note", "")) for c in raw]
    if only:
        wanted = set(only)
        cases = [c for c in cases if c.name in wanted]
        missing = wanted - {c.name for c in cases}
        if missing:
            raise SystemExit(f"unknown case(s): {sorted(missing)}; available: {[c.name for c in cases]}")
    if not cases:
        raise SystemExit("no cases selected")
    return cases


_WARNED_CONFIGS: set[str] = set()


def warn_bad_additional_config(cmd: list[str]) -> None:
    """shlex silently strips unquoted JSON quotes - catch it before a run.

    ``--server-cmd '... --additional-config {"a": true} {extra}'`` loses the
    inner double quotes.  The docstring example shows the correct nested-quote
    form; this just makes a mistake loud instead of failing at server startup.
    """
    for i, arg in enumerate(cmd):
        if arg == "--additional-config" and i + 1 < len(cmd):
            val = cmd[i + 1]
            if not val.lstrip().startswith("{"):
                continue
            try:
                json.loads(val)
            except (json.JSONDecodeError, ValueError):
                if val not in _WARNED_CONFIGS:
                    _WARNED_CONFIGS.add(val)
                    print(
                        f"    [WARN] --additional-config is not valid JSON: {val!r}\n"
                        "           quote the JSON inside --server-cmd, e.g.\n"
                        "           --server-cmd '... --additional-config "
                        "'\"'\"'{\"a\": true}'\"'\"' {extra}'",
                        file=sys.stderr,
                    )


def build_server_cmd(template: str, case: Case) -> list[str]:
    extra = " ".join(shlex.quote(a) for a in case.extra_args)
    if "{extra}" in template:
        rendered = template.replace("{extra}", extra)
    else:
        rendered = template + (" " + extra if extra else "")
    cmd = shlex.split(rendered)
    warn_bad_additional_config(cmd)
    return cmd


def wait_healthy(url: str, timeout_s: float, proc: subprocess.Popen | None) -> None:
    deadline = time.time() + timeout_s
    health = url.rstrip("/") + "/health"
    last = ""
    while time.time() < deadline:
        if proc is not None and proc.poll() is not None:
            raise RuntimeError(f"server exited early with code {proc.returncode}")
        try:
            with urllib.request.urlopen(health, timeout=3) as r:
                if r.status == 200:
                    return
        except Exception as exc:  # noqa: BLE001 - startup polling
            last = str(exc)
        time.sleep(2.0)
    raise RuntimeError(f"server not healthy after {timeout_s}s (last error: {last})")


def start_server(cmd: list[str], log_path: Path, url: str, timeout_s: float) -> tuple[subprocess.Popen, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log = log_path.open("w", encoding="utf-8")
    print(f"    server cmd: {' '.join(shlex.quote(a) for a in cmd)}")
    proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    try:
        wait_healthy(url, timeout_s, proc)
    except Exception:
        stop_server(proc, log)
        raise
    return proc, log


def stop_server(proc: subprocess.Popen | None, log) -> None:
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=10)
    try:
        log.close()
    except Exception:
        pass


def synth_prompts(n: int, prompt_tokens: int, tokenizer: str | None) -> list[str]:
    """Build n prompts of ~prompt_tokens tokens.

    With --tokenizer the length is exact (encode/decode round-trip).  Without
    it we fall back to ~4 chars/token, which is only an approximation - supply
    a real --dataset for publishable numbers.
    """
    if tokenizer:
        try:
            from transformers import AutoTokenizer  # type: ignore

            tok = AutoTokenizer.from_pretrained(tokenizer, trust_remote_code=True)
            base = "The quick brown fox jumps over the lazy dog. "
            ids = (tok.encode(base, add_special_tokens=False) * ((prompt_tokens // 12) + 2))[:prompt_tokens]
            text = tok.decode(ids)
            return [text] * n
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] tokenizer unavailable ({exc}); using char approximation", file=sys.stderr)
    unit = "The quick brown fox jumps over the lazy dog. "
    text = (unit * ((prompt_tokens * 4 // len(unit)) + 1))[: prompt_tokens * 4]
    return [f"[request {i}] {text}" for i in range(n)]


def load_prompts(args) -> list[str]:
    if args.dataset:
        prompts = []
        with open(args.dataset, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line) if line.startswith("{") else {"prompt": line}
                prompts.append(obj["prompt"])
        if len(prompts) < args.num_requests:
            raise SystemExit(f"dataset has {len(prompts)} prompts, need >= {args.num_requests}")
        return prompts[: args.num_requests]
    return synth_prompts(args.num_requests, args.prompt_tokens, args.tokenizer)


def extract_delta(obj: dict, chat: bool) -> str:
    try:
        ch = obj["choices"][0]
        if chat:
            return ch.get("delta", {}).get("content") or ""
        return ch.get("text") or ""
    except (KeyError, IndexError, TypeError):
        return ""


def one_request(url: str, endpoint: str, model: str, prompt: str, chat: bool,
                max_tokens: int, timeout_s: float, case: str, repeat: int,
                req_id: int) -> Sample:
    body = {
        "model": model,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": True,
        "ignore_eos": True,
    }
    if chat:
        body["messages"] = [{"role": "user", "content": prompt}]
    else:
        body["prompt"] = prompt
    data = json.dumps(body).encode("utf-8")
    req = urllib.request.Request(
        url.rstrip("/") + endpoint,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t_send = time.perf_counter()
    first = last = None
    tokens = 0
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        for raw in resp:
            line = raw.decode("utf-8", errors="replace").strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                break
            try:
                obj = json.loads(payload)
            except json.JSONDecodeError:
                continue
            if not extract_delta(obj, chat):
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            last = now
            tokens += 1
    if first is None or last is None:
        raise RuntimeError(f"case={case} repeat={repeat} req={req_id}: no tokens streamed")
    ttft = (first - t_send) * 1000.0
    e2e = (last - t_send) * 1000.0
    tpot = ((last - first) * 1000.0 / (tokens - 1)) if tokens > 1 else float("nan")
    return Sample(case, repeat, req_id, ttft, tpot, e2e, tokens)


def run_requests(args, prompts: list[str], case: str, repeat: int) -> list[Sample]:
    endpoint = "/v1/chat/completions" if args.chat else "/v1/completions"
    out: list[Sample] = []
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = {
            pool.submit(
                one_request, args.url, endpoint, args.model, p, args.chat,
                args.max_tokens, args.request_timeout, case, repeat, i,
            ): i
            for i, p in enumerate(prompts)
        }
        for fut in as_completed(futures):
            out.append(fut.result())
    return sorted(out, key=lambda s: s.req_id)


def warmup(args, prompts: list[str]) -> None:
    endpoint = "/v1/chat/completions" if args.chat else "/v1/completions"
    for p in prompts[: args.warmup]:
        try:
            one_request(args.url, endpoint, args.model, p, args.chat,
                        args.max_tokens, args.request_timeout, "warmup", 0, -1)
        except Exception as exc:  # noqa: BLE001
            print(f"    [warn] warmup failed: {exc}", file=sys.stderr)


def pct(values: list[float], p: float) -> float:
    """Linear-interpolation percentile, matching numpy / vllm bench --metric-percentiles."""
    vals = sorted(v for v in values if v == v)  # drop NaN
    if not vals:
        return float("nan")
    if len(vals) == 1:
        return vals[0]
    pos = (len(vals) - 1) * p
    lo = int(pos)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (pos - lo)


def summarize(samples: list[Sample]) -> dict:
    def block(key):
        v = [getattr(s, key) for s in samples]
        clean = [x for x in v if x == x]  # drop NaN (single-token responses)
        return {
            "mean": statistics.mean(clean) if clean else float("nan"),
            "p50": pct(v, 0.50),
            "p90": pct(v, 0.90),
            "p99": pct(v, 0.99),
        }
    return {"n": len(samples), "ttft_ms": block("ttft_ms"), "tpot_ms": block("tpot_ms"), "e2e_ms": block("e2e_ms")}


def fmt(block: dict) -> str:
    return f"{block['p50']:8.1f}/{block['p90']:8.1f}/{block['p99']:8.1f}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--server-cmd", default="", help="server command template; {extra} is replaced per case")
    ap.add_argument("--url", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen35", help="served-model-name")
    ap.add_argument("--chat", action="store_true", help="use /v1/chat/completions instead of /v1/completions")
    ap.add_argument("--cases-file", default=None, help="JSON list of {name, extra_args, note}")
    ap.add_argument("--cases", nargs="*", default=None, help="run only these case names")
    ap.add_argument("--no-manage-server", action="store_true", help="server already running; only 1 case allowed")
    ap.add_argument("--startup-timeout", type=float, default=900.0)
    ap.add_argument("--request-timeout", type=float, default=300.0)
    ap.add_argument("--warmup", type=int, default=2)
    ap.add_argument("--concurrency", type=int, default=4)
    ap.add_argument("--num-requests", type=int, default=20)
    ap.add_argument("--max-tokens", type=int, default=10)
    ap.add_argument("--prompt-tokens", type=int, default=2252)
    ap.add_argument("--dataset", default=None, help="JSONL, one prompt per line (string or {prompt: ...})")
    ap.add_argument("--tokenizer", default=None, help="optional HF tokenizer for exact prompt length")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--out", default="./p0_sweep_results")
    ap.add_argument("--dry-run", action="store_true", help="print commands/plan only; no server, no requests")
    ap.add_argument("--list-cases", action="store_true")
    args = ap.parse_args()

    cases = load_cases(args.cases_file, args.cases)
    if args.list_cases:
        for c in cases:
            print(f"{c.name:28s} {' '.join(c.extra_args) or '(none)':70s} {c.note}")
        return 0
    if args.no_manage_server and len(cases) != 1:
        raise SystemExit("--no-manage-server requires exactly one --cases entry (config cannot be switched)")
    if not args.no_manage_server and not args.server_cmd:
        raise SystemExit("--server-cmd is required unless --no-manage-server is set")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    prompts = load_prompts(args)

    print("=" * 118)
    print(f"P0 sweep | con={args.concurrency} reqs={args.num_requests} repeats={args.repeats} "
          f"in~{args.prompt_tokens} out={args.max_tokens} | {len(cases)} cases | out={out_dir}")
    print(f"         prompts: {'dataset=' + args.dataset if args.dataset else 'synthetic (~4 chars/token)'}")
    print("=" * 118)

    if args.dry_run:
        for c in cases:
            cmd = build_server_cmd(args.server_cmd, c) if args.server_cmd else []
            print(f"\n[{c.name}] {c.note}")
            print(f"  extra_args : {' '.join(shlex.quote(a) for a in c.extra_args) or '(none)'}")
            if cmd:
                print(f"  server     : {' '.join(shlex.quote(a) for a in cmd)}")
            else:
                print("  server     : (managed externally)")
            print(f"  requests   : {args.concurrency} workers x {args.num_requests} reqs x {args.repeats} repeats, "
                  f"warmup={args.warmup}, endpoint={'chat' if args.chat else 'completions'}")
        print("\n[dry-run] no server started, no requests sent.")
        return 0

    all_samples: list[Sample] = []
    summaries: list[dict] = []
    for c in cases:
        print(f"\n--- case: {c.name} --- {c.note}")
        proc, log = (None, None)
        try:
            if not args.no_manage_server:
                proc, log = start_server(
                    build_server_cmd(args.server_cmd, c),
                    out_dir / f"{c.name}.server.log",
                    args.url,
                    args.startup_timeout,
                )
            else:
                wait_healthy(args.url, 30.0, None)
            warmup(args, prompts)
            case_samples: list[Sample] = []
            for rep in range(args.repeats):
                print(f"    repeat {rep + 1}/{args.repeats} ...", end="", flush=True)
                s = run_requests(args, prompts, c.name, rep)
                case_samples.extend(s)
                print(f" {len(s)} reqs, TTFT p90={pct([x.ttft_ms for x in s], .9):.1f}ms "
                      f"E2E p90={pct([x.e2e_ms for x in s], .9):.1f}ms")
            all_samples.extend(case_samples)
            sm = summarize(case_samples)
            sm.update({"name": c.name, "note": c.note, "extra_args": c.extra_args})
            summaries.append(sm)
        except Exception as exc:  # noqa: BLE001
            print(f"    [ERROR] case {c.name} failed: {exc}", file=sys.stderr)
            summaries.append({"name": c.name, "note": c.note, "extra_args": c.extra_args, "error": str(exc)})
        finally:
            stop_server(proc, log)

    # ---- per-request detail ----
    detail = out_dir / "per_request.csv"
    with detail.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "repeat", "req_id", "ttft_ms", "tpot_ms", "e2e_ms", "tokens"])
        for s in all_samples:
            w.writerow([s.case, s.repeat, s.req_id, f"{s.ttft_ms:.3f}",
                        f"{s.tpot_ms:.3f}", f"{s.e2e_ms:.3f}", s.tokens])

    # ---- summary table ----
    ok = [s for s in summaries if "error" not in s]
    base = next((s for s in ok if s["name"] == "baseline"), None)
    print("\n" + "=" * 118)
    print(f"{'case':28s} {'TTFT p50/p90/p99 (ms)':>26s} {'TPOT p50/p90/p99 (ms)':>26s} "
          f"{'E2E p50/p90/p99 (ms)':>26s} {'P90 E2E vs base':>16s}")
    print("-" * 118)
    for s in ok:
        delta = ""
        if base and base is not s:
            d = (s["e2e_ms"]["p90"] / base["e2e_ms"]["p90"] - 1) * 100
            delta = f"{d:+.1f}%"
        elif base is s:
            delta = "baseline"
        print(f"{s['name']:28s} {fmt(s['ttft_ms']):>26s} {fmt(s['tpot_ms']):>26s} "
              f"{fmt(s['e2e_ms']):>26s} {delta:>16s}")
    for s in summaries:
        if "error" in s:
            print(f"{s['name']:28s} FAILED: {s['error'][:70]}")
    print("=" * 118)

    (out_dir / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    with (out_dir / "summary.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["case", "note", "n", "ttft_p50", "ttft_p90", "ttft_p99",
                    "tpot_p50", "tpot_p90", "tpot_p99", "e2e_p50", "e2e_p90", "e2e_p99"])
        for s in ok:
            w.writerow([s["name"], s["note"], s["n"],
                        *[round(s[k][q], 3) for k in ("ttft_ms", "tpot_ms", "e2e_ms")
                          for q in ("p50", "p90", "p99")]])
    print(f"\nwritten: {detail} | {out_dir / 'summary.csv'} | {out_dir / 'summary.json'}")
    print("Interpretation: a good P0 case cuts P90 TTFT/E2E by >=30% with TPOT unchanged;")
    print("if mbt-2252 also raises TPOT, decode starvation is worse than the merged prefill gain.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
