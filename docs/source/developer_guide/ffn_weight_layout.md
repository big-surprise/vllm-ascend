# Qwen3.5 BF16 FFN and attention input weight layouts

Set `ffn_weight_layout` to `"nz_kn"` in `additional_config` and retain
`weight_nz_mode=0` to enable the opt-in load-time FFN layout on Ascend950.
The default value `"default"` preserves existing behavior.

Only unquantized Qwen3.5 dense BF16 `mlp.gate_up_proj` and `mlp.down_proj`
weights are converted, after TP sharding and before graph capture. The logical
Parameter remains `[N,K]`; its storage is a transposed view of a `[K,N]`
FRACTAL_NZ base. Existing custom `unquantized_gemm` and weight-prefetch code
use the same Parameter, with no conversion in the decode path and no duplicate
resident weight cache. This is a weight-layout optimization, not SwiGLU fusion.

Packed FFN weights retain separate MatMul and AllReduce, including at larger
M, because the fused communication operator does not support this operand
contract. Other contiguous linear weights remain eligible for fusion.

LoRA, RL/weight transfer, CPU offload/sleep and context-parallel weight switching
are not supported by this option. Reload the model and recapture graphs after
changing the option or weights. Keep the option off for unvalidated hardware or
models; incompatible requested configurations fail validation.

To revert, set `"ffn_weight_layout":"default"` (or remove the key) and restart
the model. Do not set the global `weight_nz_mode=2` as a substitute: it uses a
different physical weight direction and regressed this workload in measurement.

Validation in the 184 qwen container: exact conversion roundtrip; actual loader
hook on all 64 real BF16 layers and four TP4 shards; M=1/4 checks and sampled
M=16/128/512/2048; custom-op/FakeTensor tracing, repeated ACL graph replay and
matching prefetch pointers. A TP1 whole-model check generated the same 96 tokens
and selected-token log probabilities for four fixed prompts. These checks do not
establish complete TP4 service throughput or accuracy on all tasks.

## Optional attention input layouts

The attention input projections have independent options, both disabled by
default:

```json
{
  "weight_nz_mode": 0,
  "ffn_weight_layout": "nz_kn",
  "linear_attention_qkvz_layout": "nz_kn",
  "full_attention_qkv_layout": "nz_kn",
  "enable_gdn_qkv_split": false
}
```

`linear_attention_qkvz_layout` selects only `linear_attn.in_proj_qkvz`.
`full_attention_qkv_layout` selects only `self_attn.qkv_proj`, including its
packed output-gate projection. The merged QKVZ option requires
`enable_gdn_qkv_split=false`. Attention output projections, `in_proj_ba`, and
`lm_head` retain their existing layouts.

Like FFN, these options replace `layer.weight.data` once, after checkpoint
loading and TP sharding and before graph capture. The Parameter identity,
loader metadata, and logical `[N,K]` shape are retained; storage is a
transposed view of a `[K,N]` NZ base. Prefill and every decode batch size use
the same weight allocation. There is no M=1/4 gate, runtime conversion, ND
fallback copy, or extra resident NZ buffer. Existing prefetch sites fetch
`layer.weight` directly and therefore consume the same allocation as matmul.
Temporary allocations are needed during conversion; allocator-reserved memory
can remain higher after their tensors are released.

The same model, dtype, hardware, and weight-switching restrictions as FFN
apply. Both TP-sharded dimensions must be divisible by 16. Restart the model
and recapture graphs when changing these options. To disable one attention
layout, set its option to `"default"`; the other options are independent.

Format conversion preserves every BF16 weight value. Matmul results are not
guaranteed to be bitwise identical across all shapes: CANN can select a
different accumulation order for NZ, and large-M BF16 outputs can differ at
rounding boundaries. The tests cover exact conversion roundtrips, Parameter
identity, single-allocation storage, prefetch identity, graph replay, and
numerical accuracy at TP4 projection dimensions. Numerical and whole-model
validation results for a deployment must distinguish these rounding differences
from changes in generated output or task accuracy.

The measured performance benefit was established for M=1/4, not for all
prefill or decode sizes. Enabling a layout for all shapes does not imply a
speedup for every shape. Do not replace these options with global
`weight_nz_mode=2`, which packs the opposite physical weight direction.

## Explicit serialization setting in additional_config

This local Ascend build also accepts `"VLLM_ALLOW_INSECURE_SERIALIZATION": true`
in `additional_config`, as requested. It maps the typed boolean to the existing
vLLM environment setting before engine serialization and is inherited by worker
processes. Explicit `false` disables it; omitting the key preserves the external
environment. Stale vLLM environment caches are refreshed when explicitly set.
This is a local connector added here, not an upstream vLLM additional_config key.
The full-model apply_model diagnostic verified the mapping without an export.
