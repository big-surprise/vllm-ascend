# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Dense MLP prefetch with stream dependencies contained in compute ops.

Each op returns the real vector computation result consumed by the next
projection. This keeps both stream fences in the compiled graph without
aliasing an input or relying on unused, side-effect-only custom ops.
"""

import torch
import torch_npu
from vllm.distributed import tensor_model_parallel_all_reduce
from vllm.utils.torch_utils import direct_register_custom_op

from vllm_ascend.ascend_config import (
    WEIGHT_PREFETCH_WINDOW_ATTN_O,
    WEIGHT_PREFETCH_WINDOW_MLP_DOWN,
    WEIGHT_PREFETCH_WINDOW_MLP_GATE_UP,
    WEIGHT_PREFETCH_WINDOW_NEXT_QKV,
    max_prefetch_weight_size_bytes,
)
from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops.gemma_norm_weight import gemma_norm_weight
from vllm_ascend.ops.weight_prefetch import MAX_PREFETCH_WEIGHT_SIZE, layer_idx_from_prefix, layers_of_model_instance
from vllm_ascend.utils import enable_custom_op, get_weight_prefetch_method, npu_stream_switch, prefetch_stream

MLP_PREFETCH_TOKEN_THRESHOLD = 500
# This tree has no allreduce_rmsnorm_fusion_pass module; only the constant
# is needed. 512 matches the reference tree's definition.
ALLREDUCE_NORM_FUSE_THRESHOLD = 512

DEFAULT_WINDOW_CAP_BYTES = max_prefetch_weight_size_bytes(None)


def _all_reduce_prefetch_window(reduce_first: bool) -> str:
    """W4 is issued after the down_proj collective, W2 before it."""
    return WEIGHT_PREFETCH_WINDOW_NEXT_QKV if reduce_first else WEIGHT_PREFETCH_WINDOW_MLP_GATE_UP


def _window_cap_bytes(window: str | None) -> int | None:
    """Byte budget for one window; ``None`` means prefetch the whole weight.

    ``window is None`` keeps the legacy single global cap for callers that do
    not belong to the four dense-decode windows.
    """
    if window is None:
        return MAX_PREFETCH_WEIGHT_SIZE
    method = get_weight_prefetch_method()
    caps = getattr(method, "max_prefetch_weight_size_bytes", None)
    if not caps or window not in caps:
        caps = DEFAULT_WINDOW_CAP_BYTES
    return caps[window]


def _uses_all_reduce_prefetch(projection):
    """Only suppress the Norm fallback for projections with a real issue site."""
    from vllm_ascend.ops.linear import AscendRowParallelLinear
    from vllm_ascend.ops.linear_op import MatmulAllreduceRowParallelOp

    if not getattr(projection, "reduce_results", False) or getattr(projection, "tp_size", 0) <= 1:
        return False
    op = getattr(projection, "custom_op", None)
    if isinstance(op, MatmulAllreduceRowParallelOp):
        return True
    return op is None and isinstance(projection, AscendRowParallelLinear) and projection.input_is_parallel


def _attention_gate_up_target(prefix):
    """Resolve only the current decoder's unfused Attention output reduction."""
    method = get_weight_prefetch_method()
    if method is None or not method.use_qwen_dense_mlp_prefetch:
        return None
    if not prefix.endswith((".self_attn.o_proj", ".linear_attn.out_proj")):
        return None
    idx = layer_idx_from_prefix(prefix)
    layers = layers_of_model_instance(_EXTRA_CTX.model_instance)
    if layers is None or idx is None or not 0 <= idx < len(layers):
        return None
    layer = layers[idx]
    attention = (
        getattr(layer, "self_attn", None)
        if prefix.endswith(".self_attn.o_proj")
        else getattr(layer, "linear_attn", None)
    )
    projection = (
        getattr(attention, "o_proj", None)
        if prefix.endswith(".self_attn.o_proj")
        else getattr(attention, "out_proj", None)
    )
    if (
        not _uses_all_reduce_prefetch(projection)
        or projection.prefix != prefix
        or not hasattr(layer.mlp, "gate_up_proj")
    ):
        return None
    return layer.mlp.gate_up_proj.weight, float(method.mlp.prefetch_ratio.get("gate_up", 0))


def _next_layer_input_proj_target(prefix):
    """Resolve a down_proj to the NEXT decoder layer's input projection.

    The MLP down_proj all-reduce is the last collective of layer N, and the
    next large weight read is layer N+1's input projection, one AddRmsNorm
    later. That is the same fork/consume shape the gate_up prefetch already
    uses at o_proj, so reuse it. Measured at TP4: 22.48us idle window against
    a 17.25us MTE2 for in_proj_qkvz (40MiB).
    """
    method = get_weight_prefetch_method()
    if method is None or not method.is_qwen_dense:
        return None
    if not method.attn.enable or not prefix.endswith(".mlp.down_proj"):
        return None
    idx = layer_idx_from_prefix(prefix)
    layers = layers_of_model_instance(_EXTRA_CTX.model_instance)
    if layers is None or idx is None or not 0 <= idx + 1 < len(layers):
        return None
    # Qwen3.5 alternates linear_attention and full_attention layers.
    nxt = layers[idx + 1]
    attention = getattr(nxt, "linear_attn", None) or getattr(nxt, "self_attn", None)
    projection = (
        getattr(attention, "in_proj_qkv", None)
        or getattr(attention, "in_proj_qkvz", None)
        or getattr(attention, "qkv_proj", None)
    )
    weight = getattr(projection, "weight", None)
    if weight is None:
        return None
    # The ordinary row-parallel and matmul_and_reduce paths both issue at
    # this boundary. Do not add another prefetch around mlp.down_proj.
    down = getattr(getattr(layers[idx], "mlp", None), "down_proj", None)
    if not _uses_all_reduce_prefetch(down):
        return None
    return weight, float(method.attn.prefetch_ratio.get("qkv", 0))


def _all_reduce_prefetch_target(prefix):
    """Pick what this row-parallel all-reduce should overlap a prefetch of.

    Two disjoint fork points, both landing one AddRmsNorm before their reader:
      o_proj / out_proj -> this layer's mlp.gate_up_proj
      mlp.down_proj     -> next layer's in_proj_qkvz / qkv_proj
    """
    return _attention_gate_up_target(prefix) or _next_layer_input_proj_target(prefix)


def attention_all_reduce_with_prefetch(output, prefix):
    # MatmulAllreduceRowParallelOp already calls this inside a custom op.
    target = _all_reduce_prefetch_target(prefix)
    if target is None:
        return tensor_model_parallel_all_reduce(output)
    return all_reduce_with_weight_prefetch(output, target[0], target[1], prefix.endswith(".mlp.down_proj"))


def all_reduce_with_weight_prefetch(
    output: torch.Tensor, weight: torch.Tensor, ratio: float, reduce_first: bool
) -> torch.Tensor:
    """Keep the token gate and stream operations opaque to torch.compile.

    profile_run may trace with a prefill shape. The ordinary RowParallel
    path must call this through torch.ops so a later decode capture still
    evaluates the gate with its actual shape. Returning the reduction puts
    the op on the real data path; no unused side-effect-only node is needed.
    """
    window = _all_reduce_prefetch_window(reduce_first)
    size = _prefetch_size(weight, ratio, window)
    if size <= 0 or not 0 < output.shape[0] < MLP_PREFETCH_TOKEN_THRESHOLD:
        return tensor_model_parallel_all_reduce(output)

    if reduce_first:
        compute = torch_npu.npu.current_stream()
        stream = prefetch_stream()
        # Snapshot the MatMul boundary before submitting the collective.
        # Prefetch must not wait for, or depend on, its in-place updates.
        stream.wait_stream(compute)
        output = tensor_model_parallel_all_reduce(output)
        with npu_stream_switch(stream):
            torch_npu.npu_prefetch(weight, None, size, 0)
    else:
        stream = _start_prefetch(weight, output, ratio, window=window)
        output = tensor_model_parallel_all_reduce(output)
    _finish_prefetch(stream)
    return output


def _all_reduce_with_weight_prefetch_fake(
    output: torch.Tensor, weight: torch.Tensor, ratio: float, reduce_first: bool
) -> torch.Tensor:
    return torch.empty_like(output)


def _prefetch_size(weight: torch.Tensor, ratio: float, window: str | None = None) -> int:
    weight_bytes = max(0, int(weight.numel() * weight.element_size() * ratio))
    cap = _window_cap_bytes(window)
    if cap is None:
        return weight_bytes
    return min(weight_bytes, cap)


def _start_prefetch(
    weight: torch.Tensor,
    dependency: torch.Tensor,
    ratio: float,
    num_tokens: int | None = None,
    window: str | None = None,
):
    # Decide inside the op: profile_run may trace at a prefill size while
    # ACL graphs are subsequently captured at smaller decode sizes.
    size = _prefetch_size(weight, ratio, window)
    tokens = dependency.shape[0] if num_tokens is None else num_tokens
    if size <= 0 or not 0 < tokens < MLP_PREFETCH_TOKEN_THRESHOLD:
        return None
    compute = torch_npu.npu.current_stream()
    stream = prefetch_stream()
    stream.wait_stream(compute)
    with npu_stream_switch(stream):
        torch_npu.npu_prefetch(weight, dependency, size, 0)
    return stream


def _finish_prefetch(stream):
    if stream is not None:
        torch_npu.npu.current_stream().wait_stream(stream)


def prefetch_gemma_rms_norm(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    ratio: float,
    attention_prefix: str = "",
    effective_norm_weight: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    # Repeat the runtime eligibility check, not a Python trace-time token
    # decision. Other reductions retain the Norm-overlap fallback.
    prefetched = (
        attention_prefix != ""
        and 0 < x.shape[0] < MLP_PREFETCH_TOKEN_THRESHOLD
        and _attention_gate_up_target(attention_prefix) is not None
    )
    stream = (
        None
        if prefetched
        else _start_prefetch(weight, x, ratio, window=WEIGHT_PREFETCH_WINDOW_MLP_GATE_UP)
    )
    scale = norm_weight if effective_norm_weight else 1.0 + norm_weight
    if enable_custom_op():
        output, _, residual_out = torch.ops._C_ascend.npu_add_rms_norm_bias(x, residual, scale, None, epsilon)
    else:
        output, _, residual_out = torch_npu.npu_add_rms_norm(x, residual, scale, epsilon)
    _finish_prefetch(stream)
    return output, residual_out


def _prefetch_gemma_rms_norm_fake(
    x: torch.Tensor,
    residual: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    epsilon: float,
    ratio: float,
    attention_prefix: str = "",
    effective_norm_weight: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.empty_like(x), torch.empty_like(residual)


def prefetch_swiglu(x: torch.Tensor, weight: torch.Tensor, ratio: float) -> torch.Tensor:
    stream = _start_prefetch(weight, x, ratio, window=WEIGHT_PREFETCH_WINDOW_MLP_DOWN)
    output = torch_npu.npu_swiglu(x)
    _finish_prefetch(stream)
    return output


def attention_gate_with_prefetch(
    attn_output: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, ratio: float
) -> torch.Tensor:
    """Apply full_attention's output gate, then join o_proj's prefetch.

    triton_split_qkv_rmsnorm_mrope() issued that fetch before the attention
    core. This is the last real computation before o_proj reads the weight, so
    it is where the fence belongs -- the same issue/join split GDN uses for
    linear_attn's out_proj. Returning the real gated result keeps the fence in
    the compiled graph. The guard must match the issue site's.
    """
    output = attn_output * torch.sigmoid(gate)
    if (
        _prefetch_size(weight, ratio, WEIGHT_PREFETCH_WINDOW_ATTN_O) > 0
        and 0 < attn_output.shape[0] < MLP_PREFETCH_TOKEN_THRESHOLD
    ):
        _finish_prefetch(prefetch_stream())
    return output


def _attention_gate_with_prefetch_fake(
    attn_output: torch.Tensor, gate: torch.Tensor, weight: torch.Tensor, ratio: float
) -> torch.Tensor:
    return torch.empty_like(attn_output)


def prefetch_gated_norm(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    ratio: float,
    num_tokens: int,
) -> torch.Tensor:
    """Join out_proj prefetch launched inside the GDN core, after gated norm.

    The real norm output keeps the join in the compiled graph. Token counts
    refer to tokens, not the flattened token/head rows used by this norm.
    """
    from vllm_ascend.ops.layernorm import LayerNormFn

    out = LayerNormFn.apply(
        x, norm_weight, None, z, eps, None if group_size < 0 else group_size, norm_before_gate, True
    )
    if (
        ratio > 0
        and _prefetch_size(weight, ratio, WEIGHT_PREFETCH_WINDOW_ATTN_O) > 0
        and 0 < num_tokens < MLP_PREFETCH_TOKEN_THRESHOLD
    ):
        _finish_prefetch(prefetch_stream())
    return out


def _prefetch_gated_norm_fake(
    x: torch.Tensor,
    z: torch.Tensor,
    norm_weight: torch.Tensor,
    weight: torch.Tensor,
    eps: float,
    group_size: int,
    norm_before_gate: bool,
    ratio: float,
    num_tokens: int,
) -> torch.Tensor:
    return torch.empty_like(x)


def _prefetch_swiglu_fake(x: torch.Tensor, weight: torch.Tensor, ratio: float) -> torch.Tensor:
    return x.new_empty((*x.shape[:-1], x.shape[-1] // 2))


def dense_mlp_with_prefetch(layer, hidden_states, residual, method):
    """Use this decoder's weights directly, including for PP and MTP layers."""
    mlp = layer.mlp
    norm = layer.post_attention_layernorm
    attention = getattr(layer, "self_attn", None)
    projection = getattr(attention, "o_proj", None)
    if projection is None:
        projection = getattr(getattr(layer, "linear_attn", None), "out_proj", None)
    attention_prefix = getattr(projection, "prefix", "")
    # maybe_chunk_residual is not registered in this tree, and the dense MLP
    # prefetch path is gated on `not use_sp`, so there is no sequence-parallel
    # residual to chunk here.
    hidden_states, residual = torch.ops.vllm.prefetch_gemma_rms_norm(
        hidden_states,
        residual,
        gemma_norm_weight(norm),
        mlp.gate_up_proj.weight,
        norm.variance_epsilon,
        float(method.mlp.prefetch_ratio.get("gate_up", 0)),
        attention_prefix,
        True,
    )
    gate_up, _ = mlp.gate_up_proj(hidden_states)
    hidden_states = torch.ops.vllm.prefetch_swiglu(
        gate_up, mlp.down_proj.weight, float(method.mlp.prefetch_ratio.get("down", 0))
    )
    # The row-parallel reduction owns next-layer QKV prefetch. Keeping a
    # second issue site here both duplicates it and exposes a token gate to
    # tracing at the prefill shape.
    hidden_states, _ = mlp.down_proj(hidden_states)
    return hidden_states, residual


direct_register_custom_op(
    op_name="all_reduce_with_weight_prefetch",
    op_func=all_reduce_with_weight_prefetch,
    fake_impl=_all_reduce_with_weight_prefetch_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="prefetch_gemma_rms_norm",
    op_func=prefetch_gemma_rms_norm,
    fake_impl=_prefetch_gemma_rms_norm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="attention_gate_with_prefetch",
    op_func=attention_gate_with_prefetch,
    fake_impl=_attention_gate_with_prefetch_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="prefetch_gated_norm",
    op_func=prefetch_gated_norm,
    fake_impl=_prefetch_gated_norm_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
direct_register_custom_op(
    op_name="prefetch_swiglu",
    op_func=prefetch_swiglu,
    fake_impl=_prefetch_swiglu_fake,
    mutates_args=[],
    dispatch_key="PrivateUse1",
)
