# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
"""One-kernel materialisation of the fused GDN qkvz+ba column slices.

The ba fusion emits a single ``[M, main_cols + ba_cols]`` GEMM output (4120
columns at TP4).  Splitting it along the last dimension gives *views* whose
row stride is still 4120.  ``F.linear`` output itself is contiguous, but the
slices are not as soon as M > 1, so every downstream op that insists on a
contiguous input (``causal_conv1d_update``, the gating / recurrent kernels)
materialises its slice with a separate ``aclnnInplaceCopy_SliceAiCore_Slice``
or ``aclnnContiguous_SliceAiCore_Slice`` launch.

At M == 1 those views are still reported contiguous (stride(0) is irrelevant
for a single row), so no copy happens at all.  That is exactly why the cost is
invisible at single concurrency and only shows up for batched decode: msprof
on 950DT / TP4 / con4 measured 4320 + 4320 + 2160 = 10800 such kernels in the
M=4 decode forwards, ~254us per forward spread over 48 GDN layers.

This module folds the three copies into one vector-core launch.  ``z`` is
deliberately excluded: downstream only reshapes it, and a reshape of a
last-dimension-contiguous view is free, so copying it would add HBM traffic
without removing any kernel.

The same kernel also serves the decode conv1d q/k/v split in ``gdn.py``.
``_rearrange_decode_qkv`` exposes the conv output ``[M, 2560]`` as three
``[M, N]`` column views that the recurrent / gating kernels require to be
contiguous.  At M == 1 those views are already contiguous; at M > 1 msprof
counted another 4320 + 2160 = 6480 SliceAiCore launches (~258us per forward
over 48 GDN layers) for q/k/v.  ``fused_split_contiguous`` /
``can_fuse_split`` are the layout-generic aliases used by both split sites.
"""

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _split_contiguous_kernel(
    S_QKV,
    S_B,
    S_A,
    D_QKV,
    D_B,
    D_A,
    M,
    RS_QKV: tl.constexpr,
    N_QKV: tl.constexpr,
    BLK_QKV: tl.constexpr,
    RS_B: tl.constexpr,
    N_B: tl.constexpr,
    BLK_B: tl.constexpr,
    RS_A: tl.constexpr,
    N_A: tl.constexpr,
    BLK_A: tl.constexpr,
    ROWS: tl.constexpr,
):
    """Gather three 2-D column views into three contiguous 2-D outputs.

    Each source may have a different row stride but must have ``stride(1)==1``
    (guaranteed by :func:`can_fuse_ba_split`).  Outputs are freshly allocated,
    so their row stride is exactly their column count.
    """
    rows = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
    row_mask = rows < M

    c_qkv = tl.arange(0, BLK_QKV)
    m_qkv = row_mask[:, None] & (c_qkv[None, :] < N_QKV)
    v_qkv = tl.load(S_QKV + rows[:, None] * RS_QKV + c_qkv[None, :], mask=m_qkv, other=0)
    tl.store(D_QKV + rows[:, None] * N_QKV + c_qkv[None, :], v_qkv, mask=m_qkv)

    c_b = tl.arange(0, BLK_B)
    m_b = row_mask[:, None] & (c_b[None, :] < N_B)
    v_b = tl.load(S_B + rows[:, None] * RS_B + c_b[None, :], mask=m_b, other=0)
    tl.store(D_B + rows[:, None] * N_B + c_b[None, :], v_b, mask=m_b)

    c_a = tl.arange(0, BLK_A)
    m_a = row_mask[:, None] & (c_a[None, :] < N_A)
    v_a = tl.load(S_A + rows[:, None] * RS_A + c_a[None, :], mask=m_a, other=0)
    tl.store(D_A + rows[:, None] * N_A + c_a[None, :], v_a, mask=m_a)


def can_fuse_ba_split(
    mixed_qkv: torch.Tensor, b: torch.Tensor, a: torch.Tensor
) -> bool:
    """Guard against a custom ``split_ba`` returning something exotic.

    All three must be 2-D, share the row count / dtype / device and be simple
    column slices (last-dim stride 1, identical row stride).  Anything else
    falls back to the proven per-slice ``.contiguous()`` path.
    """
    m = mixed_qkv.shape[0]
    rs = mixed_qkv.stride(0)
    for v in (mixed_qkv, b, a):
        if (
            v.dim() != 2
            or v.shape[0] != m
            or v.stride(1) != 1
            or v.stride(0) != rs
            or v.dtype != mixed_qkv.dtype
            or v.device != mixed_qkv.device
        ):
            return False
    return True


def fused_ba_split_contiguous(
    mixed_qkv: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialise three strided column views as contiguous tensors in one launch.

    ``z`` is intentionally not passed in: it is only reshaped downstream and
    a reshape of a last-dim-contiguous view is free, so copying it would just
    add HBM traffic.  The bytes moved here match the original three Slice
    kernels exactly (qkv + b + a columns), only the launch count drops 3 -> 1.
    """
    m = mixed_qkv.shape[0]
    n_qkv, n_b, n_a = mixed_qkv.shape[1], b.shape[1], a.shape[1]

    out_qkv = torch.empty((m, n_qkv), dtype=mixed_qkv.dtype, device=mixed_qkv.device)
    out_b = torch.empty((m, n_b), dtype=b.dtype, device=b.device)
    out_a = torch.empty((m, n_a), dtype=a.dtype, device=a.device)

    # One program per row keeps the QKV block (~8 KiB for 2560 bf16 columns)
    # well inside the vector-core UB.  This path is only used for small decode
    # batches (see _FUSED_SPLIT_MAX_M in gdn.py), so grid sizes stay tiny.
    rows = 1
    _split_contiguous_kernel[(m,)](
        mixed_qkv,
        b,
        a,
        out_qkv,
        out_b,
        out_a,
        m,
        mixed_qkv.stride(0),
        n_qkv,
        triton.next_power_of_2(n_qkv),
        b.stride(0),
        n_b,
        triton.next_power_of_2(n_b),
        a.stride(0),
        n_a,
        triton.next_power_of_2(n_a),
        rows,
    )
    return out_qkv, out_b, out_a


def can_fuse_split(v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor) -> bool:
    """Layout guard shared by the qkvz+ba and decode q/k/v split sites."""
    return can_fuse_ba_split(v0, v1, v2)


def fused_split_contiguous(
    v0: torch.Tensor, v1: torch.Tensor, v2: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Materialise three strided 2-D column views in one vector-core launch.

    ``fused_ba_split_contiguous`` keeps its name for the fused-weight path, but
    the underlying kernel is layout-generic: it takes a separate row stride
    and column count for each view.  This neutral alias lets
    ``_rearrange_decode_qkv`` reuse it for q/k/v without implying anything
    about ba, and lets both sites share a single guard.
    """
    return fused_ba_split_contiguous(v0, v1, v2)
