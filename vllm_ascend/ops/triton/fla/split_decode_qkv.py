# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

"""Copy the Q/K/V parts of a multi-token GDN decode result in one launch."""

import torch
from vllm.triton_utils import tl, triton

_BLOCK = 1024


@triton.jit
def _split_decode_qkv_kernel(
    source,
    query,
    key,
    value,
    SOURCE_ROW_STRIDE: tl.constexpr,
    SOURCE_COL_STRIDE: tl.constexpr,
    KEY_DIM: tl.constexpr,
    VALUE_DIM: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    columns = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    key_mask = columns < KEY_DIM
    value_mask = columns < VALUE_DIM

    q = tl.load(source + row * SOURCE_ROW_STRIDE + columns * SOURCE_COL_STRIDE, mask=key_mask, other=0)
    k = tl.load(source + row * SOURCE_ROW_STRIDE + (KEY_DIM + columns) * SOURCE_COL_STRIDE, mask=key_mask, other=0)
    v = tl.load(
        source + row * SOURCE_ROW_STRIDE + (2 * KEY_DIM + columns) * SOURCE_COL_STRIDE,
        mask=value_mask,
        other=0,
    )
    tl.store(query + row * KEY_DIM + columns, q, mask=key_mask)
    tl.store(key + row * KEY_DIM + columns, k, mask=key_mask)
    tl.store(value + row * VALUE_DIM + columns, v, mask=value_mask)


def split_decode_qkv(
    mixed_qkv: torch.Tensor,
    key_dim: int,
    value_dim: int,
    head_k_dim: int,
    head_v_dim: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    num_tokens = mixed_qkv.shape[0]
    query = torch.empty(
        (1, num_tokens, key_dim // head_k_dim, head_k_dim), device=mixed_qkv.device, dtype=mixed_qkv.dtype
    )
    key = torch.empty_like(query)
    value = torch.empty(
        (1, num_tokens, value_dim // head_v_dim, head_v_dim), device=mixed_qkv.device, dtype=mixed_qkv.dtype
    )
    _split_decode_qkv_kernel[(num_tokens, triton.cdiv(max(key_dim, value_dim), _BLOCK))](
        mixed_qkv,
        query,
        key,
        value,
        mixed_qkv.stride(0),
        mixed_qkv.stride(1),
        key_dim,
        value_dim,
        _BLOCK,
    )
    return query, key, value
