# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import torch
import torch_npu

from vllm_ascend.ops import gdn
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ


def _attn_with_weights(monkeypatch, npu_format):
    monkeypatch.setattr(gdn, "_ba_fused_proj_enabled", lambda: True)
    monkeypatch.setattr(gdn.torch_npu, "get_npu_format", lambda tensor: npu_format)
    main = SimpleNamespace(
        weight=torch.empty(8, 4, dtype=torch.bfloat16),
        bias=None,
        weight_scale=None,
    )
    ba = SimpleNamespace(
        weight=torch.empty(2, 4, dtype=torch.bfloat16),
        bias=None,
        weight_scale=None,
    )
    attn = torch.nn.Module()
    attn.in_proj_qkv = main
    attn.in_proj_ba = ba
    return attn, main, ba


def test_internal_format_skips_ba_fusion(monkeypatch):
    attn, main, ba = _attn_with_weights(monkeypatch, ACL_FORMAT_FRACTAL_NZ)
    calls = []

    def forbidden_cat(*args, **kwargs):
        calls.append(args)
        return torch.empty(10, 4)

    monkeypatch.setattr(gdn.torch, "cat", forbidden_cat)
    gdn.fuse_ba_into_qkvz(attn)
    assert not calls
    assert not getattr(attn, "_ba_fused", False)
    assert attn.in_proj_qkv.weight.shape == (8, 4)
    assert attn.in_proj_ba.weight.shape == (2, 4)


def test_nd_weights_still_fuse(monkeypatch):
    attn, main, ba = _attn_with_weights(monkeypatch, torch_npu.Format.ND)
    gdn.fuse_ba_into_qkvz(attn)
    assert attn._ba_fused is True
    assert attn._fused_qkvz_ba_w.shape == (10, 4)
    assert attn._main_cols == 8
    assert attn._ba_cols == 2
