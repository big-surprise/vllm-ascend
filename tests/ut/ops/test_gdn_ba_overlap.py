# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from dataclasses import replace
from types import SimpleNamespace

import pytest
from vllm.forward_context import override_forward_context
from vllm.v1.attention.backends.gdn_attn import GDNAttentionMetadata

from vllm_ascend.ascend_config import AscendFusionConfig
from vllm_ascend.ops.gdn_ba_overlap import _is_single_decode


def test_gdn_ba_overlap_is_opt_in():
    assert not AscendFusionConfig().overlap_gdn_qkvz_ba


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, True),
        ({"num_prefills": 1}, False),
        ({"num_spec_decodes": 1}, False),
        ({"num_decodes": 0}, False),
        ({"num_decodes": 2, "num_actual_tokens": 2}, False),
    ],
)
def test_gdn_ba_overlap_requires_actual_single_token_decode(overrides, expected):
    metadata = GDNAttentionMetadata(
        num_prefills=0,
        num_prefill_tokens=0,
        num_decodes=1,
        num_decode_tokens=1,
        num_spec_decodes=0,
        num_spec_decode_tokens=0,
        num_actual_tokens=1,
    )
    context = SimpleNamespace(attn_metadata={"gdn": replace(metadata, **overrides)})
    with override_forward_context(context):
        assert _is_single_decode("gdn") is expected
        assert not _is_single_decode("other")
    assert not _is_single_decode("gdn")


@pytest.mark.parametrize("tp_size", [1, 2, 4, 8])
def test_gdn_ba_overlap_checks_shard_geometry(monkeypatch, tp_size):
    import torch

    from vllm_ascend.ops import gdn_ba_overlap as overlap

    monkeypatch.setattr(overlap, "gdn_ba_overlap_enabled", lambda: True)
    monkeypatch.setattr(overlap, "_is_local_unquantized_projection", lambda projection: True)
    qkvz = SimpleNamespace(
        tp_size=tp_size, weight=torch.empty(16384 // tp_size, 5120)
    )
    ba = SimpleNamespace(
        tp_size=tp_size, weight=torch.empty(96 // tp_size, 5120)
    )
    layer = SimpleNamespace(
        tp_size=tp_size, num_spec=0, disable_tp_for_ba_proj=False,
        key_dim=2048, value_dim=6144, num_v_heads=48,
        in_proj_qkvz=qkvz, in_proj_ba=ba,
    )
    activation = SimpleNamespace(
        shape=(1, 5120), dtype=torch.bfloat16,
        device=SimpleNamespace(type="npu"),
    )
    assert overlap.can_overlap_gdn_qkvz_ba(layer, activation)
    ba.weight = torch.empty(ba.weight.shape[0] + 1, 5120)
    assert not overlap.can_overlap_gdn_qkvz_ba(layer, activation)
