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
