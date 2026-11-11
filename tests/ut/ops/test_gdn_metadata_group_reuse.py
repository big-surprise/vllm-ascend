#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Tests for cross-group GDN attention metadata template reuse."""

from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest
import torch

import vllm_ascend.worker.v2.attn_utils as attn_utils
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ops import gdn_attn_builder as gdn_module
from vllm_ascend.ops.gdn_attn_builder import AscendGDNAttentionMetadataBuilder


def _make_builder(*, supports: bool = True, use_full_graph: bool = False, spec: Any = None):
    """Create a builder without running the heavyweight upstream __init__."""
    builder = AscendGDNAttentionMetadataBuilder.__new__(AscendGDNAttentionMetadataBuilder)
    builder.supports_update_block_table = supports
    builder.use_full_cuda_graph = use_full_graph
    builder.decode_cudagraph_max_bs = 8
    builder.kv_cache_spec = spec if spec is not None else ("mamba", 128)
    builder.vllm_config = SimpleNamespace(cache_config=SimpleNamespace(mamba_cache_mode="none"))
    builder._pad_decode_metadata = MagicMock()
    builder._attach_non_spec_decode_metadata = MagicMock()
    return builder


def _make_template(**overrides):
    template = SimpleNamespace(
        num_prefills=0,
        num_spec_decodes=0,
        num_decodes=2,
        num_decode_tokens=2,
        non_spec_query_start_loc=torch.tensor([0, 1, 2]),
        non_spec_state_indices_tensor=torch.tensor([99, 99]),
        spec_decode_metadata=None,
        non_spec_prefill_metadata=None,
        non_spec_decode_metadata=SimpleNamespace(name="decode"),
    )
    for name, value in overrides.items():
        setattr(template, name, value)
    return template


def _make_common(*, rows: int = 2, block_table: torch.Tensor | None = None):
    table = torch.tensor([[10, 100], [11, 101]])[:rows] if block_table is None else block_table
    common = SimpleNamespace(
        num_reqs=rows,
        num_actual_tokens=rows,
        block_table_tensor=table,
        seq_lens=torch.ones(rows, dtype=torch.int32),
        query_start_loc=torch.arange(rows + 1),
    )
    common.unpadded = lambda num_tokens, num_actual: common
    return common


@pytest.fixture
def patched_gdn_helpers(monkeypatch):
    monkeypatch.setattr(
        gdn_module,
        "mamba_get_block_table_tensor",
        lambda block_table, seq_lens, spec, mode: block_table,
    )
    monkeypatch.setattr(
        gdn_module,
        "_treat_single_token_prefills_with_state_as_decodes",
        lambda common: common,
    )


def test_update_block_table_rebinds_state_to_current_group(patched_gdn_helpers):
    builder = _make_builder()
    template = _make_template()
    common = _make_common()

    result = builder.update_block_table(template, common, num_actual_reqs=2)

    assert result is not template
    assert result.num_prefills == 0
    assert result.non_spec_query_start_loc is common.query_start_loc
    assert torch.equal(result.non_spec_state_indices_tensor, torch.tensor([10, 11]))
    builder._pad_decode_metadata.assert_not_called()
    builder._attach_non_spec_decode_metadata.assert_called_once()
    metadata_arg, cache_indices = builder._attach_non_spec_decode_metadata.call_args.args
    assert metadata_arg is result
    assert torch.equal(cache_indices, common.block_table_tensor)
    # The owner template must stay untouched for the remaining groups.
    assert torch.equal(template.non_spec_state_indices_tensor, torch.tensor([99, 99]))
    assert template.non_spec_query_start_loc is not common.query_start_loc


def test_full_graph_rebinds_to_this_builders_graph_buffer(patched_gdn_helpers):
    builder = _make_builder(use_full_graph=True)
    current_graph_buffer = torch.tensor([20, 21, 0, 0])

    def materialize(metadata, graph_request_count):
        metadata.non_spec_state_indices_tensor = current_graph_buffer

    builder._pad_decode_metadata.side_effect = materialize

    result = builder.update_block_table(_make_template(), _make_common(), 2)

    assert result is not None
    builder._pad_decode_metadata.assert_called_once()
    _, cache_indices = builder._attach_non_spec_decode_metadata.call_args.args
    # A group must never read the owner builder's graph buffer.
    assert cache_indices is current_graph_buffer


@pytest.mark.parametrize(
    "overrides",
    [
        {"num_prefills": 1},
        {"num_spec_decodes": 1},
        {"num_decodes": 0},
        {"spec_decode_metadata": object()},
        {"non_spec_prefill_metadata": object()},
        {"non_spec_decode_metadata": None},
        {"non_spec_query_start_loc": None},
    ],
)
def test_update_block_table_falls_back_for_non_decode_templates(patched_gdn_helpers, overrides):
    builder = _make_builder()
    assert builder.update_block_table(_make_template(**overrides), _make_common(), 2) is None
    builder._attach_non_spec_decode_metadata.assert_not_called()


def test_update_block_table_disabled_without_v2_runner(patched_gdn_helpers):
    builder = _make_builder(supports=False)
    assert builder.update_block_table(_make_template(), _make_common(), 2) is None


def test_update_block_table_falls_back_on_shape_mismatch(patched_gdn_helpers, monkeypatch):
    monkeypatch.setattr(
        gdn_module,
        "mamba_get_block_table_tensor",
        lambda block_table, seq_lens, spec, mode: torch.ones(3, 2, dtype=torch.int32),
    )
    builder = _make_builder()
    assert builder.update_block_table(_make_template(), _make_common(), 2) is None


def test_reuse_key_requires_static_conditions():
    builder = _make_builder()

    def reuse_key(**overrides):
        params = dict(
            enabled=True,
            builder=builder,
            group_spec="mamba",
            group_causal=True,
            num_metadata_variants=1,
            pcp_context=None,
            extra_kwargs={},
        )
        params.update(overrides)
        return attn_utils._gdn_template_reuse_key(**params)

    assert reuse_key() is not None
    assert reuse_key() == reuse_key()
    assert reuse_key() != reuse_key(builder=_make_builder(spec=("mamba", 256)))
    assert reuse_key() != reuse_key(group_causal=False)
    assert reuse_key(enabled=False) is None
    assert reuse_key(builder=object()) is None
    assert reuse_key(builder=_make_builder(supports=False)) is None
    assert reuse_key(num_metadata_variants=2) is None
    assert reuse_key(pcp_context=object()) is None
    assert reuse_key(extra_kwargs={"num_actual_reqs": 2}) is None


def test_reuse_key_ignores_none_valued_spec_decode_kwargs():
    """MambaHybridAttnMetadata always emits the spec-decode kwargs for GDN
    builders, with None values when speculative decoding is off. That plain
    decode shape must not block template reuse at runtime."""
    builder = _make_builder()

    def reuse_key(extra_kwargs):
        return attn_utils._gdn_template_reuse_key(
            enabled=True,
            builder=builder,
            group_spec="mamba",
            group_causal=True,
            num_metadata_variants=1,
            pcp_context=None,
            extra_kwargs=extra_kwargs,
        )

    no_mtp_runtime = {"num_accepted_tokens": None, "num_decode_draft_tokens_cpu": None}
    assert reuse_key(no_mtp_runtime) == reuse_key({})
    mtp_runtime = {"num_accepted_tokens": torch.ones(2), "num_decode_draft_tokens_cpu": None}
    assert reuse_key(mtp_runtime) is None


def test_reuse_config_is_declared_and_opt_in_by_default():
    # init_ascend_config forwards additional_config keys to AscendConfig and
    # rejects anything absent from __dataclass_fields__, so declaring the
    # typed field is what makes --additional-config accept the switch.
    field = AscendConfig.__dataclass_fields__["enable_gdn_metadata_reuse"]
    assert field.default is False


def test_reuse_config_is_read_from_additional_config(monkeypatch):
    monkeypatch.setattr(
        attn_utils,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_gdn_metadata_reuse=True),
    )
    assert attn_utils._gdn_metadata_reuse_enabled() is True

    monkeypatch.setattr(
        attn_utils,
        "get_ascend_config",
        lambda: SimpleNamespace(enable_gdn_metadata_reuse=False),
    )
    assert attn_utils._gdn_metadata_reuse_enabled() is False


def test_reuse_without_config_init_keeps_original_path(monkeypatch):
    def uninitialized():
        raise RuntimeError("Ascend config is not initialized. Please call init_ascend_config first.")

    monkeypatch.setattr(attn_utils, "get_ascend_config", uninitialized)
    assert attn_utils._gdn_metadata_reuse_enabled() is False


def test_interleaved_hybrid_groups_share_one_template_key():
    """Interleaved hybrid KV cache groups wrap layers in UniformTypeKVCacheSpecs
    keyed by per-group layer names, and may mix the group's full-attention
    layers into the same wrapper. The template key must follow the builder's
    own unwrapped spec, not the group wrapper."""
    mamba_spec = ("mamba", 128)
    builder = _make_builder(spec=mamba_spec)

    def reuse_key(builder, group_spec):
        return attn_utils._gdn_template_reuse_key(
            enabled=True,
            builder=builder,
            group_spec=group_spec,
            group_causal=True,
            num_metadata_variants=1,
            pcp_context=None,
            extra_kwargs={},
        )

    # Homogeneous wrappers with different layer names.
    first_segment = SimpleNamespace(kv_cache_specs={"layer.0": mamba_spec, "layer.1": mamba_spec})
    second_segment = SimpleNamespace(kv_cache_specs={"layer.4": mamba_spec, "layer.5": mamba_spec})
    # Mixed sandwich groups: full attention and GDN layers share one wrapper.
    sandwich = SimpleNamespace(kv_cache_specs={"layer.6": ("full", 128), "layer.7": mamba_spec})

    assert reuse_key(builder, first_segment) == reuse_key(builder, second_segment)
    assert reuse_key(builder, first_segment) == reuse_key(builder, sandwich)
    other_builder = _make_builder(spec=("mamba", 256))
    assert reuse_key(builder, first_segment) != reuse_key(other_builder, first_segment)


def test_reuse_key_falls_back_to_group_spec_without_builder_spec():
    builder = _make_builder()
    builder.kv_cache_spec = None
    mamba_spec = ("mamba", 128)

    def reuse_key(group_spec):
        return attn_utils._gdn_template_reuse_key(
            enabled=True,
            builder=builder,
            group_spec=group_spec,
            group_causal=True,
            num_metadata_variants=1,
            pcp_context=None,
            extra_kwargs={},
        )

    first_segment = SimpleNamespace(kv_cache_specs={"layer.0": mamba_spec, "layer.1": mamba_spec})
    second_segment = SimpleNamespace(kv_cache_specs={"layer.4": mamba_spec, "layer.5": mamba_spec})
    heterogeneous = SimpleNamespace(kv_cache_specs={"layer.0": mamba_spec, "layer.1": ("full", 128)})

    assert reuse_key(first_segment) == reuse_key(second_segment)
    assert reuse_key(first_segment) != reuse_key(heterogeneous)
