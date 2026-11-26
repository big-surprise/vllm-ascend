# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from tests.ut.ops.test_ffn_weight_layout import config_pair
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ops import attention_weight_layout as layout


@pytest.mark.parametrize("field", ["linear_attention_qkvz_layout", "full_attention_qkv_layout"])
def test_attention_layout_config_is_independent_and_typed(field):
    assert AscendConfig.__dataclass_fields__[field].default == "default"
    config, vc = config_pair()
    config.ffn_weight_layout = "default"
    setattr(config, field, "nz_kn")
    AscendConfig._validate_ffn_weight_layout(config, vc)
    vc.model_config.dtype = torch.float16
    with pytest.raises(ValueError, match=field):
        AscendConfig._validate_ffn_weight_layout(config, vc)
    with pytest.raises(ValueError, match=field):
        AscendConfig(sparse_kv_offload_config=None, **{field: "nz"})


def test_merged_qkvz_option_rejects_split_mode():
    config, vc = config_pair()
    config.ffn_weight_layout = "default"
    config.linear_attention_qkvz_layout = "nz_kn"
    config.enable_gdn_qkv_split = True
    with pytest.raises(ValueError, match="enable_gdn_qkv_split=false"):
        AscendConfig._validate_ffn_weight_layout(config, vc)
    config.linear_attention_qkvz_layout = "default"
    config.full_attention_qkv_layout = "nz_kn"
    AscendConfig._validate_ffn_weight_layout(config, vc)


@pytest.mark.parametrize(
    "prefix,field",
    [
        ("language_model.model.layers.0.linear_attn.in_proj_qkvz", "linear_attention_qkvz_layout"),
        ("language_model.model.layers.3.self_attn.qkv_proj", "full_attention_qkv_layout"),
    ],
)
@pytest.mark.parametrize("skip,precast,enabled", [(True, False, True), (False, True, True), (False, False, False)])
def test_skipped_projection_does_not_convert(monkeypatch, prefix, field, skip, precast, enabled):
    config = SimpleNamespace(**{field: "nz_kn" if enabled else "default"})
    monkeypatch.setattr(layout, "get_ascend_config", lambda: config)
    cast = Mock(side_effect=AssertionError("must not convert"))
    monkeypatch.setattr(layout.torch_npu, "npu_format_cast", cast)
    layer = SimpleNamespace(prefix=prefix, skip_weight_nz_conversion=skip, precast_fp32_weight=precast)
    assert not layout.maybe_prepare_attention_nz_weight(layer)
    cast.assert_not_called()


@pytest.mark.parametrize(
    "suffix",
    [
        ".linear_attn.in_proj_qkv",
        ".linear_attn.in_proj_z",
        ".linear_attn.in_proj_ba",
        ".linear_attn.out_proj",
        ".self_attn.o_proj",
        ".mlp.gate_up_proj",
        ".mlp.down_proj",
        ".lm_head",
    ],
)
def test_other_projections_remain_unchanged(monkeypatch, suffix):
    getter = Mock(side_effect=AssertionError("must not resolve config for an excluded projection"))
    monkeypatch.setattr(layout, "get_ascend_config", getter)
    assert not layout.maybe_prepare_attention_nz_weight(SimpleNamespace(prefix="model.layers.0" + suffix))
    getter.assert_not_called()


def test_cpu_weight_is_not_moved(monkeypatch):
    monkeypatch.setattr(layout, "get_ascend_config", lambda: SimpleNamespace(full_attention_qkv_layout="nz_kn"))
    layer = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    layer.prefix = "model.layers.3.self_attn.qkv_proj"
    original = layer.weight
    assert not layout.maybe_prepare_attention_nz_weight(layer)
    assert layer.weight is original


@pytest.mark.parametrize(
    "attention_name,projection_name",
    [
        ("linear_attn", "in_proj_qkvz"),
        ("self_attn", "qkv_proj"),
    ],
)
def test_next_layer_prefetch_uses_replaced_parameter(monkeypatch, attention_name, projection_name):
    from vllm_ascend.ops import mlp_weight_prefetch as prefetch

    weight = torch.nn.Parameter(torch.randn(256, 128).t(), requires_grad=False)
    projection = SimpleNamespace(weight=weight)
    following = SimpleNamespace(**{attention_name: SimpleNamespace(**{projection_name: projection})})
    layers = [SimpleNamespace(mlp=SimpleNamespace(down_proj=object())), following]
    method = SimpleNamespace(is_qwen_dense=True, attn=SimpleNamespace(enable=True, prefetch_ratio={"qkv": 1.0}))
    monkeypatch.setattr(prefetch, "get_weight_prefetch_method", lambda: method)
    monkeypatch.setattr(prefetch, "layers_of_model_instance", lambda model: layers)
    monkeypatch.setattr(prefetch, "_EXTRA_CTX", SimpleNamespace(model_instance=object()))
    monkeypatch.setattr(prefetch, "_uses_all_reduce_prefetch", lambda projection: True)
    selected, ratio = prefetch._next_layer_input_proj_target("model.layers.0.mlp.down_proj")
    assert selected is weight and selected.data_ptr() == weight.data_ptr()
    assert ratio == 1.0
