# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.ops import ffn_weight_layout as layout


def config_pair():
    config = SimpleNamespace(ffn_weight_layout="nz_kn", weight_nz_mode=0, rl_config=SimpleNamespace(enabled=False))
    vc = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_text"),
                                     dtype=torch.bfloat16, is_moe=False, quantization=None),
        lora_config=None, weight_transfer_config=None, cache_config=SimpleNamespace(),
        parallel_config=SimpleNamespace(prefill_context_parallel_size=1, decode_context_parallel_size=1),
    )
    return config, vc


def test_config_default_and_validation():
    assert AscendConfig.__dataclass_fields__["ffn_weight_layout"].default == "default"
    config, vc = config_pair()
    AscendConfig._validate_ffn_weight_layout(config, vc)
    config.ffn_weight_layout = "default"
    AscendConfig._validate_ffn_weight_layout(config, SimpleNamespace())


@pytest.mark.parametrize("target,field,value", [
    ("config", "weight_nz_mode", 2), ("model", "dtype", torch.float16),
    ("model", "is_moe", True), ("model", "quantization", "ascend"),
    ("model", "enable_sleep_mode", True), ("vc", "lora_config", object()),
    ("vc", "weight_transfer_config", object()), ("rl", "enabled", True),
    ("cache", "cpu_offload_gb", 1), ("cache", "offload_group_size", 1),
    ("parallel", "prefill_context_parallel_size", 2),
    ("parallel", "decode_context_parallel_size", 2),
])
def test_reject_incompatible_config(target, field, value):
    config, vc = config_pair()
    objects = {"config": config, "model": vc.model_config, "vc": vc, "rl": config.rl_config,
               "cache": vc.cache_config, "parallel": vc.parallel_config}
    setattr(objects[target], field, value)
    with pytest.raises(ValueError, match="ffn_weight_layout"):
        AscendConfig._validate_ffn_weight_layout(config, vc)


def test_reject_other_model():
    config, vc = config_pair()
    vc.model_config.hf_text_config.model_type = "qwen3"
    with pytest.raises(ValueError, match="Qwen3.5"):
        AscendConfig._validate_ffn_weight_layout(config, vc)


@pytest.mark.parametrize("enabled,prefix,skip,precast", [
    (False, "model.layers.0.mlp.gate_up_proj", False, False),
    (True, "model.layers.0.self_attn.qkv_proj", False, False),
    (True, "model.layers.0.mlp.shared_expert.down_proj", False, False),
    (True, "model.layers.0.mlp.down_proj", True, False),
    (True, "model.layers.0.mlp.down_proj", False, True),
])
def test_layout_gate_does_not_touch_other_weights(monkeypatch, enabled, prefix, skip, precast):
    monkeypatch.setattr(layout, "get_ascend_config", lambda: SimpleNamespace(ffn_weight_layout="nz_kn" if enabled else "default"))
    cast = Mock(side_effect=AssertionError("must not convert"))
    monkeypatch.setattr(layout.torch_npu, "npu_format_cast", cast)
    layer = SimpleNamespace(prefix=prefix, skip_weight_nz_conversion=skip, precast_fp32_weight=precast)
    assert not layout.maybe_prepare_ffn_nz_weight(layer)
    cast.assert_not_called()


def test_cpu_weight_is_not_moved(monkeypatch):
    monkeypatch.setattr(layout, "get_ascend_config", lambda: SimpleNamespace(ffn_weight_layout="nz_kn"))
    layer = torch.nn.Linear(16, 32, bias=False, dtype=torch.bfloat16)
    layer.prefix = "model.layers.0.mlp.down_proj"
    original = layer.weight
    assert not layout.maybe_prepare_ffn_nz_weight(layer)
    assert layer.weight is original


def test_ffn_layout_enum_rejects_typos():
    with pytest.raises(ValueError, match="ffn_weight_layout"):
        AscendConfig(sparse_kv_offload_config=None, ffn_weight_layout="nz")



@pytest.mark.parametrize("packed", [False, True])
def test_matmul_allreduce_pattern_keeps_packed_view_separate(packed):
    from vllm_ascend.compilation.passes.matmul_comm_fusion_pass import MatmulAllReducePattern

    pattern = MatmulAllReducePattern.__new__(MatmulAllReducePattern)
    pattern.min_tokens = 512
    graph = torch.fx.Graph()
    x, weight = graph.placeholder("x"), graph.placeholder("weight")
    x.meta["val"] = torch.empty(2048, 4, dtype=torch.bfloat16)
    weight.meta["val"] = (
        torch.empty(4, 8, dtype=torch.bfloat16).t()
        if packed else torch.empty(8, 4, dtype=torch.bfloat16)
    )
    gemm = graph.call_function(torch.ops.vllm.unquantized_gemm.default, (x, weight, None))
    assert pattern.get_extra_check()(SimpleNamespace(nodes=[gemm])) is (not packed)


@pytest.mark.parametrize("packed", [False, True])
def test_large_m_row_parallel_fallback_retains_reduction(monkeypatch, packed):
    from vllm_ascend.ops import linear_op

    x = torch.randn(512, 4)
    expected = x + 1
    gemm = Mock(return_value=expected)
    fused = Mock(return_value=x + 2)
    reduce = Mock(return_value=x + 3)
    monkeypatch.setattr(linear_op, "DeviceOperator", SimpleNamespace(npu_mm_all_reduce_base=fused))
    monkeypatch.setattr("vllm_ascend.ops.mlp_weight_prefetch.attention_all_reduce_with_prefetch", reduce)
    layer = SimpleNamespace(weight=torch.empty(8, 4), _ascend_ffn_nz_transposed=packed)
    op = SimpleNamespace(reduce_results=True, tp_size=4, layer=layer, quant_method=SimpleNamespace(apply=gemm),
                         hcomm_info="test", matmul_comm_mode="test", prefix="model.layers.0.mlp.down_proj")
    actual = linear_op.MatmulAllreduceRowParallelOp.matmul_and_reduce(op, x, None)
    if packed:
        gemm.assert_called_once_with(layer, x, bias=None)
        reduce.assert_called_once_with(expected, op.prefix)
        fused.assert_not_called()
        assert torch.equal(actual, x + 3)
    else:
        fused.assert_called_once()
        gemm.assert_not_called()
        reduce.assert_not_called()
        assert torch.equal(actual, x + 2)
