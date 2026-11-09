# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM Ascend project

from types import SimpleNamespace

import pytest
import torch
from vllm.config import CUDAGraphMode
from vllm.forward_context import ForwardContext, override_forward_context
from vllm.v1.worker.gpu.model_runner import GPUModelRunner

from vllm_ascend.ascend_forward_context import _EXTRA_CTX
from vllm_ascend.ops import mlp_weight_prefetch as prefetch
from vllm_ascend.ops.linear import AscendRowParallelLinear
from vllm_ascend.ops.weight_prefetch import WeightPrefetchMethod
from vllm_ascend.worker.v2.aclgraph_utils import ModelWithContext
from vllm_ascend.worker.v2.model_runner import NPUModelRunner


class PrefetchModel(torch.nn.Module):
    def __init__(self, start_layer=0, multimodal=False, fail=False):
        super().__init__()
        self.fail = fail
        self.seen = []
        self.model = torch.nn.Module()
        self.model.start_layer = start_layer
        self.model.layers = torch.nn.ModuleList([torch.nn.Module() for _ in range(start_layer + 2)])
        layer = self.model.layers[start_layer]
        layer.mlp = torch.nn.Module()
        layer.mlp.gate_up_proj = torch.nn.Linear(8, 16, bias=False)
        layer.self_attn = torch.nn.Module()
        projection = AscendRowParallelLinear.__new__(AscendRowParallelLinear)
        torch.nn.Module.__init__(projection)
        projection.prefix = f"model.layers.{start_layer}.self_attn.o_proj"
        projection.custom_op = None
        projection.tp_size = 4
        projection.reduce_results = True
        projection.input_is_parallel = True
        layer.self_attn.o_proj = projection
        self.prefix = projection.prefix
        self.expected_weight = layer.mlp.gate_up_proj.weight
        if multimodal:
            self.language_model = torch.nn.Module()
            self.language_model.model = self.model
            del self.model
            self.prefix = "language_model." + self.prefix
            projection.prefix = self.prefix

    def forward(self, x):
        self.seen.append(
            (
                _EXTRA_CTX.model_instance,
                _EXTRA_CTX.layer_idx,
                _EXTRA_CTX.prefetch_mlp_gate_up_proj,
                _EXTRA_CTX.prefetch_mlp_down_proj,
            )
        )
        target = prefetch._attention_gate_up_target(self.prefix)
        _EXTRA_CTX.layer_idx = 999
        _EXTRA_CTX.prefetch_mlp_gate_up_proj = True
        _EXTRA_CTX.prefetch_mlp_down_proj = True
        if self.fail:
            raise RuntimeError("forward failed")
        return target


def load_runner(monkeypatch, target, draft, enabled=True):
    runner = NPUModelRunner.__new__(NPUModelRunner)
    runner.vllm_config = SimpleNamespace()
    runner.ascend_config = SimpleNamespace(weight_prefetch_config=SimpleNamespace(enabled=enabled))
    runner.compilation_config = SimpleNamespace(cudagraph_mode=CUDAGraphMode.NONE)

    def parent_load(self, *args, **kwargs):
        self.model = target

    monkeypatch.setattr(GPUModelRunner, "load_model", parent_load)
    monkeypatch.setattr(runner, "get_draft_model", lambda: draft)
    monkeypatch.setattr("vllm_ascend.worker.v2.model_runner.get_pd_dspark_aux_layer_ids", lambda config: ())
    runner.load_model()
    return runner


@pytest.fixture
def v2_context(monkeypatch):
    monkeypatch.setenv("VLLM_USE_V2_MODEL_RUNNER", "1")
    method = SimpleNamespace(
        is_qwen_dense=True,
        use_qwen_dense_mlp_prefetch=True,
        mlp=SimpleNamespace(prefetch_ratio={"gate_up": 1.0}),
    )
    monkeypatch.setattr(prefetch, "get_weight_prefetch_method", lambda: method)
    monkeypatch.setattr(torch.npu, "is_current_stream_capturing", lambda: False)
    return ForwardContext(
        no_compile_layers={}, attn_metadata=None, slot_mapping=None, additional_kwargs={"num_tokens": 1}
    )


@pytest.mark.parametrize("multimodal", [False, True])
@pytest.mark.parametrize("start_layer", [0, 2])
@pytest.mark.parametrize("capturing", [False, True])
def test_v2_prefetch_resolves_model_weights_and_resets_each_forward(
    monkeypatch, v2_context, multimodal, start_layer, capturing
):
    target = PrefetchModel(start_layer, multimodal)
    draft = PrefetchModel(0, False)
    runner = load_runner(monkeypatch, target, draft)
    model = ModelWithContext(runner.model) if capturing else runner.model
    with override_forward_context(v2_context):
        for current in (model, draft, model):
            result = current(torch.empty(1, 8))
            original = current.original_model if isinstance(current, ModelWithContext) else current
            assert result is not None
            assert result[0] is original.expected_weight
            expected_start = start_layer if original is target else 0
            assert original.seen[-1] == (original, expected_start, False, False)


def test_v2_prefetch_disabled_does_not_install_hooks(monkeypatch):
    target, draft = PrefetchModel(), PrefetchModel()
    load_runner(monkeypatch, target, draft, enabled=False)
    assert not target._forward_pre_hooks
    assert not draft._forward_pre_hooks


def test_v2_prefetch_resets_after_failed_forward(monkeypatch, v2_context):
    model = PrefetchModel(start_layer=2, fail=True)
    load_runner(monkeypatch, model, None)
    with override_forward_context(v2_context):
        with pytest.raises(RuntimeError, match="forward failed"):
            model(torch.empty(1, 8))
        model.fail = False
        assert model(torch.empty(1, 8))[0] is model.expected_weight
        assert model.seen[-1] == (model, 2, False, False)


def test_v2_prefetch_model_without_decoder_layers(monkeypatch, v2_context):
    class Model(torch.nn.Module):
        def forward(self, x):
            return _EXTRA_CTX.model_instance, _EXTRA_CTX.layer_idx

    model = Model()
    load_runner(monkeypatch, model, None)
    with override_forward_context(v2_context):
        assert model(torch.empty(1)) == (model, None)
        assert _EXTRA_CTX.prefetch_mlp_gate_up_proj is False
        assert _EXTRA_CTX.prefetch_mlp_down_proj is False


def test_v2_legacy_dense_prefetch_uses_local_pp_layer(monkeypatch, v2_context):
    model = PrefetchModel(start_layer=2)
    mlp = model.model.layers[2].mlp
    mlp.down_proj = torch.nn.Linear(16, 8, bias=False)
    method = WeightPrefetchMethod.__new__(WeightPrefetchMethod)
    method.mlp = SimpleNamespace(enable=True, prefetch_ratio={"gate_up": 1.0, "down": 1.0})
    method.use_qwen_dense_mlp_prefetch = False
    issued = []
    joined = []
    monkeypatch.setattr(torch.ops.vllm, "prefetch_preprocess", lambda **kwargs: issued.append(kwargs["weight"]))
    monkeypatch.setattr(torch.ops.vllm, "prefetch_postprocess", lambda x: joined.append(x))

    def forward(x):
        method.maybe_prefetch_mlp_weight_preprocess(method.MLP_GATE_UP, x, model.prefix)
        method.maybe_prefetch_mlp_weight_postprocess(x)
        method.maybe_prefetch_mlp_weight_preprocess(method.MLP_DOWN, x)
        method.maybe_prefetch_mlp_weight_postprocess(x)

    model.forward = forward
    load_runner(monkeypatch, model, None)
    with override_forward_context(v2_context):
        model(torch.empty(1, 8))
        assert _EXTRA_CTX.layer_idx == 3
        assert _EXTRA_CTX.prefetch_mlp_gate_up_proj is False
        assert _EXTRA_CTX.prefetch_mlp_down_proj is False
    assert len(issued) == 2
    assert issued[0] is mlp.gate_up_proj.weight
    assert issued[1] is mlp.down_proj.weight
    assert len(joined) == 2


def test_v2_moe_prefetch_resolves_experts_after_layernorm(monkeypatch, v2_context):
    model = PrefetchModel(start_layer=2)
    experts = torch.nn.Module()
    experts.w13_weight = torch.nn.Parameter(torch.empty(2, 16, 8))
    model.model.layers[2].mlp.experts = experts
    method = WeightPrefetchMethod.__new__(WeightPrefetchMethod)

    def forward(x):
        # RMSNorm advances the MoE cursor before prefetch reads the experts.
        _EXTRA_CTX.layer_idx += 1
        return method._get_moe_gate_up_prefetch_weight()

    model.forward = forward
    load_runner(monkeypatch, model, None)
    with override_forward_context(v2_context):
        assert model(torch.empty(1, 8)) is experts.w13_weight
        assert model(torch.empty(1, 8)) is experts.w13_weight
