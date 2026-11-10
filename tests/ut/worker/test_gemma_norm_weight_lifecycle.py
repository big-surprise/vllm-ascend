# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest.mock import Mock

from vllm_ascend.worker import worker as worker_module


def test_worker_refreshes_target_and_distinct_draft(monkeypatch):
    target, draft = object(), object()
    refresh = Mock()
    monkeypatch.setattr(worker_module, "is_950", lambda: True)
    monkeypatch.setattr(worker_module, "refresh_model_gemma_norm_weights", refresh)
    fake = SimpleNamespace(
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(model_type="qwen3_5_text")),
        model_runner=SimpleNamespace(get_model=lambda: target, get_draft_model=lambda: draft),
    )
    worker_module.NPUWorker._refresh_gemma_norm_weights(fake)
    assert [c.args[0] for c in refresh.call_args_list] == [target, draft]


def test_reload_retires_reads_before_refresh():
    order = []
    fake = SimpleNamespace(
        synchronize_device=lambda: order.append("sync"),
        model_runner=SimpleNamespace(reload_weights=lambda *a, **kw: order.append("reload")),
        _refresh_gemma_norm_weights=lambda: order.append("refresh"),
    )
    worker_module.NPUWorker.reload_weights(fake)
    assert order == ["sync", "reload", "refresh", "sync"]


def test_weight_transfer_refreshes_after_finalization():
    order = []
    fake = SimpleNamespace(
        _check_weight_transfer_engine=lambda: None,
        _weight_update_active=True,
        weight_transfer_engine=SimpleNamespace(finish_weight_update=lambda: order.append("finish")),
        model_runner=SimpleNamespace(reset_lora_state=lambda: order.append("reset_lora")),
        _refresh_gemma_norm_weights=lambda: order.append("refresh"),
        synchronize_device=lambda: order.append("sync"),
    )
    worker_module.NPUWorker.finish_weight_update(fake)
    assert order == ["finish", "refresh", "sync", "reset_lora"]
    assert not fake._weight_update_active
