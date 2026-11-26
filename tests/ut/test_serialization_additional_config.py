# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
from vllm import envs
from vllm_ascend.ascend_config import AscendConfig


@pytest.mark.parametrize("requested,ambient,expected", [
    (None, "0", False), (None, "1", True), (True, "0", True), (False, "1", False),
])
def test_serialization_additional_config_mapping(monkeypatch, requested, ambient, expected):
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", ambient)
    config = SimpleNamespace(VLLM_ALLOW_INSECURE_SERIALIZATION=requested)
    AscendConfig._apply_serialization_setting(config)
    assert envs.VLLM_ALLOW_INSECURE_SERIALIZATION is expected


def test_serialization_setting_clears_stale_env_cache(monkeypatch):
    from functools import lru_cache
    original = envs.__getattr__
    cached = lru_cache()(original)
    monkeypatch.setattr(envs, "__getattr__", cached)
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "0")
    assert not envs.VLLM_ALLOW_INSECURE_SERIALIZATION
    AscendConfig._apply_serialization_setting(SimpleNamespace(VLLM_ALLOW_INSECURE_SERIALIZATION=True))
    assert envs.VLLM_ALLOW_INSECURE_SERIALIZATION


def test_serialization_key_has_typed_validation():
    config = AscendConfig(sparse_kv_offload_config=None, VLLM_ALLOW_INSECURE_SERIALIZATION="false")
    assert config.VLLM_ALLOW_INSECURE_SERIALIZATION is False
    with pytest.raises(ValueError, match="VLLM_ALLOW_INSECURE_SERIALIZATION"):
        AscendConfig(sparse_kv_offload_config=None, VLLM_ALLOW_INSECURE_SERIALIZATION=2)
