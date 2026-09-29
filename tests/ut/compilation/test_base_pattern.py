from unittest.mock import Mock, patch

import torch

from vllm_ascend.compilation.passes import base_pattern
from vllm_ascend.compilation.passes.base_pattern import BasePattern


class _TestPattern(BasePattern):
    def get_inputs(self):
        return [Mock()]

    def get_pattern(self):
        return lambda x: x

    def get_replacement(self):
        return lambda x: x


def _pattern():
    config = Mock()
    config.model_config.dtype = torch.bfloat16
    return _TestPattern(config)


def test_local_only_pattern_is_not_registered_with_backend():
    base_pattern._registered_patterns.clear()

    with (
        patch.object(base_pattern.pm, "register_replacement") as local_register,
        patch.object(base_pattern.nge, "register_replacement") as backend_register,
    ):
        pass_manager = Mock()
        _pattern().register(pass_manager, register_with_backend=False)

    local_register.assert_called_once()
    assert local_register.call_args.args[-1] is pass_manager
    backend_register.assert_not_called()


def test_backend_deduplication_does_not_skip_new_local_pass():
    base_pattern._registered_patterns.clear()

    with (
        patch.object(base_pattern.pm, "register_replacement") as local_register,
        patch.object(base_pattern.nge, "register_replacement") as backend_register,
    ):
        _pattern().register(Mock())
        _pattern().register(Mock())

    assert local_register.call_count == 2
    backend_register.assert_called_once()
