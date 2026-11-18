#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
#
import json
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
import torch
from vllm.distributed.device_communicators.base_device_communicator import DeviceCommunicatorBase

from tests.ut.base import TestBase
from vllm_ascend.ascend_config import AscendConfig
from vllm_ascend.distributed.device_communicators import hccl_outplace
from vllm_ascend.distributed.device_communicators.hccl_outplace import (
    OutplaceHcclAllReduce,
    _as_device,
)

BF16 = torch.bfloat16
COMM_HANDLE = 4242


def _comm(stream=None, lib=None):
    # device="cpu" so the tensor.device comparison runs on a device-less UT box.
    return OutplaceHcclAllReduce(
        lib=lib if lib is not None else MagicMock(),
        comm=COMM_HANDLE,
        stream=stream,
        device=torch.device("cpu"),
        world_size=2,
        rank_in_group=0,
    )


def _patched_call(comm, tensor):
    """Drive all_reduce with the ctypes wrappers neutralised."""
    with (
        patch.object(hccl_outplace, "_format_supported", return_value=True),
        patch.object(hccl_outplace, "buffer_type", side_effect=lambda p: p),
        patch.object(hccl_outplace, "aclrtStream_t", side_effect=lambda s: s),
        patch.object(hccl_outplace, "hcclComm_t", side_effect=lambda c: c),
    ):
        return comm.all_reduce(tensor)


class TestAllReduceOutplaceConfig(TestBase):
    """The switch is a flat ``additional_config`` key, not a sub-config."""

    def test_default_is_off(self):
        self.assertFalse(AscendConfig().allreduce_outplace)

    def test_lax_bool_coercion(self):
        self.assertTrue(AscendConfig(allreduce_outplace="true").allreduce_outplace)
        self.assertTrue(AscendConfig(allreduce_outplace=1).allreduce_outplace)
        self.assertFalse(AscendConfig(allreduce_outplace="false").allreduce_outplace)

    def test_cli_form_is_a_bare_key(self):
        # Exactly what ``--additional-config '{"allreduce_outplace": true}'``
        # delivers: json.loads -> kwargs.
        self.assertTrue(AscendConfig(**json.loads('{"allreduce_outplace": true}')).allreduce_outplace)

    def test_typo_rejected(self):
        with pytest.raises(Exception):
            AscendConfig(allreduce_outplac=True)

    def test_nested_form_is_gone(self):
        # The old ``{"allreduce_config": {"enable_outplace": true}}`` shape is
        # no longer a field, so extra="forbid" rejects it loudly instead of
        # being silently ignored.
        self.assertIn("allreduce_outplace", AscendConfig.__dataclass_fields__)
        self.assertNotIn("allreduce_config", AscendConfig.__dataclass_fields__)
        with pytest.raises(Exception):
            AscendConfig(allreduce_config={"enable_outplace": True})


class TestOutplaceHcclAllReduce(TestBase):
    def test_as_device_normalises_int(self):
        self.assertEqual(_as_device(3), torch.device("npu", 3))
        self.assertEqual(_as_device(torch.device("npu:1")), torch.device("npu:1"))

    def test_dtype_gate(self):
        comm = _comm()
        with patch.object(hccl_outplace, "_format_supported", return_value=True):
            self.assertTrue(comm.supported(torch.empty(8, dtype=BF16)))
            self.assertTrue(comm.supported(torch.empty(8, dtype=torch.float16)))
            self.assertTrue(comm.supported(torch.empty(8, dtype=torch.float32)))
            self.assertFalse(comm.supported(torch.empty(8, dtype=torch.int32)))

    def test_internal_format_rejected(self):
        comm = _comm()
        x = torch.empty(8, dtype=BF16)
        with patch.object(hccl_outplace.torch_npu, "get_npu_format", return_value=29):
            self.assertFalse(comm.supported(x), "NZ/Fractal formats pad rows and must be rejected")
        with patch.object(hccl_outplace.torch_npu, "get_npu_format", return_value=2):
            self.assertTrue(comm.supported(x))

    def test_non_contiguous_rejected(self):
        comm = _comm()
        x = torch.empty(16, dtype=BF16)[::2]
        with patch.object(hccl_outplace.torch_npu, "get_npu_format", return_value=2):
            self.assertFalse(comm.supported(x))

    def test_available_property(self):
        comm = _comm()
        self.assertTrue(comm.available)
        comm._comm = None
        self.assertFalse(comm.available)
        with pytest.raises(RuntimeError):
            comm.all_reduce(torch.empty(4, dtype=BF16))

    @patch("torch.npu.device")
    @patch("torch.npu.current_stream")
    @patch.object(torch.Tensor, "record_stream")
    def test_distinct_send_recv_buffers(self, _record_stream, current_stream, _device):
        lib = MagicMock()
        stream = MagicMock()
        current_stream.return_value = stream
        comm = _comm(stream=stream, lib=lib)
        x = torch.empty(64, dtype=BF16)

        out = _patched_call(comm, x)

        self.assertIsNot(out, x, "the op must return a fresh tensor")
        args = lib.hcclAllReduce.call_args[0]
        self.assertNotEqual(args[0], args[1], "send and recv must not alias")
        self.assertEqual(args[2], 64)  # count == numel
        self.assertEqual(args[3], 11)  # hcclBfloat16
        self.assertEqual(args[4], 0)  # hcclSum
        self.assertEqual(args[5], COMM_HANDLE)  # borrowed handle, never a new one

    @patch("torch.npu.device")
    @patch("torch.npu.current_stream")
    @patch.object(torch.Tensor, "record_stream")
    def test_cross_stream_handoff(self, record_stream, current_stream, _device):
        compute = MagicMock()
        comm_stream = MagicMock()
        current_stream.return_value = compute
        comm = _comm(stream=comm_stream)

        out = _patched_call(comm, torch.empty(64, dtype=BF16))

        comm_stream.wait_stream.assert_called_once_with(compute)
        compute.wait_stream.assert_called_once_with(comm_stream)
        self.assertEqual(record_stream.call_count, 2, "both buffers must be pinned to the comm stream")
        self.assertIsNot(out, compute)

    @patch("torch.npu.device")
    @patch("torch.npu.current_stream")
    @patch.object(torch.Tensor, "record_stream")
    def test_no_stream_falls_back_to_caller_stream(self, _record_stream, current_stream, _device):
        compute = MagicMock()
        current_stream.return_value = compute
        comm = _comm(stream=None)  # stream wrapping failed at construction time

        _patched_call(comm, torch.empty(64, dtype=BF16))

        self.assertFalse(compute.wait_stream.called, "nothing to hand off when there is no dedicated stream")

    def test_try_create_without_backend_api_returns_none(self):
        group = MagicMock()
        group._get_backend.return_value = object()
        self.assertIsNone(OutplaceHcclAllReduce.try_create(group, torch.device("cpu"), 0))

    def test_try_create_survives_backend_query_failure(self):
        group = MagicMock()
        group._get_backend.side_effect = RuntimeError("not a hccl group")
        self.assertIsNone(OutplaceHcclAllReduce.try_create(group, torch.device("cpu"), 0))

    @patch("torch.npu.device")
    def test_destroy_is_idempotent(self, _device):
        stream = MagicMock()
        comm = _comm(stream=stream)
        comm.destroy()
        comm.destroy()
        stream.synchronize.assert_called_once()


class TestNPUCommunicatorDispatch(TestBase):
    """The dispatch layer: fast path when supported, default path otherwise."""

    @staticmethod
    def _bare():
        from vllm_ascend.distributed.device_communicators import npu_communicator as mod

        obj = mod.NPUCommunicator.__new__(mod.NPUCommunicator)
        obj.outplace_ar = None
        return obj

    @patch.object(DeviceCommunicatorBase, "all_reduce", autospec=True, side_effect=lambda self, t: t)
    def test_fast_path_used_when_supported(self, base_all_reduce):
        obj = self._bare()
        fast = MagicMock()
        fast.supported.return_value = True
        expected = torch.empty(4, dtype=BF16)
        fast.all_reduce.return_value = expected
        obj.outplace_ar = fast

        self.assertIs(obj.all_reduce(torch.empty(4, dtype=BF16)), expected)
        base_all_reduce.assert_not_called()

    @patch.object(DeviceCommunicatorBase, "all_reduce", autospec=True, side_effect=lambda self, t: t)
    def test_fallback_when_unsupported(self, base_all_reduce):
        obj = self._bare()
        fast = MagicMock()
        fast.supported.return_value = False
        obj.outplace_ar = fast

        obj.all_reduce(torch.empty(4, dtype=torch.int32))

        fast.all_reduce.assert_not_called()
        base_all_reduce.assert_called_once()

    @patch.object(DeviceCommunicatorBase, "all_reduce", autospec=True, side_effect=lambda self, t: t)
    def test_default_path_when_switch_off(self, base_all_reduce):
        obj = self._bare()
        obj.all_reduce(torch.empty(4, dtype=BF16))
        base_all_reduce.assert_called_once()

    def test_destroy_without_fast_path(self):
        obj = self._bare()
        with patch.object(DeviceCommunicatorBase, "destroy", autospec=True):
            obj.destroy()

    def test_destroy_releases_fast_path(self):
        obj = self._bare()
        fast = MagicMock()
        obj.outplace_ar = fast
        with patch.object(DeviceCommunicatorBase, "destroy", autospec=True):
            obj.destroy()
        fast.destroy.assert_called_once()
        self.assertIsNone(obj.outplace_ar)

    def test_switch_read_is_fail_safe_before_config_init(self):
        from vllm_ascend.distributed.device_communicators.npu_communicator import (
            _outplace_allreduce_enabled,
        )

        with patch("vllm_ascend.ascend_config.get_ascend_config", side_effect=RuntimeError("not init")):
            self.assertFalse(_outplace_allreduce_enabled())

    def test_switch_read_follows_the_flat_key(self):
        from vllm_ascend.distributed.device_communicators.npu_communicator import (
            _outplace_allreduce_enabled,
        )

        with patch("vllm_ascend.ascend_config.get_ascend_config") as get_cfg:
            get_cfg.return_value = SimpleNamespace(allreduce_outplace=True)
            self.assertTrue(_outplace_allreduce_enabled())
            get_cfg.return_value = SimpleNamespace(allreduce_outplace=False)
            self.assertFalse(_outplace_allreduce_enabled())
