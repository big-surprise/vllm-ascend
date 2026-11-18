#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
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

from unittest.mock import MagicMock, patch

import torch

from tests.ut.base import TestBase
from vllm_ascend.distributed.device_communicators import hccl_outplace_ar
from vllm_ascend.distributed.device_communicators.hccl_outplace_ar import (
    HcclOutOfPlaceAllReduce,
    _normalize_device,
    _tensor_in_base_format,
)

FAKE_HANDLE = 0xC0FFEE


def _make_reducer(lib=None, stream=None, device=None):
    """Reducer on the CPU "device" so eligibility logic runs on a UT box."""
    return HcclOutOfPlaceAllReduce(
        hccl_lib=lib if lib is not None else MagicMock(),
        comm_handle=FAKE_HANDLE,
        coll_stream=stream if stream is not None else MagicMock(),
        device=device if device is not None else torch.device("cpu"),
    )


class TestDeviceNormalization(TestBase):
    def test_int_becomes_npu_device(self):
        self.assertEqual(_normalize_device(2), torch.device("npu", 2))

    def test_torch_device_passthrough(self):
        dev = torch.device("npu:1")
        self.assertIs(_normalize_device(dev), dev)


class TestEligibility(TestBase):
    def test_dtype_gate(self):
        reducer = _make_reducer()
        with patch.object(hccl_outplace_ar, "_tensor_in_base_format", return_value=True):
            self.assertTrue(reducer.accepts(torch.zeros(4, dtype=torch.bfloat16)))
            self.assertTrue(reducer.accepts(torch.zeros(4, dtype=torch.float16)))
            self.assertTrue(reducer.accepts(torch.zeros(4, dtype=torch.float32)))
            self.assertFalse(reducer.accepts(torch.zeros(4, dtype=torch.int32)))
            self.assertFalse(reducer.accepts(torch.zeros(4, dtype=torch.float64)))

    def test_device_mismatch_rejected(self):
        reducer = _make_reducer(device=torch.device("npu", 0))
        self.assertFalse(reducer.accepts(torch.zeros(4, dtype=torch.bfloat16)))

    def test_shutdown_reducer_accepts_nothing(self):
        reducer = _make_reducer()
        reducer._comm = None
        self.assertFalse(reducer.accepts(torch.zeros(4, dtype=torch.bfloat16)))

    def test_non_contiguous_rejected(self):
        strided = torch.zeros(8, dtype=torch.float32)[::2]
        self.assertFalse(_tensor_in_base_format(strided))

    def test_internal_acl_format_rejected(self):
        x = torch.zeros(4, dtype=torch.bfloat16)
        with patch.object(hccl_outplace_ar.torch_npu, "get_npu_format", return_value=29):
            self.assertFalse(
                _tensor_in_base_format(x),
                "NZ/Fractal formats pad rows and must stay on the default path",
            )
        with patch.object(hccl_outplace_ar.torch_npu, "get_npu_format", return_value=2):
            self.assertTrue(_tensor_in_base_format(x))

    def test_format_probe_failure_is_conservative(self):
        x = torch.zeros(4, dtype=torch.bfloat16)
        with patch.object(
            hccl_outplace_ar.torch_npu, "get_npu_format", side_effect=RuntimeError("boom")
        ):
            self.assertFalse(_tensor_in_base_format(x))


class TestBuild(TestBase):
    def test_missing_accessors_returns_none(self):
        group = MagicMock()
        group._get_backend.return_value = object()  # no HCCL accessors at all
        result = HcclOutOfPlaceAllReduce.build(group, device=0, rank_in_group=0)
        self.assertIsNone(result)

    def test_backend_query_failure_returns_none(self):
        group = MagicMock()
        group._get_backend.side_effect = RuntimeError("no backend")
        result = HcclOutOfPlaceAllReduce.build(group, device=0, rank_in_group=0)
        self.assertIsNone(result)

    def test_hccl_library_failure_returns_none(self):
        backend = MagicMock()
        group = MagicMock()
        group._get_backend.return_value = backend
        with patch.object(hccl_outplace_ar, "HCCLLibrary", side_effect=OSError("no lib")):
            result = HcclOutOfPlaceAllReduce.build(group, device=0, rank_in_group=0)
        self.assertIsNone(result)


class TestAllReduce(TestBase):
    def test_out_of_place_and_distinct_buffers(self):
        lib = MagicMock()
        stream = MagicMock()
        stream.npu_stream = 1234
        reducer = _make_reducer(lib=lib, stream=stream)

        x = torch.arange(4, dtype=torch.float32)
        fake_stream_ctx = MagicMock()
        with (
            patch("torch.npu.device", return_value=fake_stream_ctx),
            patch("torch.npu.current_stream", return_value=stream),
        ):
            out = reducer.all_reduce(x)

        self.assertIsNot(out, x, "fast path must not alias its input")
        self.assertEqual(out.shape, x.shape)
        self.assertEqual(out.dtype, x.dtype)
        lib.hcclAllReduce.assert_called_once()
        args = lib.hcclAllReduce.call_args.args
        send_ptr, recv_ptr = args[0], args[1]
        self.assertNotEqual(
            getattr(send_ptr, "value", send_ptr),
            getattr(recv_ptr, "value", recv_ptr),
            "send and receive buffers must be distinct",
        )
        self.assertEqual(args[2], x.numel())  # count

    def test_empty_tensor_short_circuits(self):
        lib = MagicMock()
        reducer = _make_reducer(lib=lib)
        out = reducer.all_reduce(torch.empty(0, dtype=torch.float32))
        self.assertEqual(out.numel(), 0)
        lib.hcclAllReduce.assert_not_called()

    def test_shutdown_is_idempotent(self):
        reducer = _make_reducer()
        reducer.shutdown()
        reducer.shutdown()  # second call must be a no-op
        self.assertIsNone(reducer._comm)
        with self.assertRaises(RuntimeError):
            reducer.all_reduce(torch.zeros(4, dtype=torch.float32))


class TestCommunicatorWiring(TestBase):
    def _communicator(self, requested: bool):
        from vllm_ascend.distributed.device_communicators import npu_communicator

        with (
            patch.object(
                npu_communicator.DeviceCommunicatorBase, "__init__", lambda self, *a, **k: None
            ),
            patch("torch.npu.current_device", return_value=0),
            patch.object(
                npu_communicator, "_allreduce_outplace_requested", return_value=requested
            ),
        ):
            comm = npu_communicator.NPUCommunicator.__new__(npu_communicator.NPUCommunicator)
            comm.world_size = 2
            comm.device_group = MagicMock()
            comm.rank_in_group = 0
            npu_communicator.NPUCommunicator.__init__(comm, cpu_group=MagicMock())
        return comm

    def test_switch_off_leaves_fast_path_unbuilt(self):
        comm = self._communicator(requested=False)
        self.assertIsNone(comm.oop_all_reduce)

    def test_switch_on_builds_fast_path(self):
        with patch(
            "vllm_ascend.distributed.device_communicators.hccl_outplace_ar."
            "HcclOutOfPlaceAllReduce.build"
        ) as build:
            build.return_value = MagicMock()
            comm = self._communicator(requested=True)
        self.assertIsNotNone(comm.oop_all_reduce)
        build.assert_called_once()
        self.assertEqual(build.call_args.kwargs["rank_in_group"], 0)

    def test_unsupported_tensor_falls_back_to_super(self):
        comm = self._communicator(requested=False)
        reducer = MagicMock()
        reducer.accepts.return_value = False
        comm.oop_all_reduce = reducer
        x = torch.zeros(4, dtype=torch.int32)
        with patch(
            "vllm_ascend.distributed.device_communicators.npu_communicator."
            "DeviceCommunicatorBase.all_reduce",
            return_value=x,
        ) as fallback:
            result = comm.all_reduce(x)
        fallback.assert_called_once()
        self.assertIs(result, x)
        reducer.all_reduce.assert_not_called()
