# Mega GDN decode custom operators

The `mega_gdn_decode`, `mega_gdn_decode_strided`, and
`mega_gdn_decode_strided_batch` sources come from xLLM-AI/xllm-ops commit
`9b17618cbd4a984dd18419ecaba97808b7e55d3d` (Apache-2.0). The batch
kernel includes the `RecurrentVector128` optimization. The PTO ISA submodule is
pinned to the commit used by that source checkout.

On Ascend 950, `python setup.py build_aclnn` builds all three operators in the
existing `custom_transformer` vendor. Initialize submodules before an offline
build. An independently built `mega_gdn_trio_transformer` vendor installed
under `vllm_ascend/_cann_ops_custom` is also discovered at runtime. The C++
bridge and Python dispatch are in `csrc/mega_gdn_decode*.cpp` and
`vllm_ascend/ops/mega_gdn_decode.py`.

Runtime dispatch requires `fusion_ops_mega_gdn_decode: true`,
`--mamba-ssm-cache-dtype float32`, and a supported pure decode batch. The
fusion remains disabled by default.
