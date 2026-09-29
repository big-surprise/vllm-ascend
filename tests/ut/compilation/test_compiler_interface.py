from unittest.mock import Mock, patch

from vllm.config.utils import Range

from vllm_ascend.compilation.compiler_interface import AscendCompiler, _configure_backend
from vllm_ascend.utils import COMPILATION_PASS_KEY


def test_configure_npugraph_backend_registers_pre_optimization_pass():
    config = Mock()
    compilation_config = Mock(enable_static_kernel=False)
    post_grad_pass = Mock()
    process_options = Mock()

    _configure_backend(
        config,
        compilation_config,
        Mock(),
        process_kwargs_options=process_options,
        post_grad_pass=post_grad_pass,
    )

    options = process_options.call_args.args[1]["options"]
    assert options["post_grad_custom_pre_pass"] is post_grad_pass


def test_cache_miss_recompile_restores_fusion_pass_manager():
    compiler = AscendCompiler()
    compiler.vllm_config = Mock()
    compiler.cache_dir = "/tmp/compiler-cache"
    compile_range = Range(start=1024, end=16384)

    with (
        patch(
            "vllm_ascend.compilation.compiler_interface.os.path.exists",
            return_value=False,
        ),
        patch(
            "vllm_ascend.compilation.compiler_interface.get_ascend_config",
        ) as get_config,
        patch(
            "vllm_ascend.compilation.compiler_interface.npugraph_ex_compile",
            return_value=("compiled", None),
        ) as compile_graph,
        patch(
            "vllm_ascend.compilation.graph_fusion_pass_manager.GraphFusionPassManager",
        ) as pass_manager_cls,
    ):
        result = compiler.load(
            ("cache-key", "/missing/artifact.py"),
            Mock(),
            [],
            graph_index=0,
            compile_range=compile_range,
        )

    assert result == "compiled"
    pass_manager = pass_manager_cls.return_value
    pass_manager.configure.assert_called_once_with(compiler.vllm_config)
    assert compile_graph.call_args.args[2][COMPILATION_PASS_KEY] is pass_manager
    assert compile_graph.call_args.args[4] is get_config.return_value.ascend_compilation_config
