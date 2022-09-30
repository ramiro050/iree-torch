# Copyright 2022 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import functools
import time
from typing import List, Optional
import torch
from torch.fx.experimental.proxy_tensor import make_fx
from functorch._src.compile_utils import strip_overloads

import torch_mlir
import iree_torch


DEVICE_TO_IREE_BACKEND = { "cpu" : "llvm-cpu",
                           "cuda" : "cuda" }


def timeit(*, append_time_to: Optional[List] = None):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time_ns()
            result = func(*args, **kwargs)
            end_time = time.time_ns()

            if append_time_to is not None:
                append_time_to.append(end_time - start_time)
            return result
        return wrapper
    return decorator


def _returns_nothing(fx_g: torch.fx.GraphModule) -> bool:
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                return len(node_arg) == 0
    return False


def _unwrap_single_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    """
    Replace tuple with tuple element in functions that return one-element tuples.
    Returns true if an unwrapping took place, and false otherwise.
    """
    unwrapped_tuple = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, tuple):
                if len(node_arg) == 1:
                    node.args = (node_arg[0],)
                    unwrapped_tuple = True
                    break

    if unwrapped_tuple:
        fx_g.graph.lint()
        fx_g.recompile()
    return unwrapped_tuple


def _list_return_to_tuple_return(fx_g: torch.fx.GraphModule) -> bool:
    replaced_list = False
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, list):
                node.args = (tuple(node_arg),)
                replaced_list = True
                break

    if replaced_list:
        fx_g.graph.lint()
        fx_g.recompile()
    return replaced_list


def _remove_nones(fx_g: torch.fx.GraphModule) -> List[int]:
    removed_indexes = []
    for node in fx_g.graph.nodes:
        if node.op == "output":
            assert len(node.args) == 1, "Output node must have a single argument"
            node_arg = node.args[0]
            if isinstance(node_arg, (list, tuple)):
                node_arg = list(node_arg)
                node_args_len = len(node_arg)
                for i in range(node_args_len):
                    curr_index = node_args_len - (i + 1)
                    if node_arg[curr_index] is None:
                        removed_indexes.append(curr_index)
                        node_arg.pop(curr_index)
                node.args = (tuple(node_arg),)
                break

    if len(removed_indexes) > 0:
        fx_g.graph.lint()
        fx_g.graph.eliminate_dead_code()
        fx_g.recompile()
    removed_indexes.sort()
    return removed_indexes


def _insert_random_numbers(mlir_graph: str) -> str:
    return mlir_graph.replace("torch_c.get_next_seed : () -> i64",
                              f"arith.constant 3 : i64")


def make_torch_mlir_compiler(use_tracing: bool, device: str, verbose=False):
    def compiler(fx_graph: torch.fx.GraphModule,
                 example_inputs: List[torch.Tensor]):
        """Compile GraphModule using torch-mlir + IREE."""
        if verbose:
            print("Compiling graph...")

        if _returns_nothing(fx_graph):
            return fx_graph

        was_unwrapped = _unwrap_single_tuple_return(fx_graph)
        was_list_replaced = _list_return_to_tuple_return(fx_graph)
        removed_none_indexes = _remove_nones(fx_graph)
        fx_graph = make_fx(fx_graph)(*example_inputs)
        strip_overloads(fx_graph)

        if verbose:
            print("torch.fx graph:")
            #print(fx_graph.graph)

        ts_compiler = torch.jit.trace if use_tracing else torch.jit.script
        ts_graph = ts_compiler(fx_graph, example_inputs)

        if verbose:
            torch_mlir_module = torch_mlir.compile(
                ts_graph, example_inputs,
                output_type=torch_mlir.OutputType.RAW)
            print("\n\ntorch-mlir backend contract graph:")
            print(torch_mlir_module)

        linalg_module = torch_mlir.compile(
            ts_graph, example_inputs,
            output_type=torch_mlir.OutputType.LINALG_ON_TENSORS)
        linalg_module = _insert_random_numbers(str(linalg_module))
        backend = DEVICE_TO_IREE_BACKEND[device]
        arch = "sm_80" if device == "cuda" else None
        compiled_module = iree_torch.compile_to_vmfb(linalg_module, backend, arch)
        loaded_module = iree_torch.load_vmfb(compiled_module, backend)

        def forward(*inputs):
            result = loaded_module.forward(*inputs)
            result = tuple() if result is None else result
            result = (result,) if was_unwrapped else result
            # Turn into list to insert back nones
            result = list(result)
            for removed_index in removed_none_indexes:
                result.insert(removed_index, None)
            return result
        return forward

    return compiler


def check_results(compiled_results, eager_results):
    for compiled_result, eager_result in zip(compiled_results, eager_results):
        if not torch.allclose(compiled_result.to("cpu"),
                              eager_result.to("cpu"), atol=1e-5):
            print("Compiled result does not match eager result")
            return
    print("Compiled result matches eager result!")


def print_time_stats(times):
    times_tensor = torch.tensor(times)
    def quantile_ms(q):
        return torch.quantile(times_tensor.to(float), q).item() / 1e6
    print(f"Median: {quantile_ms(0.5)} ms")
    print(f"10%ile: {quantile_ms(0.1)} ms")
    print(f"90%ile: {quantile_ms(0.9)} ms")
    print(f"Total: {torch.sum(times_tensor) / 1e6} ms")
    print()
