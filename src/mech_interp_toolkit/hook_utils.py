from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from typing import Callable, Iterable

import torch
import torch.nn as nn
from torch.utils.hooks import RemovableHandle
from transformers import PreTrainedModel

from .activation_dict import ActivationDict

type Position = slice | int | Iterable | None
type LayerComponent = tuple[int, str]

ModuleDict = dict[str, PreTrainedModel]


@dataclass
class HookSpec:
    layer_component: LayerComponent
    fn: Callable[[nn.Module, torch.Tensor, torch.Tensor], torch.Tensor | None]


def _detach_first_tensor(
    value: torch.Tensor | tuple | list,
    clone_tensors: bool = False,
) -> torch.Tensor:
    # Hooks can receive tuple/list inputs or outputs; capture the first tensor payload.
    # Optionally clone after detach to avoid shared storage aliasing.
    if isinstance(value, torch.Tensor):
        result = value.detach()
        return result.clone() if clone_tensors else result

    if isinstance(value, (tuple, list)):
        for item in value:
            if isinstance(item, torch.Tensor):
                result = item.detach()
                return result.clone() if clone_tensors else result

    raise TypeError(f"Expected tensor or sequence containing a tensor, got {type(value)!r}")


def layer_component_to_hookloc(layer_component: LayerComponent) -> str:
    layer, component = layer_component

    if component == "layer_in":
        return f"model.layers.{layer}"

    elif component == "layer_out":
        return f"model.layers.{layer}"

    elif component == "attn":
        return f"model.layers.{layer}.self_attn"

    elif component == "mlp":
        return f"model.layers.{layer}.mlp"

    elif component == "z":
        return f"model.layers.{layer}.self_attn.o_proj"

    elif component == "mlp_hidden":
        return f"model.layers.{layer}.mlp.down_proj"

    else:
        raise ValueError(f"Invalid component name {component}")


def gen_hookfn(
    layer_components: Iterable[LayerComponent],
    results: ActivationDict,
    clone_tensors: bool = False,
) -> Iterable[HookSpec]:
    hook_list = []
    for layer_component in layer_components:
        layer, component = layer_component

        if component in ["layer_out", "attn", "mlp"]:

            def hook_output_fn(layer_component, _, output):
                results[layer_component] = _detach_first_tensor(
                    output, clone_tensors=clone_tensors
                )

            hook_list.append(HookSpec(layer_component, hook_output_fn))
        else:

            def hook_input_fn(layer_component, inputs, _):
                results[layer_component] = _detach_first_tensor(
                    inputs, clone_tensors=clone_tensors
                )

            hook_list.append(HookSpec(layer_component, hook_input_fn))

    return hook_list


@contextmanager
def temporary_hooks(
    module_dict: ModuleDict,
    hook_specs_dict: dict[str, Iterable[HookSpec]],
):
    """
    Register forward hooks temporarily and remove them on exit.

    Hook function contract:
      fn(output, module_name, module) -> None or modified_output
    """
    handles: list[RemovableHandle] = []

    try:
        for hook_type, hook_specs in hook_specs_dict.items():
            for spec in hook_specs:
                layer_component = spec.layer_component
                module_name = layer_component_to_hookloc(layer_component)

                if module_name not in module_dict:
                    raise ValueError(f"Module {module_name!r} not found in model.")

                fn = spec.fn
                module = module_dict[module_name]

                def make_hook(fn, layer_component: LayerComponent):
                    def hook(
                        module: nn.Module,
                        inputs: tuple | torch.Tensor,
                        output: tuple | torch.Tensor,
                    ):
                        new_output = fn(layer_component, inputs, output)
                        return new_output

                    return hook

                if hook_type == "fwd":
                    handles.append(module.register_forward_hook(make_hook(fn, layer_component)))
                elif hook_type == "bwd":
                    handles.append(
                        module.register_full_backward_hook(make_hook(fn, layer_component))
                    )
                else:
                    raise ValueError(f"Invalid hook type {hook_type}")

        yield

    except Exception as e:
        print(f"Exception occurred while executing {hook_type} hook for {layer_component}: {e}")
        raise e

    finally:
        for handle in handles:
            handle.remove()
