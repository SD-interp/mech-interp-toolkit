from collections.abc import Sequence
from typing import Callable

import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812
from transformers import PreTrainedModel

from .activation_dict import ActivationDict, LayerComponent
from .hook_utils import gen_cache_hookfn, gen_patch_hookfn, temporary_hooks
from .utils import empty_dict_like, regularize_position

type Position = slice | int | Sequence | None


def _get_model_device(model: PreTrainedModel) -> torch.device:
    """Infer the model device from parameters/buffers, defaulting to CPU."""
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def _inputs_on_model_device(model: PreTrainedModel, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return a copy of inputs with all tensors moved to model device."""
    model_device = _get_model_device(model)
    output: dict[str, Tensor] = {}
    for key, value in inputs.items():
        output[key] = value if value.device == model_device else value.to(model_device)
    return output


def get_activations_and_grads(
    model: PreTrainedModel,
    inputs: dict[str, Tensor],
    layer_components: list[LayerComponent],
    metric_fn: Callable[[Tensor], Tensor] = torch.nanmean,
    positions: int | slice | Sequence | None = None,
    return_logits: bool = True,
    clone_tensors: bool = False,
) -> tuple[ActivationDict, ActivationDict, Tensor | None]:
    positions = regularize_position(positions)
    inputs = _inputs_on_model_device(model, inputs)

    module_dict = dict(model.named_modules())

    act_output = ActivationDict(model.config, positions=positions, value_type="activation")
    grad_output = ActivationDict(model.config, positions=positions, value_type="gradient")

    hook_specs_dict = {
        "fwd": gen_cache_hookfn(layer_components, act_output, clone_tensors=clone_tensors),
        "bwd": gen_cache_hookfn(layer_components, grad_output, clone_tensors=clone_tensors),
    }

    with torch.enable_grad():
        with temporary_hooks(module_dict, hook_specs_dict):
            logits = model(**inputs).logits
            metric = metric_fn(logits)
            if metric.ndim != 0:
                raise ValueError("Metric function must return a scalar.")
            metric.backward()

    model.zero_grad(set_to_none=True)

    if "attention_mask" in inputs:
        act_output.attention_mask = inputs["attention_mask"]
        grad_output.attention_mask = inputs["attention_mask"]
    act_output.extract_positions()
    grad_output.extract_positions()
    logits = logits.detach()[:, act_output.positions, :]

    if return_logits:
        return act_output, grad_output, logits
    else:
        return act_output, grad_output, None


def get_gradients(
    model: PreTrainedModel,
    inputs: dict[str, Tensor],
    layer_components: list[LayerComponent],
    metric_fn: Callable[[Tensor], Tensor] = torch.nanmean,
    positions: int | slice | Sequence | None = None,
    return_logits: bool = True,
    clone_tensors: bool = False,
) -> tuple[ActivationDict, Tensor | None]:
    positions = regularize_position(positions)
    inputs = _inputs_on_model_device(model, inputs)

    module_dict = dict(model.named_modules())

    grad_output = ActivationDict(model.config, positions=positions, value_type="gradient")

    hook_specs_dict = {
        "bwd": gen_cache_hookfn(layer_components, grad_output, clone_tensors=clone_tensors),
    }

    with torch.enable_grad():
        with temporary_hooks(module_dict, hook_specs_dict):
            logits = model(**inputs).logits
            metric = metric_fn(logits)
            if metric.ndim != 0:
                raise ValueError("Metric function must return a scalar.")
            metric.backward()

    model.zero_grad(set_to_none=True)

    if "attention_mask" in inputs:
        grad_output.attention_mask = inputs["attention_mask"]
    grad_output.extract_positions()
    logits = logits.detach()[:, grad_output.positions, :]

    if return_logits:
        return grad_output, logits
    else:
        return grad_output, None


def get_activations(
    model: PreTrainedModel,
    inputs: dict[str, Tensor],
    layer_components: list[LayerComponent],
    positions: int | slice | Sequence | None = None,
    return_logits: bool = True,
    clone_tensors: bool = False,
) -> tuple[ActivationDict, Tensor | None]:
    positions = regularize_position(positions)
    inputs = _inputs_on_model_device(model, inputs)

    module_dict = dict(model.named_modules())

    act_output = ActivationDict(model.config, positions=positions, value_type="activation")

    hook_specs_dict = {
        "fwd": gen_cache_hookfn(layer_components, act_output, clone_tensors=clone_tensors),
    }

    with torch.no_grad():
        with temporary_hooks(module_dict, hook_specs_dict):
            logits = model(**inputs).logits

    if "attention_mask" in inputs:
        act_output.attention_mask = inputs["attention_mask"]
    act_output.extract_positions()
    logits = logits.detach()[:, act_output.positions, :]

    if return_logits:
        return act_output, logits
    else:
        return act_output, None


def patch_activations(
    model: PreTrainedModel,
    inputs: dict[str, Tensor],
    layer_components: list[LayerComponent],
    patch_dict: ActivationDict,
    positions: int | slice | Sequence | None = None,
    return_logits: bool = True,
    clone_tensors: bool = False,
) -> tuple[ActivationDict, Tensor | None]:
    positions = regularize_position(positions)
    model_device = _get_model_device(model)
    inputs = _inputs_on_model_device(model, inputs)
    patch_dict = patch_dict.to(device=model_device)

    module_dict = dict(model.named_modules())

    act_output = ActivationDict(model.config, positions=positions, value_type="activation")

    hook_specs_dict = {
        "patch": gen_patch_hookfn(patch_dict),
        "fwd": gen_cache_hookfn(layer_components, act_output, clone_tensors=clone_tensors),
    }

    with torch.no_grad():
        with temporary_hooks(module_dict, hook_specs_dict):
            logits = model(**inputs).logits

    if "attention_mask" in inputs:
        act_output.attention_mask = inputs["attention_mask"]
    act_output.extract_positions()
    logits = logits.detach()[:, act_output.positions, :]

    if return_logits:
        return act_output, logits
    else:
        return act_output, None


def get_embeddings_dict(model: PreTrainedModel, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    if "inputs_embeds" not in inputs:
        embeds, _ = get_activations(
            model,
            inputs,
            [(0, "layer_in")],
            positions=None,
            return_logits=False,
        )
        inputs.pop("input_ids", None)
        inputs["inputs_embeds"] = embeds[(0, "layer_in")].detach()
    return inputs


def interpolate_activations(
    clean_activations: Tensor,
    baseline_activations: Tensor,
    alpha: float | Tensor,
) -> Tensor:
    """
    Interpolates between clean and corrupted inputs.
    """
    return (1 - alpha) * clean_activations + alpha * baseline_activations


def _pad_and_concat(tensors, padding_value, dim):
    dim = dim % tensors[0].ndim
    max_len = max(t.shape[dim] for t in tensors)

    padded = []
    ndim = tensors[0].ndim

    for t in tensors:
        pad_len = max_len - t.shape[dim]
        if pad_len > 0:
            pad = [0, 0] * ndim
            # left-padding: put pad_len on the "left" side of `dim`
            pad[2 * (ndim - dim - 1)] = pad_len
            t = F.pad(t, pad, value=padding_value)
        padded.append(t)

    return torch.cat(padded, dim=0)


def concat_activations(list_activations: list[ActivationDict], pad_value=None) -> ActivationDict:
    new_obj = empty_dict_like(list_activations[0])

    if new_obj.attention_mask.numel() > 0:
        new_obj.attention_mask = _pad_and_concat(
            [activation.attention_mask for activation in list_activations],
            padding_value=0,
            dim=1,
        )

    for key in new_obj.keys():
        if pad_value is None:
            new_obj[key] = torch.cat([activation[key] for activation in list_activations])
        else:
            new_obj[key] = _pad_and_concat(
                [activation[key] for activation in list_activations],
                padding_value=pad_value,
                dim=1,
            )

    return new_obj


def expand_mask(mask: Tensor, expansion: int):
    batch_size, seq_len = mask.shape

    padding = torch.zeros((batch_size, expansion), dtype=mask.dtype, device=mask.device)

    merged_tensor = torch.cat([padding, mask], dim=1)

    return merged_tensor[:, :seq_len]
