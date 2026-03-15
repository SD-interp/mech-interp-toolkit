from collections.abc import Sequence
from typing import Callable

import torch
from torch import Tensor
from transformers import PreTrainedModel

from .activation_dict import ActivationDict, LayerComponent
from .hook_utils import gen_cache_hookfn, gen_patch_hookfn, temporary_hooks
from .utils import regularize_position

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

    if return_logits:
        logits = logits.detach()[:, grad_output.positions, :]
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

    if return_logits:
        logits = logits.detach()[:, grad_output.positions, :]
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
    early_exit: bool = False,
) -> tuple[ActivationDict, Tensor | None]:
    positions = regularize_position(positions)
    inputs = _inputs_on_model_device(model, inputs)

    module_dict = dict(model.named_modules())

    act_output = ActivationDict(model.config, positions=positions, value_type="activation")

    hook_specs_dict = {
        "fwd": gen_cache_hookfn(layer_components, act_output, clone_tensors=clone_tensors),
    }

    logits = None

    with torch.no_grad():
        with temporary_hooks(module_dict, hook_specs_dict, early_exit=early_exit):
            result = model(**inputs)
            if not early_exit and return_logits:
                logits = result.logits
                logits = logits.detach()[:, act_output.positions, :]

    if "attention_mask" in inputs:
        act_output.attention_mask = inputs["attention_mask"]
    act_output.extract_positions()

    return act_output, logits


def patch_activations(
    model: PreTrainedModel,
    inputs: dict[str, Tensor],
    layer_components: list[LayerComponent],
    patch_dict: ActivationDict,
    positions: int | slice | Sequence | None = None,
    return_logits: bool = True,
    clone_tensors: bool = False,
    early_exit: bool = False,
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

    logits = None

    with torch.no_grad():
        with temporary_hooks(module_dict, hook_specs_dict, early_exit=early_exit):
            result = model(**inputs)
            if not early_exit and return_logits:
                logits = result.logits
                logits = logits.detach()[:, act_output.positions, :]

    if "attention_mask" in inputs:
        act_output.attention_mask = inputs["attention_mask"]
    act_output.extract_positions()

    return act_output, logits


def get_embeddings_dict(model: PreTrainedModel, inputs: dict[str, Tensor]) -> dict[str, Tensor]:
    """Return a synthetic-input-ready copy of ``inputs`` with ``inputs_embeds`` populated."""
    embeddings_inputs = inputs.copy()
    input_ids = embeddings_inputs.pop("input_ids", None)

    if "inputs_embeds" not in embeddings_inputs:
        if input_ids is None:
            raise ValueError("Expected either 'inputs_embeds' or 'input_ids' in inputs.")
        embedding_layer = model.get_input_embeddings()
        if embedding_layer is None:
            raise ValueError("Model does not expose an input embedding layer.")
        embedding_device = embedding_layer.weight.device
        embeddings_inputs["inputs_embeds"] = embedding_layer(
            input_ids.to(device=embedding_device)
        ).detach()

    return embeddings_inputs


def interpolate_activations(
    clean_activations: Tensor,
    baseline_activations: Tensor,
    alpha: float | Tensor,
) -> Tensor:
    """
    Interpolates between clean and corrupted inputs.
    """
    return (1 - alpha) * clean_activations + alpha * baseline_activations
