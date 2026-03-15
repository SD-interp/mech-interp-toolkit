import gc
from collections.abc import Callable
from typing import Literal, cast

import torch
from transformers import PreTrainedModel

from .activation_dict import ActivationDict, LayerComponent
from .activation_utils import (
    get_activations,
    get_embeddings_dict,
    get_gradients,
    interpolate_activations,
)
from .utils import get_layer_components


def _validate_embeddings(
    input_embeddings: torch.Tensor,
    baseline_embeddings: torch.Tensor,
) -> None:
    """Validate that input and baseline embeddings have matching shape, device, and dtype."""
    if input_embeddings.shape != baseline_embeddings.shape:
        raise ValueError(
            f"Input and baseline embeddings must have identical shape. "
            f"Got input: {input_embeddings.shape}, baseline: {baseline_embeddings.shape}"
        )
    if input_embeddings.device != baseline_embeddings.device:
        raise ValueError(
            f"Input and baseline embeddings must be on the same device. "
            f"Got input: {input_embeddings.device}, baseline: {baseline_embeddings.device}"
        )
    if input_embeddings.dtype != baseline_embeddings.dtype:
        raise ValueError(
            f"Input and baseline embeddings must have the same dtype. "
            f"Got input: {input_embeddings.dtype}, baseline: {baseline_embeddings.dtype}"
        )


def _prepare_synthetic_inputs(input_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Create synthetic input dict by removing input_ids and inputs_embeds."""
    synthetic_inputs = input_dict.copy()
    synthetic_inputs.pop("input_ids", None)
    synthetic_inputs.pop("inputs_embeds", None)
    return synthetic_inputs


def _get_alpha_values(steps: int, dtype: torch.dtype) -> torch.Tensor:
    """Generate alpha values for midpoint Riemann sum integration."""
    return (torch.arange(steps, dtype=dtype) + 0.5) / steps


def _cleanup_memory() -> None:
    """Clear garbage collection and CUDA cache."""
    gc.collect()
    torch.cuda.empty_cache()


def simple_integrated_gradients(
    model: PreTrainedModel,
    input_dict: dict[str, torch.Tensor],
    baseline_dict: dict[str, torch.Tensor],
    metric_fn: Callable = torch.mean,
    steps: int = 20,
) -> ActivationDict:
    """
    Computes vanilla integrated w.r.t. input embeddings.
    Implements the method from "Axiomatic Attribution for Deep Networks" by Sundararajan et al., 2017.
    https://arxiv.org/abs/1703.01365
    """

    input_embeddings = get_embeddings_dict(model, input_dict)["inputs_embeds"]
    baseline_embeddings = get_embeddings_dict(model, baseline_dict)["inputs_embeds"]

    _validate_embeddings(input_embeddings, baseline_embeddings)

    synthetic_inputs = _prepare_synthetic_inputs(input_dict)
    alphas = _get_alpha_values(steps, input_embeddings.dtype)
    accumulated_grads = torch.zeros_like(input_embeddings)

    for alpha in alphas:
        interpolated_embeddings = (
            interpolate_activations(baseline_embeddings, input_embeddings, alpha)
            .detach()
            .requires_grad_(True)
        )
        synthetic_inputs["inputs_embeds"] = interpolated_embeddings

        gradients, _ = get_gradients(
            model,
            synthetic_inputs,
            [(0, "layer_in")],
            metric_fn=metric_fn,
            positions=None,
            return_logits=False,
        )

        if (0, "layer_in") not in gradients or gradients[(0, "layer_in")] is None:
            raise RuntimeError("Failed to retrieve gradients.")
        accumulated_grads.add_(gradients[(0, "layer_in")], alpha=1.0 / steps)
        synthetic_inputs.pop("inputs_embeds", None)

        _cleanup_memory()

    ig_scores = ((input_embeddings - baseline_embeddings) * accumulated_grads).sum(dim=-1)
    output = ActivationDict(model.config, slice(None), "simple_ig_scores")
    output[(0, "layer_in")] = ig_scores
    model.zero_grad(set_to_none=True)

    return output


def edge_attribution_patching(
    model: PreTrainedModel,
    input_dict: dict[str, torch.Tensor],
    baseline_dict: dict[str, torch.Tensor],
    compute_grad_at: Literal["clean", "corrupted"] = "clean",
    metric_fn: Callable = torch.mean,
    layer_components: list[LayerComponent] | None = None,
    include_block_outputs: bool = False,
) -> tuple[ActivationDict, tuple[torch.Tensor | None, torch.Tensor | None]]:
    """
    Computes edge attributions for attention heads using simple gradient x activation.
    """

    with torch.no_grad():
        if layer_components is None:
            layer_components = get_layer_components(
                model, include_block_outputs=include_block_outputs
            )

        # Determine which inputs to use for gradient computation
        if compute_grad_at == "clean":
            grad_inputs = input_dict
        elif compute_grad_at == "corrupted":
            grad_inputs = baseline_dict
        else:
            raise ValueError(f"Unknown compute_grad_at value: {compute_grad_at}")

        # Get activations for ALL layer components, not just embeddings
        input_activations, input_logits = get_activations(
            model, input_dict, layer_components, return_logits=True
        )
        baseline_activations, baseline_logits = get_activations(
            model, baseline_dict, layer_components, return_logits=True
        )
        grads, _ = get_gradients(
            model,
            grad_inputs,
            layer_components,
            metric_fn=metric_fn,
            positions=None,
            return_logits=False,
        )

        input_logits = cast(torch.Tensor, input_logits)
        baseline_logits = cast(torch.Tensor, baseline_logits)

        input_logits = input_logits.detach().cpu()
        baseline_logits = baseline_logits.detach().cpu()

        input_activations = input_activations.split_heads()
        baseline_activations = baseline_activations.split_heads()
        grads = grads.split_heads()

        eap_scores = input_activations.empty_like()
        for layer, comp_name in eap_scores:
            if comp_name == "mlp_hidden":
                eap_scores[(layer, comp_name)] = (
                    input_activations[(layer, comp_name)] - baseline_activations[(layer, comp_name)]
                ) * grads[(layer, comp_name)]
            else:
                eap_scores[(layer, comp_name)] = (
                    (
                        input_activations[(layer, comp_name)]
                        - baseline_activations[(layer, comp_name)]
                    )
                    * grads[(layer, comp_name)]
                ).sum(dim=-1)

    model.zero_grad(set_to_none=True)
    eap_scores.value_type = "eap_scores"
    _cleanup_memory()

    return eap_scores, (input_logits, baseline_logits)


def eap_integrated_gradients(
    model: PreTrainedModel,
    input_dict: dict[str, torch.Tensor],
    baseline_dict: dict[str, torch.Tensor],
    metric_fn: Callable = torch.mean,
    layer_components: list[LayerComponent] | None = None,
    steps: int = 20,
    include_block_outputs: bool = False,
) -> tuple[ActivationDict, tuple[torch.Tensor | None, torch.Tensor | None]]:
    """
    Computes integrated gradients for edge attributions.
    Implements the method from "Have Faith in Faithfulness: Going Beyond Circuit Overlap ..."
    by Hanna et al., 2024. https://arxiv.org/pdf/2403.17806
    """

    with torch.no_grad():
        if layer_components is None:
            layer_components = get_layer_components(
                model, include_block_outputs=include_block_outputs
            )

        input_activations, input_logits = get_activations(
            model, input_dict, layer_components, return_logits=True
        )
        baseline_activations, baseline_logits = get_activations(
            model, baseline_dict, layer_components, return_logits=True
        )

        input_logits = cast(torch.Tensor, input_logits)
        baseline_logits = cast(torch.Tensor, baseline_logits)

        input_logits = input_logits.detach().cpu()
        baseline_logits = baseline_logits.detach().cpu()

        # Keep the large cached activations off GPU during the IG loop.
        input_activations = input_activations.detach().cpu()
        baseline_activations = baseline_activations.detach().cpu()

        input_activations.attention_mask = torch.empty((1, 1))
        baseline_activations.attention_mask = torch.empty((1, 1))

        _cleanup_memory()

        input_embeddings = get_embeddings_dict(model, input_dict)["inputs_embeds"]
        baseline_embeddings = get_embeddings_dict(model, baseline_dict)["inputs_embeds"]

        _validate_embeddings(input_embeddings, baseline_embeddings)

        synthetic_input_dict = _prepare_synthetic_inputs(input_dict)
        alphas = _get_alpha_values(steps, input_embeddings.dtype)
        accumulated_grads = input_activations.zeros_like()

        for alpha in alphas:
            interpolated_embeddings = interpolate_activations(
                baseline_embeddings, input_embeddings, alpha
            ).detach()
            interpolated_embeddings.requires_grad_(True)
            synthetic_input_dict["inputs_embeds"] = interpolated_embeddings

            grad_cache, _ = get_gradients(
                model,
                synthetic_input_dict,
                layer_components,
                metric_fn=metric_fn,
                positions=None,
                return_logits=False,
            )

            grad_cache.cpu()
            accumulated_grads.add_(grad_cache, alpha=1.0 / steps)

            synthetic_input_dict.pop("inputs_embeds", None)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            del interpolated_embeddings, grad_cache
            model.zero_grad(set_to_none=True)
            _cleanup_memory()

        accumulated_grads = accumulated_grads.split_heads()
        input_activations = input_activations.split_heads()
        baseline_activations = baseline_activations.split_heads()

        eap_ig_scores = input_activations.empty_like()

        for layer, comp_name in eap_ig_scores:
            if comp_name == "mlp_hidden":
                eap_ig_scores[(layer, comp_name)] = (
                    input_activations[(layer, comp_name)] - baseline_activations[(layer, comp_name)]
                ) * accumulated_grads[(layer, comp_name)]
            else:
                eap_ig_scores[(layer, comp_name)] = (
                    (
                        input_activations[(layer, comp_name)]
                        - baseline_activations[(layer, comp_name)]
                    )
                    * accumulated_grads[(layer, comp_name)]
                ).sum(dim=-1)

    eap_ig_scores.value_type = "eap_ig_scores"
    eap_ig_scores.attention_mask = input_dict["attention_mask"]
    eap_ig_scores.detach().cpu()
    model.zero_grad(set_to_none=True)
    return eap_ig_scores, (input_logits, baseline_logits)
