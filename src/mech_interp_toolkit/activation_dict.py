import warnings
from abc import ABC
from collections.abc import Sequence
from copy import deepcopy
from typing import Callable, Self

import einops
import torch
from torch import Tensor
from torch.nn import functional as F  # noqa: N812

from .utils import empty_dict_like, regularize_position, zeros_dict_like

type Position = slice | int | Sequence | None
type LayerComponent = tuple[int, str]


class FrozenError(RuntimeError):
    """Raised when attempting to modify a frozen ActivationDict."""

    pass


class FreezableDict(ABC, dict[LayerComponent, Tensor]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._frozen = False

    def freeze(self) -> Self:
        """Freeze the dictionary, making it immutable."""
        self._frozen = True
        return self

    def unfreeze(self) -> Self:
        """Unfreeze the dictionary, making it mutable."""
        self._frozen = False
        return self

    def _check_frozen(self):
        if getattr(self, "_frozen", False):
            raise FrozenError("This object is frozen and cannot be modified.")

    def __setitem__(self, key, value):
        self._check_frozen()
        return super().__setitem__(key, value)

    def __delitem__(self, key):
        self._check_frozen()
        return super().__delitem__(key)

    def clear(self) -> None:
        self._check_frozen()
        return super().clear()

    def pop(self, *args) -> Tensor:
        self._check_frozen()
        return super().pop(*args)

    def popitem(self) -> tuple[LayerComponent, Tensor]:
        self._check_frozen()
        return super().popitem()

    def setdefault(self, *args) -> Tensor:
        self._check_frozen()
        return super().setdefault(*args)

    def update(self, *args, **kwargs) -> None:
        self._check_frozen()
        return super().update(*args, **kwargs)

    def clone(self) -> Self:
        return deepcopy(self)


class ArithmeticOperation(FreezableDict):
    def __init__(self, config=None, positions=None) -> None:
        super().__init__()
        self.config = config
        self.positions = positions

    def check_act_dict_compatibility(self, other) -> None:
        if self.keys() != other.keys():
            warnings.warn(
                "ActivationDicts have different keys; only matching keys will be processed."
            )

    def __add__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            result = empty_dict_like(self)
            for key in self.keys():
                if key in other:
                    result[key] = self[key] + other[key]
            return result
        elif isinstance(other, (int, float, Tensor)):
            result = empty_dict_like(self)
            for key in self.keys():
                result[key] = self[key] + other
            return result
        else:
            raise NotImplementedError(
                f"Addition not supported for this type (received: {type(other).__name__})."
            )

    def __iadd__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            for key in self.keys():
                if key in other:
                    self[key].add_(other[key])
            return self
        elif isinstance(other, (int, float, Tensor)):
            for key in self.keys():
                self[key].add_(other)
            return self
        else:
            raise NotImplementedError(
                f"In-place addition not supported for this type (received: {type(other).__name__})."
            )

    def __radd__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            return NotImplemented
        return self.__add__(other)

    def __sub__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            result = empty_dict_like(self)
            for key in self.keys():
                if key in other:
                    result[key] = self[key] - other[key]
            return result
        elif isinstance(other, (int, float, Tensor)):
            result = empty_dict_like(self)
            for key in self.keys():
                result[key] = self[key] - other
            return result
        else:
            raise NotImplementedError(
                f"Subtraction not supported for this type (received: {type(other).__name__})."
            )

    def __isub__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            for key in self.keys():
                if key in other:
                    self[key].sub_(other[key])
            return self
        elif isinstance(other, (int, float, Tensor)):
            for key in self.keys():
                self[key].sub_(other)
            return self
        else:
            raise NotImplementedError(
                f"In-place subtraction not supported for this type (received: {type(other).__name__})."
            )

    def __mul__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            result = empty_dict_like(self)
            for key in self.keys():
                if key in other:
                    result[key] = self[key] * other[key]
            return result
        elif isinstance(other, (int, float, Tensor)):
            result = empty_dict_like(self)
            for key in self.keys():
                result[key] = self[key] * other
            return result
        else:
            raise NotImplementedError(
                f"Multiplication not supported for this type (received: {type(other).__name__})."
            )

    def __imul__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            for key in self.keys():
                if key in other:
                    self[key].mul_(other[key])
            return self
        elif isinstance(other, (int, float, Tensor)):
            for key in self.keys():
                self[key].mul_(other)
            return self
        else:
            raise NotImplementedError(
                f"In-place multiplication not supported for this type (received: {type(other).__name__})."
            )

    def __rmul__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            return NotImplemented
        return self.__mul__(other)

    def __truediv__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            result = empty_dict_like(self)
            for key in self.keys():
                if key in other:
                    result[key] = self[key] / other[key]
            return result
        elif isinstance(other, (int, float, Tensor)):
            result = empty_dict_like(self)
            for key in self.keys():
                result[key] = self[key] / other
            return result
        else:
            raise NotImplementedError(
                f"Division not supported for this type (received: {type(other).__name__})."
            )

    def __itruediv__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            for key in self.keys():
                if key in other:
                    self[key].div_(other[key])
            return self
        elif isinstance(other, (int, float, Tensor)):
            for key in self.keys():
                self[key].div_(other)
            return self
        else:
            raise NotImplementedError(
                f"In-place division not supported for this type (received: {type(other).__name__})."
            )

    def __rtruediv__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            return NotImplemented
        elif isinstance(other, (int, float, Tensor)):
            result = empty_dict_like(self)
            for key in self.keys():
                result[key] = other / self[key]
            return result
        else:
            raise NotImplementedError(
                f"Division not supported for this type (received: {type(other).__name__})."
            )

    def __matmul__(self, other) -> Self:
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            result = empty_dict_like(self)
            for key in self.keys():
                if key in other:
                    result[key] = self[key] @ other[key]
            return result
        elif isinstance(other, Tensor):
            result = empty_dict_like(self)
            for key in self.keys():
                result[key] = self[key] @ other
            return result
        else:
            raise NotImplementedError(
                f"Matrix multiplication not supported for this type (received: {type(other).__name__})."
            )

    def __rmatmul__(self, other) -> Self:
        if isinstance(other, Tensor):
            result = empty_dict_like(self)
            for key in self.keys():
                result[key] = other @ self[key]
            return result
        else:
            raise NotImplementedError(
                f"Matrix multiplication not supported for this type (received: {type(other).__name__})."
            )


class ActivationDict(ArithmeticOperation):
    """
    A dictionary-like object to store and manage model activations.

    This class extends the standard dictionary to provide features specific to handling
    activations from neural networks, such as freezing the state, managing head
    dimensions, and moving data to a GPU.

    Args:
        config: The model's configuration object.
        positions: The sequence positions of the activations.
    """

    def __init__(
        self,
        config,
        positions,
        value_type: str = "activation",
    ):
        super().__init__()
        self.config = config
        self.num_heads = config.num_attention_heads
        self.num_layers = config.num_hidden_layers
        self.head_dim = getattr(
            config, "head_dim", config.hidden_size // config.num_attention_heads
        )
        self.model_dim = config.hidden_size
        self.num_kv_heads = getattr(config, "num_key_value_heads", self.num_heads)
        self.fused_heads = True
        self.positions = regularize_position(positions)
        self.value_type = value_type  # e.g., 'activation' or 'gradient' or 'scores'
        # Keep placeholder device-agnostic; callers can set/transfer explicitly.
        self.attention_mask = torch.empty(0)

    def empty_like(self) -> Self:
        return empty_dict_like(self)

    def zeros_like(self) -> Self:
        return zeros_dict_like(self)

    def add_(self, other, *, alpha: int | float = 1) -> Self:
        """In-place add, mirroring Tensor.add_ semantics where relevant."""
        if isinstance(other, ActivationDict):
            self.check_act_dict_compatibility(other)
            for key in self.keys():
                if key in other:
                    self[key].add_(other[key], alpha=alpha)
            return self
        elif isinstance(other, (int, float, Tensor)):
            for key in self.keys():
                self[key].add_(other, alpha=alpha)
            return self
        else:
            raise NotImplementedError(
                f"In-place addition not supported for this type (received: {type(other).__name__})."
            )

    def reorganize(self) -> Self:
        execution_order = {
            "layer_in": 0,
            "z": 1,
            "attn": 2,
            "mlp_hidden": 3,
            "mlp": 4,
            "layer_out": 5,
        }

        new_dict = empty_dict_like(self)
        keys = list(self.keys())
        keys = sorted(
            keys,
            key=lambda x: (x[0], execution_order[x[1]]),
        )

        for key in keys:
            new_dict[key] = self[key]
        return new_dict

    def split_heads(self) -> Self:
        """
        Splits the 'z' activations into individual heads.
        Assumes 'z' activations are stored with fused heads.
        """
        if self.value_type not in ["activation", "gradient"]:
            warnings.warn(
                f"Splitting heads is typically only relevant for activations or gradients, not '{self.value_type}'."
            )

        pre_state = self._frozen
        self.unfreeze()
        if not self.fused_heads:
            return self

        for layer, component in self.keys():
            if component != "z":
                continue
            # Rearrange from (batch, pos, n_heads * d_head) to (batch, pos, n_heads, d_head)
            tensor = self[(layer, "z")]
            self[(layer, "z")] = einops.rearrange(
                tensor,
                "batch pos (head d_head) -> batch pos head d_head",
                head=self.num_heads,
                d_head=self.head_dim,
            )
            if tensor.grad is not None:
                self[(layer, "z")].grad = einops.rearrange(
                    tensor.grad,
                    "batch pos (head d_head) -> batch pos head d_head",
                    head=self.num_heads,
                    d_head=self.head_dim,
                )
        self.fused_heads = False
        self._frozen = pre_state
        return self

    def merge_heads(self) -> Self:
        """
        Merges the 'z' activations from individual heads back into a single tensor.
        """
        if self.value_type not in ["activation", "gradient"]:
            warnings.warn(
                f"Splitting heads is typically only relevant for activations or gradients, not '{self.value_type}'."
            )

        pre_state = self._frozen
        self.unfreeze()
        if self.fused_heads:
            return self
        for layer, component in self.keys():
            if component != "z":
                continue
            # Rearrange from (batch, pos, n_heads, d_head) to (batch, pos, n_heads * d_head)
            tensor = self[(layer, "z")]
            self[(layer, "z")] = einops.rearrange(
                tensor,
                "batch pos head d_head -> batch pos (head d_head)",
                head=self.num_heads,
                d_head=self.head_dim,
            )
            if tensor.grad is not None:
                self[(layer, "z")].grad = einops.rearrange(
                    tensor.grad,
                    "batch pos head d_head -> batch pos (head d_head)",
                    head=self.num_heads,
                    d_head=self.head_dim,
                )
        self.fused_heads = True
        self._frozen = pre_state
        return self

    def apply(
        self,
        function: Callable,
        *args,
        **kwargs,
    ) -> Self:
        mask_aware = kwargs.pop("mask_aware", False)

        if mask_aware:
            warnings.warn("Using .apply() with mask_aware is not grad safe")

            # Check if attention mask is properly set
            if self.attention_mask.numel() == 0:
                raise ValueError(
                    "attention_mask must be set before using mask_aware=True. "
                    "Set it via: activation_dict.attention_mask = your_mask"
                )

            base_mask = self.attention_mask.bool()

            def apply_func(x: Tensor, *args, **kwargs) -> Tensor:
                mask = base_mask.to(x.device)
                while mask.ndim < x.ndim:
                    mask = mask.unsqueeze(-1)

                x_masked = torch.where(mask, x, torch.nan)
                return function(x_masked, *args, **kwargs)
        else:
            apply_func = function

        output = empty_dict_like(self)
        output.value_type = self.value_type
        output.attention_mask = self.attention_mask
        for layer, component in self.keys():
            output[(layer, component)] = apply_func(self[(layer, component)], *args, **kwargs)

        return output

    def to(self, *args, **kwargs) -> Self:
        """Moves all tensors in the ActivationDict to the requested device/dtype."""
        self.attention_mask = self.attention_mask.to(*args, **kwargs)
        for key in self.keys():
            self[key] = self[key].to(*args, **kwargs)
        return self

    def cuda(self) -> Self:
        """Moves all activation tensors to CUDA."""
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is not available, cannot move ActivationDict to GPU.")
        return self.to(device="cuda")

    def cpu(self) -> Self:
        """Moves all activation tensors to the CPU."""
        return self.to(device="cpu")

    def extract_positions(self, keys: list[LayerComponent] | None = None) -> Self:
        pre_state = self._frozen
        self.unfreeze()

        if self.attention_mask.numel() > 0:
            if self.attention_mask.ndim == 1:
                self.attention_mask = self.attention_mask.unsqueeze(0)
            if self.attention_mask.ndim >= 2:
                self.attention_mask = self.attention_mask[:, self.positions]

        if keys is None:
            keys = list(self.keys())

        for key in keys:
            if key in self:
                tensor = self[key]
                sliced = tensor[:, self.positions, :]
                self[key] = sliced
                if tensor.grad is not None:
                    self[key].grad = tensor.grad[:, self.positions, :]  # type: ignore
            else:
                warnings.warn(f"Key {key} not found in ActivationDict. Skipping.")

        self._frozen = pre_state
        return self

    def detach(self) -> Self:
        """Detaches all activation tensors and the attention mask from the computation graph."""
        self.attention_mask = self.attention_mask.detach()
        for key in self.keys():
            self[key] = self[key].detach()
        return self

    def get_grads(self, keys: list[LayerComponent] | None = None) -> Self:
        new_obj = empty_dict_like(self)
        new_obj.value_type = "gradient"
        new_obj.attention_mask = self.attention_mask

        if keys is None:
            keys = list(self.keys())

        for key in keys:
            if key in self:
                if self[key].grad is not None:
                    new_obj[key] = self[key].grad.detach()  # type: ignore
                    self[key].grad = None
                else:
                    warnings.warn(f"Key {key} has no gradient. Skipping.")
            else:
                warnings.warn(f"Key {key} not found in ActivationDict. Skipping.")

        return new_obj


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
