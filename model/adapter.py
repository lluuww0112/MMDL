"""Parameter-efficient adapter layers."""

import math
from typing import Tuple

import torch
import torch.nn as nn


class LoRA(nn.Module):
    """A learnable low-rank weight update.

    LoRA represents an additive update for a weight shaped
    (out_features, in_features) as (B @ A) * (alpha / rank). B
    is initialized to zero, so adding LoRA does not alter the base model
    before training begins.
    """

    def __init__(self, size: Tuple[int, int], rank: int, alpha: float | None = None):
        super().__init__()
        if len(size) != 2 or any(not isinstance(dimension, int) or isinstance(dimension, bool) or dimension <= 0 for dimension in size):
            raise ValueError("LoRA size must contain two positive integer dimensions.")
        if not isinstance(rank, int) or isinstance(rank, bool) or rank <= 0:
            raise ValueError("LoRA rank must be a positive integer.")
        if alpha is None:
            alpha = float(rank)
        if not isinstance(alpha, (int, float)) or isinstance(alpha, bool) or not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("LoRA alpha must be a finite positive number.")

        self.size = (int(size[0]), int(size[1]))
        self.rank = rank
        self.alpha = float(alpha)
        self.scaling = self.alpha / self.rank
        self.B = nn.Parameter(torch.empty(self.size[0], rank))
        self.A = nn.Parameter(torch.empty(rank, self.size[1]))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Initialize factors without changing the unadapted model output."""
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        nn.init.zeros_(self.B)

    def update(self, w: torch.Tensor | None = None) -> torch.Tensor:
        return self.get_update(w)

    def get_update(self, w: torch.Tensor | None = None) -> torch.Tensor:
        """Return the scaled additive update in (out_features, in_features) order.

        w is accepted for compatibility with the original adapter API;
        the update is independent of the base weight.
        """
        return (self.B @ self.A) * self.scaling

    def extra_repr(self) -> str:
        return f"size={self.size}, rank={self.rank}, alpha={self.alpha}"


class KRAdapter(nn.Module):
    """Official Khatri--Rao product adapter wrapped around one ``nn.Linear``."""

    def __init__(self, original_layer: nn.Linear, scaling: float = 2.0):
        if not isinstance(original_layer, nn.Linear):
            raise TypeError("KRAdapter can wrap only torch.nn.Linear layers.")
        if scaling <= 0:
            raise ValueError("KRAdapter scaling must be positive.")

        super().__init__()
        self.original_layer = original_layer
        self.scaling = float(scaling)
        self.merged = False
        self.s = (original_layer.out_features, original_layer.in_features)

        for parameter in self.original_layer.parameters():
            parameter.requires_grad_(False)

        min_shape, max_shape = min(self.s), max(self.s)
        self.r = int(math.sqrt(min_shape))
        self.d = 0
        while self.r * (self.r + self.d) < max_shape:
            self.d += 1

        # Adapter parameters remain fp32 for optimizer stability and are cast
        # to the activation dtype only for the matrix multiplication.
        device = original_layer.weight.device
        self.weight = nn.Parameter(torch.zeros(min_shape, self.r, device=device))
        self.v_weight = nn.Parameter(torch.zeros(min_shape, self.r + self.d, device=device))
        nn.init.kaiming_uniform_(self.v_weight, a=5)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.merged:
            return self.original_layer(x)
        update = self.get_update().to(dtype=x.dtype)
        return x @ update * self.scaling + self.original_layer(x)

    def get_update(self) -> torch.Tensor:
        """Return the additive update in ``(in_features, out_features)`` order."""
        update = (self.weight.unsqueeze(1) * self.v_weight.unsqueeze(-1)).reshape(self.weight.shape[0], -1)
        if min(range(2), key=lambda index: self.s[index]) != min(range(2), key=lambda index: update.shape[index]):
            update = update.T
        update = update[: self.s[0], : self.s[1]]
        return update.T

    @torch.no_grad()
    def merge(self) -> None:
        """Permanently add the adapter update to the wrapped layer weight."""
        if not self.merged:
            self.original_layer.weight.add_(self.get_update().to(dtype=self.original_layer.weight.dtype) * self.scaling)
            self.merged = True

    @torch.no_grad()
    def unmerge(self) -> None:
        """Remove a previously merged adapter update."""
        if self.merged:
            self.original_layer.weight.sub_(self.get_update().to(dtype=self.original_layer.weight.dtype) * self.scaling)
            self.merged = False
