"""Exact decomposition of a projection with fixed residual channel scales.

Adapted from the ReSpike research implementation for this partial release.
The projection is independent of the speech backbone and neuron library.
Licensed under the Apache License, Version 2.0. See LICENSE and NOTICE.
"""

from torch import Tensor, nn
from torch.nn import functional as F


class ReparameterizedLinear(nn.Module):
    """Frozen projection of ``x * (1 + scale * gate)`` through two paths.

    Stores ``weight`` and ``scaled_weight = weight * scale[None, :]``.
    Bias is added once. Inputs and gates must have matching shapes
    ``(..., in_features)``. Binary inputs make both paths spike inputs;
    the algebraic identity also holds for real inputs and gates.

    These are dense PyTorch reference operations, not sparse hardware kernels.
    Floating-point operation ordering can introduce numerical differences.
    """

    def __init__(self, projection: nn.Linear, channel_scale: Tensor):
        super().__init__()
        if not isinstance(projection, nn.Linear):
            raise TypeError("projection must be an nn.Linear")
        if channel_scale.ndim != 1 or channel_scale.numel() != projection.in_features:
            raise ValueError("channel_scale must have shape (in_features,)")
        if channel_scale.device != projection.weight.device:
            raise ValueError("channel_scale and projection must use the same device")
        if channel_scale.dtype != projection.weight.dtype:
            raise ValueError("channel_scale and projection must use the same dtype")

        self.in_features = projection.in_features
        self.out_features = projection.out_features
        weight = projection.weight.detach().clone()
        scale = channel_scale.detach()
        self.register_buffer("weight", weight)
        self.register_buffer("scaled_weight", (weight * scale.unsqueeze(0)).clone())
        self.register_buffer(
            "bias",
            None if projection.bias is None else projection.bias.detach().clone(),
        )
        self.eval()

    def train(self, mode: bool = True):
        if mode:
            raise RuntimeError("reparameterized projections are for inference only")
        return super().train(False)

    def forward(self, inputs: Tensor, gate: Tensor) -> Tensor:
        if inputs.ndim < 1 or inputs.shape[-1] != self.in_features:
            raise ValueError("inputs must have shape (..., in_features)")
        if inputs.shape != gate.shape:
            raise ValueError("gate must have the same shape as inputs")
        return F.linear(inputs, self.weight, self.bias) + F.linear(
            inputs * gate, self.scaled_weight, bias=None
        )


def reparameterize_linear(
    projection: nn.Linear, channel_scale: Tensor
) -> ReparameterizedLinear:
    """Copy a trained projection and precompute its second weight matrix.

    Neither the source projection nor its scale is modified. The result stores
    both weight matrices as buffers, with no trainable parameters. Its
    ``state_dict`` contains everything needed to restore the converted layer.
    Call this after training, once the channel scales have been fixed.
    """
    return ReparameterizedLinear(projection, channel_scale)
