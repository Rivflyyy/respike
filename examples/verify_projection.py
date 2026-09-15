"""Verify the projection identity with synthetic binary inputs on CPU."""

import torch
from torch import nn

from respike import reparameterize_linear


def main():
    torch.manual_seed(7)
    torch.set_num_threads(1)
    projection = nn.Linear(32, 64, bias=True).eval()
    channel_scale = torch.linspace(-0.5, 0.5, 32)
    inputs = (torch.rand(12, 4, 32) < 0.2).float()
    gate = (torch.rand_like(inputs) < 0.4).float()
    converted = reparameterize_linear(projection, channel_scale)

    with torch.inference_mode():
        reference = projection(inputs * (1.0 + channel_scale * gate))
        actual = converted(inputs, gate)
    torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)

    print(f"Maximum absolute projection error: {(actual - reference).abs().max():.8e}")
    print(f"Extra stored weights: {converted.scaled_weight.numel()}")
    print("Projection equivalence check passed (FP32, synthetic inputs, CPU).")


if __name__ == "__main__":
    main()
