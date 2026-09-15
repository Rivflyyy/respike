# ReSpike

Core code for **ReSpike: Reparameterized Spiking Networks for Efficient Speech Command Recognition**.

This repository provides the core projection reparameterization for review. **The complete code will be released after acceptance.** The current release includes the conversion implementation, a runnable example, and tests. The full model architecture, training and data pipelines, and pretrained checkpoints will be included in the complete release.

## Projection reparameterization

Let `x` be the input spikes, `g` the binary gate, and `s` the learned scale for each input channel. The original projection is

```text
u = x * (1 + s * g)
y = linear(u, W, b)
```

Once training is complete, the channel scales are fixed. We precompute a second weight matrix by multiplying each input column of `W` by its corresponding scale

```text
Ws = W * s[None, :]
y = linear(x, W, b) + linear(x * g, Ws, None)
```

The two expressions are algebraically equivalent. Both paths receive binary inputs when `x` and `g` are binary, and the bias is added only once. Conversion preserves the source layer and stores a separate frozen copy with `C_out * C_in` additional weights. The gate is still evaluated for each input by the surrounding model.

This implementation uses dense PyTorch reference kernels. It demonstrates the projection identity and weight storage, rather than hardware speed or energy savings. Floating-point operation order can produce small numerical differences.

## Installation

Python 3.10 or later and PyTorch 2.1 or later are required. No GPU or spiking neuron library is needed. This release was checked on CPU with PyTorch 2.1.2 under Python 3.10 and 3.11.

```bash
git clone https://github.com/Rivflyyy/respike.git
cd respike
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[test]'
```

## Quick start

```python
import torch
from torch import nn
from respike import reparameterize_linear

projection = nn.Linear(32, 64).eval()
channel_scale = torch.full((32,), 0.1)
converted = reparameterize_linear(projection, channel_scale)

x = torch.randint(0, 2, (8, 4, 32)).float()
g = torch.randint(0, 2, x.shape).float()

with torch.inference_mode():
    reference = projection(x * (1 + channel_scale * g))
    actual = converted(x, g)

torch.testing.assert_close(actual, reference, rtol=1e-5, atol=1e-6)
```

The example uses synthetic inputs and an initialized layer. In a trained model, pass its learned projection and channel scales to `reparameterize_linear` and supply the same gate to the converted layer. Inputs and gates have matching shapes `(..., C_in)`. The scale has shape `(C_in,)` and shares the projection's device and dtype. Conversion is performed once after training, not on every forward pass.

Run the CPU example and tests

```bash
python examples/verify_projection.py
python -m pytest -q
```

Tests cover binary and real inputs, positive and negative scales, projections with and without bias, FP32 and FP64, source independence, and serialization of both weight matrices. These checks verify the released projection module and do not reproduce the paper's speech recognition experiments.

## Files

- `respike/reparameterization.py` contains the conversion and inference module.
- `examples/verify_projection.py` provides a minimal projection equivalence check.
- `tests/test_reparameterization.py` verifies the projection behavior and stored weights.

## License

Apache License 2.0. See `LICENSE` and `NOTICE`.
