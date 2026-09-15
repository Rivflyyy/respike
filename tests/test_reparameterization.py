"""Check numerical equivalence, storage, and independence from source weights."""

import io

import pytest
import torch
from torch import nn

from respike import reparameterize_linear


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("shape", [(7,), (5, 7), (6, 3, 7)])
@pytest.mark.parametrize("binary", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_projection_equivalence(bias, shape, binary, dtype):
    torch.manual_seed(19)
    projection = nn.Linear(7, 11, bias=bias, dtype=dtype).eval()
    scale = torch.linspace(-1.5, 1.5, 7, dtype=dtype)
    inputs = torch.randn(shape, dtype=dtype)
    gate = torch.randn(shape, dtype=dtype)
    if binary:
        inputs = (inputs > 0).to(dtype)
        gate = (gate > 0).to(dtype)
    converted = reparameterize_linear(projection, scale)
    with torch.no_grad():
        reference = projection(inputs * (1.0 + scale * gate))
        actual = converted(inputs, gate)
    tolerance = 1e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("gate_value", [0.0, 1.0])
def test_bias_is_added_once(gate_value):
    projection = nn.Linear(3, 2)
    with torch.no_grad():
        projection.weight.fill_(2.0)
        projection.bias.copy_(torch.tensor([3.0, -4.0]))
    scale = torch.tensor([-1.0, 0.0, 2.0])
    inputs = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    gate = torch.full_like(inputs, gate_value)
    expected_second = 6.0 + 2.0 * gate_value
    expected = torch.tensor([[3.0, -4.0], [expected_second + 3.0, expected_second - 4.0]])
    actual = reparameterize_linear(projection, scale)(inputs, gate)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_conversion_is_a_frozen_copy():
    projection = nn.Linear(4, 3)
    scale = torch.tensor([0.0, -0.2, 0.1, 0.3], requires_grad=True)
    original = {name: value.detach().clone() for name, value in projection.state_dict().items()}
    converted = reparameterize_linear(projection, scale)
    for name, value in original.items():
        torch.testing.assert_close(projection.state_dict()[name], value, rtol=0, atol=0)
    assert projection.training
    assert projection.weight.requires_grad and scale.requires_grad
    assert not list(converted.parameters())
    assert not converted.training
    assert converted.scaled_weight.numel() == 12
    assert all(not value.requires_grad for value in converted.buffers())
    with torch.no_grad():
        projection.weight.zero_()
        projection.bias.zero_()
        scale.zero_()
    torch.testing.assert_close(converted.weight, original["weight"], rtol=0, atol=0)
    torch.testing.assert_close(converted.bias, original["bias"], rtol=0, atol=0)
    torch.testing.assert_close(
        converted.scaled_weight,
        original["weight"] * torch.tensor([0.0, -0.2, 0.1, 0.3]),
        rtol=0, atol=0,
    )
    with pytest.raises(RuntimeError, match="inference only"):
        converted.train()


@pytest.mark.parametrize("bias", [False, True])
def test_serialized_state_restores_both_paths(bias):
    torch.manual_seed(23)
    converted = reparameterize_linear(nn.Linear(4, 6, bias=bias), torch.randn(4))
    buffer = io.BytesIO()
    torch.save(converted.state_dict(), buffer)
    buffer.seek(0)
    restored = reparameterize_linear(nn.Linear(4, 6, bias=bias), torch.zeros(4))
    restored.load_state_dict(torch.load(buffer, weights_only=True), strict=True)
    inputs = torch.randint(0, 2, (3, 4)).float()
    gate = torch.randint(0, 2, (3, 4)).float()
    torch.testing.assert_close(restored(inputs, gate), converted(inputs, gate), rtol=0, atol=0)


def test_invalid_shapes_are_rejected():
    projection = nn.Linear(4, 6)
    with pytest.raises(ValueError, match="channel_scale"):
        reparameterize_linear(projection, torch.ones(6))
    converted = reparameterize_linear(projection, torch.ones(4))
    with pytest.raises(ValueError, match="same shape"):
        converted(torch.ones(2, 4), torch.ones(4))
    with pytest.raises(ValueError, match="in_features"):
        converted(torch.ones(2, 3), torch.ones(2, 3))
