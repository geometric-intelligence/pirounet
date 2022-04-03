"""Testing neural architectures."""

import numpy as np
import torch

from move.nn import RotationLayer

def test_rotation_layer():
    theta = np.pi
    rotation_layer = RotationLayer(theta)
    x = torch.Tensor([1., 0., 0.]).reshape((1, 1, 3))
    result = rotation_layer(x).squeeze().detach().cpu().numpy()
    expected = np.array([-1., 0., 0.])
    assert np.allclose(result, expected, atol=1e-6)