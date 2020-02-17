import torch
from torch import nn
from collections.abc import Iterable


class Transform(nn.Module):
    """Base class for Transform"""

    has_inverse = True
    changes_numel = False

    def forward(self, x):
        """
        Forward transform.
        Computes `z = f(x)` and `log|det J|` for `J = df(x)/dx`
        such that `log p_x(x) = log p_z(f(x)) + log|det J|`.

        Args:
            x: Tensor, shape (batch_size, ...)

        Returns:
            z: Tensor, shape (batch_size, ...)
            ldj: Tensor, shape (batch_size,)
        """
        raise NotImplementedError()

    def inverse(self, z):
        """
        Inverse transform.
        Computes `x = f^{-1}(z)`.

        Args:
            z: Tensor, shape (batch_size, ...)

        Returns:
            x: Tensor, shape (batch_size, ...)
        """
        raise NotImplementedError()

    def z_shape(self, x_shape):
        """
        Computes z.shape for a given x.shape.

        Args:
            x_shape: torch.Size or iterable

        Returns:
            z_shape: torch.Size or iterable
        """
        return torch.Size(x_shape)

    def x_shape(self, z_shape):
        """
        Computes x.shape for a given z.shape.

        Args:
            z_shape: torch.Size or iterable

        Returns:
            x_shape: torch.Size or iterable
        """
        return torch.Size(z_shape)


class SequentialTransform(Transform):
    """
    Chains multiple Transform objects sequentially.

    Args:
        transforms: Transform or iterable with each element being a Transform object
    """

    def __init__(self, transforms):
        super(SequentialTransform, self).__init__()
        if isinstance(transforms, Transform): transforms = [transforms]
        assert isinstance(transforms, Iterable)
        assert all(isinstance(transform, Transform) for transform in transforms)
        self.has_inverse = all(transform.has_inverse for transform in transforms)
        self.changes_numel = any(transform.changes_numel for transform in transforms)
        self.transforms = nn.ModuleList(transforms)

    def forward(self, x):
        batch_size = x.shape[0]
        ldj = torch.zeros(batch_size, device=x.device)
        for transform in self.transforms:
            x, l = transform.forward(x)
            ldj += l
        return x, ldj

    def inverse(self, z):
        for transform in reversed(self.transforms):
            z = transform.inverse(z)
        return z

    def z_shape(self, x_shape):
        for transform in self.transforms:
            x_shape = transform.z_shape(x_shape)
        return x_shape

    def x_shape(self, z_shape):
        for transform in reversed(self.transforms):
            z_shape = transform.x_shape(z_shape)
        return z_shape
