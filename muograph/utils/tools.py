import torch
from torch import Tensor
from scipy.ndimage import gaussian_filter
import numpy as np
from typing import Union, List


def normalize(x: Union[Tensor, np.ndarray]) -> Union[Tensor, np.ndarray]:
    r"""Normalize the input array.

    Args:
        x (Union[Tensor, np.ndarray]): Input array.

    Returns:
        array (Union [Tensor, np.ndarray]): The normalized array.
    """
    if isinstance(x, Tensor):
        return (x - torch.min(x)) / (torch.max(x) - torch.min(x))
    elif isinstance(x, np.ndarray):
        return (x - np.min(x)) / (np.max(x) - np.min(x))
    else:
        raise TypeError(
            f"Input type {type(x)} is not supported. Expected Tensor or np.ndarray."
        )


def apply_gaussian_filter(
    x: Union[Tensor, np.ndarray], sigma: List[float]
) -> Union[Tensor, np.ndarray]:
    r"""Applies a Guassian filter to the provided array:
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html).

    Args:
        x (Union[Tensor, np.dnarray]): Input array.

        sigma (Union[float, Tuple[float, ...]]): Standard deviation for Gaussian kernel.The standard deviations of the Gaussian filter are given for each axis as a sequence,
        or as a single number, in which case it is equal for all axes.

    Returns:
        array (Union[Tensor, np.ndarray]): Returned array of same shape as input.
    """

    if isinstance(x, Tensor):
        ndim = len(x.size())
    elif isinstance(x, np.ndarray):
        ndim = len(x.shape)

    if ndim != len(sigma):
        raise ValueError("`sigma` must have one element for each dimension of `x`.")
    elif isinstance(x, Tensor):
        return torch.tensor(gaussian_filter(x.numpy(), sigma=sigma))
    elif isinstance(x, np.ndarray):
        return gaussian_filter(x, sigma=sigma)
    else:
        raise TypeError(
            f"Input type {type(x)} is not supported. Expected Tensor or np.ndarray."
        )
