import torch
from torch import Tensor
from scipy.ndimage import gaussian_filter
import numpy as np
from typing import Union, List


def normalize(s: Tensor) -> Tensor:
    s_max, s_min = torch.max(s), torch.min(s)

    return (s - s_min) / (s_max - s_min)


def apply_gaussian_filter(
    x: Union[Tensor, np.ndarray], sigma: List[float]
) -> Union[Tensor, np.ndarray]:
    """
    Applies a Guassian filter to the provided array.
    (https://docs.scipy.org/doc/scipy/reference/generated/scipy.ndimage.gaussian_filter.html)

    Args:
        x (Union[Tensor, np.dnarray]): The input array.
        sigma (Union[float, Tuple[float, ...]]): Standard deviation for Gaussian kernel.
        The standard deviations of the Gaussian filter are given for each axis as a sequence,
        or as a single number, in which case it is equal for all axes.

    Returns:
        Union[Tensor, np.ndarray]: Returned array of same shape as input.
    """

    if isinstance(x, Tensor):
        ndim = len(x.size())
    elif isinstance(x, np.ndarray):
        ndim = len(x.shape)

    if (ndim != len(sigma)) & (len(sigma) > 1):
        raise ValueError(
            "`sigma` should have one element for each dimension of `x`, or have a single element."
        )
    else:
        return gaussian_filter(x, sigma=sigma)
