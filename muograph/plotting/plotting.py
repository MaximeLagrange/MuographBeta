import matplotlib.pyplot as plt
from torch import Tensor
from typing import Tuple, Optional
from volume.volume import Volume
from plotting.params import d_unit


def plot_n_poca_per_voxel(
    n_poca_per_vox: Tensor, dim: int, plot_name: Optional[str] = None
) -> None:
    r"""
    Plot the average number of poca point per voxel along a certain direction.

    Args:
        - n_poca_per_vox (Tensor): The number of poca point per voxel,
        with size (nx, ny, nz) with ni the number of voxels along the i direction.
        - dim (int): Integer defining the projection. dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """

    xlabels = ["Y", "X", "X"]
    ylabels = ["Z", "Z", "Y"]

    fig, ax = plt.subplots()
    fig.suptitle(
        "Average {}{} distribution of POCA points".format(xlabels[dim], ylabels[dim]),
        fontweight="bold",
        fontsize=15,
    )
    im = ax.imshow(
        n_poca_per_vox.sum(dim=dim).T / n_poca_per_vox.size()[dim],
        interpolation="nearest",
        origin="lower",
    )

    ax.set_xlabel(xlabels[dim] + f" [{d_unit}]")
    ax.set_ylabel(ylabels[dim] + f" [{d_unit}]")

    cbar_ax = fig.add_axes([1.0, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=" Average # poca per voxel")
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()


def plot_poca_points_per_pixel(
    poca_points: Tensor,
    dim: int,
    plot_name: Optional[str] = None,
    figsize: Tuple[float, float] = (6.0, 6.0),
) -> None:
    r"""
    Plot the number of poca point per pixel (screen pixel).

    Args:
        - poca_points (Tensor): Tensor poca points location.
        - dim (int): Integer defining the projection. dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """

    from matplotlib.colors import LinearSegmentedColormap

    # "Viridis-like" colormap with white background
    white_viridis = LinearSegmentedColormap.from_list(
        "white_viridis",
        [
            (0, "#ffffff"),
            (1e-20, "#440053"),
            (0.2, "#404388"),
            (0.4, "#2a788e"),
            (0.6, "#21a784"),
            (0.8, "#78d151"),
            (1, "#fde624"),
        ],
        N=256,
    )

    xlabels = ["Y", "X", "X"]
    ylabels = ["Z", "Z", "Y"]
    dimx = [1, 0, 0]
    dimy = [2, 2, 1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")

    density = ax.scatter_density(
        poca_points[:, dimx[dim]].numpy(),
        poca_points[:, dimy[dim]].numpy(),
        cmap=white_viridis,
    )

    fig.suptitle(
        "Average {}{} distribution of POCA points".format(xlabels[dim], ylabels[dim]),
        fontweight="bold",
        fontsize=15,
    )

    ax.set_xlabel(xlabels[dim] + f" [{d_unit}]")
    ax.set_ylabel(ylabels[dim] + f" [{d_unit}]")
    ax.set_aspect("equal")
    cbar_ax = fig.add_axes([1.0, 0.15, 0.05, 0.7])
    fig.colorbar(density, cax=cbar_ax, label="# points per pixel")
    plt.tight_layout()
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()


def plot_poca_points(
    poca_points: Tensor,
    dim: int,
    plot_name: Optional[str],
    figsize: Tuple[float, float] = (8, 8),
) -> None:
    r"""
    Plot the poca points location.

    Args:
        - poca_points (Tensor): poca points location.
        - dim (int): Integer defining the projection. dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """

    xlabels = ["Y", "X", "X"]
    ylabels = ["Z", "Z", "Y"]
    dimx = [1, 0, 0]
    dimy = [2, 2, 1]

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(1, 1, 1, projection="scatter_density")

    ax.scatter(
        poca_points[:, dimx[dim]].numpy(), poca_points[:, dimy[dim]].numpy(), alpha=0.01
    )

    fig.suptitle(
        "Average {}{} distribution of POCA points".format(xlabels[dim], ylabels[dim]),
        fontweight="bold",
        fontsize=15,
    )

    ax.set_xlabel(xlabels[dim] + f" [{d_unit}]")
    ax.set_ylabel(ylabels[dim] + f" [{d_unit}]")
    ax.set_aspect("equal")

    plt.tight_layout()
    if plot_name is not None:
        plt.savefig(plot_name)
    plt.show()


def plot_poca_points_hist2d(poca_points: Tensor, voi: Volume, dim: int = 2) -> None:
    r"""
    Plot the poca points location as 2D histogram.

    Args:
        - poca_points: Tensor poca points location.
        - dim: integer defining the projection. dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """
    poca_points = poca_points.numpy()

    dims = [_ for _ in [0, 1, 2] if _ != dim]
    labels = ["x", "y", "z"]
    fig, ax = plt.subplots()
    fig.suptitle("POCA points location", fontsize=15, fontweight="bold")

    im = ax.hist2d(
        poca_points[:, dims[0]],
        poca_points[:, dims[1]],
        bins=(voi.n_vox_xyz[dims[0]], voi.n_vox_xyz[dims[1]]),
    )

    ax.set_xlabel(labels[dims[0]] + " [{}]".format(d_unit))
    ax.set_ylabel(labels[dims[1]] + " [{}]".format(d_unit))

    cbar_ax = fig.add_axes([1.01, 0.15, 0.05, 0.7])
    fig.colorbar(im[3], cax=cbar_ax, label="# POCA points per pixel")
    ax.set_aspect("equal")

    plt.show()


def plot_density_pred(pred: Tensor, dim: int, plot_name: Optional[str] = None) -> None:
    r"""
    Plot the average density prediction per voxel averaged along a certain direction.

    Args:
        - pred (Tensor): The scattering denisyt predictions,
        with size (nx, ny, nz) with ni the number of voxels along the i direction.
        - dim (int): Integer defining the projection: dim = 0 -> YZ, dim = 1 -> XZ, dim = 2 -> XY.
    """

    xlabels = ["Y", "X", "X"]
    ylabels = ["Z", "Z", "Y"]

    fig, ax = plt.subplots()
    fig.suptitle(
        "Average {}{} density predictions".format(xlabels[dim], ylabels[dim]),
        fontweight="bold",
        fontsize=15,
    )
    im = ax.imshow(
        pred.sum(dim=dim).T / pred.size()[dim], interpolation="nearest", origin="lower"
    )

    ax.set_xlabel(xlabels[dim] + f" [{d_unit}]")
    ax.set_ylabel(ylabels[dim] + f" [{d_unit}]")

    cbar_ax = fig.add_axes([1.0, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax, label=" Average voxel density [a.u]")
    if plot_name is not None:
        plt.savefig(plot_name)
        plt.tight_layout()
    plt.show()
