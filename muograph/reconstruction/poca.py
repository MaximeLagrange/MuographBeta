import torch
from torch import Tensor
from copy import deepcopy
from typing import Optional, List
from fastprogress import progress_bar
import numpy as np

from utils.save import AbsSave
from utils.device import DEVICE
from utils.params import dtype_track
from volume.volume import Volume
from tracking.tracking import TrackingMST
from plotting.voxel import VoxelPlotting


def are_parallel(v1: Tensor, v2: Tensor, tol: float = 1e-5) -> bool:
    cross_prod = torch.cross(v1, v2)
    return torch.all(torch.abs(cross_prod) < tol)


class POCA(AbsSave, VoxelPlotting):
    r"""
    A class for Point Of Closest Approach computation in the context of a Muon Scattering Tomography analysis.
    """

    _parallel_mask: Optional[Tensor] = None  # (mu)
    _poca_points: Optional[Tensor] = None  # (mu, 3)
    _n_poca_per_vox: Optional[Tensor] = None  # (nx, ny, nz)
    _poca_indices: Optional[Tensor] = None  # (mu, 3)
    _mask_in_voi: Optional[Tensor] = None  # (mu)

    _vars_to_save = [
        "poca_points",
        "n_poca_per_vox",
        "poca_indices",
    ]

    _vars_to_load = [
        "poca_points",
        "n_poca_per_vox",
        "poca_indices",
    ]

    def __init__(
        self,
        tracking: Optional[TrackingMST] = None,
        voi: Optional[Volume] = None,
        poca_file: Optional[str] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        r"""
        Initializes the POCA object with either an instance of the TrackingMST class or a
        poca.hdf5 file.

        Args:
            - tracking (Optional[TrackingMST]): Instance of the TrackingMST class.
            - voi (Optional[Volume]): Instance of the Volume class. If provided, muon events with
            poca locations outside the voi will be filtered out, the number of poca locations per voxel
            `n_poca_per_vox` as well as the voxel indices of each poca location will be computed.
            poca_file (Optional[str]): The path to the poca.hdf5 to load attributes from.
            - output_dir (Optional[str]): Path to a directory where to save POCA attributes
            in a hdf5 file.
        """
        AbsSave.__init__(self, output_dir)
        VoxelPlotting.__init__(self, voi)

        if tracking is None and poca_file is None:
            raise ValueError("Provide either poca.hdf5 file of TrackingMST instance.")

        # Compute poca attributes if TrackingMST is provided
        elif tracking is not None:
            self.tracks = deepcopy(tracking)
            self.voi = voi

            # Remove parallel events
            self.tracks._filter_muons(self.parallel_mask)

            # Remove POCAs outside voi
            if voi is not None:
                self.tracks._filter_muons(self.mask_in_voi)
                self._filter_pocas(self.mask_in_voi)

            # Save attributes to hdf5
            if voi is None:
                self.save_attr(["poca_points"], self.output_dir, filename="poca")
            else:
                self.save_attr(self._vars_to_save, self.output_dir, filename="poca")

        # Load poca attributes from hdf5 if poca_file is provided
        elif tracking is None and poca_file is not None:
            self.load_attr(self._vars_to_load, poca_file)

    @staticmethod
    def compute_parallel_mask(
        tracks_in: Tensor, tracks_out: Tensor, tol: float = 1e-7
    ) -> Tensor:
        """
        Compute a mask to filter out events with parallel or nearly parallel tracks.

        Arguments:
            tracks_in: Tensor containing the incoming tracks, with size (n_mu, 3).
            tracks_out: Tensor containing the outgoing tracks, with size (n_mu, 3).
            tol: Tolerance value for determining if tracks are parallel.

        Returns:
            mask: Boolean tensor with size (n_mu), where True indicates the event is not parallel.
        """
        # Compute the cross product for all pairs at once
        cross_prod = torch.cross(tracks_in, tracks_out, dim=1)

        # Compute the mask by checking if the cross product magnitude is below the tolerance
        mask = torch.all(torch.abs(cross_prod) >= tol, dim=1)

        return mask

    def _filter_pocas(self, mask: Tensor) -> None:
        r"""
        Remove poca points specified as False in `mask`.

        Arguments:
            mask: (N,) Boolean tensor. poca points with False elements will be removed.
        """
        self.poca_points = self.poca_points[mask]

    @staticmethod
    def compute_poca_points(
        points_in: Tensor, points_out: Tensor, tracks_in: Tensor, tracks_out: Tensor
    ) -> Tensor:
        """
        @MISC {3334866,
        TITLE = {Closest points between two lines},
        AUTHOR = {Brian (https://math.stackexchange.com/users/72614/brian)},
        HOWPUBLISHED = {Mathematics Stack Exchange},
        NOTE = {URL:https://math.stackexchange.com/q/3334866 (version: 2019-08-26)},
        EPRINT = {https://math.stackexchange.com/q/3334866},
        URL = {https://math.stackexchange.com/q/3334866}
        }

        Compute POCA points.

        Arguments:
            points_in: xyz coordinates of a point on the incomming track, with size (n_mu, 3).
            points_out: xyz coordinates of a point on the outgoing track, with size (n_mu, 3).
            tracks_in: The incomming track, with size (n_mu, 3).
            tracks_out: The outgoing track, with size (n_mu, 3).

        Returns:
            POCA points' coordinate(n_mu, 3)

        Given 2 lines L1, L2 aka incoming and outgoing tracks with parametric equation:
        L1 = P1 + t*V1

        1- A segment of shortest length between two 3D lines L1 L2 is perpendicular to both lines (if L1 L2 are neither parallele or in the same plane). One must compute V3, vector perpendicular to L1 and L2
        2- Search for points where L3 = P1 + t1*V1 +t3*V3 crosses L2. One must find t1 and t2 for which:
        L3 = P1 + t1*V1 +t3*V3 = P2 + t2*V2

        3- Then POCA location M is the middle of the segment Q1-Q2 where Q1,2 = P1,2 +t1,2*V1,2
        """
        from numpy import cross

        P1, P2 = points_in[:], points_out[:]
        V1, V2 = tracks_in[:], tracks_out[:]

        V3 = torch.tensor(cross(V2, V1), dtype=dtype_track, device=DEVICE)

        if are_parallel(V1, V2):
            raise ValueError("Tracks are parallel or nearly parallel")

        RES = P2 - P1
        LES = torch.transpose(torch.stack([V1, -V2, V3]), 0, 1)
        LES = torch.transpose(LES, -1, 1)

        try:
            ts = torch.linalg.solve(LES, RES)
        except RuntimeError as e:
            if "singular" in str(e):
                raise ValueError(
                    f"Singular matrix encountered for points: {P1}, {P2} with tracks: {V1}, {V2}"
                )
            else:
                raise e

        t1 = torch.stack([ts[:, 0], ts[:, 0], ts[:, 0]], -1)
        t2 = torch.stack([ts[:, 1], ts[:, 1], ts[:, 1]], -1)

        Q1s, Q2s = P1 + t1 * V1, P2 + t2 * V2
        M = (Q2s - Q1s) / 2 + Q1s

        return M

    @staticmethod
    def assign_voxel_to_pocas(poca_points: Tensor, voi: Volume) -> List[List[int]]:
        """
        Get the indinces of the voxel corresponding to each poca point.

        Arguments:
            - poca_points: Tensor with size (n_mu, 3).
            - voi: An instance of the VolumeInterest class.

        Returns:
            - poca points voxel indices as List[List[int]] with length n_mu.
        """
        indices = (
            torch.ones((len(poca_points), 3), dtype=torch.int16, device=DEVICE) - 2
        )

        print("\nAssigning voxel to each POCA point:")

        for i in progress_bar(range(len(poca_points))):
            mask_vox_x1 = poca_points[i, 0] >= voi.voxel_edges[:, :, :, 0, 0]
            mask_vox_x2 = poca_points[i, 0] <= voi.voxel_edges[:, :, :, 1, 0]

            mask_vox_y1 = poca_points[i, 1] >= voi.voxel_edges[:, :, :, 0, 1]
            mask_vox_y2 = poca_points[i, 1] <= voi.voxel_edges[:, :, :, 1, 1]

            mask_vox_z1 = poca_points[i, 2] >= voi.voxel_edges[:, :, :, 0, 2]
            mask_vox_z2 = poca_points[i, 2] <= voi.voxel_edges[:, :, :, 1, 2]

            mask_x_ = mask_vox_x1 & mask_vox_x2
            mask_y_ = mask_vox_y1 & mask_vox_y2
            mask_z_ = mask_vox_z1 & mask_vox_z2

            mask_vox = mask_x_ & mask_y_ & mask_z_

            indice = (mask_vox == 1).nonzero()

            # because of float precision (I assume?), more than 1 vox can be triggered.
            # if it is the case, only keep one voxel
            if len(indice) == 1:
                indices[i] = indice
            elif len(indice) > 1:
                indices[i] = indice[0]

        return indices.int()

    @staticmethod
    def compute_n_poca_per_vox(poca_points: Tensor, voi: Volume) -> Tensor:
        """
        Computes the number of POCA points per voxel, given a voxelized volume VOI.

        Arguments:
         - voi:VolumeIntrest, an instance of the VOI class.
         - poca_points: Tensor containing the poca points location, with size (n_mu, 3).

        Returns:
         - n_poca_per_vox: torch.tensor(dtype=int64) with size (nvox_x,nvox_y,nvox_z),
         the number of poca points per voxel.
        """

        n_poca_per_vox = torch.zeros(
            tuple(voi.n_vox_xyz), device=DEVICE, dtype=torch.int16
        )

        for i in range(voi.n_vox_xyz[2]):
            z_min = voi.xyz_min[2] + i * voi.vox_width
            z_max = z_min + voi.vox_width
            mask_slice = (poca_points[:, 2] >= z_min) & ((poca_points[:, 2] <= z_max))

            H, _, __ = np.histogram2d(
                poca_points[mask_slice, 0].numpy(),
                poca_points[mask_slice, 1].numpy(),
                bins=(int(voi.n_vox_xyz[0]), int(voi.n_vox_xyz[1])),
                range=(
                    (voi.xyz_min[0], voi.xyz_max[0]),
                    (voi.xyz_min[1], voi.xyz_max[1]),
                ),
            )

            n_poca_per_vox[:, :, i] = torch.tensor(H, dtype=torch.int16)

        return n_poca_per_vox

    @staticmethod
    def compute_mask_in_voi(poca_points: Tensor, voi: Volume) -> Tensor:
        """
        Compute the mask of POCA points located within the volumne of interest.

        Arguments:
            - poca_points: Tensor with size (n_mu, 3).
            - voi: An instance of the VolumeInterest class.

        Returns:
            - mask_in_voi: Tensor of bool with size (n_mu) with True if
            the POCA point is within the voi, False otherwise.
        """

        masks_xyz = [
            (poca_points[:, i] >= voi.xyz_min[i])
            & (poca_points[:, i] <= voi.xyz_max[i])
            for i in range(3)
        ]
        return masks_xyz[0] & masks_xyz[1] & masks_xyz[2]

    @property
    def poca_points(self) -> Tensor:
        if self._poca_points is None:
            self._poca_points = self.compute_poca_points(
                points_in=self.tracks.points_in,
                points_out=self.tracks.points_out,
                tracks_in=self.tracks.tracks_in,
                tracks_out=self.tracks.tracks_out,
            )
        return self._poca_points

    @poca_points.setter
    def poca_points(self, value: Tensor) -> None:
        self._poca_points = value

    @property
    def mask_in_voi(self) -> Tensor:
        if self._mask_in_voi is None:
            self._mask_in_voi = self.compute_mask_in_voi(
                poca_points=self.poca_points, voi=self.voi
            )
        return self._mask_in_voi

    @property
    def n_poca_per_vox(self) -> Tensor:
        if self._n_poca_per_vox is None:
            self._n_poca_per_vox = self.compute_n_poca_per_vox(
                poca_points=self.poca_points, voi=self.voi
            )
        return self._n_poca_per_vox

    @n_poca_per_vox.setter
    def n_poca_per_vox(self, value: Tensor) -> None:
        self._n_poca_per_vox = value

    @property
    def poca_indices(self) -> Tensor:
        if self._poca_indices is None:
            self._poca_indices = self.assign_voxel_to_pocas(
                poca_points=self.poca_points, voi=self.voi
            )
        return self._poca_indices

    @poca_indices.setter
    def poca_indices(self, value: Tensor) -> None:
        self._poca_indices = value

    @property
    def parallel_mask(self) -> Tensor:
        if self._parallel_mask is None:
            self._parallel_mask = self.compute_parallel_mask(
                self.tracks.tracks_in, self.tracks.tracks_out
            )
        return self._parallel_mask
