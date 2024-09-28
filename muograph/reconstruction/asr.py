from typing import Optional, Tuple, Dict, Union, List
import numpy as np
from functools import partial
import math
import torch
from torch import Tensor
from pathlib import Path
from fastprogress import progress_bar
import h5py

from utils.save import AbsSave
from tracking.tracking import TrackingMST
from volume.volume import Volume
from plotting.voxel import VoxelPlotting
from utils.tools import normalize

value_type = Union[partial, Tuple[float, float], bool]


class ASR(AbsSave, VoxelPlotting):
    _xyz_voxel_pred: Optional[Tensor] = None  # (Nx, Ny, Nz)
    _triggered_voxels: Optional[List[Tensor]] = None
    _n_mu_per_vox: Optional[Tensor] = None  # (Nx, Ny, Nz)
    _recompute_preds = True

    _ars_params: Dict[str, value_type] = {
        "score_method": partial(np.quantile, q=0.5),
        "p_range": (0.0, 10000000),  # MeV
        "dtheta_range": (0.0, math.pi / 3),
        "use_p": False,
    }

    _vars_to_save = ["triggered_voxels"]

    _vars_to_load = ["trigerred_voxels"]

    def __init__(
        self,
        voi: Volume,
        tracking: TrackingMST,
        output_dir: Optional[str] = None,
        triggered_vox_file: Optional[str] = None,
    ) -> None:
        AbsSave.__init__(self, output_dir=output_dir)
        VoxelPlotting.__init__(self, voi=voi)

        self.voi = voi
        self.tracks = tracking

        if triggered_vox_file is None:
            self.save_triggered_vox(
                self.triggered_voxels, self.output_dir, "triggered_voxels.hdf5"
            )

        elif triggered_vox_file is not None:
            self.tracks = tracking
            self.triggered_voxels = self.load_triggered_vox(triggered_vox_file)

    @staticmethod
    def save_triggered_vox(
        triggered_voxels: List[Tensor], directory: Path, filename: str
    ) -> None:
        r"""
        Method for saving triggered voxel as a hdf5 file.
        """
        with h5py.File(directory / filename, "w") as f:
            print("Saving trigerred voxels to {}".format(directory / filename))
            for i, indices in enumerate(progress_bar(triggered_voxels)):
                f.create_dataset("{}".format(i), data=indices)
        f.close()

    @staticmethod
    def load_triggered_vox(triggered_vox_file: str) -> List[np.ndarray]:
        r"""
        Method for loading triggered voxel from hdf5 file.
        """
        with h5py.File(triggered_vox_file, "r") as f:
            print("Loading trigerred voxels from {}".format(triggered_vox_file))
            triggered_voxels = [
                f["{}".format(i)][:] for i, _ in enumerate(progress_bar(f.keys()))
            ]
        f.close()
        return triggered_voxels

    @staticmethod
    def _compute_xyz_in_out(
        points_in: Tensor,
        points_out: Tensor,
        voi: Volume,
        theta_xy_in: Tuple[Tensor],
        theta_xy_out: Tuple[Tensor],
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Compute muon position (x,y,z) when enters/exits the volume,
        both for the incoming and outgoing tracks.

        Args:
             - points_in (Tensor): Points on incoming muon tracks.
             - points_out (Tensor): Points on outgoing muon tracks.
             - voi (Volume): Instance of the volume class.
             - theta_xy_in (Tensor): The incoming projected zenith angle in XZ and YZ plane.
             - theta_xy_out (Tensor): The outgoing projected zenith angle in XZ and YZ plane.

        Returns:
             - xyz_in_VOI [3, 2, mu] the muon position when entering/exiting the volume,
            for the INCOMING tracks.
             - xyz_out_VOI [3, 2, mu] the muon position when entering/exiting the volume,
            for the OUTGOING tracks.
        """
        n_mu = theta_xy_in[0].size(0)
        xyz_in_voi, xyz_out_voi = torch.zeros((n_mu, 2, 3)), torch.zeros((n_mu, 2, 3))

        for point, theta_xy, pm, xyz in zip(
            [points_in, points_out],
            [theta_xy_in, theta_xy_out],
            [1, -1],
            [xyz_in_voi, xyz_out_voi],
        ):
            dz = (abs(point[:, 2] - voi.xyz_max[2]), abs(point[:, 2] - voi.xyz_min[2]))

            for coord, theta in zip([0, 1], theta_xy):
                xyz[:, 0, coord] = point[:, coord] - dz[1] * torch.tan(theta) * (pm)
                xyz[:, 1, coord] = point[:, coord] - dz[0] * torch.tan(theta) * (pm)
            xyz[:, 0, 2], xyz[:, 1, 2] = voi.xyz_min[2], voi.xyz_max[2]

        return xyz_in_voi, xyz_out_voi

    @staticmethod
    def _compute_discrete_tracks(
        voi: Volume,
        xyz_in_out_voi: Tuple[Tensor, Tensor],
        theta_xy_in: Tuple[Tensor, Tensor],
        theta_xy_out: Tuple[Tensor, Tensor],
        n_points_per_z_layer: int = 3,
    ) -> Tuple[Tensor, Tensor]:
        r"""
        Computes a discretized version of the muon incoming and outgoing track
        within the volume.
        The number of points is defined as n_points = n_points_per_z_layer * n_z_layer,
        where n_z_layer is the number of voxels along z.

        Args:
             - voi (Volume): Instance of the volume class.
             - xyz_in_out_voi (Tensor): The location of muons when entering/exiting the volume for incoming and outgoing tracks
             - theta_xy_in (Tuple[Tensor, Tensor]):  The incoming muon zenith angle projections in the XZ and YZ plane.
             - theta_xy_out (Tuple[Tensor, Tensor]):  The outcoming muon zenith angle projections in the XZ and YZ plane.
             - n_points_per_z_layer (int): The number of locations per voxel. Must be not too small (all the voxels are not triggered),
            nor too large (computationaly expensive). Default value is set at 3  point per voxel.

        Returns:
             - The discretized incoming and outgoing tracks with size (3, n_points, n_mu)
        """
        n_mu = theta_xy_in[0].size(0)
        n_points = (voi.n_vox_xyz[2] + 1) * n_points_per_z_layer

        # Compute the z locations cross the voi
        z_discrete = (
            torch.linspace(
                torch.min(voi.voxel_edges[0, 0, :, :, 2]),
                torch.max(voi.voxel_edges[0, 0, :, :, 2]),
                n_points,
            )[:, None]
        ).expand(-1, n_mu)

        xyz_discrete_in, xyz_discrete_out = torch.ones((3, n_points, n_mu)), torch.ones(
            (3, n_points, n_mu)
        )

        for xyz_discrete, theta_in_out, xyz_in_out in zip(
            [xyz_discrete_in, xyz_discrete_out],
            [theta_xy_in, theta_xy_out],
            xyz_in_out_voi,
        ):
            for dim, theta in zip([0, 1], theta_in_out):
                xyz_discrete[dim] = (
                    abs(z_discrete - xyz_in_out[:, 0, 2]) * torch.tan(theta)
                    + xyz_in_out[:, 0, dim]
                )

            xyz_discrete[2] = z_discrete

        return xyz_discrete_in, xyz_discrete_out

    @staticmethod
    def _find_sub_volume(
        voi: Volume, xyz_in_voi: torch.Tensor, xyz_out_voi: torch.Tensor
    ) -> List[torch.Tensor]:
        r"""
        Find the xy voxel indices of the sub-volume which contains both incoming and outgoing tracks.

        Args:
             - voi (Volume): Instance of the volume class.
             - xyz_in_voi (Tensor): The location of muons when entering/exiting the volume for the incoming track.
             - xyz_out_voi (Tensor): The location of muons when entering/exiting the volume for the outgoing track.

        Returns:
             - sub_vol_indices_min_max (List[Tensor]): List containing the voxel indices.
        """

        print("\nSub-volumes")
        sub_vol_indices_min_max = []
        n_mu = xyz_in_voi.size(0)

        for event in progress_bar(range(n_mu)):
            x_min = torch.min(
                torch.min(xyz_in_voi[event, :, 0]), torch.min(xyz_out_voi[event, :, 0])
            )
            x_max = torch.max(
                torch.max(xyz_in_voi[event, :, 0]), torch.max(xyz_out_voi[event, :, 0])
            )

            y_min = torch.min(
                torch.min(xyz_in_voi[event, :, 1]), torch.min(xyz_out_voi[event, :, 1])
            )
            y_max = torch.max(
                torch.max(xyz_in_voi[event, :, 1]), torch.max(xyz_out_voi[event, :, 1])
            )

            sub_vol_indices = (
                (voi.voxel_edges[:, :, 0, 1, 0] > x_min)
                & (voi.voxel_edges[:, :, 0, 1, 1] > y_min)
                & (voi.voxel_edges[:, :, 0, 0, 0] < x_max)
                & (voi.voxel_edges[:, :, 0, 0, 1] < y_max)
            ).nonzero()

            if len(sub_vol_indices) != 0:
                sub_vol_indices_min_max.append(
                    [sub_vol_indices[0], sub_vol_indices[-1]]
                )

            else:
                sub_vol_indices_min_max.append([])

        return sub_vol_indices_min_max

    @staticmethod
    def _find_triggered_voxels(
        voi: Volume,
        sub_vol_indices_min_max: List[Tensor],
        xyz_discrete_in: Tensor,
        xyz_discrete_out: Tensor,
    ) -> List[Tensor]:
        r"""
        For each muon incoming and outgoing tracks, find the associated triggered voxels.
        Only voxels triggered by both INCOMING and OUTGOING tracks are kept.

        Args:
             - voi (Volume): Instance of the volume class.
             - sub_vol_indices_min_max (List[Tensor]):  the xy voxel indices of the sub-volume
             which contains both incoming and outgoing tracks.
             - xyz_discrete_in (Tensor): The discretized incoming tracks with size (3, n_points, n_mu)
             - xyz_discrete_out (Tensor): The discretized outgoing tracks with size (3, n_points, n_mu)

        Returns:
             - triggererd_voxels (List[Tensor]): List with len() = n_mu, containing the indices
             of the triggered voxels as a Tensor (with size [n_triggered_vox,3])
        """

        triggered_voxels = []
        n_mu = xyz_discrete_in.size(2)

        print("\nVoxel triggering")
        for event in progress_bar(range(n_mu)):
            if len(sub_vol_indices_min_max[event]) != 0:
                ix_min, iy_min = (
                    sub_vol_indices_min_max[event][0][0],
                    sub_vol_indices_min_max[event][0][1],
                )
                ix_max, iy_max = (
                    sub_vol_indices_min_max[event][1][0],
                    sub_vol_indices_min_max[event][1][1],
                )

                sub_voi_edges = voi.voxel_edges[
                    ix_min : ix_max + 1, iy_min : iy_max + 1
                ]
                sub_voi_edges = sub_voi_edges[:, :, :, :, None, :].expand(
                    -1, -1, -1, -1, xyz_discrete_out.size()[1], -1
                )

                sub_mask_in = (
                    (sub_voi_edges[:, :, :, 0, :, 0] < xyz_discrete_in[0, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 0] > xyz_discrete_in[0, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 1] < xyz_discrete_in[1, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 1] > xyz_discrete_in[1, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 2] < xyz_discrete_in[2, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 2] > xyz_discrete_in[2, :, event])
                )

                sub_mask_out = (
                    (sub_voi_edges[:, :, :, 0, :, 0] < xyz_discrete_out[0, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 0] > xyz_discrete_out[0, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 1] < xyz_discrete_out[1, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 1] > xyz_discrete_out[1, :, event])
                    & (sub_voi_edges[:, :, :, 0, :, 2] < xyz_discrete_out[2, :, event])
                    & (sub_voi_edges[:, :, :, 1, :, 2] > xyz_discrete_out[2, :, event])
                )

                vox_list = (sub_mask_in & sub_mask_out).nonzero()[:, :-1].unique(dim=0)
                vox_list[:, 0] += ix_min
                vox_list[:, 1] += iy_min
                triggered_voxels.append(vox_list.numpy())
            else:
                triggered_voxels.append([])
        return triggered_voxels

    @staticmethod
    def get_asr_name(
        asr_params: Dict[str, value_type],
    ) -> str:
        r"""
        Returns the name of the bca given its parameters.
        """

        def get_partial_name_args(func: partial) -> str:
            r"""
            Returns the name, arguments and their value of a partial method as a string.
            """
            func_name = func.func.__name__
            args, values = list(func.keywords.keys()), list(func.keywords.values())
            for i, arg in enumerate(args):
                func_name += "_{}={}".format(arg, values[i])
            return func_name

        method = "method_{}_".format(
            get_partial_name_args(asr_params["score_method"])  # type: ignore
        )
        dtheta = "{:.2f}_{:.2f}_rad_".format(
            asr_params["dtheta_range"][0], asr_params["dtheta_range"][1]  # type: ignore
        )
        dp = "{:.0f}_{:.0f}_MeV_".format(
            asr_params["p_range"][0], asr_params["p_range"][1]  # type: ignore
        )
        use_p = "use_p_{}".format(asr_params["use_p"])

        asr_name = method + dtheta + dp + use_p

        return asr_name

    @staticmethod
    def get_triggered_voxels(
        points_in: Tensor,
        points_out: Tensor,
        voi: Volume,
        theta_xy_in: Tensor,
        theta_xy_out: Tensor,
    ) -> List[np.ndarray]:
        """
        Gets the `xyz` indices of the voxels along each muon path, as a list of np.ndarray.
        e.g triggered_voxels[i] is an np.ndarray with shape (n, 3) where n is the number
        of voxels along the muon path for the muon event i.

        Args:
             - points_in (Tensor): Points on incoming muon tracks.
             - points_out (Tensor): Points on outgoing muon tracks.
             - voi (Volume): Instance of the volume class.
             - theta_xy_in (Tensor): The incoming projected zenith angle in XZ and YZ plane.
             - theta_xy_out (Tensor): The outgoing projected zenith angle in XZ and YZ plane.

        Returns:
             - triggered_voxels (List[np.ndarray]): the list of triggered voxels.
        """

        xyz_in_out_voi = ASR._compute_xyz_in_out(
            points_in=points_in,
            points_out=points_out,
            voi=voi,
            theta_xy_in=theta_xy_in,
            theta_xy_out=theta_xy_out,
        )

        xyz_discrete_in, xyz_discrete_out = ASR._compute_discrete_tracks(
            voi=voi,
            theta_xy_in=theta_xy_in,
            theta_xy_out=theta_xy_out,
            xyz_in_out_voi=xyz_in_out_voi,
        )

        sub_vol_indices_min_max = ASR._find_sub_volume(
            voi=voi, xyz_in_voi=xyz_in_out_voi[0], xyz_out_voi=xyz_in_out_voi[1]
        )

        return ASR._find_triggered_voxels(
            voi=voi,
            sub_vol_indices_min_max=sub_vol_indices_min_max,
            xyz_discrete_in=xyz_discrete_in,
            xyz_discrete_out=xyz_discrete_out,
        )

    def get_density_pred(self) -> Tensor:
        r"""
        Computes the density predictions per voxel.

        Returns:
            vox_density_pred (Tensor): voxelwise density predictions
        """

        score_list: List[List[List[List]]] = [
            [
                [[] for _ in range(self.voi.n_vox_xyz[2])]
                for _ in range(self.voi.n_vox_xyz[1])
            ]
            for _ in range(self.voi.n_vox_xyz[0])
        ]

        if self._ars_params["use_p"]:
            score = np.log(self.tracks.dtheta.numpy() * self.tracks.E.numpy())
        else:
            score = self.tracks.dtheta.numpy()

        mask_E = (self.tracks.E > self.asr_params["p_range"][0]) & (  # type: ignore
            self.tracks.E < self.asr_params["p_range"][1]  # type: ignore
        )
        mask_theta = (self.tracks.dtheta > self.asr_params["dtheta_range"][0]) & (  # type: ignore
            self.tracks.dtheta < self.asr_params["dtheta_range"][1]  # type: ignore
        )

        if self.asr_params["use_p"]:  # type: ignore
            mask = mask_E & mask_theta
        else:
            mask = mask_E

        print("\nAssigning voxels score")
        for i, vox_list in enumerate(progress_bar(self.triggered_voxels)):
            for vox in vox_list:
                if mask[i]:
                    score_list[vox[0]][vox[1]][vox[2]].append(score[i])

        vox_density_preds = torch.zeros(tuple(self.voi.n_vox_xyz))

        print("Compute final score")
        for i in progress_bar(range(self.voi.n_vox_xyz[0])):
            for j in range(self.voi.n_vox_xyz[1]):
                for k in range(self.voi.n_vox_xyz[2]):
                    if score_list[i][j][k] != []:
                        vox_density_preds[i, j, k] = self.asr_params["score_method"](  # type: ignore
                            score_list[i][j][k]
                        )

        if vox_density_preds.isnan().any():
            raise ValueError("Prediction contains NaN values")

        self._recompute_preds = False

        if self.asr_params["use_p"]:  # type: ignore
            return torch.exp(vox_density_preds)
        else:
            return vox_density_preds

    def get_n_mu_per_vox(
        self,
    ) -> Tensor:
        n_mu_per_vox = torch.zeros(self.voi.n_vox_xyz)

        all_voxels = np.vstack([ev for ev in self.triggered_voxels if len(ev) > 0])
        unique_voxels, counts = np.unique(all_voxels, axis=0, return_counts=True)
        for (x, y, z), count in zip(unique_voxels, counts):
            n_mu_per_vox[x, y, z] += count

        return n_mu_per_vox

    @property
    def theta_xy_in(self) -> Tuple[Tensor, Tensor]:
        r"""
        Returns:
            Tuple[(mu, )] tensors of muons incoming projected zenith angles in XZ and YZ plane.
        """
        return (self.tracks.theta_xy_in[0], self.tracks.theta_xy_in[1])

    @property
    def theta_xy_out(self) -> Tuple[Tensor, Tensor]:
        r"""
        Returns:
            Tuple[(mu, )] tensors of muons outgoing projected zenith angles in XZ and YZ plane.
        """
        return (self.tracks.theta_xy_out[0], self.tracks.theta_xy_out[0])

    @property
    def asr_params(self) -> Dict[str, value_type]:
        r"""
        The parameters of the ASR algorithm.
        """
        return self._ars_params

    @asr_params.setter
    def asr_params(self, value: Dict[str, value_type]) -> None:
        r"""
        Sets the parameters of the ASR algorithm.
        Args:
            - Dict containing the parameters name and value. Only parameters with
            valid name and non `None` values wil be updated.
        """
        for key in value.keys():
            if key in self._ars_params.keys():
                if value[key] is not None:
                    self._ars_params[key] = value[key]

        self._recompute_preds = True

    @property
    def xyz_voxel_pred(self) -> Tensor:
        r"""
        The scattering density predictions.
        """
        if (self._xyz_voxel_pred is None) | (self._recompute_preds):
            self._xyz_voxel_pred = self.get_density_pred()
        return self._xyz_voxel_pred

    @property
    def density_pred_norm(self) -> Tensor:
        r"""
        The normalized scattering density predictions.
        """
        return normalize(self.density_pred)

    @property
    def triggered_voxels(self) -> List[np.ndarray]:
        r"""
        The list of triggered voxels.
        """
        if self._triggered_voxels is None:
            self._triggered_voxels = self.get_triggered_voxels(
                self.tracks.points_in,
                self.tracks.points_out,
                self.voi,
                self.theta_xy_in,
                self.theta_xy_out,
            )
        return self._triggered_voxels

    @triggered_voxels.setter
    def triggered_voxels(self, value: List[Tensor]) -> None:
        self._triggered_voxels = value

    @property
    def n_mu_per_vox(self) -> Tensor:
        if self._n_mu_per_vox is None:
            self._n_mu_per_vox = self.get_n_mu_per_vox()
        return self._n_mu_per_vox

    @n_mu_per_vox.setter
    def n_mu_per_vox(self, value: Tensor) -> Tensor:
        self._n_mu_per_vox = value
