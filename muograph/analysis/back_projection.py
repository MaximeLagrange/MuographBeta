from utils.save import AbsSave
from volume.volume import Volume
from reconstruction.back_projection import BackProjection
from hits.hits import Hits
from tracking.tracking import Tracking
from plotting.voxel import VoxelPlotting

from typing import Optional, Tuple
from torch import Tensor
import torch


class BackProjectionAnalysis(AbsSave, VoxelPlotting):
    r"""
    Class for voxel-wise muon absorption ratio computation using Back Projection algorithm.

    The voxel-wise muon absorption ratio is computed as the ratio of the voxel-wise muon counts
    between the `freesky` measurement and `absorption` measurement.
    """

    _back_proj_freesky: Optional[BackProjection] = None
    _back_proj_absorption: Optional[BackProjection] = None

    _voxel_xyz_transmission_ratio: Optional[Tensor] = None
    _voxel_xyz_transmission_ratio_uncs: Optional[Tensor] = None

    _files_to_load = [
        "tracks_below_absorption",
        "tracks_below_freesky",
        "triggered_voxels_freesky",
        "triggered_voxels_absorption",
    ]

    def __init__(
        self,
        voi: Volume,
        freesky_hits: Optional[Hits] = None,
        absorption_hits: Optional[Hits] = None,
        input_dir: Optional[str] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        output_dir: Optional[str] = None,
    ):
        r"""
        Instanciate the `BackProjectionAnalysis` class.

        The instanciation can be done in 2 ways:
         - Provide `freesky_hits` and `absorption_hits`. Associated instances of `Tracking` and
         `BackProjection` will be computed from those files. They are then saved in `output_dir` under
         the follwing file names: `tracks_below_absorption.hdf5`, `tracks_below_freesky.hdf5`, `triggered_voxels_freesky.hdf5`
         and `triggered_voxels_absorption.hdf5`.
         - Provide `input_dir`, the Path to the directory where to load `tracks_below_absorption.hdf5`, `tracks_below_freesky.hdf5`, `triggered_voxels_freesky.hdf5`
         and `triggered_voxels_absorption.hdf5`.

        Args:
            - voi (`Volume`) Instance of the Volume class.
            - freesky_hits (`Hits`) Instance of the Hits class,
            containing the muon hits for the freesky meaurement campaign.
            - absorption_hits (`Hits`) Instance of the Hits class,
            containing the muon hits for the absorption meaurement campaign.
            - input_dir (`str`) Path to the directory where to load the `Tracking`
            and `triggered_voxels` hdf5 files.
            - energy_range (`Tuple[float, float]`) The muon energy range to consider, in MeV.
            Muon events outside of energy range are discarded.
            - output_dir (`str`) Path to the directoty where to save the `Tracking`
            and `triggered_voxels` hdf5 files.
        """

        AbsSave.__init__(self, output_dir=output_dir)
        VoxelPlotting.__init__(self, voi=voi)

        # Volume to scan
        self.voi = voi

        # Input hits
        self.freesky_hits = freesky_hits
        self.absorption_hits = absorption_hits

        # Muons energy range
        self.energy_range = energy_range

        if input_dir is None:
            # If freesky_hits and absorption_hits are provided, copmute the BackProjection instances
            if freesky_hits is not None and absorption_hits is not None:
                (
                    self._back_proj_freesky,
                    self._back_proj_absorption,
                ) = self.get_back_projs_from_hits(freesky_hits, absorption_hits)
            else:
                raise ValueError(
                    "Provide either `input_dir` or `freesky_hits` and `absorption_hits`."
                )
        else:
            # Check if `input_dir` contains the required hdf5 files
            all_exist = AbsSave.files_in_dir(dir=input_dir, files=self._files_to_load)

            # If yes, instanciate `BackProjection` classes by loading the hdf5 files.
            if all_exist:
                (
                    self._back_proj_freesky,
                    self._back_proj_absorption,
                ) = self.get_back_projs_from_files(input_dir=input_dir)

            else:
                raise ValueError(
                    "`input_dir` must contain the following hdf5 files:",
                    [file for file in self._files_to_load],
                )

    def __repr__(self) -> str:
        return "Back Projection Analysis with {} (freesky) and {} (absorption) muons between {:.1f} and {:.1f} GeV".format(
            self._back_proj_freesky.tracks.n_mu,  # type: ignore
            self._back_proj_absorption.tracks.n_mu,  # type: ignore
            self.energy_range[0] / 1000,  # type: ignore
            self.energy_range[1] / 1000,  # type: ignore
        )

    def get_back_projs_from_hits(
        self, freesky_hits: Hits, absorption_hits: Hits
    ) -> Tuple[BackProjection, BackProjection]:
        r"""
        Instanciate `BackProjection` classes from the freesky and absorption Hits instances.

        First the `Tracking` instances are computed from the respective hits, then the
        `BackProjection` instances are computed from the respective tracks.

        Args:
            - freesky_hits (`Hits`) Instance of the `Hits` class,
            containing the muon hits for the freesky meaurement campaign.
            - absorption_hits (`Hits`) Instance of the `Hits` class,
            containing the muon hits for the absorption meaurement campaign.

        Returns:
            - back_proj_freesky (`BackProjection`) Instance of the `BackProjection` class
            for the freesky measuerment campaign.
            - back_proj_absorption (`BackProjection`) Instance of the `BackProjection` class
            for the absorption measuerment campaign.
        """

        tracks_freesky = Tracking(
            hits=freesky_hits, label="below", output_dir=self.output_dir, type="freesky"
        )

        tracks_absorption = Tracking(
            hits=absorption_hits,
            label="below",
            output_dir=self.output_dir,
            type="absorption",
        )

        back_proj_freesky = BackProjection(
            voi=self.voi,
            tracking=tracks_freesky,
            output_dir=self.output_dir,
            label="freesky",
            energy_range=self.energy_range,
        )

        back_proj_absorption = BackProjection(
            voi=self.voi,
            tracking=tracks_absorption,
            output_dir=self.output_dir,
            label="absorption",
            energy_range=self.energy_range,
        )

        return back_proj_freesky, back_proj_absorption

    def get_back_projs_from_files(
        self, input_dir: str
    ) -> Tuple[BackProjection, BackProjection]:
        r"""
        Instanciate `BackProjection` classes from the "tracks_below_absorption.hdf5",
        "tracks_below_freesky.hdf5", "triggered_voxels_freesky.hdf5", "triggered_voxels_absorption.hdf5"
        in the `input_dir` directory.

        Args:
            - input_dir (str) Path to the hdf5 files.

        Returns:
            - back_proj_freesky (`BackProjection`) Instance of the `BackProjection` class
            for the freesky measuerment campaign.
            - back_proj_absorption (`BackProjection`) Instance of the `BackProjection` class
            for the absorption measuerment campaign.
        """

        tracks_freesky = Tracking(
            tracks_hdf5=input_dir + "tracks_below_freesky.hdf5",
            label="below",
            type="freesky",
        )

        tracks_absorption = Tracking(
            tracks_hdf5=input_dir + "tracks_below_absorption.hdf5",
            label="below",
            type="absorption",
        )

        back_proj_freesky = BackProjection(
            voi=self.voi,
            tracking=tracks_freesky,
            label="freesky",
            energy_range=self.energy_range,
            triggered_vox_file=input_dir + "triggered_voxels_freesky.hdf5",
        )

        back_proj_measurement = BackProjection(
            voi=self.voi,
            tracking=tracks_absorption,
            label="absorption",
            energy_range=self.energy_range,
            triggered_vox_file=input_dir + "triggered_voxels_absorption.hdf5",
        )

        return back_proj_freesky, back_proj_measurement

    @staticmethod
    def compute_voxel_xyz_transmission_ratio_uncs(
        voxel_xyz_muon_count_absorption: Tensor,
        voxel_xyz_muon_count_absorption_uncs: Tensor,
        voxel_xyz_muon_count_freesky: Tensor,
        voxel_xyz_muon_count_freesky_uncs: Tensor,
        voxel_xyz_transmission_ratio: Tensor,
    ) -> Tensor:
        relative_absorption_uncs = (
            voxel_xyz_muon_count_absorption_uncs / voxel_xyz_muon_count_absorption
        )
        relative_freesky_uncs = (
            voxel_xyz_muon_count_freesky_uncs / voxel_xyz_muon_count_freesky
        )

        quad_sum_uncs = torch.sqrt(
            relative_absorption_uncs**2 + relative_freesky_uncs**2
        )
        return voxel_xyz_transmission_ratio * quad_sum_uncs

    @property
    def voxel_xyz_transmission_ratio(self) -> Tensor:
        r"""
        The voxel-wise transmission ratio predictions.
        """
        if self._voxel_xyz_transmission_ratio is None:
            self._voxel_xyz_transmission_ratio = (
                self._back_proj_absorption.voxel_xyz_muon_count  # type: ignore
                / self._back_proj_freesky.voxel_xyz_muon_count  # type: ignore
            )
        return self._voxel_xyz_transmission_ratio

    @property
    def voxel_xyz_transmission_ratio_uncs(self) -> Tensor:
        r"""
        The uncertainties associated to the voxel-wise transmission ratio predictions.
        """
        if self._voxel_xyz_transmission_ratio_uncs is None:
            self._voxel_xyz_transmission_ratio_uncs = self.compute_voxel_xyz_transmission_ratio_uncs(
                self._back_proj_absorption.voxel_xyz_muon_count,  # type: ignore
                self._back_proj_absorption.voxel_xyz_muon_count_uncs,  # type: ignore
                self._back_proj_freesky.voxel_xyz_muon_count,  # type: ignore
                self._back_proj_freesky.voxel_xyz_muon_count_uncs,  # type: ignore
                self.voxel_xyz_transmission_ratio,
            )
        return self._voxel_xyz_transmission_ratio_uncs
