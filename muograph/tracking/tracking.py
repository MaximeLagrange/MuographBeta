import torch
from torch import Tensor
from typing import Tuple, Optional
import numpy as np
import math
import matplotlib.pyplot as plt

from utils.save import AbsSave
from hits.hits import Hits
from plotting.params import n_bins, alpha


class Tracking(AbsSave):
    r"""
    A class for tracking muons based on hits data.
    """
    _tracks: Optional[Tensor] = None  # (mu, 3)
    _points: Optional[Tensor] = None  # (mu, 3)
    _theta: Optional[Tensor] = None  # (mu)
    _theta_xy: Optional[Tensor] = None  # (mu)
    _angular_error: Optional[None] = None  # (mu)
    _angular_res: Optional[float] = None

    _vars_to_save = [
        "tracks",
        "points",
        "angular_res",
    ]

    def __init__(
        self,
        hits: Hits,
        label: str,
        output_dir: Optional[str] = None,
    ) -> None:
        r"""
        Initializes the Tracking object.

        Args:
            hits (Hits): An instance of the Hits class.
            label (str): The position of the hits with respect to the passive volume.
            either 'above' or 'below'
            output_dir (str): The name of the directory where to save the Tracking attributes
            in _vars_to_save.
        """

        super().__init__(output_dir)

        self.hits = hits
        if label in ["above", "below"]:
            self.label = label
        else:
            raise ValueError("Provide either 'above' or 'below' as label")

        self.save_attr(
            attributes=self._vars_to_save,
            directory=self.output_dir,
            filename="tracks_" + self.label,
        )

    @staticmethod
    def get_tracks_points_from_hits(hits: Tensor) -> Tuple[Tensor, Tensor]:
        r"""
        Extract tracks and points from hits data.

        Args:
            - hits (Tensor): The hits data with shape (3, n_plane, mu).

        Returns:
            - tracks, points (Tuple[Tensor, Tensor]): The points and tracks tensors
            with respective size (mu, 3).
        """

        from skspatial.objects import Line, Points
        from joblib import Parallel, delayed

        hits = torch.transpose(hits, 0, 1).numpy()

        def fit_line(hits_ev):  # type: ignore
            fit = Line.best_fit(Points(hits_ev))
            return fit.direction, fit.point

        # Extract number of hits
        num_hits = hits.shape[-1]

        # Prepare data for parallel processing
        hits_list = [hits[:, :, i] for i in range(num_hits)]

        # Use joblib to parallelize the fitting process
        results = Parallel(n_jobs=-1)(
            delayed(fit_line)(hits_ev) for hits_ev in hits_list
        )

        # Separate the results into tracks and points
        tracks, points = zip(*results)

        # Convert the results to NumPy arrays
        tracks_np = np.array(tracks)
        points_np = np.array(points)

        # Convert the results back to torch tensors if needed
        tracks_tensor = Tensor(tracks_np)
        points_tensor = Tensor(points_np)

        return tracks_tensor, points_tensor

    @staticmethod
    def get_theta_from_tracks(tracks: Tensor) -> Tensor:
        r"""
        Compute muons' zenith angle in radiants, from the direction vector of the track.

        Args:
            tracks (Tensor): Direction vector of the tracks.

        Returns:
            theta (Tensor): Muons' zenith angle.
        """

        x, y, z = tracks[:, 0], tracks[:, 1], tracks[:, 2]
        r = torch.sqrt(x**2 + y**2 + z**2)
        theta = torch.acos(z / r)
        return torch.where(math.pi - theta < theta, math.pi - theta, theta)

    @staticmethod
    def get_theta_xy_from_tracks(tracks: Tensor) -> Tensor:
        r"""
        Compute muons' projected zenith angle in XZ and YZ plane in radiants,
        from the direction vector of the track.

        Args:
            tracks (Tensor): Direction vector of the tracks.

        Returns:
            theta_xy (Tensor): Muons' zenith angle in XZ and YZ plane.
        """

        theta_xy = torch.empty([2, tracks.size()[0]])

        theta_xy[0] = torch.atan(tracks[:, 0] / tracks[:, 2])
        theta_xy[1] = torch.atan(tracks[:, 1] / tracks[:, 2])

        theta_xy = torch.where(theta_xy > math.pi / 2, math.pi - theta_xy, theta_xy)

        return theta_xy

    def get_angular_error(self, reco_theta: Tensor) -> Tensor:
        r"""
        Compute the angular error between the generated and reconstructed tracks.
        Args:
            reco_theta (Tensor): Zenith angle of the reconstructed tracks.

        Returns:
            (Tensor): angular error with size (mu).
        """

        gen_tracks, _ = self.get_tracks_points_from_hits(hits=self.hits.gen_hits)
        gen_theta = self.get_theta_from_tracks(tracks=gen_tracks)
        return gen_theta - reco_theta

    def plot_muon_features(self, figname: Optional[str] = None) -> None:
        r"""
        Plot the zenith angle and energy of the reconstructed tracks.
        Args:
            figname (Tensor): If provided, save the figure at self.output / figname.
        """

        fig, axs = plt.subplots(ncols=2, figsize=(10, 4))

        # Fig title
        fig.suptitle(
            f"Batch of {self.tracks.size()[0]} muons", fontsize=15, fontweight="bold"
        )

        # Zenith angle
        axs[0].hist(self.theta.numpy() * 180 / math.pi, bins=n_bins, alpha=alpha)
        axs[0].axvline(
            x=self.theta.mean().numpy() * 180 / math.pi,
            label=f"mean = {self.theta.mean().numpy() * 180 / math.pi:.1f}",
            color="red",
        )
        axs[0].set_xlabel(r" Zenith angle $\theta$ [deg]")

        # Energy
        axs[1].hist(self.E.numpy(), bins=n_bins, alpha=alpha, log=True)
        axs[1].axvline(
            x=self.E.mean().numpy(),
            label=f"mean = {self.E.mean().numpy():.3E}",
            color="red",
        )
        axs[1].set_xlabel(r" Energy [MeV]")

        for ax in axs:
            ax.grid("on")
            ax.set_ylabel("Frequency [a.u]")
            ax.legend()
        plt.tight_layout()

        if figname is not None:
            plt.savefig(self.output_dir / figname, bbox_inches="tight")

        plt.show()

    @property
    def tracks(self) -> Tensor:
        r"""
        The muons' direction
        """
        if self._tracks is None:
            self._tracks, self._points = self.get_tracks_points_from_hits(
                hits=self.hits.reco_hits
            )
        return self._tracks

    @property
    def points(self) -> Tensor:
        r"""
        Point on muons' trajectory.
        """
        if self._points is None:
            self._tracks, self._points = self.get_tracks_points_from_hits(
                hits=self.hits.reco_hits
            )
        return self._points

    @property
    def theta_xy(self) -> Tensor:
        r"""
        The muons' projected zenith angle in XZ and YZ plane.
        """
        if self._theta_xy is None:
            self._theta_xy = self.get_theta_xy_from_tracks(self.tracks)
        return self._theta_xy

    @property
    def theta(self) -> Tensor:
        r"""
        The muons' zenith angle.
        """

        if self._theta is None:
            self._theta = self.get_theta_from_tracks(self.tracks)
        return self._theta

    @property
    def E(self) -> Tensor:
        r"""The muons' energy."""
        return self.hits.E

    @property
    def n_mu(self) -> int:
        r"""
        The number of muons.
        """
        return len(self.theta)

    @property
    def angular_error(self) -> Tensor:
        r"""
        The angular error between the generated and reconstructed tracks.
        """
        if self._angular_error is None:
            if self.hits.spatial_res is None:
                self._angular_error = torch.zeros_like(self.theta)
            else:
                self._angular_error = self.get_angular_error(self.theta)
        return self._angular_error

    @property
    def angular_res(self) -> float:
        r"""
        The angular resolution, computed as the standard deviation of the
        angular error distribution.
        """
        if self._angular_res is None:
            self._angular_res = self.angular_error.std().item()
        return self._angular_res


class TrackingMST(AbsSave):
    r"""
    A class for tracking muons in the context of a Muon Scattering Tomography analysis.
    """

    _theta_in: Optional[Tensor] = None  # (mu)
    _theta_out: Optional[Tensor] = None  # (mu)
    _theta_xy_in: Optional[Tensor] = None  # (2, mu)
    _theta_xy_out: Optional[Tensor] = None  # (2, mu)
    _dtheta: Optional[Tensor] = None  # (mu)

    _vars_to_load = [
        "tracks",
        "points",
        "angular_res",
    ]

    def __init__(
        self,
        tracking_files: Optional[Tuple[str, str]] = None,
        trackings: Optional[Tuple[Tracking, Tracking]] = None,
        output_dir: Optional[str] = None,
    ) -> None:
        r"""
        Initializes the TrackingMST object with either 2 instances of the Tracking class
        (with tags 'above' and 'below') or with hdf5 files where Tracking attributes where saved.

        Args:
            - tracking_files (Optional[Tuple[str, str]]): path to hdf5 files
            where Tracking class attributes are saved
            - trackings (Optional[Tuple[Tracking, Tracking]]): instances of the Tracking class
            for the incoming muon tracks (Tracking.label = 'above') and outgoing tracks
            (Tracking.label = 'below')
            -  output_dir (Optional[str]): Path to a directory where to sav TrackingMST attributes
            in a hdf5 file. (Not Implemented Yet).
        """
        super().__init__(output_dir)

        if tracking_files is None and trackings is None:
            raise ValueError(
                "Provide either a list of tracking files or a list of Tracking instances."
            )

        # Load data from tracking hdf5 files
        elif trackings is None and tracking_files is not None:
            for tracking_file, tag in zip(tracking_files, ["_in", "_out"]):
                self.load_attr(self._vars_to_load, tracking_file, tag=tag)

        # Load data from Tracking instances
        elif trackings is not None and tracking_files is None:
            for tracking, tag in zip(trackings, ["_in", "_out"]):
                self.load_attr_from_tracking(tracking, tag)

    def load_attr_from_tracking(self, tracking: Tracking, tag: str) -> None:
        r"""
        Load class attributes in TrackingMST._vars_to_load from the input Tracking class.
        Attributes name are modified according to the tag as `attribute_name` + `tag`,
        so that incoming and outgoing muon features can be treated independently.

        Args:
            - tracking (Tracking): Instance of the Tracking class.
            - tag (str): tag to add to the attribuites name (either `_in` or `_out`)
        """

        for attr in self._vars_to_load:
            data = getattr(tracking, attr)
            attr += tag
            setattr(self, attr, data)

    @staticmethod
    def compute_dtheta_from_tracks(
        tracks_in: Tensor, tracks_out: Tensor, tol: float = 1.0e-12
    ) -> Tensor:
        r"""
        Computes the scattering angle between the incoming and outgoing muon tracks.

        Args:
            - tracks_in (Tensor): The incoming muon tracks with size (mu, 3)
            - tracks_out (Tensor): The outgoing muon tracks with size (mu, 3)
            - tol (float): A tolerance parameter to avoid errors when computing acos(dot_prod).
            the dot_prod is clamped between (-1 + tol, 1 - tol). Default value is 1.e12.

        Returns:
            - dtheta (Tensor): The scattering angle between the incoming and outgoing muon
            tracks in [rad], with size (mu).
        """

        def norm(x: Tensor) -> Tensor:
            return torch.sqrt((x**2).sum(dim=-1))

        dot_prod = torch.abs(tracks_in * tracks_out).sum(dim=-1) / (
            norm(tracks_in) * norm(tracks_out)
        )
        dot_prod = torch.clamp(dot_prod, -1.0 + tol, 1.0 - tol)
        dtheta = torch.acos(dot_prod)
        return dtheta

    def _filter_muons(self, mask: Tensor) -> None:
        r"""
        Remove muons specified as False in `mask`.

        Arguments:
            keep_mask: (N,) Boolean tensor. Muons with False elements will be removed.
        """

        # Set attributes without setter method to None
        self._reset_vars()

        n_muons = self.tracks_in.size()[0]
        # Loop over class attributes and apply the mask is Tensor
        for var in vars(self).keys():
            data = getattr(self, var)
            if isinstance(data, Tensor):
                if data.size()[0] == n_muons:
                    setattr(self, var, data[mask])

    def _reset_vars(self) -> None:
        r"""
        Reset attributes to None.
        """

        self._theta_in = None  # (mu)
        self._theta_out = None  # (mu)
        self._theta_xy_in = None  # (2, mu)
        self._theta_xy_out = None  # (2, mu)
        self._dtheta = None  # (mu)

    # Scattering angle
    @property
    def dtheta(self) -> Tensor:
        r"""Muon scattering angle measured between the incoming and outgoing tracks"""
        if self._dtheta is None:
            self._dtheta = self.compute_dtheta_from_tracks(
                self.tracks_in, self.tracks_out
            )
        return self._dtheta

    # Tracks
    @property
    def tracks_in(self) -> Tensor:
        r"""Incoming muon tracks, with size (mu, 3)"""
        return self._tracks_in

    @tracks_in.setter
    def tracks_in(self, value: Tensor) -> None:
        self._tracks_in = value

    @property
    def tracks_out(self) -> Tensor:
        r"""Outgoing muon tracks, with size (mu, 3)"""
        return self._tracks_out

    @tracks_out.setter
    def tracks_out(self, value: Tensor) -> None:
        self._tracks_out = value

    # Points
    @property
    def points_in(self) -> Tensor:
        r"""Points on the incoming muon tracks, with size (mu, 3)"""
        return self._points_in

    @points_in.setter
    def points_in(self, value: Tensor) -> None:
        self._points_in = value

    @property
    def points_out(self) -> Tensor:
        r"""Points on the outgoing muon tracks, with size (mu, 3)"""
        return self._points_out

    @points_out.setter
    def points_out(self, value: Tensor) -> None:
        self._points_out = value

    @property
    def theta_in(self) -> Tensor:
        r"""
        Zenith angle of the incoming tracks.
        """
        if self._theta_in is None:
            self._theta_in = Tracking.get_theta_from_tracks(self.tracks_in)
        return self._theta_in

    @property
    def theta_out(self) -> Tensor:
        r"""
        Zenith angle of the outgoing tracks.
        """
        if self._theta_out is None:
            self._theta_out = Tracking.get_theta_from_tracks(self.tracks_out)
        return self._theta_out

    @property
    def theta_xy_in(self) -> Tensor:
        r"""
        Projected zenith angles of the incoming tracks.
        """
        if self._theta_xy_in is None:
            self._theta_xy_in = Tracking.get_theta_xy_from_tracks(self.tracks_in)
        return self._theta_xy_in

    @property
    def theta_xy_out(self) -> Tensor:
        r"""
        Projected zenith angles of the outgoing tracks.
        """
        if self._theta_xy_out is None:
            self._theta_xy_out = Tracking.get_theta_xy_from_tracks(self.tracks_out)
        return self._theta_xy_out

    # Resolutions
    @property
    def angular_res_in(self) -> float:
        r"""
        Angular resolution of the incoming tracks.
        """
        return self._angular_res_in

    @angular_res_in.setter
    def angular_res_in(self, value: float) -> None:
        self._angular_res_in = value

    @property
    def angular_res_out(self) -> float:
        r"""
        Angular resolution of the outgoing tracks.
        """
        return self._angular_res_out

    @angular_res_out.setter
    def angular_res_out(self, value: float) -> None:
        self._angular_res_out = value
