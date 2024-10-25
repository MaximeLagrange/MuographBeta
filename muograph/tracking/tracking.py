import torch
from torch import Tensor
from typing import Tuple, Optional
import math
import matplotlib.pyplot as plt
import matplotlib

from utils.save import AbsSave
from utils.device import DEVICE
from utils.params import dtype_track
from hits.hits import Hits
from plotting.params import n_bins, alpha, font, titlesize, hist_figsize, labelsize


class Tracking(AbsSave):
    r"""
    A class for tracking muons based on hits data.

    The muon hits on detector planes are plugged into a linear fit
    to compute a track T(tx, ty, tz) and a point on that track P(px, py, pz).

    From T(tx, ty, tz), one computes the muons' zenith angles, defined as the
    angle between the vertical axis z and the muon track. A vertical muon has
    a 0 [rad] zenith angle.

    The projections of the zenth angle in the XZ and YZ planes,
    theta_x and theta_y respectively, are also computed.

    If Hits used as input were smeared, the tracking angular resolution is compted as
    the standard deviation of the distribution of the error on theta. The error on theta,
    is copmuted by comparing the values of theta computed from the generated hits and
    from the smeared hits.
    """
    _tracks: Optional[Tensor] = None  # (mu, 3)
    _points: Optional[Tensor] = None  # (mu, 3)
    _theta: Optional[Tensor] = None  # (mu)
    _theta_xy: Optional[Tensor] = None  # (mu)
    _angular_error: Optional[Tensor] = None  # (mu)
    _angular_res: Optional[float] = None
    _E: Optional[Tensor] = None  # (mu)
    _tracks_eff: Optional[Tensor] = None  # (mu)

    _vars_to_save = [
        "tracks",
        "points",
        "angular_res",
        "E",
        "label",
        "measurement_type",
        "tracks_eff",
    ]

    def __init__(
        self,
        label: str,
        hits: Optional[Hits] = None,
        output_dir: Optional[str] = None,
        tracks_hdf5: Optional[str] = None,
        measurement_type: Optional[str] = None,
        save: bool = True,
        compute_angular_res: bool = False,
    ) -> None:
        r"""
        Initializes the Tracking object.

        The instantiation can be done in two ways:
        - By providing `hits`: Computes tracks and saves them as HDF5 files in `output_dir`.
        - By providing `tracks_hdf5`: Loads tracking features from the specified HDF5 file.

        Args:
            label (str): The position of the hits relative to the passive volume ('above' or 'below').
            hits (Optional[Hits]): An instance of the Hits class, required if `tracks_hdf5` is not provided.
            output_dir (Optional[str]): Directory to save Tracking attributes if `save` is True.
            tracks_hdf5 (Optional[str]): Path to an HDF5 file with previously saved Tracking data.
            measurement_type (Optional[str]): Type of measurement campaign, either 'absorption' or 'freesky'.
            save (bool): If True, saves attributes to `output_dir`. Default is True.
            compute_angular_res (bool): If True, computes angular resolution. Default is False.
        """

        self._compute_angular_res = compute_angular_res

        self._label = self._validate_label(label)
        self._measurement_type = self._validate_measurement_type(measurement_type)
        self._compute_angular_res = compute_angular_res
        super().__init__(output_dir=output_dir, save=save)

        if (hits is not None) & (tracks_hdf5 is None):
            self.hits = hits

            if save:
                self.save_attr(
                    attributes=self._vars_to_save,
                    directory=self.output_dir,
                    filename="tracks_" + self.label + "_" + self._measurement_type,
                )
        elif tracks_hdf5 is not None:
            self.load_attr(attributes=self._vars_to_save, filename=tracks_hdf5)

    @staticmethod
    def _validate_label(label: str) -> str:
        if label not in ["above", "below"]:
            raise ValueError("Label must be either 'above' or 'below'.")
        return label

    @staticmethod
    def _validate_measurement_type(measurement_type: Optional[str]) -> str:
        valid_types = ["absorption", "freesky", None]
        if measurement_type not in valid_types:
            raise ValueError(
                "Measurement type must be 'absorption', 'freesky', or None."
            )
        return measurement_type or ""

    @staticmethod
    def get_tracks_points_from_hits(
        hits: Tensor, chunk_size: int = 200_000
    ) -> Tuple[Tensor, Tensor]:
        r"""
        The muon hits on detector planes are plugged into a linear fit
        to compute a track T(tx, ty, tz) and a point on that track P(px, py, pz).

        Args:
            - hits (Tensor): The hits data with shape (3, n_plane, mu).
            - chunk_size (int): Size of chunks for processing in case mu is very large.

        Returns:
            - tracks, points (Tuple[Tensor, Tensor]): The points and tracks tensors
            with respective size (mu, 3).
        """

        _, __, mu = hits.shape

        tracks = torch.empty((mu, 3), dtype=hits.dtype, device=hits.device)
        points = torch.empty((mu, 3), dtype=hits.dtype, device=hits.device)

        # Process in chunks to manage memory
        for start in range(0, mu, chunk_size):
            end = min(start + chunk_size, mu)

            hits_chunk = hits[:, :, start:end]  # Shape: (3, n_plane, chunk_size)

            # Calculate the mean point for each set of hits in the chunk
            points_chunk = hits_chunk.mean(dim=1)  # Shape: (3, chunk_size)

            # Center the data
            centered_hits_chunk = hits_chunk - points_chunk.unsqueeze(
                1
            )  # Shape: (3, n_plane, chunk_size)

            # Perform SVD in batch mode
            centered_hits_chunk = centered_hits_chunk.permute(
                2, 1, 0
            )  # Shape: (chunk_size, n_plane, 3)
            _, _, vh = torch.linalg.svd(
                centered_hits_chunk, full_matrices=False
            )  # vh shape: (chunk_size, 3, 3)

            # Extract the principal direction (first right singular vector) for each set
            tracks_chunk = vh[:, 0, :]  # Shape: (chunk_size, 3)

            # Store the chunk results in the main output tensors
            tracks[start:end, :] = tracks_chunk
            points[start:end, :] = points_chunk.T

        return tracks, points

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

        theta_xy = torch.empty([2, tracks.size()[0]], dtype=dtype_track, device=DEVICE)

        theta_xy[0] = torch.atan(tracks[:, 0] / tracks[:, 2])
        theta_xy[1] = torch.atan(tracks[:, 1] / tracks[:, 2])

        theta_xy = torch.where(theta_xy > math.pi / 2, math.pi - theta_xy, theta_xy)

        return theta_xy

    @staticmethod
    def get_tracks_eff_from_hits_eff(hits_eff: Tensor) -> Tensor:
        r"""
        Computes the tracks efficiency.
        """

        tracks_eff = torch.where(hits_eff.sum(dim=0) == 3, 1, 0)
        return tracks_eff

    def get_angular_error(self, reco_theta: Tensor) -> Tensor:
        r"""
        Compute the angular error between the generated and reconstructed tracks.
        Args:
            reco_theta (Tensor): Zenith angle of the reconstructed tracks.

        Returns:
            (Tensor): angular error with size (mu).
        """

        gen_tracks, _ = self.get_tracks_points_from_hits(hits=self.hits.gen_hits)  # type: ignore
        gen_theta = self.get_theta_from_tracks(tracks=gen_tracks)
        return gen_theta - reco_theta

    def plot_muon_features(
        self,
        figname: Optional[str] = None,
        dir: Optional[str] = None,
        save: bool = True,
    ) -> None:
        r"""
        Plot the zenith angle and energy of the reconstructed tracks.
        Args:
            figname (Tensor): If provided, save the figure at self.output / figname.
        """

        # Set default figname
        if figname is None:
            figname = "tracks_theta_E_" + self.label

        # Set default output directory
        if dir is None:
            dir = str(self.output_dir) + "/"

        # Set default font
        matplotlib.rc("font", **font)

        fig, axs = plt.subplots(ncols=2, figsize=(2 * hist_figsize[0], hist_figsize[1]))

        # Fig title
        fig.suptitle(
            f"Batch of {self.tracks.size()[0]} muons",
            fontsize=titlesize,
            fontweight="bold",
        )

        # Zenith angle
        axs[0].hist(
            self.theta.detach().cpu().numpy() * 180 / math.pi, bins=n_bins, alpha=alpha
        )
        axs[0].axvline(
            x=self.theta.mean().detach().cpu().numpy() * 180 / math.pi,
            label=f"mean = {self.theta.mean().detach().cpu().numpy() * 180 / math.pi:.1f}",
            color="red",
        )
        axs[0].set_xlabel(r" Zenith angle $\theta$ [deg]", fontweight="bold")

        # Energy
        axs[1].hist(self.E.detach().cpu().numpy(), bins=n_bins, alpha=alpha, log=True)
        axs[1].axvline(
            x=self.E.mean().detach().cpu().numpy(),
            label=f"mean = {self.E.mean().detach().cpu().numpy():.3E}",
            color="red",
        )
        axs[1].set_xlabel(r" Energy [MeV]", fontweight="bold")

        for ax in axs:
            ax.grid(visible=True, color="grey", linestyle="--", linewidth=0.5)
            ax.tick_params(axis="both", labelsize=labelsize)
            ax.set_ylabel("Frequency [a.u]", fontweight="bold")
            ax.legend()
        plt.tight_layout()

        if save:
            plt.savefig(dir + figname, bbox_inches="tight")

        plt.show()

    def plot_angular_error(
        self,
        figname: Optional[str] = None,
        dir: Optional[str] = None,
        save: bool = True,
    ) -> None:
        """Plot the angular error of the tracks.

        Args:
            filename (Optional[str], optional): Path to a file where to save the figure. Defaults to None.
        """

        # Set default figname
        if figname is None:
            figname = "tracks_angular_error_" + self.label

        # Set default output directory
        if dir is None:
            dir = str(self.output_dir) + "/"

        # Set default font
        matplotlib.rc("font", **font)

        fig, ax = plt.subplots(figsize=(hist_figsize))

        # Fig title
        fig.suptitle(
            f"Batch of {self.tracks.size()[0]} muons\nAngular resolution = {self.angular_error.std().detach().cpu().numpy() * 180 / math.pi:.2f} deg",
            fontsize=titlesize,
            fontweight="bold",
        )

        # Projected zenith angle error
        ax.hist(
            self.angular_error.detach().cpu().numpy() * 180 / math.pi,
            bins=n_bins,
            alpha=alpha,
        )

        # Mean angular error
        ax.axvline(
            x=self.angular_error.mean().detach().cpu().numpy() * 180 / math.pi,
            label=f"mean = {self.angular_error.mean().detach().cpu().numpy() * 180 / math.pi:.1f}",
            color="red",
        )

        # Highlight 1 sigma region
        std = self.angular_error.std().detach().cpu().numpy() * 180 / math.pi
        mean = self.angular_error.mean().detach().cpu().numpy() * 180 / math.pi

        ax.axvline(x=mean - std, color="green")
        ax.axvline(x=mean + std, color="green", label=r"$\pm 1 \sigma$")

        # Grid
        ax.grid(visible=True, color="grey", linestyle="--", linewidth=0.5)

        # Axis labels
        ax.set_ylabel("Frequency [a.u]", fontweight="bold")
        ax.set_xlabel(r" Angular error $\delta\theta$ [deg]", fontweight="bold")
        ax.tick_params(axis="both", labelsize=labelsize)

        ax.legend()
        plt.tight_layout()

        if save:
            plt.savefig(dir + figname, bbox_inches="tight")
        plt.show()

    def _reset_vars(self) -> None:
        r"""
        Reset attributes to None.
        """
        self._theta = None  # (mu)
        self._theta_xy = None  # (2, mu)
        self._dtheta = None  # (mu)

    def _filter_muons(self, mask: Tensor) -> None:
        r"""
        Remove muons specified as False in `mask`.

        Args:
            - mask (Boolean tensor) Muons with False elements will be removed.
        """

        # Set attributes without setter method to None
        self._reset_vars()

        n_muons = self.tracks.size()[0]
        # Loop over class attributes and apply the mask is Tensor
        for var in vars(self).keys():
            data = getattr(self, var)
            if isinstance(data, Tensor):
                if data.size()[0] == n_muons:
                    setattr(self, var, data[mask])

    @property
    def tracks(self) -> Tensor:
        r"""
        The muons' direction
        """
        if self._tracks is None:
            self._tracks, self._points = self.get_tracks_points_from_hits(
                hits=self.hits.reco_hits  # type: ignore
            )
        return self._tracks

    @tracks.setter
    def tracks(self, value: Tensor) -> None:
        self._tracks = value

    @property
    def points(self) -> Tensor:
        r"""
        Point on muons' trajectory.
        """
        if self._points is None:
            self._tracks, self._points = self.get_tracks_points_from_hits(
                hits=self.hits.reco_hits  # type: ignore
            )
        return self._points

    @points.setter
    def points(self, value: Tensor) -> None:
        self._points = value

    @property
    def tracks_eff(self) -> Tensor:
        r"""
        The tracks efficiency.
        """
        if self._tracks_eff is None:
            self._tracks_eff = self.get_tracks_eff_from_hits_eff(self.hits.hits_eff)  # type: ignore
        return self._tracks_eff

    @tracks_eff.setter
    def tracks_eff(self, value: Tensor) -> None:
        self._tracks_eff = value

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
        if self._E is None:
            self._E = self.hits.E  # type: ignore
        return self._E

    @E.setter
    def E(self, value: Tensor) -> None:
        self._E = value

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
            if (self.hits.spatial_res is None) | (self._compute_angular_res is False):  # type: ignore
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
            if self._compute_angular_res:
                self._angular_res = self.angular_error.std().item()
            else:
                self._angular_res = 0.0
        return self._angular_res

    @angular_res.setter
    def angular_res(self, value: Tensor) -> None:
        self._angular_res = value

    @property
    def label(self) -> str:
        return self._label

    @label.setter
    def label(self, value: str) -> None:
        self._label = value

    @property
    def measurement_type(self) -> str:
        return self._measurement_type

    @measurement_type.setter
    def measurement_type(self, value: str) -> None:
        self._measurement_type = value


class TrackingMST(AbsSave):
    r"""
    A class for tracking muons in the context of a Muon Scattering Tomography analysis.
    """

    _theta_in: Optional[Tensor] = None  # (mu)
    _theta_out: Optional[Tensor] = None  # (mu)
    _theta_xy_in: Optional[Tensor] = None  # (2, mu)
    _theta_xy_out: Optional[Tensor] = None  # (2, mu)
    _dtheta: Optional[Tensor] = None  # (mu)
    _muon_eff: Optional[Tensor] = None  # (mu)

    _vars_to_load = ["tracks", "points", "angular_res", "E", "tracks_eff"]

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
            -  output_dir (Optional[str]): Path to a directory where to save TrackingMST attributes
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

        # Filter muon event due to detector efficiency
        print(f"{(self.muon_eff==False).sum()} muon removed due to detector efficiency")
        self._filter_muons(self.muon_eff)

    def load_attr_from_tracking(self, tracking: Tracking, tag: str) -> None:
        r"""
        Load class attributes in TrackingMST._vars_to_load from the input Tracking class.
        Attributes name are modified according to the tag as `attribute_name` + `tag`,
        so that incoming and outgoing muon features can be treated independently (except for the kinetic energy).

        Args:
            - tracking (Tracking): Instance of the Tracking class.
            - tag (str): tag to add to the attribuites name (either `_in` or `_out`)
        """

        for attr in self._vars_to_load:
            data = getattr(tracking, attr)
            if attr != "E":
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

        IMPORTANT: Change nfrom cosine to tan to avoid floating precision issues.

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

    @staticmethod
    def get_muon_eff(tracks_eff_in: Tensor, tracks_eff_out: Tensor) -> Tensor:
        """Computes muon-wise efficiency through all detector panels, based on the
        muon-wise efficiency through the set of panels before and after the object.
        Muon is detected => efficency = 1, muon not detected => efficiency = 0.

        Args:
            tracks_eff_in (Tensor): muon-wise efficiency through the set of panels before the object.
            tracks_eff_out (Tensor): muon-wise efficiency through the set of panels after the object.

        Returns:
            muon_wise_eff: muon-wise efficiency through all detector panels.
        """
        muon_wise_eff = (tracks_eff_in + tracks_eff_out) == 2
        return muon_wise_eff

    def _filter_muons(self, mask: Tensor) -> None:
        r"""
        Remove muons specified as False in `mask`.

        Arguments:
            mask: (N,) Boolean tensor. Muons with False elements will be removed.
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

    def plot_muon_features(
        self,
        figname: Optional[str] = None,
        dir: Optional[str] = None,
        save: bool = True,
    ) -> None:
        r"""
        Plot the zenith angle and energy of the reconstructed tracks.
        Args:
            figname (str): If provided, save the figure at self.output / figname.
        """
        # Set default figname
        if figname is None:
            figname = "tracks_theta_E_dtheta"

        # Set default output directory
        if dir is None:
            dir = str(self.output_dir) + "/"

        # Set default font
        matplotlib.rc("font", **font)

        fig, axs = plt.subplots(
            ncols=2, nrows=2, figsize=(2 * hist_figsize[0], 2 * hist_figsize[1])
        )
        axs = axs.ravel()

        # Fig title
        fig.suptitle(
            f"Batch of {self.n_mu} muons", fontsize=titlesize, fontweight="bold"
        )

        # Zenith angle
        axs[0].hist(
            self.theta_in.detach().cpu().numpy() * 180 / math.pi,
            bins=n_bins,
            alpha=alpha,
        )
        axs[0].axvline(
            x=self.theta_in.mean().detach().cpu().numpy() * 180 / math.pi,
            label=f"mean = {self.theta_in.mean().detach().cpu().numpy() * 180 / math.pi:.1f}",
            color="red",
        )
        axs[0].set_xlabel(r" Zenith angle $\theta$ [deg]", fontweight="bold")

        # Energy
        axs[1].hist(self.E.detach().cpu().numpy(), bins=n_bins, alpha=alpha, log=True)
        axs[1].axvline(
            x=self.E.mean().detach().cpu().numpy(),
            label=f"mean = {self.E.mean().detach().cpu().numpy():.3E}",
            color="red",
        )
        axs[1].set_xlabel(r" Energy [MeV]", fontweight="bold")

        # Scattering angle
        axs[2].hist(
            self.dtheta.detach().cpu().numpy() * 180 / math.pi,
            bins=n_bins,
            alpha=alpha,
            log=True,
        )
        axs[2].axvline(
            x=self.dtheta.mean().detach().cpu().numpy() * 180 / math.pi,
            label=f"mean = {self.dtheta.mean().detach().cpu().numpy() * 180 / math.pi:.3E}",
            color="red",
        )
        axs[2].set_xlabel(r" Scattering angle $\delta\theta$ [deg]", fontweight="bold")

        for ax in axs[:-1]:
            ax.grid(visible=True, color="grey", linestyle="--", linewidth=0.5)
            ax.set_ylabel("Frequency [a.u]", fontweight="bold")
            ax.tick_params(axis="both", labelsize=labelsize)
            ax.legend()

        axs[-1].remove()
        axs[-1] = None

        plt.tight_layout()

        if save:
            plt.savefig(dir + figname, bbox_inches="tight")

        plt.show()

    # Number of muons
    @property
    def n_mu(self) -> int:
        """The number of muons."""
        return self.dtheta.size()[0]

    # Energy
    @property
    def E(self) -> Tensor:
        r"""
        Muons kinetic energy.
        """
        return self._E

    @E.setter
    def E(self, value: Tensor) -> None:
        self._E = value

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

    @property
    def tracks_eff_in(self) -> Tensor:
        return self._tracks_eff_in

    @tracks_eff_in.setter
    def tracks_eff_in(self, value: Tensor) -> None:
        self._tracks_eff_in = value

    @property
    def tracks_eff_out(self) -> Tensor:
        return self._tracks_eff_out

    @tracks_eff_out.setter
    def tracks_eff_out(self, value: Tensor) -> None:
        self._tracks_eff_out = value

    # Muon efficiency
    @property
    def muon_eff(self) -> Tensor:
        """The muon efficiencies."""
        if self._muon_eff is None:
            self._muon_eff = self.get_muon_eff(self.tracks_eff_in, self.tracks_eff_out)
        return self._muon_eff

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
