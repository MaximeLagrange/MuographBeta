import torch
from torch import Tensor
from typing import Tuple, Optional
import numpy as np
import math

from utils.save import AbsSave
from hits.hits import Hits


class Tracking:
    r"""
    A class for tracking muons based on hits data.
    """
    _tracks = None  # (mu, 3)
    _points = None  # (mu, 3)
    _theta = None  # (mu)
    _theta_xy = None  # (mu)
    _angular_error = None  # (mu)
    _angular_res = None

    def __init__(
        self,
        hits: Hits,
    ) -> None:
        r"""
        Initializes the Tracking object.

        Args:
            hits (Hits): An instance of the Hits class.
        """
        self.hits = hits

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
        tracks_tensor = torch.tensor(tracks_np)
        points_tensor = torch.tensor(points_np)

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
    def __init__(
        self,
        tracking_in: Tracking,
        tracking_out: Tracking,
        output_dir: Optional[str] = None,
    ) -> None:
        self.tracking_in = tracking_in
        self.tracking_out = tracking_out
