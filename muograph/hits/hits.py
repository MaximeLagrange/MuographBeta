from pathlib import Path
import pandas as pd
import torch
from torch import Tensor
from typing import Optional, Tuple
import matplotlib
import matplotlib.pyplot as plt

from plotting.plotting import get_n_bins_xy_from_xy_span
from plotting.params import (
    font,
    d_unit,
    n_bins_2D,
    hist_figsize,
    labelsize,
)


class Hits:
    r"""
    A class to handle and process muon hit data from a CSV file.
    """
    # Muon hits
    _gen_hits = None  # (3, n_plane, mu)
    _reco_hits = None  # (3, n_plane, mu)

    # Muon energy
    _E = None  # (mu)

    def __init__(
        self,
        plane_labels: Optional[Tuple[int, ...]] = None,
        csv_filename: Optional[str] = None,
        df: Optional[pd.DataFrame] = None,
        spatial_res: Optional[Tensor] = None,
        energy_range: Optional[Tuple[float, float]] = None,
        efficiency: float = 1.0,
    ) -> None:
        r"""
        Initializes the Hits object with the path to the CSV file or a pd.DataFrame.

        Args:
            csv_filename (str): The path to the CSV file containing
            hit and energy data.
            df (pd.DataFrame): The CSV file containing
            hit and energy data.
        """
        # Detector panel parameters
        self.spatial_res = spatial_res  # in `d_unit``
        self.efficiency = efficiency  # in %

        # Energy range
        self.energy_range = energy_range

        # Load or create hits DataFrame
        if csv_filename is not None and df is not None:
            raise ValueError("Provide either csv_filename or df, not both.")

        if csv_filename is not None:
            self._df = self.get_data_frame_from_csv(csv_filename)
        elif df is not None:
            self._df = df
        else:
            raise ValueError("Either csv_filename or df must be provided.")

        # Panels label
        self.plane_labels = (
            plane_labels
            if plane_labels is not None
            else self.get_panels_labels_from_df(self._df)
        )

        # Filter events with E out of energy_range
        if self.energy_range is not None:
            energy_mask = (self.E > self.energy_range[0]) & (
                self.E < self.energy_range[-1]
            )
            self._filter_events(energy_mask)

        # detector efficiency
        if (self.efficiency > 0.0) & (self.efficiency <= 1.0):
            self.total_efficiency = self.efficiency**self.n_panels
        else:
            raise ValueError("Panels efficiency must be in [0., 1.].")

    @staticmethod
    def get_data_frame_from_csv(csv_filename: str) -> pd.DataFrame:
        r"""
        Reads a CSV file into a DataFrame.

        Args:
            csv_filename (Path): The path to the CSV file containing
            hit and energy data.

        Returns:
            pd.DataFrame: The DataFrame containing the data from the CSV file.
        """
        if not Path(csv_filename).exists():
            raise FileNotFoundError(f"The file {csv_filename} does not exist.")

        return pd.read_csv(csv_filename)

    @staticmethod
    def get_hits_from_df(
        df: pd.DataFrame, plane_labels: Tuple[int, ...] = None
    ) -> Tensor:
        r"""
        Extracts hits data from a DataFrame and returns it as a Tensor.

        IMPORTANT:
            The DataFrame must have the following columns:
            "X0, Y0, Z0, ..., Xi, Yi, Zi", where Xi is the muon hit x position on plane i.

        Args:
            df (pd.DataFrame): DataFrame containing the hit data.

        Returns:
            hits (Tensor): Hits, with size (3, n_plane, n_mu)
        """
        # Extract plane count and validate columns

        n_plane = len(plane_labels)  # type: ignore
        hits = torch.zeros((3, n_plane, len(df)))

        for i, plane in enumerate(plane_labels):  # type: ignore
            x_col = f"X{plane}"
            y_col = f"Y{plane}"
            z_col = f"Z{plane}"

            if x_col not in df or y_col not in df or z_col not in df:
                raise KeyError(
                    f"Missing columns for plane {plane}: {x_col}, {y_col}, {z_col}"
                )

            hits[0, i, :] = torch.tensor(df[x_col].values)
            hits[1, i, :] = torch.tensor(df[y_col].values)
            hits[2, i, :] = torch.tensor(df[z_col].values)

        return hits

    @staticmethod
    def get_energy_from_df(df: pd.DataFrame) -> Tensor:
        r"""
        Extracts energy data from a DataFrame and returns it as a Tensor.

        IMPORTANT:
            The DataFrame must have the following column:
            "E"

        Args:
            df (pd.DataFrame): DataFrame where the hits and muons energy are saved.

        Returns:
            E (Tensor): Muons energy, with size (n_mu)
        """
        if "E" not in df:
            raise KeyError(
                "Column 'E' not found in the DataFrame. Muon energy set to 0."
            )
        return torch.tensor(df["E"].values)

    @staticmethod
    def get_panels_labels_from_df(df: pd.DataFrame) -> Tuple[int, ...]:
        r"""
        Get the labels of ALL detector panels from the csv file.

        IMPORTANT:
            The DataFrame must have the following column 'X':

        Args:
            df (pd.DataFrame): DataFrame where the hits and muons energy are saved.

        Returns:
            plane_labels (Tuple[int, ...]): The labels of the detector panels.
        """

        planes = [col for col in df.columns if col.startswith("X")]
        plane_labels = tuple([int(s[1:]) for s in planes])

        return plane_labels

    @staticmethod
    def get_reco_hits_from_gen_hits(gen_hits: Tensor, spatial_res: Tensor) -> Tensor:
        r"""
        Smear the gen_hits position using a Normal distribution centered at 0,
        and with standard deviation equal to the spatial resolution along a given dimension

        Args:
            gen_hits (Tensor): The generated level hits, with size (3, n_plane, mu).
            spatial_res (Tensor): The spatial resolution along x,y,z with size (3).

        Returns:
            reco_hits (Tensor): The reconstructed hits, with size (3, n_plane, mu)
        """
        reco_hits = torch.ones_like(gen_hits) * gen_hits

        for i in range(spatial_res.size()[0]):
            if spatial_res[i] != 0.0:
                reco_hits[i] += torch.normal(
                    mean=0.0, std=torch.ones_like(reco_hits[i]) * spatial_res[i]
                )

        return reco_hits

    def _filter_events(self, mask: Tensor) -> None:
        r"""
        Remove muons specified as False in `mask`.

        Args:
            mask: (N,) Boolean tensor. Muons with False elements will be removed.
        """

        self.reco_hits = self.reco_hits[:, :, mask]
        self.gen_hits = self.gen_hits[:, :, mask]
        self.E = self.E[mask]

    def plot_hits(
        self,
        plane_label: int = 0,
        reco_hits: bool = True,
        n_bins: int = n_bins_2D,
        filename: Optional[str] = None,
    ) -> None:
        # Set default font
        matplotlib.rc("font", **font)

        # Create figure
        fig, ax = plt.subplots(figsize=hist_figsize)

        # Get true hits or real hits
        hits = self.reco_hits if reco_hits is True else self.gen_hits

        # The span of the detector in x and y
        dx = (hits[0, plane_label].max() - hits[0, plane_label].min()).item()
        dy = (hits[1, plane_label].max() - hits[1, plane_label].min()).item()

        # Get the number of bins as function of the xy ratio
        bins_x, bins_y, pixel_size = get_n_bins_xy_from_xy_span(
            dx=dx, dy=dy, n_bins=n_bins
        )

        # Plot hits as 2D histogram
        h = ax.hist2d(
            hits[0, plane_label],
            hits[1, plane_label],
            bins=(bins_x, bins_y),
        )

        ax.set_aspect("equal")

        # Set axis labels
        ax.set_xlabel(f"x [{d_unit}]", fontweight="bold")
        ax.set_ylabel(f"y [{d_unit}]", fontweight="bold")
        ax.tick_params(axis="both", labelsize=labelsize)

        # Set figure title
        fig.suptitle(
            f"Muon hits on plane {plane_label} \nat z = {hits[2,plane_label,0]:.0f} [{d_unit}]",
            fontweight="bold",
            y=1,
        )

        # Add colorbar
        cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
        fig.colorbar(
            h[3], cax=cbar_ax, label=f"# hits / {pixel_size**2:.0f} {d_unit}$^2$"
        )

        # Save plot
        if filename is not None:
            plt.savefig(filename, bbox_inches="tight")
        plt.show()

    @property
    def n_panels(self) -> int:
        return self.gen_hits.size()[1]

    @property
    def E(self) -> Tensor:
        r"""
        Muon's energy as a Tensor. If is not provided in the input csv/DataFrame,
        it is automatically set to zero.
        """
        if self._E is None:
            try:
                self._E = self.get_energy_from_df(self._df)
            except Exception as e:
                print(f"An error occurred: {e}")
                self._E = torch.zeros(self.reco_hits.size()[-1])  # Setting _E to zero
        return self._E

    @E.setter
    def E(self, value: Tensor) -> None:
        self._E = value

    @property
    def gen_hits(self) -> Tensor:
        r"""
        Hits data as a Tensor with size (3, n_plane, mu).
        """
        if self._gen_hits is None:
            self._gen_hits = self.get_hits_from_df(self._df, self.plane_labels)
        return self._gen_hits

    @gen_hits.setter
    def gen_hits(self, value: Tensor) -> None:
        self._gen_hits = value

    @property
    def reco_hits(self) -> Tensor:
        r"""
        Reconstructed hits data as Tensor with size (3, n_plane, mu).
        """
        if self.spatial_res is None:
            return self.gen_hits
        elif self._reco_hits is None:
            self._reco_hits = self.get_reco_hits_from_gen_hits(
                gen_hits=self.gen_hits, spatial_res=self.spatial_res
            )
        return self._reco_hits

    @reco_hits.setter
    def reco_hits(self, value: Tensor) -> None:
        self._reco_hits = value
