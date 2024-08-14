from pathlib import Path
import pandas as pd
import torch
from torch import Tensor


class Hits:
    _hits = None
    _E = None

    def __init__(self, csv_filename: Path) -> None:
        self.csv_filename = csv_filename
        self._df = self.get_data_frame_from_csv(csv_filename)

    @staticmethod
    def get_data_frame_from_csv(csv_filename: Path) -> pd.DataFrame:
        if not csv_filename.exists():
            raise FileNotFoundError(f"The file {csv_filename} does not exist.")

        return pd.read_csv(csv_filename)

    @staticmethod
    def get_hits_from_df(df: pd.DataFrame) -> Tensor:
        r"""
        Method to get hits DataFrame as a Tensor.

        IMPORTANT:
            The DataFrame must have the following columns:
            "X0, Y0, Z0, ..., Xi, Yi, Zi", where Xi is the muon hit x position on plane i.

        Args:
            df (pd.DataFrame): DataFrame where the hits are saved.

        Returns:
            hits (Tensor): Hits, with size (3, n_plane, n_mu)
        """
        # Extract plane count and validate columns
        planes = [col for col in df.columns if col.startswith("X")]
        n_plane = len(planes)

        if not planes or len(planes) == 0:
            raise KeyError("No columns starting with 'X' found in the DataFrame.")

        hits = torch.zeros((3, n_plane, len(df)))

        for plane in range(n_plane):
            x_col = f"X{plane}"
            y_col = f"Y{plane}"
            z_col = f"Z{plane}"

            if x_col not in df or y_col not in df or z_col not in df:
                raise KeyError(
                    f"Missing columns for plane {plane}: {x_col}, {y_col}, {z_col}"
                )

            hits[0, plane, :] = torch.tensor(df[x_col].values)
            hits[1, plane, :] = torch.tensor(df[y_col].values)
            hits[2, plane, :] = torch.tensor(df[z_col].values)

        return hits

    @staticmethod
    def get_energy_from_df(df: pd.DataFrame) -> Tensor:
        r"""
        Method to get hits DataFrame as a Tensor.

        IMPORTANT:
            The DataFrame must have the following column:
            "E"

        Args:
            df (pd.DataFrame): DataFrame where the hits and muons energy are saved.

        Returns:
            E (Tensor): Muons energy, with size (n_mu)
        """
        if "E" not in df:
            raise KeyError("Column 'E' not found in the DataFrame.")
        return torch.tensor(df["E"].values)

    @property
    def E(self) -> Tensor:
        if self._E is None:
            self._E = self.get_energy_from_df(self._df)
        return self._E

    @property
    def hits(self) -> Tensor:
        if self._hits is None:
            self._hits = self.get_hits_from_df(self._df)
        return self._hits
