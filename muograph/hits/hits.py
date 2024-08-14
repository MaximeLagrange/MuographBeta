from pathlib import Path
import pandas as pd
import torch
from torch import Tensor
from typing import Optional


class Hits:
    r"""
    A class to handle and process muon hit data from a CSV file.
    """

    _hits = None
    _E = None

    def __init__(
        self, csv_filename: Optional[Path] = None, df: Optional[pd.DataFrame] = None
    ) -> None:
        r"""
        Initializes the Hits object with the path to the CSV file or a pd.DataFrame

        Args:
            csv_filename (Path): The path to the CSV file containing
            hit and energy data.
            df (pd.DataFrame): The CSV file containing
            hit and energy data.
        """
        if csv_filename is not None and df is not None:
            raise ValueError("Provide either csv_filename or df, not both.")

        if csv_filename is not None:
            self._df = self.get_data_frame_from_csv(csv_filename)
        elif df is not None:
            self._df = df
        else:
            raise ValueError("Either csv_filename or df must be provided.")

    @staticmethod
    def get_data_frame_from_csv(csv_filename: Path) -> pd.DataFrame:
        r"""
        Reads a CSV file into a DataFrame.

        Args:
            csv_filename (Path): The path to the CSV file containing
            hit and energy data.

        Returns:
            pd.DataFrame: The DataFrame containing the data from the CSV file.
        """
        if not csv_filename.exists():
            raise FileNotFoundError(f"The file {csv_filename} does not exist.")

        return pd.read_csv(csv_filename)

    @staticmethod
    def get_hits_from_df(df: pd.DataFrame) -> Tensor:
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
            raise KeyError("Column 'E' not found in the DataFrame.")
        return torch.tensor(df["E"].values)

    @property
    def E(self) -> Tensor:
        r"""
        Muon's energy as a Tensor.
        """
        if self._E is None:
            self._E = self.get_energy_from_df(self._df)
        return self._E

    @property
    def hits(self) -> Tensor:
        r"""
        Hits data as a Tensor.
        """
        if self._hits is None:
            self._hits = self.get_hits_from_df(self._df)
        return self._hits
