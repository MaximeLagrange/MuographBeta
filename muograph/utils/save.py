from pathlib import Path
from typing import List
from torch import Tensor
import numpy as np
import h5py

name = Path(__file__)


class AbsSave:
    """
    A base class for managing directory creation and handling class attributes
    saving/loading.
    """

    def __init__(self, output_dir: str) -> None:
        """
        Initializes the AbsSave object and ensures the output directory exists.

        Args:
            output_dir (str): The path to the directory where output files will be saved.
        """
        self.output_dir = Path(output_dir)
        self.create_directory(self.output_dir)

    @staticmethod
    def create_directory(directory: Path) -> None:
        r"""
        Creates a directory at the specified path if it does not already exist.

        Args:
            directory (Path): The path to the directory to be created.

        Notes:
            If the directory already exists, this method will not raise an exception.
            It will simply indicate that the directory already exists.
        """
        print(
            f"\n{directory} directory {'created' if not directory.exists() else 'already exists'}"
        )
        directory.mkdir(parents=True, exist_ok=True)

    def save_attr(self, attributes: List[str], directory: Path, filename: str) -> None:
        r"""
        Saves class attributes to hdf5 file.

        Args:
            attributes (List[str]): The list of the class attributes to save.
            directory (Path): The path to the directory to be created.
            filename (str): the name of the file where to save the attributes.
        """

        filename += ".hdf5"
        with h5py.File(directory / filename, "w") as f:
            for attr in attributes:
                if type(getattr(self, attr)) is Tensor:
                    f.create_dataset(attr, data=getattr(self, attr).numpy())

        f.close()
        print("Class attributes saved at {}".format(directory / filename))

    def load_attr(self, attributes: List[str], directory: Path, filename: str) -> None:
        r"""
        Loads class attributes from hdf5 file.

        Args:
            attributes (List[str]): the list of the class attributes to save.
            directory (Path): The path to the directory to be created.
            filename (str): the name of the file where to save the attributes.
        """

        with h5py.File(directory / filename, "r") as f:
            for attr in attributes:
                data = f[attr]
                if type(data[:]) is np.ndarray:
                    setattr(self, attr, Tensor(data[:]))
        f.close()
        print("\nTracking instance loaded from {}".format(directory / filename))
