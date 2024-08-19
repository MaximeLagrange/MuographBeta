from pathlib import Path
from typing import List, Optional
from torch import Tensor
import numpy as np
import h5py

muograph_path = str(Path(__file__).parent.parent.parent)
default_output_dir = muograph_path + "/output/"


class AbsSave:
    """
    A base class for managing directory creation and handling class attributes
    saving/loading.
    """

    def __init__(self, output_dir: Optional[str] = None) -> None:
        """
        Initializes the AbsSave object and ensures the output directory exists.

        Args:
            output_dir (str): The path to the directory where output files will be saved.
        """
        if output_dir is None:
            output_dir = default_output_dir

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
                value = getattr(self, attr)
                if isinstance(value, Tensor):
                    f.create_dataset(attr, data=value.numpy())
                elif isinstance(value, (np.ndarray, float)):
                    f.create_dataset(attr, data=value)

        f.close()
        print("Class attributes saved at {}".format(directory / filename))

    def load_attr(self, attributes: List[str], filename: str, tag: str = None) -> None:
        r"""
        Loads class attributes from hdf5 file.

        Args:
            attributes (List[str]): the list of the class attributes to save.
            directory (Path): The path to the directory to be created.
            filename (str): the name of the file where to save the attributes.
            tag (str): tag to add to the attributes to that it matches
            the parent class attribute names. e.g: TrackingMST differentiate
            incoming and outgoing track attributes.
        """

        with h5py.File(filename, "r") as f:
            for attr in attributes:
                data = f[attr]
                if tag is not None:
                    attr += tag
                if data.ndim == 0:
                    setattr(self, attr, data[()])
                elif type(data[:]) is np.ndarray:
                    setattr(self, attr, Tensor(data[:]))
        f.close()
        print("\nTracking attributes loaded from {}".format(filename))
