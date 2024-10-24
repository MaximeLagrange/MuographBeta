from typing import Dict, Optional, Tuple, Union, List
import numpy as np
from pathlib import Path
import logging
import pandas as pd
import h5py

from volume.volume import Volume
from tracking.tracking import Tracking, TrackingMST
from hits.hits import Hits
from reconstruction.binned_clustered import BCA, bca_params_type
from hits.hits import allowed_d_units

voi_dict_type = Dict[str, Union[Tuple[float, float, float], int]]
file_dict_type = Dict[str, str]
preds_dict_type = Dict[str, Dict[str, np.ndarray]]
det_dic_type = Dict[str, Union[Tuple[float, float, float], float, int]]
data_dict_type = Dict[
    str, Union[voi_dict_type, file_dict_type, preds_dict_type, det_dic_type, str]
]


class MetaData:
    """
    A class to manage metadata associated with volume-based predictions, input files, and output directories.

    The MetaData class is responsible for validating file paths, organizing predictions,
    handling Volume of Interest (VOI) metadata, and ensuring the correct distance unit is used.
    It also provides easy access to the various metadata components through the `data_dict`,
    `voi_dict`, and `file_dict` properties.

    Attributes:
    ----------
    voi : Volume
        The volume of interest (VOI) object used to compute the predictions.
    input_file : Path
        Path to the input file from which hits are read.
    output_dir : Path
        Path to the output directory where metadata is saved.
    preds_dict : Dict[str, np.ndarray]
        A dictionary of predictions where each key corresponds to a prediction identifier
        and the value is a numpy array of the prediction.
    d_unit : str, optional
        The unit of measurement for distance in the input data file, default is meters ("m").
    """

    def __init__(
        self,
        voi: Volume,
        input_file: str,
        output_dir: str,
        n_panels: int,
        spatial_res: Tuple[float, float, float],
        efficency: float,
        preds_dict: Dict[str, np.ndarray],
        d_unit: Optional[str] = "m",
    ) -> None:
        """
        Initialize the MetaData object with volume information, input/output paths, and predictions.

        This method validates the input file, output directory, and predictions provided by the user.
        It ensures that the necessary file paths exist and that the predictions are in the correct format
        (numpy arrays). The distance unit is also validated against a predefined list of allowed units.

        Args:
            voi (Volume): The Volume of Interest (VOI) object containing the voxel size,
                        dimension, and position data.
            input_file (str): The file path to the input data file (e.g., where hits are stored).
            output_dir (str): The directory path where metadata should be saved.
            preds (Dict[str, np.ndarray]): A dictionary of predictions with string keys
                                        representing prediction names and numpy arrays as the corresponding values.
            d_unit (Optional[str]): The distance unit for the input file. Must be one of the allowed
                                    units defined in `allowed_d_units`. Default is "m" for meters.

        Raises:
            FileNotFoundError: If the input file or output directory does not exist.
            TypeError: If any value in the `preds` dictionary is not a numpy array.
            ValueError: If the distance unit (`d_unit`) is not in the list of allowed distance units.
        """
        # The volume of interest
        self.voi = voi

        # Detector parameters
        self.n_panels = n_panels
        self.spatial_res = spatial_res
        self.efficiency = efficency

        # Input file path validation
        self.input_file = Path(input_file)
        if not self.input_file.exists():
            raise FileNotFoundError(
                f"Input file {self.input_file.absolute()} does not exist!"
            )

        # Output directory validation
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            raise FileNotFoundError(
                f"Output directory {self.output_dir.absolute()} does not exist!"
            )

        # Predictions validation
        self.preds_dict = preds_dict
        # Distance unit validation
        self.d_unit = d_unit
        if self.d_unit not in allowed_d_units:
            raise ValueError(
                f"The input data file must have the following distance units: {allowed_d_units}"
            )

    @staticmethod
    def save_dict_to_hdf5(data_dict: data_dict_type, output_dir: Path) -> None:
        """
        Recursively saves a dictionary to an HDF5 file.

        Args:
            dictionary (dict): The dictionary to save.
            filename (str): The name of the output HDF5 file.
        """

        def recursively_save_dict(group: h5py.Group, dict_obj: data_dict_type) -> None:
            for key, value in dict_obj.items():
                if isinstance(value, dict):
                    # Create a subgroup and recursively save the dictionary
                    subgroup = group.create_group(key)
                    recursively_save_dict(subgroup, value)  # type: ignore
                elif isinstance(value, (list, tuple)):
                    # Convert lists and tuples to NumPy arrays and save
                    group.create_dataset(key, data=np.array(value))
                elif isinstance(value, (int, float, str, np.ndarray)):
                    # Directly save int, float, str, or numpy arrays
                    group.create_dataset(key, data=value)
                elif value is None:
                    # Save None as an attribute since HDF5 can't store None natively
                    group.attrs[key] = "None"
                else:
                    raise TypeError(
                        f"Unsupported data type for key: {key}, value: {value}"
                    )

        # Ensure proper path handling
        output_file_path = output_dir / (data_dict["input_file"]["name"] + ".h5")  # type: ignore

        with h5py.File(output_file_path) as hdf_file:
            recursively_save_dict(hdf_file, data_dict)

    @staticmethod
    def load_dict_from_hdf5(filename: Path) -> data_dict_type:
        """
        Recursively loads a dictionary from an HDF5 file.

        Args:
            filename (str): The name of the input HDF5 file.

        Returns:
            dict: The loaded dictionary.
        """

        def recursively_load_dict(group: h5py.Group) -> Dict:
            result_dict = {}
            for key, item in group.items():
                if isinstance(item, h5py.Group):
                    # Recursively load subgroups (dictionaries)
                    result_dict[key] = recursively_load_dict(item)
                elif isinstance(item, h5py.Dataset):
                    # Load datasets and decode byte strings if necessary
                    data = item[()]
                    if isinstance(data, bytes):
                        result_dict[key] = data.decode("utf-8")  # type: ignore
                    else:
                        result_dict[key] = data
            # Load attributes (e.g., "None" saved as an attribute)
            for key, value in group.attrs.items():
                if isinstance(value, bytes):
                    result_dict[key] = value.decode("utf-8")  # type: ignore
                else:
                    result_dict[key] = value if value != "None" else None
            return result_dict

        with h5py.File(filename, "r") as hdf_file:
            return recursively_load_dict(hdf_file)

    def _get_file_name(self) -> str:
        """Helper to extract the file name without extension."""
        return self.input_file.stem

    def _get_file_extension(self) -> str:
        """Helper to extract the file extension."""
        return self.input_file.suffix

    @property
    def voi_dict(self) -> voi_dict_type:
        """Returns metadata related to the Volume of Interest (VOI)."""
        return {
            "voxelsize": self.voi.vox_width,
            "dimension": tuple(self.voi.dxyz.detach().cpu().numpy()),  # type: ignore
            "position": tuple(self.voi.xyz.detach().cpu().numpy()),  # type: ignore
        }

    @property
    def file_dict(self) -> file_dict_type:
        """Returns metadata related to the input file."""
        return {
            "dir": f"{self.input_file.parent.absolute()}/",
            "name": self._get_file_name(),
            "extension": self._get_file_extension(),
            "file": str(self.input_file.absolute()),  # type: ignore
            "d_unit": self.d_unit,  # type: ignore
        }

    @property
    def detector_dict(self) -> det_dic_type:
        return {
            "n_panels": self.n_panels,
            "spatial_res": self.spatial_res,
            "efficiency": self.efficiency,
        }

    @property
    def data_dict(self) -> data_dict_type:
        """Combines various metadata into a dictionary."""
        return {
            "voi": self.voi_dict,
            "input_file": self.file_dict,
            "preds": self.preds_dict,
            "output_dir": f"{self.output_dir}/",
            "detector": self.detector_dict,
        }


class PredictorBCA:
    """A class for extracting BCA voxel-wise predictions based on voxelized volume and tracking data."""

    _bca: Optional[BCA] = None
    _preds_dict: Optional[preds_dict_type] = None
    _predictions_attr: List[str] = ["n_poca_per_vox", "xyz_voxel_pred"]

    def __init__(
        self,
        voi: Volume,
        tracking_mst: TrackingMST,
        bca_params: Union[bca_params_type, List[bca_params_type]],
    ) -> None:
        """Initialize the PredictorBCA with volume, tracking data, and BCA parameters.

        Args:
            voi (Volume): The volume object containing spatial information.
            tracking_mst (TrackingMST): The instance of the TrackingMST class for tracking data.
            bca_params (Union[bca_params_type, List[bca_params_type]]): BCA parameters, which can be a single instance or a list of parameters.
        """

        # The instance of the TrackingMST class
        self.tracking_mst = tracking_mst
        # The instance of the volume class
        self.voi = voi
        # The BCA parameters
        self.bca_params = (
            bca_params if isinstance(bca_params, (list, tuple)) else [bca_params]
        )

    def get_bca_preds_as_dict(self) -> preds_dict_type:
        """Compute and return BCA predictions as a dictionary."""
        preds_dict = {}

        for attr in self._predictions_attr:
            if attr == "xyz_voxel_pred":
                preds_dict.update(self._compute_xyz_voxel_pred())
            elif attr == "n_poca_per_vox":
                preds_dict[attr] = self._compute_n_poca_per_vox()

        return preds_dict

    def _compute_xyz_voxel_pred(self) -> dict:
        """Compute and return the xyz voxel predictions."""
        results = {}
        for bca_param in self.bca_params:
            self.bca.bca_params = bca_param
            logging.info(
                f"Compute BCA predictions with parameters {self.bca.bca_params}"
            )
            results[self.bca.bca_name] = self.bca.xyz_voxel_pred.detach().cpu().numpy()
        return results

    def _compute_n_poca_per_vox(self) -> int:
        """Compute and return the number of POCA points per voxel."""
        logging.info("Compute n_poca_per_vox")
        return self.bca.n_poca_per_vox.detach().cpu().numpy()

    @property
    def bca(self) -> BCA:
        if self._bca is None:
            self._bca = BCA(voi=self.voi, tracking=self.tracking_mst, save=False)
        return self._bca

    @property
    def preds_dict(self) -> preds_dict_type:
        if self._preds_dict is None:
            self._preds_dict = self.get_bca_preds_as_dict()
        return self._preds_dict


class DataGen:
    def __init__(
        self,
        files: List[str],
        n_panels: int,
        voi: Volume,
        output_dir: str,
        bca_params: Union[bca_params_type, List[bca_params_type]],
        use_energy: bool = True,
        d_unit: str = "m",
        spatial_res: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        efficiency: float = 1.0,
    ) -> None:
        # Detector parameters
        self.spatial_res = spatial_res  # in mm
        self.efficiency = efficiency

        # The number of detection planes
        self.n_panels = n_panels
        if not isinstance(n_panels, int) | (n_panels < 0) | (n_panels % 2 != 0):
            raise ValueError(
                f"The number of panels {n_panels} must be a positive and even integer"
            )
        self.upper_panel_indices = tuple([i for i in range(int(self.n_panels / 2))])
        self.lower_panel_indices = tuple(
            [i for i in range(int(self.n_panels / 2), self.n_panels)]
        )

        # Include energy
        self.use_energy = use_energy

        # Volume of interest
        self.voi = voi

        # The hits file to analyse
        self.files = files
        if not self.check_files_exist(self.files):
            raise FileNotFoundError(
                f"File(s) {[str(Path(file).absolute()) for file in self.files if not Path(file).exists()]} do(es) not exist!"
            )

        # Distance unit validation
        self.d_unit = d_unit
        if self.d_unit not in allowed_d_units:
            raise ValueError(
                f"The input data file must have the following distance units: {allowed_d_units}"
            )

        # Directory where to save metadata
        self.output_dir = Path(output_dir)
        if not self.output_dir.exists():
            raise FileNotFoundError(
                f"Output directory {self.output_dir.absolute()} does not exist!"
            )

        # Parameters of the BCA
        self.bca_params = (
            bca_params if isinstance(bca_params, (list, tuple)) else [bca_params]
        )

    def generate_metadata_from_files(self) -> None:
        for file in self.files:
            print(f"Analysing file {Path(file).name}")
            self.generate_metadata_from_file(hit_file_npy=Path(file))

    def generate_metadata_from_file(self, hit_file_npy: Path) -> None:
        tracking_mst = self.get_tracking_mst_df_from_B2G4_npy(hit_file_npy=hit_file_npy)

        predictor = PredictorBCA(
            tracking_mst=tracking_mst, voi=self.voi, bca_params=self.bca_params
        )

        metadata = MetaData(
            voi=self.voi,
            input_file=str(hit_file_npy),
            output_dir=str(self.output_dir),
            preds_dict=predictor.preds_dict,
            n_panels=self.n_panels,
            spatial_res=self.spatial_res,
            efficency=self.efficiency,
            d_unit=self.d_unit,
        )

        metadata.save_dict_to_hdf5(metadata.data_dict, output_dir=self.output_dir)

    def get_tracking_mst_df_from_B2G4_npy(
        self, hit_file_npy: Path
    ) -> Tuple[Hits, Hits]:
        # Load the hits as numpy array
        hits_npy = np.load(str(hit_file_npy))

        # Get hits as DataFrame
        hits_df = DataGen.convert_B2G4_hits_npy_to_df(
            hits_npy=hits_npy, n_panels=self.n_panels, use_energy=self.use_energy
        )

        # Get the upper and lower Hits instances
        hits = self.get_hits_from_df(hits_df=hits_df)

        # Get tracking MST from upper and lower hits instances
        tracking_mst = self.get_tracking_mst_from_hits(hits=hits)

        return tracking_mst

    def get_hits_from_df(self, hits_df: pd.DataFrame) -> Tuple[Hits, Hits]:
        hits_in = Hits(
            plane_labels=self.upper_panel_indices,
            input_unit=self.d_unit,
            df=hits_df.iloc[:50000],
            spatial_res=self.spatial_res,
            efficiency=self.efficiency,
        )

        hits_out = Hits(
            plane_labels=self.lower_panel_indices,
            input_unit=self.d_unit,
            df=hits_df.iloc[:50000],
            spatial_res=self.spatial_res,
            efficiency=self.efficiency,
        )

        return hits_in, hits_out

    @staticmethod
    def convert_B2G4_hits_npy_to_df(
        hits_npy: np.ndarray, n_panels: int, use_energy: bool
    ) -> pd.DataFrame:
        """
        Converts a numpy array of B2G4 hits data into a pandas DataFrame.

        Args:
            hits_npy (np.ndarray): A 2D numpy array where each row represents an event,
                                and columns contain hit data. The number of columns
                                must match `n_panels * 3`, with an optional energy
                                column if `use_energy` is True.
            n_panels (int): The number of panels contributing to the hit data. Each
                            panel adds 3 columns (X, Y, Z) to the DataFrame.
            use_energy (bool): If True, an additional column for energy ("E") will be
                            included in the DataFrame.

        Returns:
            pd.DataFrame: A DataFrame with the hit data. Contains `n_panels * 3` columns
                        for panel coordinates (X, Y, Z for each panel), and an optional
                        "E" column for energy if `use_energy` is True.

        Raises:
            ValueError: If the shape of `hits_npy` does not match the expected shape
                        based on `n_panels` and the `use_energy` flag.
        """

        # Determine the number of variables based on the presence of energy data
        n_vars = n_panels * 3 + (1 if use_energy else 0)

        # Validate shape of hits_npy
        if hits_npy.shape[1] != n_vars:
            raise ValueError(
                f"Hits numpy array must have shape (n_events, {n_vars}), but has shape (n_events, {hits_npy.shape[1]})"
            )

        # Generate headers for energy (if applicable) and panel coordinates
        headers = (["E"] if use_energy else []) + [
            f"{axis}{i}" for i in range(n_panels) for axis in "XYZ"
        ]

        # Create the DataFrame
        return pd.DataFrame(hits_npy, columns=headers)

    @staticmethod
    def get_tracking_mst_from_hits(hits: Tuple[Hits, Hits]) -> TrackingMST:
        tracks_in = Tracking(
            hits=hits[0],
            label="above",
            save=False,
            compute_angular_res=False,
        )

        tracks_out = Tracking(
            hits=hits[1],
            label="below",
            save=False,
            compute_angular_res=False,
        )

        return TrackingMST(trackings=(tracks_in, tracks_out))

    @staticmethod
    def check_files_exist(files: List[str]) -> bool:
        return all([Path(file).exists() for file in files])
