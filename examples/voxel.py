from typing import Dict, Tuple, List
import numpy as np
import pickle

class Element:

    def __init__(self, density: float, X0: float, name: str):

        self.density = density # g.cm-3
        self.X0 = X0 # cm
        self.name = name

Al = Element(density = 2.699, X0= 8.897, name = "Aluminum")
Fe = Element(density = 7.874, X0= 1.57, name = "Iron")
Pb = Element(density = 11.35, X0= 0.5612, name = "Lead")
U = Element(density = 18.95, X0 = 0.3166, name = "Uranium")
tissue = Element(density = 1.127, X0 = 0., name = "Tissue")
glass = Element(density = 2.23, X0 = 0., name = "Glass")
water = Element(density =1., X0 = 36.08, name = "Water")

class Voxel:

    _n_poca: int = None
    _dtheta_params: Dict[str, float] = None
    _asr_params: Dict[str, float] = None
    _bca_params: Dict[str, float] = None

    def __init__(self, 
                 element: Element,
                 pocas: np.ndarray, 
                 bca_scores: np.ndarray, 
                 asr_scores: np.ndarray,
                 dtheta: np.ndarray, 
                 index: Tuple[int], 
                 lumuosity: int,
                 relative_lumuosity: float) -> None:

        self.element = element
        self.pocas = pocas # (n_poca, 3)
        self.bca_scores = bca_scores # (n_bca_scores) 
        self.asr_scores = asr_scores # (n_asr_scores)
        self.dtheta = dtheta # (n_poca)
        self.index = index # (x, y, z)
        self.lumuosity = lumuosity
        self.relative_lumuosity = relative_lumuosity

    
    @staticmethod
    def get_n_poca(poca_scores) -> int:
        return len(poca_scores)
    
    @staticmethod
    def get_dtheta_params(dtheta) -> Dict[str, float]:
        return {"std": np.std(dtheta), 
                "mean":np.mean(dtheta)}
    
    @staticmethod
    def get_asr_params(asr_scores) -> Dict[str, float]:
        return {"std": np.std(asr_scores), 
                "mean": np.mean(asr_scores),
                "q25": np.quantile(asr_scores, .25),
                "q50": np.quantile(asr_scores, .5),
                "q75": np.quantile(asr_scores, .75),}
    
    @staticmethod
    def get_bca_params(bca_scores) -> Dict[str, float]:
        return {"std": np.std(bca_scores), 
                "mean": np.mean(bca_scores),
                "q25": np.quantile(bca_scores, .25),
                "q5": np.quantile(bca_scores, .5),
                "q75": np.quantile(bca_scores, .75),}
    
    @property
    def dtheta_params(self) -> Dict[str, float]:

        if self._dtheta_params is None:
            self._dtheta_params = self.get_dtheta_params(self.dtheta)
        return self._dtheta_params

    @property
    def asr_params(self) -> Dict[str, float]:

        if self._asr_params is None:
            self._asr_params = self.get_asr_params(self.asr_scores)
        return self._asr_params
    
    @property
    def bca_params(self) -> Dict[str, float]:
        if self._bca_params is None:
            self._bca_params = self.get_bca_params(self.bca_scores)
        return self._bca_params
    
    @property
    def n_poca(self) -> int:
        if self._n_poca is None:
            self._n_poca = self.get_n_poca(self.pocas)
        return self._n_poca    

class Material:

    def __init__(self, name: str, output_dir: str, voxels: List[Voxel] = None, input_file: str = None) -> None:

        self.output_dir = output_dir
        self.name = name

        if voxels is None and input_file is not None:
            
            self.input_file = input_file
            self.voxels = self.load_voxels(filename = input_file)
        
        elif voxels is not None and input_file is None:
            self.voxels = voxels
            self.save_voxels(filename = self.output_dir + self.name)

        # self.n_pocas = np.sum([vox.n_poca for vox in self.voxels])
        
    
    def save_voxels(self, filename: str) -> None:

        with open(filename + '.pickle', 'wb') as file:
            pickle.dump(self.voxels, file)

    def load_voxels(self, filename: str) -> None:

            with open(filename, 'rb') as file:
                return pickle.load(file)
