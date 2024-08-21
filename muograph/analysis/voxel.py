from typing import Dict, Tuple, Optional
import numpy as np

from analysis.element import Element


class Voxel:
    _n_poca: Optional[int] = None
    _dtheta_params: Optional[Dict[str, float]] = None
    _asr_params: Optional[Dict[str, float]] = None
    _bca_params: Optional[Dict[str, float]] = None

    def __init__(
        self,
        element: Element,
        pocas: np.ndarray,
        bca_scores: np.ndarray,
        asr_scores: np.ndarray,
        dtheta: np.ndarray,
        index: Tuple[int],
        lumuosity: int,
        relative_lumuosity: float,
    ) -> None:
        self.element = element
        self.pocas = pocas  # (n_poca, 3)
        self.bca_scores = bca_scores  # (n_bca_scores)
        self.asr_scores = asr_scores  # (n_asr_scores)
        self.dtheta = dtheta  # (n_poca)
        self.index = index  # (x, y, z)
        self.lumuosity = lumuosity
        self.relative_lumuosity = relative_lumuosity

    @staticmethod
    def get_n_poca(poca_scores: np.ndarray) -> int:
        return len(poca_scores)

    @staticmethod
    def get_dtheta_params(dtheta: np.ndarray) -> Dict[str, float]:
        return {"std": np.std(dtheta), "mean": np.mean(dtheta)}

    @staticmethod
    def get_asr_params(asr_scores: np.ndarray) -> Dict[str, float]:
        return {
            "std": np.std(asr_scores),
            "mean": np.mean(asr_scores),
            "q25": np.quantile(asr_scores, 0.25),
            "q50": np.quantile(asr_scores, 0.5),
            "q75": np.quantile(asr_scores, 0.75),
        }

    @staticmethod
    def get_bca_params(bca_scores: np.ndarray) -> Dict[str, float]:
        return {
            "std": np.std(bca_scores),
            "mean": np.mean(bca_scores),
            "q25": np.quantile(bca_scores, 0.25),
            "q5": np.quantile(bca_scores, 0.5),
            "q75": np.quantile(bca_scores, 0.75),
        }

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
