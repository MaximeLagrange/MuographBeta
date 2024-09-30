import numpy as np
from numba import int32, float32, float64
import numba
Tuple_n = numba.types.Tuple
from typing import Tuple

from numba.experimental import jitclass
from numpy import ndarray
from typing import Optional

spec = [
    # ('gen_hits', float32[:, :]),
    ('spatial_res', Tuple((float64, float64, float64)))
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),        # an array field

]

@jitclass(spec)
class Hits(object):

    def __init__(self, value, spatial_res: Tuple[float, float, float]):
        self.spatial_res = spatial_res
        # self.csv_filename = csv_filename
        self.value = value
        self.array = np.zeros(value, dtype=np.float32)

    @property
    def size(self):
        return self.array.size

    def increment(self, val):
        for i in range(self.size):
            self.array[i] += val
        return self.array

    @staticmethod
    def add(x, y):
        return x + y
    
    @property
    def gen_hits(self) -> ndarray:
        return np.zeros((self.value, self.value), dtype = np.float32)


