import numpy as np
from numba import njit, jit
##Function that sets the random seed on the compiled side.
@njit
def set_seed(seed):
  np.random.seed(seed)
  return