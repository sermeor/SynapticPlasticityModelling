import numpy as np
from numba import njit, jit

@njit
def alpha_c(Vm):
  return 0.01 * (Vm + 20.0) / (1.0 - np.exp(-(Vm + 20.0) / 2.5))
