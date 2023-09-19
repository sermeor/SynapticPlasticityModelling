import numpy as np
from numba import njit, jit

@njit
def beta_c(Vm):
  return 0.125 * np.exp(-(Vm + 50.0) / 80.0)
