import numpy as np
from numba import njit, jit
##Function of rate constant increase of n.
@njit
def alpha_n(Vm):
  return 0.01 * (Vm + 55.0) / (1.0 - np.exp(-(Vm + 55.0) / 10.0))