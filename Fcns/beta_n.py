import numpy as np
from numba import njit, jit
##Function of rate constant decrease of n.
@njit
def beta_n(Vm):
  return 0.125 * np.exp(-(Vm + 65) / 80.0)