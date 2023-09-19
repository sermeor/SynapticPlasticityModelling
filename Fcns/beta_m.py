import numpy as np
from numba import njit, jit
##Function of rate constant decrease of m.
@njit
def beta_m(Vm):
  return 4.0 * np.exp(-(Vm + 65.0) / 18.0)