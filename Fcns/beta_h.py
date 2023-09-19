import numpy as np
from numba import njit, jit
##Function of rate constant decrease of h.
@njit
def beta_h(Vm):
  return 1.0 / (1.0 + np.exp(-(Vm + 35.0) / 10.0))