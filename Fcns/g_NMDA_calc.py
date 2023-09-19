import numpy as np
from numba import njit, jit
#Function that calculates g_NMDA, from NMDA input weights and connectivity weights.
@njit
def g_NMDA_calc(a3, C, w, N, NE):
  return a3 * np.dot(C[:NE, :].T, w[:NE]) / NE
