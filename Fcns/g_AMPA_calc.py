import numpy as np
from numba import njit, jit
#Function that calculates g_AMPA, from AMPA input weights and connectivity weights.
@njit
def g_AMPA_calc(a1, C, w, N, NE):
  return a1 * np.dot(C[:NE, :].T, w[:NE]) / NE
