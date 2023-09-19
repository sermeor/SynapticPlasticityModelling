import numpy as np
from numba import njit, jit
#Function that calculates g_GABA B, from GABA B input weights and connectivity weights.
@njit
def g_GABA_B_calc(a4, C, w, N, NE):
  return a4 * np.dot(C[NE:, :].T, w[NE:]) / (N - NE)
