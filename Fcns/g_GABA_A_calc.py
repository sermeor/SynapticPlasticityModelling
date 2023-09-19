import numpy as np
from numba import njit, jit
#Function that calculates g_GABA_A, from GABA A input weights and connectivity weights.
@njit
def g_GABA_A_calc(a2, C, w, N, NE):
  return a2 * np.dot(C[NE:, :].T, w[NE:]) / (N - NE)
