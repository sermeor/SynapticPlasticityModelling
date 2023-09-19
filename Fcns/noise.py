import numpy as np
from numba import njit, jit
#Function of current given by Poison noise inputs (mA).
@njit
def noise(w, rate, N):
  return w * np.random.poisson(rate, N)