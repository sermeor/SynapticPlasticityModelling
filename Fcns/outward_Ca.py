import numpy as np
from numba import njit, jit
##Function of outward calcium rate (uM/ms).
@njit
def outward_Ca(Cai, Cai_eq):
  c = 0.05  #Rate of calcium pump buffering (ms^-1).
  return + c * (Cai - Cai_eq)