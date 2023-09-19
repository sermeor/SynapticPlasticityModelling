import numpy as np
from numba import njit, jit
##Function to determine if action potential is backpropagating.
@njit
def bAP(Vm, act):
  return np.logical_or(Vm > 35, act > 0.7)