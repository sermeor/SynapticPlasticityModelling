import numpy as np
from numba import njit, jit
#Function that flattens the input array to comp_model.
@njit
def flatten_y(y, N, var_number):
  return y.reshape(var_number + N, N)