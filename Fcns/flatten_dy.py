import numpy as np
from numba import njit, jit
#Function to flatten dy arrays.
@njit
def flatten_dy(dy):
  return dy.flatten()