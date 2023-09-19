import numpy as np
from numba import njit, jit
##Function of fraction of BDNF bound to TrkB receptor (sigmoid).
#bdnf: levels of BDNF protein in the extracellular space.
@njit
def TrkB(bdnf):
  s = 10 #Slope of sigmoid.
  return 1 / (1 + np.exp(-s*(bdnf - 0.5)))